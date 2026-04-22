## 用 CUDA VMM 干掉 CUDA Graph 的静态显存绑定

## 1 引言

### 1.1 CUDA Graph：高性能推理的必经之路

做过 LLM 推理优化的人多半都碰过 CUDA Graph。它的原理不复杂：把一段 GPU 工作录制成一张有向无环图（DAG），replay 时跳过所有 CPU 端的 launch 开销，kernel 接近背靠背地执行。对于 decode 阶段这类 kernel 数量多但每个 kernel 计算量小的场景，CUDA Graph 可以把 CPU launch overhead 从几百微秒压到个位数微秒。

但这份性能不是白拿的。CUDA Graph 有一个核心约束：**replay 时所有 kernel 看到的 GPU 内存地址必须和 capture 时完全一致。** 这意味着 capture 时 tensor A 分配在 `0x7f0000000000`，replay 时它还得在 `0x7f0000000000`。如果 allocator 在两次 replay 之间把这块内存回收了再分配给别人，graph replay 就会读写到错误的数据——轻则结果乱码，重则 CUDA error。

PyTorch 的默认做法是 "pool isolation"：为每个 graph 预留一个 memory pool，capture 期间的分配全在这个 pool 里完成，replay 时保证这些地址不被其他人占用。代价是这些内存在 graph 生命周期内是**常驻的**——哪怕你有 10 个 graph 各自只需要 500MB，也得常驻 5GB 显存。在 batch size 多变、graph 变体多的推理服务里，这个代价相当可观。

### 1.2 CUDA VMM：虚拟地址与物理内存的解耦

CUDA 从 10.2 开始引入了 Virtual Memory Management (VMM) API。传统的 `cudaMalloc` 返回的指针是虚拟地址和物理内存绑死的——你没有任何能力单独操控物理页。VMM 把这两件事拆开了：

- `cuMemAddressReserve` — 预留一段虚拟地址空间，此时不占任何物理显存
- `cuMemCreate` — 创建一个物理内存 handle（`CUmemGenericAllocationHandle`），实际占用显存
- `cuMemMap` — 把物理 handle 贴到某段虚拟地址上
- `cuMemUnmap` + `cuMemMap` — 同一虚拟地址，换一组物理 handle

一句话总结：**虚拟地址可以是稳定的，物理内存可以在背后自由来去。**

### 1.3 把两者结合：消除 CUDA Graph 的静态显存需求

如果我们用 VMM 来管理所有 GPU 内存分配，那么 CUDA Graph capture 时录制的虚拟地址可以在整个进程生命周期内保持不变。replay 前只需要把正确的物理内存映射到这些虚拟地址上，kernel 看到的地址没变，数据却是新的。graph 不用的时候，物理内存可以安全地 unmap 掉让给别人。

这就是 zero-cuda-graph（zcg）的核心思路：**构建一个完全基于 VMM API 的 PyTorch allocator，让 CUDA Graph 不再需要常驻专属显存。**

---

## 2 设计：如何将所有张量替换为 VMM 内存

### 2.1 Allocator 替换机制

PyTorch 通过 `c10::cuda::CUDACachingAllocator` 管理所有 CUDA 内存。这个组件有一个全局原子指针 `allocator`，通过 `changeCurrentAllocator()` 可以在**第一次 CUDA 分配之前**一次性替换。zcg 正是利用这个接口：`import zcg` 时自动调用注册函数，把默认 allocator 替换为我们的 `VMMAllocator`。替换之后，所有 `torch.empty()`、`torch.zeros()`、autograd 中间变量的分配和释放全部走 VMM 路径——应用代码无需任何改动。

`VMMAllocator` 继承自 `c10::cuda::CUDACachingAllocator::CUDAAllocator`，实现了完整的接口集：`allocate()`、`raw_alloc_with_stream()`、`recordStream()`、`emptyCache()`、`getDeviceStats()`、pool isolation 三件套（`beginAllocateToPool` / `endAllocateToPool` / `releasePool`）等。PyTorch 上层完全无感知底层已经从 `cudaMalloc` 换成了 VMM。

### 2.2 双通道分配

深度学习工作负载中的 tensor 大小呈明显的双峰分布：大量 1KB-10MB 的小 tensor（激活值、临时缓冲区、梯度碎片）和少量 16MB-数百 MB 的大 tensor（权重、KV Cache、大型激活）。zcg 用一个可配置的阈值（默认 16MB）将分配请求路由到两条完全独立的路径。

**SmallAllocator**（< 16MB）采用预留大段连续 VA + 按需 2MB 物理块映射的设计。启动时一次性 `cuMemAddressReserve` 32GB 虚拟地址空间（不占显存），内部维护一个 address-ordered best-fit 空闲链表做 sub-allocation。当分配触及未映射的 2MB 页面时才向 `PhysicalMemoryManager` 申请物理块并 `cuMemMap`。释放时做相邻空闲区间的 coalescing 合并。这个设计的好处是地址空间紧凑连续，locality 好，同时物理内存是弹性的——只有真正被用到的页面才会有物理后端。

**LargeAllocator**（>= 16MB）每次分配独立 reserve 一段 VA，然后从 5 个档位（16/32/64/128/256 MB）的物理块池中通过贪心算法拼接出所需大小。例如 769MB 会组合 3×256MB + 1×16MB。释放时 `cuMemUnmap` 所有物理块归还池中，VA 段释放。

两个池的物理块严格隔离。大块永远不会被拆碎给小分配用——如果一个 256MB 块被切出 1MB 给小 tensor，剩下的 255MB 就变成了难以复用的碎片。

### 2.3 物理块生命周期管理

`PhysicalMemoryManager` 是所有物理内存的唯一入口和出口。小块和大块各有独立的 free list。分配时优先从 free list 取（命中即可复用，跳过 `cuMemCreate` 开销）；释放时物理块回到 free list 而不是立即 `cuMemRelease`——这是第一层惰性：**物理块被 sub-allocator 归还后不会真正释放回 driver，而是留在 free list 里等待复用。**

当 GPU 显存真正不够时（OOM），allocator 才会释放这些缓存的物理块。这里 zcg 实现了一个三级 OOM 回收策略：

1. **选择性 purge**：`purge(needed_size)` 从 free list 中按大小降序释放物理块，释放到够用就停。优先释放大块，因为每次 `cuMemRelease` 都有 driver 调用开销，释放少量大块比释放大量小块效率高。
2. **全量 trim**：如果选择性 purge 后重试仍然 OOM，再做一轮 deferred free 刷新和 sub-allocator cache 清空，然后 `trim()` 释放 free list 中所有剩余物理块。
3. **真 OOM**：如果全量释放后还是分配不出来，才抛出 OOM 异常。

这个设计的核心想法是：**大部分 OOM 只是临时的内存压力波峰，释放一小部分缓存就能度过。保留其余缓存可以显著减少后续的 `cuMemCreate` 调用。**

### 2.4 Stream 感知与 Event-based 延迟回收

PyTorch 的 `recordStream()` 告诉 allocator "这块内存在某个 stream 上仍有 in-flight 操作"。free 时不能立即回收。zcg 的方案是在每个 recorded stream 上 record 一个轻量级 CUDA event（`cudaEventDisableTiming`，没有 timing 开销），把 `{ptr, meta, events}` 放进 pending_frees 列表。后续每次 alloc 时顺便 poll 一轮 `cudaEventQuery`——event 全部完成才真正释放。CUDA event 通过 `EventPool` 对象池复用，避免反复 `cudaEventCreate`/`Destroy`。

关键实现细节：**`forceProcessPendingFrees` 在锁内收集需要同步的 entry，然后在锁外执行 `cudaEventSynchronize`。** 如果在锁内同步，所有 allocator 操作都会被阻塞直到 GPU 完成——这在多线程 data-loading pipeline 里会造成严重的 contention。

### 2.5 CUDA Graph 池隔离与 VMM 重映射

zcg 通过 `beginAllocateToPool` / `endAllocateToPool` / `releasePool` 三件套支持 PyTorch 的 CUDA Graph 池隔离协议。capture 期间的分配通过 stream filter 函数标记到指定 pool，graph 销毁时 `releasePool` 一次性释放所有属于该 pool 的分配。

在此之上，zcg 暴露了 `vmm_remap(va, size, old_handles, new_handles, new_sizes)` 接口：在同一虚拟地址上执行 `cuMemUnmap` + `cuMemMap`。所有引用这段 VA 的 tensor 完全无感知，因为地址没变。这是 CUDA Graph replay 的理想路径。

---

## 3 VMM Raw API 性能剖析与优化手段

VMM API 给了我们虚拟/物理解耦的能力，但它不是免费的。每个 API 调用都是一次进入 CUDA driver 的开销。如果每次 `torch.empty()` 都要走完 reserve → create → map → setAccess 的完整链路，性能会远差于默认 allocator 的 free-list 命中路径。这一章剖析各 API 的真实开销，以及 zcg 如何规避它们。

### 3.1 Raw API 延迟基准

以下是在 **NVIDIA H100 80GB HBM3** 上，对各 VMM API 独立调用的延迟测量结果。每个操作执行 200 次，报告中位数（med）和 p99，单位为微秒（μs）。

| 操作 | 2 MB | 16 MB | 64 MB | 256 MB |
|------|------|-------|-------|--------|
| `cuMemAddressReserve` | 0.98 | 1.01 | 0.96 | 0.95 |
| `cuMemCreate` | 9.75 | 11.96 | 10.55 | 14.70 |
| `cuMemMap` | 0.43 | 0.44 | 0.45 | 0.47 |
| `cuMemSetAccess` | 77.00 | 77.16 | 77.48 | 97.61 |
| `cuMemUnmap` | 36.30 | 36.50 | 36.23 | 36.69 |
| `cuMemRelease` | 35.85 | 38.89 | 49.41 | 90.83 |
| `cuMemAddressFree` | 0.86 | 0.86 | 0.82 | 0.83 |
| **完整链路 (reserve→create→map→access)** | **88.77** | **88.39** | **156.51** | **629.02** |

p99 延迟（捕捉尾部抖动）：

| 操作 | 2 MB | 16 MB | 64 MB | 256 MB |
|------|------|-------|-------|--------|
| `cuMemCreate` | 12.69 | 468.21 | 19.72 | 49.26 |
| `cuMemSetAccess` | 82.82 | 595.23 | 1024.00 | 1028.26 |
| `cuMemRelease` | 39.21 | 1193.13 | 59.10 | 127.88 |
| `cuMemUnmap` | 37.32 | 808.13 | 37.92 | 1850.63 |

这组数据揭示了几个关键事实。

**`cuMemSetAccess` 是绝对瓶颈。** 中位数 77-98 μs，远超其他所有操作。它占了完整链路耗时的 80% 以上（2MB 时完整链路 88.77 μs，其中 setAccess 就 77 μs）。这个开销来自 driver 内部修改 GPU 页表的 access permission 位——即使只有单设备单 descriptor，开销也在 77 μs 量级。这意味着任何依赖 VMM 的 allocator，如果每次分配都调 setAccess，性能天花板就在这里。

**`cuMemCreate`/`cuMemRelease` 是第二梯队的重操作。** create 中位数 10-15 μs，release 36-91 μs，且随块大小增长。值得注意的是 release 比 create 贵 3-6 倍——driver 内部回收物理页的开销远高于分配。这直接解释了为什么 zcg 把物理块缓存在 free list 里而不是立即 release：省掉 create+release 的往返就是省掉 45-106 μs。

**`cuMemMap` 极其便宜，`cuMemUnmap` 却很贵。** map 只要 0.43-0.47 μs（纯页表映射），而 unmap 要 36 μs（需要 flush TLB）。这个不对称性意味着 "map 新物理块" 几乎免费，但 "换出旧物理块" 有真实代价。SmallAllocator 的设计——只 map 不 unmap，页面常驻——在这个数据下是完全合理的。

**`cuMemAddressReserve`/`cuMemAddressFree` 几乎零开销。** 不到 1 μs，与块大小无关。这证实了 SmallAllocator 启动时预留 32GB VA 空间的策略代价极低——VA 空间是虚拟的，预留多少都不影响性能。

**p99 尾部延迟剧烈抖动。** 中位数 77 μs 的 `cuMemSetAccess`，p99 可以飙到 1028 μs（13 倍）。`cuMemRelease` 16MB 的 p99 高达 1193 μs（中位数的 30 倍），最大值甚至到 5626 μs。这种长尾几乎肯定来自 driver 内部的 GC 或页表刷新操作，在大规模并发分配时会造成显著的延迟抖动。对于延迟敏感的推理服务，这进一步强调了缓存复用的必要性——热路径上绝对不能碰这些 driver 调用。

**完整链路开销与块大小正相关。** 2MB 块 89 μs，256MB 块 629 μs——相差 7 倍。这意味着如果用 2MB 小块拼出 256MB 的分配（128 个块），即使每个 map 只要 0.43 μs，光 setAccess 就要调 128 次 × 77 μs ≈ 9856 μs。而用单个 256MB 块只需 1 次 setAccess ≈ 98 μs。LargeAllocator 的大块拼接策略在这组数据下有 **100 倍** 的量级优势。

### 3.2 优化手段一：物理块缓存（消除 create/release 开销）

从上面的数据看，`cuMemCreate` + `cuMemRelease` 的往返开销在 46-106 μs（取决于块大小），而 release 的 p99 尾部延迟可达毫秒级。最直接的优化就是**尽量不调用它们**。zcg 的 `PhysicalMemoryManager` 维护 per-size-class 的 free list：物理块释放时归还到 free list 而不是 `cuMemRelease`，后续分配优先从 free list 取。在稳态工作负载（如推理服务的连续 batch 处理）下，物理块几乎 100% 命中缓存，热路径上完全不触碰 driver——一次 create/release 都不需要。

### 3.3 优化手段二：选择性 purge（最小化 release 调用数）

当内存压力确实需要释放缓存时，zcg 的 `purge(min_bytes)` 从最大 size class 开始释放。实测 release 一个 256MB 块耗时 91 μs，而 release 128 个 2MB 块耗时 128 × 36 μs = 4608 μs——**50 倍的差距**。purge 只释放到"够用"就停，保留其余缓存供后续复用。

### 3.4 优化手段三：Lock-outside 释放路径

`purge()` 的实现分两阶段：锁内收集待释放的物理块（只操作 free list 的 vector，纯内存操作），锁外执行 `cuMemRelease`（实测 36-91 μs/次）。这样 driver 调用不会阻塞其他线程的 alloc/free 操作。同样的模式也用在 `forceProcessPendingFrees` 中：锁内收集 pending entries，锁外执行 `cudaEventSynchronize`。考虑到 release 的 p99 可达毫秒级，如果在锁内执行这些调用，并发分配线程会被严重阻塞。

### 3.5 优化手段四：Event 池复用

CUDA event 的创建和销毁也有 driver 开销。zcg 使用一个 `EventPool` 对象池，event 用完后 release 回池中而不是 destroy。在多 stream 交叉使用的场景（data-parallel training、pipeline parallelism）下，这可以避免大量的 `cudaEventCreate`/`Destroy` 调用。

### 3.6 优化手段五：SmallAllocator 的连续 VA + 按需映射

实测 `cuMemAddressReserve` 不到 1 μs，与大小无关——预留 32GB VA 空间和预留 2MB 的开销一样。SmallAllocator 正是利用这一点：启动时一次性预留 32GB VA，此后小 tensor 的分配路径极其轻量：best-fit 查找 + 可能的 `cuMemMap`（仅 0.43 μs，仅在触及新页面时）+ `cuMemSetAccess`（77 μs，仅在首次 map 时）。在稳态下，高水位线以下的所有页面都已映射，分配退化为一次 free-list 查找——纯 CPU 操作，零 driver 调用。

### 3.7 优化手段六：大块贪心拼接（减少 map + setAccess 调用）

实测 `cuMemSetAccess` 占完整链路 80% 以上的耗时。LargeAllocator 用 5 个档位的贪心算法拼接物理块，而不是用大量小块拼。分配 769MB 用 3×256MB + 1×16MB（4 次 map + 4 次 setAccess ≈ 4 × 98 μs = 392 μs），而不是 384×2MB（384 次 map + 384 次 setAccess ≈ 384 × 77 μs ≈ 29568 μs）。**75 倍的差距。** 更少的物理块也意味着后续 unmap/remap 更快。

---

## 4 优化效果：torch.empty Profile 对比

> **TODO: 运行 `benchmarks/bench_torch_empty.py` 和 `benchmarks/bench_alloc_patterns.py` 填入实测数据**

### 4.1 测试环境

| 项目 | 值 |
|------|-----|
| GPU | **[填入]** |
| CUDA | **[填入]** |
| PyTorch | **[填入]** |
| Driver | **[填入]** |

### 4.2 torch.empty 延迟对比

单次 `torch.empty(size, device="cuda")` 调用的 CUDA event 计时（单位 μs），cold 表示首次分配（缓存未命中），warm 表示稳态重复分配（缓存命中）。

| Tensor 大小 | Default (cold) | Default (warm) | zcg (cold) | zcg (warm) |
|-------------|----------------|-----------------|------------|------------|
| 1 KB | — | — | — | — |
| 64 KB | — | — | — | — |
| 1 MB | — | — | — | — |
| 8 MB | — | — | — | — |
| 16 MB | — | — | — | — |
| 64 MB | — | — | — | — |
| 256 MB | — | — | — | — |
| 1 GB | — | — | — | — |

### 4.3 分配模式综合延迟

模拟多种真实工作负载的分配模式，总耗时（ms）对比。详见 `benchmarks/bench_alloc_patterns.py`。

| 模式 | Default Allocator | zcg Allocator | 比值 |
|------|-------------------|---------------|------|
| Alloc-Free 循环 (小张量 1KB-8MB) | — | — | — |
| Alloc-Free 循环 (大张量 16MB-256MB) | — | — | — |
| Alloc-Free 循环 (混合 13 种大小) | — | — | — |
| 随机交错 alloc/free (小张量) | — | — | — |
| 随机交错 alloc/free (混合) | — | — | — |
| 峰值(512MB)后稳态(8MB, 1000 cycles) | — | — | — |
| 模拟推理 (32层, hidden=4096, 100 batch) | — | — | — |

### 4.4 物理内存效率

在连续推理 100 个 batch 后的物理内存使用情况对比。

| 指标 | Default Allocator | zcg Allocator |
|------|-------------------|---------------|
| 峰值物理内存 | — | — |
| 稳态物理内存 | — | — |
| 碎片率 | — | — |
| 缓存命中率 | — | — |
| purge 次数 | N/A | — |

### 4.5 预期分析

在冷启动场景下，zcg 因为需要走 VMM 的 reserve → create → map → setAccess 完整链路，首次分配延迟预计高于默认 allocator 的 `cudaMalloc`。但在稳态（warm）场景下，两者的分配路径都退化为 free-list 查找，延迟应当接近。zcg 的优势主要体现在物理内存效率和 CUDA Graph 兼容性上——不再需要为每个 graph 常驻预留显存，物理内存可以在多个 graph 之间动态共享。

---

## 写在最后

zero-cuda-graph 目前还是一个学习和探索性质的项目。代码全部开源，欢迎阅读、试用、提 issue。如果你在做 CUDA Graph 相关的优化工作，或者对 PyTorch 内存管理的内部机制感兴趣，希望这个项目能给你一些启发。

项目地址：[zero-cuda-graph](https://github.com/nastyapple/zero-cuda-graph)
