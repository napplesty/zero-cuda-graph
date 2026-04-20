## 当 PyTorch 的显存分配器成为 CUDA Graph 的绊脚石

如果你做过 LLM 推理优化或者用 `torch.compile` 压过延迟，大概率遇到过这样的场景：CUDA Graph capture 好好的，replay 时却因为内存地址变了而翻车。你开始和 `torch.cuda.graph()` 斗智斗勇，手动预分配 warmup tensor，祈祷 allocator 不要在 replay 时给你换个地址。

这背后的根本原因很简单——PyTorch 默认的 CUDACachingAllocator 是为通用场景设计的，它在内部维护了一套基于 `cudaMalloc` 的块缓存机制。分配出去的虚拟地址和物理内存绑死在一起，free 之后虚拟地址回到空闲池，下次分配可能拿到完全不同的地址。CUDA Graph 录制的是确定性的地址序列，这种不确定性天然和它冲突。

我们做了一个叫 **zero-cuda-graph**（简称 `zcg`）的项目来彻底解决这件事。它是一个基于 CUDA Virtual Memory Management (VMM) API 的 PyTorch allocator 扩展，核心思想只有一句话：**虚拟地址稳定不变，物理内存在背后自由切换。**

---

## VMM 是什么，为什么它是解药

CUDA 从 10.2 开始提供了一组 Virtual Memory Management API（`cuMemAddressReserve`、`cuMemCreate`、`cuMemMap`、`cuMemSetAccess` 等）。传统的 `cudaMalloc` 返回的指针既是虚拟地址也隐含了物理绑定——你没法控制物理页的生命周期。VMM 把这两件事拆开了：

1. **预留虚拟地址**（`cuMemAddressReserve`）—— 只占用地址空间，不分配物理内存
2. **创建物理内存**（`cuMemCreate`）—— 拿到一个 handle，实际占用显存
3. **映射**（`cuMemMap`）—— 把物理 handle 贴到某段虚拟地址上
4. **解映射+重映射** —— 同一个虚拟地址，换一组物理 handle

这意味着你可以在 CUDA Graph capture 时录制一组虚拟地址，replay 时只换底层物理内存，上层 kernel 看到的地址完全一样。不需要 warmup hack，不需要祈祷。

---

## zcg 的架构设计

整个 allocator 用 `import zcg` 一行代码激活，在任何 CUDA tensor 分配之前替换掉默认的 CUDACachingAllocator。

架构分成三层。最底层是一个 `CUDADriverAPI` 单例，运行时通过 `dlopen("libcuda.so")` 加载所有 VMM 函数指针——这样做的好处是不需要 nvcc 编译，整个项目用纯 C++17 + CppExtension 就能构建。中间层是 `PhysicalMemoryManager`，管理物理块的创建、缓存和回收。最上层是 `VMMAllocator`，继承自 PyTorch 的 `c10::cuda::CUDACachingAllocator::CUDAAllocator`，通过原子指针替换注册为全局分配器。

### 双通道分配

我们观察到深度学习工作负载里的 tensor 大小呈双峰分布：大量 1KB-10MB 的小 tensor（激活值、临时缓冲区）和少量 16MB-数百 MB 的大 tensor（权重、KV Cache）。对这两类用同一种策略既低效又容易互相干扰，所以 zcg 做了严格的双通道设计：

**SmallAllocator**（< 16MB，可配）：启动时通过 VMM 预留一段 32GB 的连续虚拟地址空间，内部管理一个 address-ordered best-fit 空闲链表。物理内存以 2MB 为粒度按需映射——当分配触及未映射的页面时才向 `PhysicalMemoryManager` 申请物理块。释放时做 coalescing 合并相邻空闲区间。`emptyCache()` 会检测哪些 2MB 页面已经完全空闲，解除映射并归还物理块。

**LargeAllocator**（>= 16MB）：每次分配独立 reserve 一段虚拟地址。物理内存从 5 个档位（16/32/64/128/256 MB）的块池中取出，通过贪心算法拼接。比如分配 769MB 会组合 3×256MB + 1×16MB（补齐到最小档位）。释放时解映射、归还物理块到池中、释放虚拟地址。

两个池子的物理块严格隔离。大块永远不会被切碎给小分配用。这个决策看起来简单，但对避免碎片至关重要——如果一个 256MB 的块被拆出 1MB 给小 tensor，剩下 255MB 就成了碎片。

### CUDA Graph 支持：指针替换与 VMM 重映射

zcg 提供两个原语来支持 CUDA Graph：

`replace_data_ptr(tensor, new_ptr)` 直接修改 tensor 的 Storage 底层指针。这是 "hot-swap" 路径：tensor 的 shape、stride、dtype 全部不变，只换数据地址。适用于需要把 tensor 指向预分配池中某块内存的场景。

`vmm_remap(va, size, old_handles, new_handles, new_sizes)` 则更精巧——在同一虚拟地址上执行 unmap + remap。所有引用这段 VA 的 tensor 完全无感知，因为地址没变。这是 CUDA Graph replay 的理想路径：capture 时录制的地址在 replay 时仍然有效，只是物理页面换了。

---

## Phase 4：生产级特性

一个能跑 demo 的 allocator 和一个能上生产的 allocator 之间，差着大量的工程细节。zcg 的 Phase 4 实现了五个关键特性。

### Event-based 延迟回收

PyTorch 的 `recordStream()` 机制告诉 allocator "这块内存还在某个 stream 上被用"。如果 tensor 在 stream A 上分配，被 `recordStream` 记录到 stream B、C，那么 free 时不能立即回收——B 和 C 可能还在访问它。

朴素做法是 `cudaStreamSynchronize`，但这会把 GPU pipeline 打断。zcg 采用 event-based 方案：free 时在每个 recorded stream 上 record 一个轻量级 event（`cudaEventDisableTiming`），把 `{ptr, meta, events}` 放进 pending_frees 列表。后续每次 `alloc_impl` 时顺便 poll 一轮 `cudaEventQuery`——如果所有 event 都完成了，才真正释放。CUDA event 通过 `EventPool` 复用，避免反复创建销毁。

### OOM Retry

第一次分配失败时，zcg 不会直接抛异常。它会先强制同步并释放所有 pending frees，清空 sub-allocator 的空闲缓存，让 `PhysicalMemoryManager` trim 掉所有空闲物理块，然后重试一次。只有第二次也失败，才真正 OOM。`num_alloc_retries` 和 `num_ooms` 计数器让你能在 memory_stats 里看到这些事件。

### CUDA Graph 池隔离

通过 `beginAllocateToPool` / `endAllocateToPool` / `releasePool` 三个接口，zcg 可以把 CUDA Graph capture 期间的所有分配隔离到一个独立的 mempool。pool 使用 filter 函数匹配 capture stream，只有匹配的 stream 上的分配才会被标记到 pool 中。当 graph 被销毁时，`releasePool` 一次性释放所有属于这个 pool 的分配（包括仍在 pending_frees 中等待 event 完成的）。PyTorch 的 `torch.cuda.graph()` context manager 内部会调用这些接口。

### 完整的 DeviceStats

zcg 使用 lock-free 的 `PoolStats` 结构（基于 atomic CAS）追踪每个 pool 的 current、peak、accumulated 字节数和分配次数。`getDeviceStats()` 返回与 PyTorch 原生 allocator 兼容的 `DeviceStats` 结构，包含 AGGREGATE / SMALL_POOL / LARGE_POOL 三个维度。`torch.cuda.memory_stats()` 在 zcg 下也能正常工作。

### 多 GPU Peer Access

`enablePeerAccess(dev, dev_to_access)` 同时做两件事：启用 CUDA runtime 的 P2P 访问（`cudaDeviceEnablePeerAccess`），并且更新 VMM 层面的 `CUmemAccessDesc`——让目标设备上所有已映射和未来映射的内存都包含源设备的读写权限。sub-allocator 的 `setAccess` 方法会构建包含所有 peer 设备的 descriptor 数组，一次 `cuMemSetAccess` 调用设置完毕。

---

## 怎么用

安装就是标准的 pip install：

```bash
cd zero-cuda-graph
pip install -e .
```

使用只需要在脚本最前面加一行 import：

```python
import zcg          # allocator 已替换
import torch

x = torch.randn(1024, 1024, device="cuda")  # 通过 VMMAllocator 分配
print(zcg.memory_summary())                  # 查看内存使用详情
```

调整大小阈值：

```python
zcg.configure(threshold=32 * 1024 * 1024)  # 32MB 以上走大块路径
```

多 GPU 场景：

```python
zcg.enable_peer_access(0, 1)  # GPU 0 可以直接读写 GPU 1 的内存
```

CUDA Graph 场景的核心操作：

```python
# VMM 重映射：虚拟地址不变，换底层物理页
zcg.vmm_remap(
    va=tensor.data_ptr(),
    size=tensor.nelement() * tensor.element_size(),
    old_handles=[...],
    new_handles=[...],
    new_sizes=[...],
)
# tensor 的 data_ptr() 没变，但物理数据已经换了
```

---

## 和 PyTorch 原生 Allocator 的对比

PyTorch 的 CUDACachingAllocator 已经是一个非常成熟的实现，它用 stream-ordered block pool、event-based recycling、expandable segment 等技术在通用场景下表现优异。zcg 不是要替代它的所有场景，而是针对 CUDA Graph 兼容性和物理内存精细控制这两个痛点提供一个替代方案。

具体来说：CUDACachingAllocator 的 expandable segment 在 CUDA 11.4+ 上也用了 VMM，但它的 VA 到物理映射是由 allocator 内部管理的，不对外暴露重映射能力。zcg 把这个能力暴露为一等公民 API（`vmm_remap`），让 CUDA Graph 场景可以显式控制物理页的绑定。

另一个区别是物理块的管理粒度。CUDACachingAllocator 倾向于分配大块然后内部切分，zcg 则在大分配路径上用多个标准档位的块拼接——这意味着更好的块复用率，代价是可能有少量内部碎片（向上取整到最小档位 16MB）。

---

## 写在最后

`zero-cuda-graph` 目前还是一个学习和探索性质的项目。代码全部开源，欢迎阅读、试用、提 issue。如果你在做 CUDA Graph 相关的优化工作，或者对 PyTorch 内存管理的内部机制感兴趣，希望这个项目能给你一些启发。

项目地址：[zero-cuda-graph](https://github.com/nastyapple/zero-cuda-graph)
