# zero-cuda-graph 功能规划

## 一、总体目标

基于 CUDA Virtual Memory Management (VMM) API，实现一个 PyTorch CUDA Allocator 扩展，替换默认的 CUDACachingAllocator。核心解决两个问题：

1. **更精细的物理内存管理**：通过 VMM 的虚拟/物理分离，实现按需映射、弹性扩缩，减少碎片和浪费。
2. **CUDA Graph 零开销支持**：虚拟地址空间稳定不变，物理内存可在背后自由重映射，使 Graph Capture/Replay 天然兼容。

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Python Layer (zcg)                     │
│                                                           │
│  __init__.py          ─ import 即替换 allocator           │
│  memory.py            ─ 内存统计 / 诊断 API               │
│  ops.py               ─ 指针替换 / VMM 重映射 API          │
│  config.py            ─ 可配置参数（阈值、块大小等）         │
└───────────────────────────┬───────────────────────────────┘
                            │ pybind11
┌───────────────────────────▼───────────────────────────────┐
│                   C++ Extension (zcg_c)                    │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              VMMAllocator (CUDAAllocator)             │  │
│  │  继承 c10::cuda::CUDAAllocator                        │  │
│  │  通过 changeCurrentAllocator() 替换全局 allocator      │  │
│  │                                                        │  │
│  │  ┌──────────────┐       ┌──────────────────────┐      │  │
│  │  │ SmallAllocator│       │   LargeAllocator     │      │  │
│  │  │ (< threshold) │       │   (>= threshold)     │      │  │
│  │  │              │       │                      │      │  │
│  │  │ Enhanced     │       │ PhysicalBlockPool    │      │  │
│  │  │ Expandable   │       │ {16,32,64,128,256}MB │      │  │
│  │  │ Segment      │       │ 多块拼接映射          │      │  │
│  │  │ (2MB粒度)    │       │                      │      │  │
│  │  └──────┬───────┘       └──────────┬───────────┘      │  │
│  │         │                          │                   │  │
│  │  ┌──────▼──────────────────────────▼───────────┐      │  │
│  │  │       VirtualAddressSpace Manager            │      │  │
│  │  │  cuMemAddressReserve / cuMemMap / cuMemUnmap │      │  │
│  │  └──────────────────┬──────────────────────────┘      │  │
│  │                     │                                  │  │
│  │  ┌──────────────────▼──────────────────────────┐      │  │
│  │  │         PhysicalMemoryManager                │      │  │
│  │  │  cuMemCreate / cuMemRelease / cuMemSetAccess │      │  │
│  │  └──────────────────┬──────────────────────────┘      │  │
│  │                     │                                  │  │
│  │  ┌──────────────────▼──────────────────────────┐      │  │
│  │  │           CUDADriverAPI Wrapper              │      │  │
│  │  │  dlopen("libcuda.so") + dlsym 封装           │      │  │
│  │  │  类型安全的 C++ RAII 接口                      │      │  │
│  │  └─────────────────────────────────────────────┘      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              DataPtr Substitution                     │  │
│  │  replace_data_ptr(tensor, new_ptr)                    │  │
│  │  vmm_remap(va, old_handles, new_handles)              │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## 三、模块详细设计

### 模块 1：CUDA Driver API Wrapper

**文件**: `include/cuda_driver_api.h`, `csrc/cuda_driver_api.cc`

**职责**: 运行时 dlopen 加载 libcuda，封装所有用到的 VMM 函数为类型安全的 C++ 接口。

**需要封装的 API**:

```
cuMemAddressReserve    ─ 预留虚拟地址段
cuMemAddressFree       ─ 释放虚拟地址段
cuMemCreate            ─ 创建物理内存块（CUmemGenericAllocationHandle）
cuMemRelease           ─ 释放物理内存块
cuMemMap               ─ 将物理块映射到虚拟地址
cuMemUnmap             ─ 解除映射
cuMemSetAccess         ─ 设置访问权限
cuMemGetAllocationGranularity ─ 查询分配粒度
```

**设计要点**:
- 单例模式，首次调用时 dlopen + dlsym 初始化
- 每个 API 封装为同名方法，内部做错误码检查并抛 c10::Error
- RAII 封装：`PhysicalBlock`（析构时 cuMemRelease）、`VirtualRange`（析构时 cuMemAddressFree）

---

### 模块 2：PhysicalMemoryManager

**文件**: `include/physical_memory.h`, `csrc/physical_memory.cc`

**职责**: 管理物理内存块的创建、缓存和回收。

**数据结构**:

```cpp
struct PhysicalBlock {
    CUmemGenericAllocationHandle handle;
    size_t size;
    int device;
    bool in_use;
};

class PhysicalMemoryManager {
    // 小块池：固定 2MB 粒度
    std::vector<PhysicalBlock> small_pool_;  // free list

    // 大块池：按档位分桶
    // 档位: 16MB, 32MB, 64MB, 128MB, 256MB
    std::map<size_t, std::vector<PhysicalBlock>> large_pools_;

    PhysicalBlock allocSmall();
    PhysicalBlock allocLarge(size_t size_class);  // size_class ∈ {16M,32M,...,256M}
    void freeSmall(PhysicalBlock);
    void freeLarge(PhysicalBlock);
};
```

**设计要点**:
- 小块和大块的物理内存严格隔离，大块永远不会被切碎给小块用
- 大块池按档位管理 free list，分配时从池中取，释放时归还池中
- 物理块可复用（不 cuMemRelease），只在显式 trim 或 OOM 时才释放

---

### 模块 3：VirtualAddressSpace Manager

**文件**: `include/virtual_address_space.h`, `csrc/virtual_address_space.cc`

**职责**: 管理虚拟地址空间的预留和映射关系。

**设计**:

```cpp
struct MappingEntry {
    CUdeviceptr va_base;
    size_t total_size;
    std::vector<PhysicalBlock> mapped_blocks;  // 有序，拼接映射
};

class VirtualAddressSpaceManager {
    // 预留一大段 VA 空间（如启动时预留 256GB VA）
    CUdeviceptr va_pool_base_;
    size_t va_pool_size_;
    size_t va_cursor_;  // bump allocator 风格

    // VA → Mapping 的记录
    std::unordered_map<CUdeviceptr, MappingEntry> mappings_;

    // 为分配申请一段连续 VA，返回起始地址
    CUdeviceptr reserveVA(size_t size);

    // 将物理块列表映射到指定 VA 段
    void mapBlocks(CUdeviceptr va, const std::vector<PhysicalBlock>& blocks);

    // 解除映射，VA 段可复用
    void unmapBlocks(CUdeviceptr va);

    // 重映射：同一 VA 换绑不同物理块（CUDA Graph 场景）
    void remapBlocks(CUdeviceptr va,
                     const std::vector<PhysicalBlock>& new_blocks);
};
```

**设计要点**:
- 启动时通过 cuMemAddressReserve 预留大段 VA（可配置，默认 256GB），确保地址不与 cudaMalloc 冲突
- 小张量体系内部在自己的 Segment 里管理子分配（类似 ExpandableSegment）
- 大张量分配直接从 VA pool 中 bump 出一段连续地址，然后拼接映射多个物理块
- 重映射操作：先 cuMemUnmap 再 cuMemMap 新物理块到同一 VA，原子性保证

---

### 模块 4：SmallAllocator（增强 ExpandableSegment）

**文件**: `include/small_allocator.h`, `csrc/small_allocator.cc`

**职责**: 处理小于阈值的分配请求，基于 2MB 物理块粒度的 ExpandableSegment。

**设计思路**:
- 维护多个 Segment，每个 Segment 预留一段 VA，内部按 2MB 物理块扩展
- Segment 内部做 sub-allocation：类似经典的 best-fit / slab 策略
- 当 Segment 内有连续空闲空间不足时，通过映射新的 2MB 物理块扩展
- 当 Segment 内一段区域完全空闲时，cuMemUnmap 对应物理块并归还池
- stream-ordered：记录每个 sub-allocation 关联的 CUDA stream，free 时确保 stream 同步

---

### 模块 5：LargeAllocator（物理块池拼接）

**文件**: `include/large_allocator.h`, `csrc/large_allocator.cc`

**职责**: 处理大于等于阈值的分配请求。

**分配流程**:

```
输入: 请求 size 字节
1. 计算拼接方案：贪心从最大块(256MB)到最小块(16MB)
   例: 769MB → 3×256MB + 1×16MB（不足16MB的部分向上取整到16MB）
2. 从 PhysicalMemoryManager 的各档位池中取物理块
   - 池中有空闲块 → 复用
   - 池中无空闲块 → cuMemCreate 新建
3. 从 VA Manager 申请一段连续 VA
4. 将物理块列表按序映射到 VA 段
5. cuMemSetAccess 设置权限
6. 返回 VA 起始地址
```

**释放流程**:

```
输入: VA 起始地址
1. 查找 MappingEntry，获取物理块列表
2. cuMemUnmap 所有块
3. 物理块归还到各自档位的池中
4. VA 段标记可复用
```

**设计要点**:
- 拼接方案用贪心算法：从 256MB 开始，能放就放，不够切到下一档，最后不足 16MB 的向上补齐到 16MB
- 物理块严格按档位管理，绝不会将 256MB 块分给小于 256MB 的请求
- 这些大块物理内存空闲时可以被 trim 释放

---

### 模块 6：VMMAllocator（主 Allocator 入口）

**文件**: `include/vmm_allocator.h`, `csrc/vmm_allocator.cc`

**职责**: 继承 `c10::cuda::CUDAAllocator`，作为面向 PyTorch 的统一接口。

**需要实现的接口**（继承自 CUDAAllocator）:

```cpp
class VMMAllocator : public c10::cuda::CUDAAllocator {
public:
    // === 核心分配 ===
    DataPtr allocate(size_t n) override;
    void* raw_alloc(size_t nbytes) override;
    void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override;
    void raw_delete(void* ptr) override;

    // === Stream 感知 ===
    void recordStream(const DataPtr&, cuda::CUDAStream stream) override;

    // === 内存管理 ===
    void emptyCache() override;
    void setMemoryFraction(double fraction, c10::DeviceIndex device) override;

    // === 统计信息 ===
    DeviceStats getDeviceStats(c10::DeviceIndex device) override;
    void resetPeakStats(c10::DeviceIndex device) override;

    // === Snapshot / 诊断 ===
    SnapshotInfo snapshot() override;

    // === 其他必要接口 ===
    void init(int device_count) override;
    bool initialized() override;
    void cacheInfo(c10::DeviceIndex, size_t* maxWorkspaceGuess) override;
    std::string name() override { return "VMMAllocator"; }

private:
    SmallAllocator small_allocator_;
    LargeAllocator large_allocator_;
    size_t threshold_;  // 大小张量分界

    // 路由：根据 size 选择走哪条路径
    bool isLargeAllocation(size_t size) const {
        return size >= threshold_;
    }
};
```

**注册方式**:

```cpp
// 在 Python import zcg 时调用
void register_vmm_allocator() {
    auto* allocator = new VMMAllocator();
    allocator->init(device_count);
    c10::cuda::CUDACachingAllocator::changeCurrentAllocator(allocator);
}
```

---

### 模块 7：DataPtr Substitution

**文件**: `include/substitution.h`, `csrc/substitution.cc`

**职责**: 提供两种指针操作能力。

#### 7a. Tensor 指针热替换

直接修改 tensor 的 Storage 底层指针，使 tensor 指向新的内存区域。

```cpp
// 通过 Storage API 实现
void replace_data_ptr(at::Tensor& tensor, void* new_ptr, DeleterFn deleter) {
    // 方案：构造新的 DataPtr，通过 storage.set_data_ptr() 替换
    auto device = tensor.device();
    c10::DataPtr new_data_ptr(new_ptr, new_ptr, deleter, device);
    tensor.storage().set_data_ptr(std::move(new_data_ptr));
}
```

**使用场景**: 将 tensor 指向预分配池中的某块内存，实现 zero-copy 数据交换。

#### 7b. VMM 重映射

在同一虚拟地址上替换底层物理内存，tensor 的指针值不变，但物理数据变了。

```cpp
void vmm_remap(CUdeviceptr va, size_t size,
               const std::vector<CUmemGenericAllocationHandle>& old_handles,
               const std::vector<CUmemGenericAllocationHandle>& new_handles) {
    // 1. cuMemUnmap(va, size)
    // 2. cuMemMap(va, ..., new_handles)
    // 3. cuMemSetAccess(...)
}
```

**使用场景**: CUDA Graph Replay 时，capture 阶段录制的 VA 不变，replay 时将新的物理内存映射上去，数据就位后 replay graph 即可。

---

### 模块 8：Python 绑定

**文件**: `csrc/bindings.cc`（pybind11 模块定义）

```python
# 暴露到 Python 的接口

# 1. Allocator 注册
zcg_c.register_vmm_allocator()

# 2. 指针替换
zcg_c.replace_data_ptr(tensor, new_ptr_as_int, device_index)

# 3. VMM 重映射
zcg_c.vmm_remap(va_as_int, size, old_handle_list, new_handle_list)

# 4. 内存统计
zcg_c.get_memory_stats(device_index)
    → {"allocated": ..., "reserved": ..., "small_pool": {...}, "large_pool": {...}}

zcg_c.get_pool_snapshot(device_index)
    → 各档位物理块使用情况详情

# 5. 控制
zcg_c.empty_cache()
zcg_c.set_threshold(size_in_bytes)
zcg_c.set_va_pool_size(size_in_bytes)
```

---

### 模块 9：Python Package

**文件结构**:

```
zcg/
    __init__.py     ─ import 时自动替换 allocator
    config.py       ─ 配置管理
    memory.py       ─ 内存统计诊断的 Pythonic 封装
    ops.py          ─ 指针替换和 VMM 操作的 Pythonic 封装
```

**`__init__.py` 核心逻辑**:

```python
import torch
import zcg_c

def _init():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    # 必须在任何 CUDA 分配之前调用
    zcg_c.register_vmm_allocator()

_init()
```

**`config.py`**:

```python
# 可配置参数
DEFAULT_THRESHOLD = 16 * 1024 * 1024   # 16MB, 大小张量分界
DEFAULT_VA_POOL = 256 * 1024**3         # 256GB VA 预留空间
SMALL_BLOCK_SIZE = 2 * 1024 * 1024      # 2MB 小块粒度
LARGE_BLOCK_SIZES = [16, 32, 64, 128, 256]  # MB, 大块档位

def configure(threshold=None, va_pool_size=None, ...):
    """必须在 import zcg 之前或紧接着之后、首次分配之前调用"""
    ...
```

---

## 四、项目文件结构（最终）

```
zero-cuda-graph/
├── setup.py
├── README.md
├── DESIGN.md                       ← 本文件
├── include/
│   ├── cuda_driver_api.h           ─ CUDA Driver API dlopen 封装
│   ├── physical_memory.h           ─ 物理内存块管理
│   ├── virtual_address_space.h     ─ 虚拟地址空间管理
│   ├── small_allocator.h           ─ 小张量分配器
│   ├── large_allocator.h           ─ 大张量分配器
│   ├── vmm_allocator.h             ─ 主 Allocator（CUDAAllocator 子类）
│   └── substitution.h              ─ DataPtr 替换与 VMM 重映射
├── csrc/
│   ├── cuda_driver_api.cc
│   ├── physical_memory.cc
│   ├── virtual_address_space.cc
│   ├── small_allocator.cc
│   ├── large_allocator.cc
│   ├── vmm_allocator.cc
│   ├── substitution.cc
│   └── bindings.cc                 ─ pybind11 入口
└── zcg/
    ├── __init__.py
    ├── config.py
    ├── memory.py
    └── ops.py
```

---

## 五、关键设计决策总结

| 决策项 | 选择 | 理由 |
|--------|------|------|
| Allocator 层级 | CUDAAllocator (changeCurrentAllocator) | 需要 stream-aware、recordStream 等 CUDA 特有功能 |
| Driver API 加载 | CppExtension + dlopen | 无需 nvcc 编译，部署更灵活，封装后接口整洁 |
| 大小分界 | 可配置（默认 16MB） | 便于根据实际 workload 调优 |
| 小块粒度 | 2MB 物理块 | 与 CUDA 分配粒度对齐，细粒度控制 |
| 大块档位 | 16/32/64/128/256 MB | 覆盖常见 tensor 大小，贪心拼接策略浪费可控 |
| 大块不切碎 | 大块池与小块池严格隔离 | 避免大块被碎片化，保证大分配的服务质量 |
| VA 空间 | 启动时预留 256GB | 远大于物理显存，保证地址不枯竭，不与 cudaMalloc 冲突 |
| 物理块复用 | free 时归池不释放 | 减少 cuMemCreate/Release 开销，OOM 或显式 trim 时才释放 |

---

## 六、实现优先级建议

**Phase 1 — 基础骨架（可编译可运行）**
1. CUDA Driver API Wrapper（dlopen 封装）
2. VMMAllocator 骨架（继承 CUDAAllocator，最小可用实现）
3. setup.py 更新 + Python import 即替换
4. 基础验证：`import zcg; t = torch.zeros(100, device='cuda')` 能跑通

**Phase 2 — 双体系分配**
5. PhysicalMemoryManager + 池管理
6. VirtualAddressSpaceManager
7. SmallAllocator（ExpandableSegment 增强版）
8. LargeAllocator（多块拼接映射）
9. VMMAllocator 路由逻辑

**Phase 3 — 指针操作与诊断**
10. DataPtr 替换（tensor 指针热替换）
11. VMM 重映射（同 VA 换物理块）
12. 内存统计 / 诊断 API
13. Python 侧 Pythonic 封装

**Phase 4 — 完善与优化**
14. Stream-aware 逻辑（recordStream）
15. 线程安全（per-device mutex / lock-free 优化）
16. OOM 处理（retry + trim 策略）
17. CUDA Graph 场景端到端验证
