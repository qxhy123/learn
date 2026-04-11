# 附录C：练习答案汇总

> 本附录提供的是“答案提示”和“思考方向”，不是唯一标准答案。对 CUDA 学习来说，能说清推理过程，通常比只给出结论更重要。

---

## 第1章：为什么需要 GPU 计算

- 吞吐导向关注单位时间总处理量；延迟导向关注单次任务完成时间。
- 向量加法适合 GPU，因为数据并行强、元素独立、访问规整。
- 频繁 CPU-GPU 往返会让传输开销吞掉计算收益。
- 深度学习矩阵乘法适合 GPU，因为规则、批量大、可分块、计算密集。

---

## 第2章：环境配置与第一个 CUDA 程序

- 驱动负责让 GPU 能被系统正确管理；Toolkit 提供开发工具链；`nvcc` 负责组织 CUDA 编译流程。
- CUDA 程序既有 host 代码也有 device 代码，因此需要主机编译器与 CUDA 工具链协同。
- `nvidia-smi` 说明驱动和 GPU 基本可见；`nvcc --version` 才能说明开发工具链已安装。

---

## 第3章：线程、线程块、网格与 Warp

- thread 是最小执行单位；block 是协作边界；grid 是一次 launch 的整体线程集合。
- 常用全局索引公式：`idx = blockIdx.x * blockDim.x + threadIdx.x`。
- `if (idx < n)` 用于过滤超出有效范围的线程。
- warp 重要，因为很多性能现象都以 warp 为基本观察粒度。

---

## 第4章：Kernel 编写与启动配置

- `__global__` 表示 host 启动、device 执行的 kernel。
- `<<<grid, block>>>` 指定 block 数量和每个 block 的线程数。
- block 大小通常先从 128 / 256 / 512 这样的经验值开始试。
- 边界判断几乎总是基础正确性保障。

---

## 第5章：内存分配与数据传输

- `cudaMalloc` 分配 device 内存，`cudaFree` 释放。
- `cudaMemcpyHostToDevice`、`cudaMemcpyDeviceToHost` 方向不能写错。
- `bytes = n * sizeof(T)` 是基础习惯，避免把元素数误当字节数。
- GPU 加速效果可能被频繁 host-device 往返拖垮。

---

## 第6章：索引计算与数据布局

- row-major 中二维索引常写作 `row * width + col`。
- 转置时输入输出地址公式不同，宽高容易混淆。
- 数据布局不仅影响正确性，也影响访存合并效果。

---

## 第7章：CUDA 内存层次

- register 线程私有且快；shared memory 适合 block 内复用；global memory 容量大但访问代价更高。
- local memory 名字容易误导，它只是逻辑上的线程私有空间，不等于高速存储。
- shared memory 适合有明确数据复用的场景。

---

## 第8章：共享内存与分块优化

- tiling 的核心是把大问题切成 block 局部处理的小块。
- shared memory 的主要价值是减少重复 global memory 访问。
- `__syncthreads()` 常用于保证 tile 装载完成后再开始复用。
- bank conflict 说明 shared memory 访问模式也会影响性能。

---

## 第9章：Warp 分歧与同步

- warp divergence 首先是性能问题：同一 warp 内线程走不同路径会降低效率。
- `__syncthreads()` 只能同步当前 block，不能同步整个 grid。
- 原子操作保证正确性，但高冲突时会带来明显性能代价。

---

## 第10章：并行归约

- 归约比向量加法更难并行，因为最终需要把大量线程结果合并。
- 所有线程直接原子加到一个全局地址通常正确但冲突重。
- 更常见方案是 block 内先做局部 reduction，再进行更高层合并。

---

## 第11章：前缀和与流压缩

- inclusive scan 包含当前位置，exclusive scan 不包含当前位置。
- compaction 的典型结构是：标记 + scan + scatter。
- scan 的关键作用是为保留元素生成紧凑输出位置。

---

## 第12章：卷积、Stencil 与邻域计算

- 邻域计算的核心特征是一个输出依赖输入中的局部邻域。
- halo 是为了支持 tile 边界位置的邻域访问而额外加载的数据。
- shared memory 优化的价值来自相邻输出共享大量输入区域。

---

## 第13章：访存合并与 Occupancy

- coalescing 强调相邻线程访问相邻地址。
- occupancy 反映活跃线程 / warp 相对理论上限的比例，主要帮助隐藏延迟。
- 高 occupancy 不保证一定最快，仍需结合访存模式和实际 profiling 判断。

---

## 第14章：Profiling 与 Benchmarking

- 不同步就直接计时，结果通常不可靠。
- CUDA events 适合测 GPU 工作区间；Nsight Systems 看全局时间线；Nsight Compute 看单 kernel 细节。
- benchmark 应包含 warmup、重复测量与正确性验证。

---

## 第15章：错误处理与调试

- CUDA 错误常异步暴露，报错位置未必就是根因位置。
- 调试期常用模板：launch 后 `cudaGetLastError()` + `cudaDeviceSynchronize()`。
- 缩小输入规模、验证索引和边界，是排查错误的高效起点。

---

## 第16章：Streams、Events 与计算-传输重叠

- stream 是工作队列，同一 stream 内通常顺序执行。
- 异步不等于自动重叠；重叠需要合适 stream、依赖结构和传输条件。
- event 既能计时，也能作为时间线依赖点。

---

## 第17章：Unified Memory 与异步编程

- `cudaMallocManaged` 提供统一地址空间，但不消除数据迁移成本。
- `cudaMemPrefetchAsync` 可以让 managed memory 迁移更可控。
- Unified Memory 适合原型、教学和复杂数据结构；性能极敏感场景仍需谨慎评估。

---

## 第18章：Cooperative Groups 与 CUDA Graphs

- Cooperative Groups 用于更明确表达不同粒度线程组的协作关系。
- CUDA Graphs 适合稳定、重复的小粒度工作流，用于减少反复 launch 的开销。
- 是否引入这些高阶特性，应由实际瓶颈决定。

---

## 第19章：cuBLAS、cuDNN 与 CUB

- cuBLAS 面向线性代数，尤其是 GEMM。
- cuDNN 面向深度学习核心算子。
- CUB 面向 reduction、scan 等基础并行原语。
- 标准算子应优先考虑成熟库，自定义 kernel 更适合特殊需求或融合场景。

---

## 第20章：WMMA 与 Tensor Core

- Tensor Core 主要针对高吞吐矩阵乘加工作负载。
- WMMA 是 warp 级矩阵乘加接口。
- 混合精度常和 Tensor Core 一起出现，但必须评估数值稳定性。

---

## 第21章：PyTorch 自定义 CUDA 算子

- 最小结构通常分为 Python API、C++ 包装层、CUDA kernel 三层。
- 包装层要负责 shape、dtype、device 等系统边界检查。
- 是否值得写自定义算子，应由 benchmark 和实际热点决定。

---

## 第22章：多 GPU 与 NCCL 基础

- 多 GPU 的核心挑战不只是多几张卡算，而是通信与协同。
- 数据并行中 all-reduce 是统一梯度的关键。
- NCCL 是 GPU 集体通信的重要基础设施。

---

## 第23章：PTX、编译链路与 Runtime/Driver API

- PTX 是设备代码的重要中间表示。
- Runtime API 更常用、更高层；Driver API 更底层、更灵活。
- 理解编译链路有助于理解部署、架构差异和运行时行为。

---

## 第24章：高级优化与下一步学习路线

- 优化顺序通常应是：先正确、再测量、再定位瓶颈、最后针对性优化。
- block 大小、occupancy、shared memory 配置都应在目标硬件上验证。
- 更成熟的能力不是“继续堆优化术语”，而是能判断什么值得做、什么不值得做。

---

## 使用建议

阅读本附录时，建议你：

1. 先独立完成每章练习
2. 再回来看这里的答案提示
3. 对照自己的推理过程，而不只对照最后结论
4. 对有代码的题目，尽量在 GPU 上亲自验证

如果你发现自己总是在某一类题目上卡住，例如：

- 索引计算
- shared memory
- profiling
- stream / 异步

就说明对应章节值得再精读一遍，并配合实验重新做一次。
