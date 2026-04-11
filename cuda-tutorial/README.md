# 从零到高阶的CUDA教程

## 项目简介

本教程旨在为学习者提供一套系统、完整的 CUDA 学习路径，从 GPU 计算的基本直觉出发，逐步覆盖 CUDA 的执行模型、内存层次、常见并行算法模式、性能分析工具、高级运行时特性，以及与深度学习工程密切相关的库生态和 PyTorch 自定义算子开发。

**本教程的独特之处**：每章都包含「CUDA实验」部分，尽量用短小、可运行、可解释的 CUDA C++ 示例把抽象概念落到真实 GPU 程序上；同时在合适的地方连接 cuBLAS、cuDNN、Tensor Core、PyTorch Extension、NCCL 等工程主题。

---

## 目标受众

- 没有 CUDA 经验、希望系统学习 GPU 编程的开发者
- 有 C/C++ 或 Python 基础，想理解深度学习框架底层加速原理的工程师
- 需要编写高性能数值计算、图像处理、并行算法程序的学习者
- 对 GPU 性能优化、访存模式、异步执行、多 GPU 通信感兴趣的系统开发者
- 希望从“会调用框架”进阶到“理解底层执行机制”的 AI 工程师

---

## 章节导航目录

### 开始之前

- [前言：如何使用本教程](./00-preface.md)

### 第一部分：CUDA 与 GPU 计算基础

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第1章 | [为什么需要 GPU 计算](./part1-foundations/01-why-gpu-computing.md) | CPU vs GPU、吞吐导向、CUDA 生态、适用问题 | 分析向量加法为何天然适合并行 |
| 第2章 | [环境配置与第一个 CUDA 程序](./part1-foundations/02-environment-and-first-program.md) | 驱动、Toolkit、`nvcc`、`nvidia-smi`、Hello World | 编译并运行向量加法程序 |
| 第3章 | [线程、线程块、网格与 Warp](./part1-foundations/03-thread-block-grid-and-warp.md) | SIMT、层级结构、warp、索引直觉 | 可视化线程映射与索引计算 |

### 第二部分：CUDA 编程模型

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第4章 | [Kernel 编写与启动配置](./part2-programming-model/04-kernel-writing-and-launch.md) | `__global__`、launch 参数、边界检查 | 实现 SAXPY 并比较不同 block 大小 |
| 第5章 | [内存分配与数据传输](./part2-programming-model/05-memory-allocation-and-transfer.md) | `cudaMalloc`、`cudaMemcpy`、pinned memory | 测试 host-device 传输路径 |
| 第6章 | [索引计算与数据布局](./part2-programming-model/06-indexing-and-data-layout.md) | 1D/2D/3D 索引、row-major、pitch | 编写二维矩阵索引与转置示例 |

### 第三部分：内存层次与执行细节

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第7章 | [CUDA 内存层次](./part3-memory-and-execution/07-memory-hierarchy.md) | register、shared、global、constant、cache | 判断不同数据应放在哪一层内存 |
| 第8章 | [共享内存与分块优化](./part3-memory-and-execution/08-shared-memory-and-tiling.md) | tile、bank conflict、矩阵乘法优化 | 编写 tiled matmul 骨架 |
| 第9章 | [Warp 分歧与同步](./part3-memory-and-execution/09-warp-divergence-and-synchronization.md) | divergence、`__syncthreads()`、原子操作 | 观察分支与同步对性能和正确性的影响 |

### 第四部分：常见并行算法模式

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第10章 | [并行归约](./part4-parallel-patterns/10-parallel-reduction.md) | sum/max、树形归约、性能演进 | 从串行累加改造成 block reduction |
| 第11章 | [前缀和与流压缩](./part4-parallel-patterns/11-prefix-sum-and-compaction.md) | inclusive/exclusive scan、compaction | 用 scan 构建简单过滤器 |
| 第12章 | [卷积、Stencil 与邻域计算](./part4-parallel-patterns/12-convolution-and-stencil.md) | halo、邻域重用、共享内存模板 | 实现 1D stencil 与 2D 邻域访问 |

### 第五部分：性能分析与工具链

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第13章 | [访存合并与 Occupancy](./part5-performance-and-tooling/13-memory-coalescing-and-occupancy.md) | coalescing、occupancy、资源权衡 | 比较连续访问与跨步访问 |
| 第14章 | [Profiling 与 Benchmarking](./part5-performance-and-tooling/14-profiling-and-benchmarking.md) | CUDA events、Nsight Systems、Nsight Compute | 正确测量 kernel 时间与吞吐 |
| 第15章 | [错误处理与调试](./part5-performance-and-tooling/15-error-handling-and-debugging.md) | error API、同步调试、Compute Sanitizer | 定位越界访问与异步报错 |

### 第六部分：高级 CUDA 特性

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第16章 | [Streams、Events 与计算-传输重叠](./part6-advanced-cuda/16-streams-events-and-overlap.md) | 异步执行、多 stream、pipeline | 让 memcpy 与 kernel 重叠 |
| 第17章 | [Unified Memory 与异步编程](./part6-advanced-cuda/17-unified-memory-and-async-programming.md) | `cudaMallocManaged`、prefetch、异步 API | 观察统一内存迁移与 prefetch 效果 |
| 第18章 | [Cooperative Groups 与 CUDA Graphs](./part6-advanced-cuda/18-cooperative-groups-and-graphs.md) | 组同步、graph capture、launch 开销优化 | 将重复工作流封装为 CUDA Graph |

### 第七部分：库生态与深度学习接口

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第19章 | [cuBLAS、cuDNN 与 CUB](./part7-libraries-and-dl/19-cublas-cudnn-and-cub.md) | 何时用库替代手写 kernel、接口风格 | 调用高性能库完成 GEMM 或 reduction |
| 第20章 | [WMMA 与 Tensor Core](./part7-libraries-and-dl/20-wmma-and-tensor-cores.md) | WMMA API、混合精度、Tensor Core 条件 | 运行一个最小矩阵乘加片段 |
| 第21章 | [PyTorch 自定义 CUDA 算子](./part7-libraries-and-dl/21-pytorch-custom-cuda-ops.md) | Extension、C++ 包装、CUDA kernel 接入 | 把简单 kernel 暴露给 PyTorch |

### 第八部分：系统级主题与后续进阶

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第22章 | [多 GPU 与 NCCL 基础](./part8-systems-and-next-steps/22-multi-gpu-and-nccl.md) | peer access、all-reduce、数据并行直觉 | 构建最小双 GPU 通信示例 |
| 第23章 | [PTX、编译链路与 Runtime/Driver API](./part8-systems-and-next-steps/23-ptx-compilation-and-runtime.md) | `nvcc`、PTX、JIT、runtime vs driver | 导出 PTX 并理解编译产物 |
| 第24章 | [高级优化与下一步学习路线](./part8-systems-and-next-steps/24-advanced-optimization-and-next-steps.md) | 优化 checklist、架构差异、继续学习方向 | 用系统化方法复盘一个 kernel 的优化空间 |

### 附录

| 附录 | 标题 | 内容说明 |
|------|------|----------|
| 附录A | [CUDA 速查表](./appendix/cuda-cheatsheet.md) | 常用命令、内建变量、同步原语、优化提示 |
| 附录B | [CUDA Runtime API 速查](./appendix/cuda-runtime-api.md) | 内存、执行、事件、流、设备管理常用 API |
| 附录C | [练习答案汇总](./appendix/answers.md) | 各章练习题的要点、提示与常见误区 |

---

## 学习路径建议

### 路径一：零 CUDA 基础

适合第一次接触 GPU 编程的学习者：

1. 按顺序学习第 1-9 章，建立执行模型与内存模型的基本直觉
2. 学习第 13-15 章，掌握最基本的性能分析和调试方法
3. 再进入第 10-12 章，理解常见并行算法模式
4. 最后选择第 16-18 章中的异步执行主题继续深入

### 路径二：深度学习工程师导向

适合已经使用 PyTorch，希望理解底层加速原理的工程师：

1. 学习第 2-9 章，补齐 CUDA 编程模型与内存层次
2. 重点学习第 13-18 章，建立性能优化和异步执行意识
3. 深入第 19-21 章，理解库生态、Tensor Core 与自定义算子
4. 按需阅读第 22-24 章，进入多 GPU 和编译链路主题

### 路径三：系统与性能优化导向

适合希望专注高性能 GPU 编程的学习者：

1. 快速阅读第 1-6 章，完成概念和 API 入门
2. 重点学习第 7-18 章，系统掌握内存、同步、并行模式与 profiling
3. 补充第 19、20、22、23 章，理解工业级高性能栈
4. 用第 24 章的优化 checklist 反复复盘自己的 kernel

---

## 前置要求

学习本教程建议具备以下基础：

- **必需**：基本编程经验，能读懂循环、数组、函数和命令行操作
- **推荐**：C/C++ 基本语法（指针、数组、编译命令）
- **推荐**：线性代数基础（向量、矩阵、矩阵乘法）
- **可选**：Python / PyTorch 使用经验（学习第 21 章时会更轻松）

本教程默认**不要求**你有任何 CUDA 经验；前两章会补齐开始学习所需的最小背景。

---

## 环境配置

本教程的示例以 NVIDIA 官方 CUDA 工具链为主，推荐环境如下：

```bash
# 推荐环境
NVIDIA GPU（支持 CUDA）
CUDA Toolkit >= 12.0
C++ compiler（gcc / clang / MSVC）
Python >= 3.10（仅在 PyTorch / 附加实验中需要）
```

重要说明：

- **Windows / Linux** 是 CUDA 开发的主流平台
- **近年的 macOS 设备不支持 NVIDIA CUDA 开发**，如果你使用 Mac，建议通过远程 Linux 服务器、云 GPU 或独立的 Windows/Linux 工作站实践
- 安装完成后，至少应能运行 `nvidia-smi` 和 `nvcc --version`

官方参考文档：
- CUDA Documentation: https://docs.nvidia.com/cuda/
- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-programming-guide/index.html

---

## 如何使用本教程

1. **不要只读不跑**：每章的 CUDA 实验至少亲手敲一遍并运行
2. **先保证正确，再谈快**：GPU 优化的前提永远是结果正确
3. **用工具而不是猜测**：性能问题优先用 profiler 和 event 测量
4. **记录实验现象**：每次改 block 大小、访存模式或同步策略都记下变化
5. **多做对照实验**：比如共享内存 vs 直接访存、同步前 vs 同步后、coalesced vs strided

---

## 教程特色

- **24 章完整内容**：从 GPU 计算直觉到多 GPU、PTX 与 CUDA Graphs
- **实验驱动学习**：每章都有对应的 CUDA 实验与练习题
- **工程导向**：不仅讲原理，还讲 profiling、debugging、API 使用边界
- **深度学习连接**：覆盖 cuBLAS、cuDNN、Tensor Core、PyTorch Extension
- **中文编写**：术语统一，强调直觉解释和实际开发经验

---

## 与仓库其他教程的关系

本教程与本仓库其他系列教程形成互补关系：

- 如果你还不熟悉张量和 PyTorch，可先阅读 [Python 教程中的 PyTorch 张量基础](../python-tutorial/part6-pytorch/17-pytorch-tensors.md)
- 如果你对矩阵乘法、卷积和线性变换的数学背景不熟悉，可先阅读线性代数教程相关章节
- 如果你想理解 Flash Attention、Tensor Core、稀疏注意力等工程主题的底层背景，本教程可与 Transformer 教程搭配阅读

---

## 许可证

本项目采用 MIT 许可证开源。

---

*如有建议或发现错误，欢迎提交 Issue 或 Pull Request。*
