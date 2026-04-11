# 第21章：PyTorch 自定义 CUDA 算子

> 真正把 CUDA 能力接到深度学习工程里，关键不是单独写出一个 kernel，而是理解如何让框架世界和自定义底层实现稳妥地衔接起来。

---

## 学习目标

学完本章，你将能够：

1. 理解为什么 PyTorch 工程有时需要自定义 CUDA 算子
2. 知道一个最小自定义算子通常由哪些部分组成
3. 建立 Python 接口、C++ 包装层和 CUDA kernel 之间关系的整体认识
4. 理解形状、dtype、device 检查在框架集成中的重要性
5. 知道何时值得写自定义 CUDA 算子，何时应优先使用现有算子组合

---

## 21.1 为什么已经有 PyTorch 了，还要自己写 CUDA 算子？

大多数时候，PyTorch 内置算子已经足够强大。

你通常只有在下面场景才需要考虑自定义 CUDA 算子：

### 场景一：现有算子组合表达不自然

例如一个操作需要：

- 特殊数据访问模式
- 非标准数学逻辑
- 自定义融合流程

如果用多个现有张量操作拼接，可能会：

- launch 太多 kernel
- 中间张量过多
- 访存冗余严重

### 场景二：你确实需要一个专用高性能 kernel

例如：

- 特殊 attention 变体
- 自定义稀疏操作
- 专门为某种布局设计的融合算子

### 场景三：研究型工作

在论文复现、原型验证或框架扩展中，这也很常见。

---

## 21.2 一个最小自定义 CUDA 算子通常包含哪些部分？

高层上，通常会有三层：

1. **Python 层**：供用户或模型代码直接调用
2. **C++ 包装层**：负责和 PyTorch 扩展接口对接，做参数检查与 dispatch
3. **CUDA kernel 层**：真正执行 GPU 计算

你可以把它理解为：

```text
Python API
    ↓
C++ extension wrapper
    ↓
CUDA kernel launch
```

### 为什么需要中间 C++ 层？

因为 PyTorch extension 通常需要：

- 获取 `Tensor` 元信息
- 检查 device / dtype / contiguous 等条件
- 从张量拿到底层指针
- 组织 launch 参数
- 调用 CUDA kernel

这些工作放在 C++ 层最自然。

---

## 21.3 一个最小结构示意

### Python 侧调用

```python
y = my_ops.add_bias_cuda(x, bias)
```

### C++ 包装层的职责

高层上，它通常会做：

- 检查 `x` 是否在 CUDA 上
- 检查 dtype 是否符合预期
- 检查形状是否匹配
- 分配输出张量
- 调用 CUDA 实现函数

### CUDA 实现层的职责

- 根据张量尺寸计算 `blocks` / `threads`
- 启动 kernel
- 执行真正的逐元素或块级计算

---

## 21.4 为什么 shape / dtype / device 检查非常重要？

在纯 CUDA 教程里，我们常常默认输入都是正确的。

但一旦接入 PyTorch，算子就成了更大的系统边界，用户可能传入：

- CPU tensor
- 错误 dtype
- 非 contiguous tensor
- 维度不匹配的数据

如果你不在包装层检查这些条件，就可能导致：

- kernel 读错布局
- launch 参数不匹配
- 指针解释错误
- 程序崩溃或 silently 输出错误结果

### 所以要建立一个工程习惯

- 在 Python/C++ 系统边界做输入约束检查
- 不要把所有错误都留给底层 kernel 去“自己撞出来”

---

## 21.5 一个最小逐元素算子的高层骨架

假设我们实现一个简单算子：

```text
out[i] = x[i] + bias
```

### CUDA kernel

```cpp
__global__ void add_bias_kernel(const float* x, float* out, float bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + bias;
    }
}
```

### C++ 包装层直觉

```cpp
torch::Tensor add_bias_cuda(torch::Tensor x, float bias) {
    // 1. check CUDA / dtype / layout
    // 2. allocate output
    // 3. launch kernel
    // 4. return output
}
```

### Python 暴露层

```python
out = ext.add_bias_cuda(x, 1.0)
```

这已经构成了最基础的框架集成闭环。

---

## 21.6 为什么“一个 kernel 接进 PyTorch”比单独写 kernel 更复杂？

因为这不只是 GPU 编程问题，还涉及框架接口问题：

- 张量生命周期
- dtype dispatch
- device dispatch
- contiguous / stride 语义
- autograd 支持
- Python API 设计
- 编译与打包

所以自定义 CUDA 算子的难点，往往不是“多写几十行 kernel”，而是把它稳妥地接进框架生态。

---

## 21.7 Autograd 是什么时候变成问题的？

如果你的算子只在推理里使用，或者只作为前向实验原型，可能先不考虑反向。

但如果它要进入训练图中，就很快会遇到：

- 这个算子的 backward 怎么定义？
- 是否需要额外保存中间值？
- backward 是不是也要写自定义 CUDA kernel？

这说明自定义算子通常并不只是“前向快一点”这么简单，而是可能扩大成：

- 前向实现
- 反向实现
- 数值检查
- 框架注册与测试

初学阶段建议先把重点放在前向路径的最小闭环上。

---

## 21.8 什么时候不值得写自定义 CUDA 算子？

### 情况一：现有 PyTorch 算子组合已经足够好

如果只是几个标准逐元素操作，PyTorch 往往已经能做得不错，未必值得专门写扩展。

### 情况二：问题规模太小

如果工作量很小，扩展维护成本和编译复杂度可能得不偿失。

### 情况三：你还没证明瓶颈真的在这里

如果没有 benchmark 和 profiler 证据，直接写自定义算子往往属于过早优化。

### 情况四：算子定义还不稳定

如果算法本身经常变，先用 Python 原型保持迭代速度，通常更合理。

---

## 21.9 一个现实中的工作流建议

更稳妥的流程通常是：

1. 先用纯 PyTorch 版本做正确原型
2. 用 benchmark 找出热点
3. 证明热点无法被现有算子组合高效覆盖
4. 再实现最小自定义前向 CUDA 算子
5. 验证数值一致性
6. 如有需要，再补 backward
7. 最后做工程化整理

这能避免一开始就陷入：

- 编译配置复杂
- 结果难验证
- 还不知道到底值不值得写

---

## 21.10 常见误区

### 误区一：写了自定义 CUDA 算子就一定比 PyTorch 原生快

不对。框架内部很多算子已经高度优化，甚至会自动走到底层高性能库。

### 误区二：只要 kernel 能跑，扩展就算完成

不对。框架集成还要考虑：

- 输入检查
- dtype / device 行为
- contiguous 问题
- 测试
- backward（如果训练需要）

### 误区三：所有性能问题都该靠自定义算子解决

不对。很多性能问题也可能通过：

- 调整张量布局
- 减少不必要数据搬运
- 使用现有 fused 算子
- 更好地利用库

来解决。

### 误区四：自定义算子一定要从最复杂版本开始

不对。最好的路径通常是：

- 先做最小前向版本
- 先保证正确
- 再逐步加复杂度

---

## 本章小结

| 核心概念 | 要点 |
|----------|------|
| 自定义 CUDA 算子 | 用于补足 PyTorch 现有算子难以高效表达的特殊需求 |
| 三层结构 | Python API、C++ 包装层、CUDA kernel |
| 包装层职责 | 负责 shape / dtype / device 检查和 launch 组织 |
| 工程边界 | 接进框架比单独写 kernel 更复杂 |
| autograd | 若用于训练，往往还需考虑 backward |
| 开发流程 | 先原型、后定位热点、再决定是否写自定义扩展 |

---

## CUDA实验

### 实验 1：设计一个最小 PyTorch 自定义算子结构

假设你要实现：

```text
out[i] = x[i] * scale + bias
```

请写出三层结构草图：

1. Python 调用形式
2. C++ 包装层应检查哪些内容
3. CUDA kernel 的核心逻辑是什么

### 实验 2：分析一个算子是否值得自定义

请判断下面两种场景是否更值得写自定义 CUDA 算子：

1. 三个标准逐元素算子串联，PyTorch 已有高效实现
2. 一个特殊稀疏布局下的融合算子，现有算子组合导致大量中间张量和访存浪费

请说明原因，而不是只给结论。

---

## 练习题

1. **基础题**：一个最小 PyTorch 自定义 CUDA 算子通常包含哪三层？
2. **基础题**：为什么 C++ 包装层需要做 shape / dtype / device 检查？
3. **实现题**：写出一个逐元素自定义 CUDA 算子的高层伪流程。
4. **思考题**：为什么“写自定义算子”经常比“单独写一个 CUDA kernel”复杂得多？
5. **思考题**：为什么在没有 benchmark 证据前，不应轻易决定写自定义 CUDA 扩展？
