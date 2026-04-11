# 第15章：错误处理与调试

> 很多 CUDA 程序最麻烦的地方，不是“结果完全不对”，而是“有时候对、有时候错、错误还晚一点才冒出来”。

---

## 学习目标

学完本章，你将能够：

1. 理解 CUDA 错误为何经常表现为异步、延迟暴露
2. 掌握基础的 CUDA Runtime 错误检查模式
3. 知道 `cudaGetLastError()`、`cudaDeviceSynchronize()` 在调试中的作用
4. 了解 Compute Sanitizer 在定位内存错误和竞争问题中的价值
5. 建立一套“先定位正确性，再讨论性能”的调试习惯

---

## 15.1 为什么 CUDA 调试常常让人感觉“不直观”？

因为 CUDA 程序通常同时涉及：

- host 代码
- device 代码
- 异步 launch
- 多层内存空间
- 线程间并发行为

这会带来几个典型现象：

- 错误不一定在出问题的那一行立刻报出来
- 某个 kernel 已经越界了，但你可能到下一次 API 调用才看到错误
- 结果偶尔正确、偶尔错误，往往意味着同步或越界问题

所以 CUDA 调试的第一原则是：

- **不要只看“最后哪里报错”，要理解错误可能是在更早的异步操作中产生的。**

---

## 15.2 最基础的错误检查：检查 Runtime API 返回值

很多 CUDA Runtime API 都会返回 `cudaError_t`。

例如：

```cpp
cudaError_t err = cudaMalloc(&d_x, bytes);
if (err != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
}
```

### 为什么这一步重要？

因为如果：

- `cudaMalloc` 失败
- `cudaMemcpy` 失败
- event / stream 创建失败

而你没有立刻检查，后面看到的“奇怪结果”可能只是前面错误的连锁反应。

### 一个常见宏

很多示例会写成：

```cpp
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                      \
                      << cudaGetErrorString(err)              \
                      << std::endl;                           \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)
```

使用时：

```cpp
CHECK_CUDA(cudaMalloc(&d_x, bytes));
CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
```

初学阶段非常建议养成这种习惯。

---

## 15.3 Kernel 启动错误怎么检查？

kernel launch 和普通 Runtime API 有一点不同：

- launch 本身通常不会像普通函数那样直接返回详细执行错误
- 许多问题会在后续同步时才真正暴露

### 第一层检查：launch 后立刻看是否有启动层面的错误

```cpp
my_kernel<<<blocks, threads>>>(...);
CHECK_CUDA(cudaGetLastError());
```

这可以帮助你捕捉一些明显的 launch 配置错误，例如：

- 非法 launch 参数
- 某些立即可检测的问题

### 第二层检查：同步，逼迫异步错误尽早暴露

```cpp
my_kernel<<<blocks, threads>>>(...);
CHECK_CUDA(cudaGetLastError());
CHECK_CUDA(cudaDeviceSynchronize());
```

`cudaDeviceSynchronize()` 在调试阶段非常有价值，因为它会：

- 等待前面的 device 工作完成
- 让原本可能延迟暴露的错误尽快浮现

虽然它会影响性能，但调试正确性时这是值得的。

---

## 15.4 一个推荐的调试期检查模板

```cpp
my_kernel<<<blocks, threads>>>(d_in, d_out, n);
CHECK_CUDA(cudaGetLastError());
CHECK_CUDA(cudaDeviceSynchronize());
```

这段模板非常常见，适用于：

- 刚写好一个新 kernel
- 正在定位结果错误
- 怀疑存在越界或非法访存

### 为什么后期有时会去掉同步？

因为：

- 每次都同步会破坏异步流水和性能
- release / benchmark 阶段不应滥用同步

但在调试早期，这种“强行同步暴露错误”的方式通常更高效。

---

## 15.5 最常见的 CUDA 错误类型有哪些？

### 15.5.1 越界访问

例如：

- `idx < n` 判断漏了
- 二维索引错位
- halo / tile 边界装载错误

表现可能是：

- 程序崩溃
- 结果随机错误
- 某次运行正常，某次运行异常

### 15.5.2 host / device 指针混淆

例如：

- 把 host 指针传给 kernel
- 在 host 上直接解引用 device 指针
- `cudaMemcpy` 方向写错

### 15.5.3 同步问题

例如：

- shared memory 写完后忘记 `__syncthreads()`
- 只让部分线程进入同步点
- 错误假设不同 block 之间可以同步

### 15.5.4 竞争写问题

例如多个线程同时写同一地址，却没有：

- 原子操作
- 分阶段规约
- 正确的写冲突规避策略

---

## 15.6 Compute Sanitizer 是做什么的？

当你遇到这些问题时：

- 怀疑非法内存访问
- 怀疑越界读写
- 怀疑数据竞争
- 怀疑同步错误

Compute Sanitizer 是非常重要的工具。

### 你可以先把它理解成什么？

它类似于 CUDA 程序的“运行时诊断器”，能帮助你在执行期间发现：

- memory access 问题
- race condition
- 某些同步错误

### 为什么它有价值？

因为很多 CUDA bug 用肉眼看代码不一定立刻能发现，尤其是：

- 只在特定输入规模下出现
- 只在特定 launch 配置下出现
- 表现不稳定

这类问题非常适合借助工具定位。

---

## 15.7 结果不对时，一个实用的排查顺序

当 kernel 输出不对时，不要立刻猜“是不是 GPU 很神秘”，更有效的顺序通常是：

### 第一步：缩小输入规模

把大问题缩成：

- 8 个元素
- 16 个元素
- 一个很小矩阵

这样你可以手算预期结果。

### 第二步：检查索引与边界

重点检查：

- `idx` 公式
- `row/col` 公式
- `if (idx < n)` 或二维边界判断

### 第三步：加上错误检查与同步

使用：

```cpp
cudaGetLastError()
cudaDeviceSynchronize()
```

让错误尽早暴露。

### 第四步：把 kernel 逻辑简化

先保留最小计算，例如：

```cpp
out[idx] = in[idx];
```

如果这个都不对，问题很可能在：

- 索引
- 数据搬运
- 指针
- launch 配置

### 第五步：用 Compute Sanitizer

如果还是找不到，就用工具辅助定位。

---

## 15.8 为什么“先正确，后优化”是 CUDA 中的硬原则？

因为一旦你在结果不确定的情况下继续做优化：

- 问题会变得更难定位
- shared memory、异步执行、warp 优化只会让调试复杂度更高
- 你无法判断性能变化究竟来自优化还是 bug

正确的顺序应当是：

1. 先做一个最简单但正确的版本
2. 验证输出
3. 再逐步引入性能优化
4. 每次只改一层复杂度

这几乎是所有高质量 CUDA 工程的共同经验。

---

## 15.9 常见误区

### 误区一：程序没崩就说明没错

不对。GPU 越界、数据竞争、同步问题都可能不立即崩溃，却 silently 产生错误结果。

### 误区二：只检查 `cudaMemcpy` 或 `cudaMalloc` 就够了

不对。kernel launch 本身和 device 执行阶段也需要检查。

### 误区三：只靠打印最终结果排查

对于并行程序，这通常太粗糙。更有效的是：

- 缩小规模
- 验证中间阶段
- 用工具定位

### 误区四：为了性能，调试阶段也尽量不做同步

不对。调试阶段恰恰应该适度同步，让异步错误尽早暴露。

---

## 本章小结

| 核心概念 | 要点 |
|----------|------|
| 错误暴露方式 | CUDA 错误常常异步出现，报错位置未必就是根因位置 |
| Runtime API 检查 | `cudaMalloc`、`cudaMemcpy` 等都应检查返回值 |
| kernel 检查 | launch 后常用 `cudaGetLastError()`，调试时常配合 `cudaDeviceSynchronize()` |
| 常见 bug | 越界、指针混淆、同步错误、竞争写 |
| Compute Sanitizer | 用于定位内存错误、竞争条件等运行期问题 |
| 调试原则 | 先缩小规模、先保正确，再逐步引入优化 |

---

## CUDA实验

### 实验 1：给一个最小 kernel 加完整错误检查

把下面流程补完整：

1. 分配显存
2. 拷贝输入
3. 启动 kernel
4. 检查 launch 错误
5. 同步并检查执行错误
6. 拷回结果
7. 释放资源

要求你为每个 CUDA API 调用都加入错误检查。

### 实验 2：设计一个“小规模调试版”验证流程

针对一个二维矩阵 kernel，设计一个调试策略：

- 输入规模降到多小更适合手算
- 应先验证哪些位置
- 如果结果错误，应优先检查哪些索引公式和边界条件

---

## 练习题

1. **基础题**：为什么 CUDA 错误常常不是在真正出问题的那一行立刻暴露？
2. **基础题**：`cudaGetLastError()` 和 `cudaDeviceSynchronize()` 在调试中各起什么作用？
3. **实现题**：写一个 `CHECK_CUDA` 宏或等价的错误检查辅助函数。
4. **思考题**：为什么调试 CUDA 程序时要先把输入规模缩小？
5. **思考题**：为什么“先正确，后优化”在 CUDA 中尤其重要？
