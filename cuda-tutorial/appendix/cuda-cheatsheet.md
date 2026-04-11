# 附录A：CUDA 速查表

## 1. 常用命令

```bash
nvidia-smi              # 查看 GPU、驱动、显存占用
nvcc --version          # 查看 CUDA Toolkit / nvcc 版本
nvcc main.cu -o main    # 编译 CUDA 程序
nvcc -ptx main.cu -o main.ptx   # 导出 PTX
```

---

## 2. Kernel 启动模板

### 1D

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
my_kernel<<<blocks, threads>>>(...);
```

### 2D

```cpp
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y);
my_kernel<<<grid, block>>>(...);
```

---

## 3. 常用内建变量

### 1D

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 2D

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

### 3D

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

---

## 4. 常用内存 API

```cpp
cudaMalloc(&ptr, bytes);
cudaFree(ptr);

cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);

cudaMallocHost(&host_ptr, bytes);
cudaFreeHost(host_ptr);

cudaMallocManaged(&ptr, bytes);
```

---

## 5. 常用同步与错误检查

```cpp
cudaGetLastError();
cudaDeviceSynchronize();
__syncthreads();
```

推荐调试模板：

```cpp
my_kernel<<<blocks, threads>>>(...);
CHECK_CUDA(cudaGetLastError());
CHECK_CUDA(cudaDeviceSynchronize());
```

---

## 6. 常见函数修饰符

```cpp
__global__   // host 启动，device 执行
__device__   // device 内部调用
__host__     // host 执行
```

---

## 7. 常见性能检查清单

- 相邻线程是否访问相邻地址
- 是否存在明显重复 global memory 读取
- 能否用 shared memory 提高复用
- block 大小是否从 128 / 256 / 512 等常见值开始试
- benchmark 是否做了 warmup 和同步
- 是否先验证了结果正确

---

## 8. 常见并行模式关键词

- 向量加法：一个线程一个元素
- reduction：局部汇总后再全局合并
- scan：前缀位置生成
- compaction：标记 + scan + scatter
- stencil / convolution：tile + halo + 邻域复用

---

## 9. 常用工具

- `nvidia-smi`
- `nvcc`
- CUDA events
- Nsight Systems
- Nsight Compute
- Compute Sanitizer

---

## 10. 学习优先级建议

1. 先掌握索引、边界、内存分配、kernel launch
2. 再掌握 shared memory、同步、访存模式
3. 然后学习 profiling、并行模式、异步执行
4. 最后进入 Tensor Core、多 GPU、PTX、graphs 等高阶主题
