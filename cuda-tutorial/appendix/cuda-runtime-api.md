# 附录B：CUDA Runtime API 速查

## 1. 设备管理

### 设置当前设备

```cpp
cudaSetDevice(device_id);
```

### 获取设备数量

```cpp
int count = 0;
cudaGetDeviceCount(&count);
```

### 设备同步

```cpp
cudaDeviceSynchronize();
```

---

## 2. 内存管理

### 分配 device 内存

```cpp
cudaMalloc(&ptr, bytes);
```

### 释放 device 内存

```cpp
cudaFree(ptr);
```

### 分配 pinned host 内存

```cpp
cudaMallocHost(&host_ptr, bytes);
```

### 释放 pinned host 内存

```cpp
cudaFreeHost(host_ptr);
```

### 分配 Unified Memory

```cpp
cudaMallocManaged(&ptr, bytes);
```

---

## 3. 数据传输

### Host -> Device

```cpp
cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
```

### Device -> Host

```cpp
cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
```

### Device -> Device

```cpp
cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
```

### 异步拷贝

```cpp
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
```

### Unified Memory 预取

```cpp
cudaMemPrefetchAsync(ptr, bytes, device_id, stream);
```

---

## 4. Stream 管理

### 创建 stream

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
```

### 同步 stream

```cpp
cudaStreamSynchronize(stream);
```

### 销毁 stream

```cpp
cudaStreamDestroy(stream);
```

---

## 5. Event 管理

### 创建 event

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
```

### 记录 event

```cpp
cudaEventRecord(start, stream);
```

### 同步 event

```cpp
cudaEventSynchronize(stop);
```

### 计算耗时

```cpp
float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
```

### 销毁 event

```cpp
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

---

## 6. 错误处理

### 获取最近一次 launch 错误

```cpp
cudaGetLastError();
```

### 将错误码转成字符串

```cpp
cudaGetErrorString(err);
```

### 推荐错误检查宏

```cpp
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << cudaGetErrorString(err) << std::endl;\
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)
```

---

## 7. Kernel 启动示意

### 默认 stream

```cpp
my_kernel<<<blocks, threads>>>(args...);
```

### 指定 stream

```cpp
my_kernel<<<blocks, threads, 0, stream>>>(args...);
```

---

## 8. 常见调试模板

```cpp
CHECK_CUDA(cudaMalloc(&d_x, bytes));
CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

my_kernel<<<blocks, threads>>>(d_x, d_y, n);
CHECK_CUDA(cudaGetLastError());
CHECK_CUDA(cudaDeviceSynchronize());

CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
```

---

## 9. 使用建议

- 调试期优先加完整错误检查
- benchmark 时优先使用 CUDA events
- 引入异步执行前先确保串行版本完全正确
- 对标准算子优先考虑高性能库，而不是重复手写所有逻辑
