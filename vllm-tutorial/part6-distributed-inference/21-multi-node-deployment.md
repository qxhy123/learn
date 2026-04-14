# 第21章：多节点分布式部署

> 当单台服务器的 GPU 不够用时，你需要把推理分布到多台机器上。这引入了网络通信、集群管理和故障恢复等新的挑战。

---

## 学习目标

学完本章，你将能够：

1. 理解多节点推理的架构和通信需求
2. 使用 Ray 搭建多节点 vLLM 推理集群
3. 配置节点间的网络通信
4. 诊断多节点部署的常见问题
5. 评估多节点部署的性能开销

---

## 21.1 多节点架构

### 什么时候需要多节点？

```
单节点 8×A100 (640 GB 显存):
  可以部署: ≤ 405B 模型 (FP16, TP=8)

需要多节点的场景:
  1. 超大模型 (405B+ FP16)
  2. 需要更多 KV Cache 空间 (超高并发)
  3. 多副本高可用部署
```

### 基本架构

```
┌────────────────────────┐     ┌────────────────────────┐
│     Node 0 (Head)      │     │     Node 1 (Worker)    │
│  ┌─────┬─────┬─────┐  │     │  ┌─────┬─────┬─────┐  │
│  │GPU0 │GPU1 │GPU2 │  │     │  │GPU0 │GPU1 │GPU2 │  │
│  │GPU3 │GPU4 │GPU5 │  │ ←→  │  │GPU3 │GPU4 │GPU5 │  │
│  │GPU6 │GPU7 │     │  │ 网络 │  │GPU6 │GPU7 │     │  │
│  └─────┴─────┴─────┘  │     │  └─────┴─────┴─────┘  │
│  API Server + Scheduler │     │  Worker Processes      │
└────────────────────────┘     └────────────────────────┘
          ↑
       用户请求
```

---

## 21.2 使用 Ray 部署

### 集群设置

```bash
# --- Node 0 (Head) ---
ray start --head --port=6379

# --- Node 1 (Worker) ---
ray start --address='node0-ip:6379'
```

### 启动 vLLM

```bash
# 在 Head 节点上启动
vllm serve meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray
```

### 验证集群

```python
import ray
ray.init(address="auto")
print(f"可用节点: {ray.nodes()}")
print(f"可用 GPU: {ray.available_resources().get('GPU', 0)}")
```

---

## 21.3 网络配置

### 关键要求

```
节点间通信方式:
  1. NCCL over TCP/IP → 最简单，但较慢
  2. NCCL over InfiniBand/RDMA → 推荐，高性能

环境变量:
  export NCCL_SOCKET_IFNAME=eth0      # 网络接口
  export NCCL_IB_DISABLE=0            # 启用 InfiniBand
  export GLOO_SOCKET_IFNAME=eth0      # Gloo 通信接口
```

### 带宽需求

```
节点间通信数据量 (PP模式):
  每层激活值: hidden_size × batch_tokens × dtype_size
  Llama-70B: 8192 × 256 × 2 = 4 MB/层

  对于 PP=2，每个 iteration:
  通信量 ≈ 4 MB (激活传递) + 4 MB (梯度回传)
  
需要: 至少 10 Gbps 网络，推荐 100 Gbps+
```

---

## 21.4 常见问题

### 问题 1：节点间连接超时

```bash
# 增加超时时间
export NCCL_TIMEOUT=1800  # 30 分钟
export RAY_BACKEND_LOG_LEVEL=debug
```

### 问题 2：性能不理想

```
检查清单:
  1. 确认节点间网络带宽 (iperf3 测试)
  2. 确认 NCCL 使用了正确的网络接口
  3. 确认 GPU 之间的拓扑 (nvidia-smi topo -m)
  4. 优先在节点内使用 TP，节点间使用 PP
```

### 问题 3：GPU 内存不均匀

```bash
# 确保所有节点的 GPU 型号和显存一致
# 不一致时，vLLM 以最小显存为基准
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 何时需要 | 超大模型或超高并发 |
| 集群管理 | Ray 集群 |
| 网络要求 | 推荐 InfiniBand/RDMA |
| 并行策略 | 节点内 TP + 节点间 PP |
| 常见问题 | 超时、带宽不足、配置不一致 |

---

## 练习题

### 基础题

1. 多节点部署中，TP 和 PP 通常如何分配？
2. 为什么节点间推荐使用 PP 而不是 TP？

### 思考题

3. 如果你有 4 个节点，每个 8 张 A100，要部署 Llama-405B，你会怎么规划？
4. 多节点部署相比多副本单节点部署有什么优劣？
