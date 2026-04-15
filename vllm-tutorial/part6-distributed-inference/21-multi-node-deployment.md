# 第21章：多节点分布式部署

> 如果你还把 vLLM 的多节点理解成“先搭一个 Ray 集群，再启动服务”，那已经落后于当前仓库了。今天的 vLLM 至少有两条主路径：Ray 集群路线，以及基于 `--nnodes/--node-rank/--headless` 的 multiprocessing 路线。真正需要掌握的是拓扑规划、环境一致性、网络连通性和排障手法。

---

## 学习目标

学完本章，你将能够：

1. 理解当前 vLLM 多节点部署的两条主路径及其适用场景
2. 正确规划 `tensor_parallel_size`、`pipeline_parallel_size` 与节点关系
3. 使用 Ray 或 multiprocessing 启动多节点单副本推理
4. 理解 `--headless`、`--nnodes`、`--node-rank` 在当前源码中的作用
5. 处理 IP 选择、NCCL 网络接口和 GPUDirect RDMA 等常见问题

---

## 21.1 当前仓库里的多节点地图

`vllm/docs/serving/parallelism_scaling.md` 和 `vllm/vllm/entrypoints/cli/serve.py` 一起看，可以得到一个当前版本的更准心智模型：

### 路线 A：Ray

特点：

- 官方文档主推的多节点运行时
- 适合统一调度整个集群
- 一条命令即可看到整个 Ray cluster 的资源

### 路线 B：native multiprocessing

特点：

- 使用 `--nnodes`、`--node-rank`、`--master-addr`
- 非主节点常见写法是 `--headless`
- 更像传统 `torch.distributed` 风格的多节点启动

### 两条路线的共同前提

无论用哪条路，多节点部署都要求：

1. **每个节点的软件环境一致**
2. **模型路径一致，或都能访问同一份模型**
3. **节点间网络可达**
4. **NCCL / Ray 看到的是正确的 IP**

这也是为什么官方文档强烈建议：

- 使用容器
- 提前下载模型
- 在所有节点使用一致的挂载路径

---

## 21.2 官方推荐的资源规划方式

对于**单副本模型**，当前文档给出的常见规划是：

```text
tensor_parallel_size = 每个节点的 GPU 数
pipeline_parallel_size = 节点数
```

例如 2 个节点、每节点 8 卡：

```text
TP = 8
PP = 2
```

这意味着：

- 节点内主要靠 TP
- 节点间主要靠 PP

这是当前教程里最该记住的多节点经验法则。

### 但它不是唯一方案

文档也明确说了另一种可能：

- 直接把 `tensor_parallel_size` 设成全 cluster 的总 GPU 数

例如 16 卡 cluster 直接设：

```text
TP = 16
```

这是否更好，取决于：

- 网络带宽
- 是否有 RDMA / GPUDirect
- 模型结构
- TP 通信开销能否承受

所以当前多节点调优的态度不是“抄一份标准答案”，而是：

```text
先按 TP=每节点卡数、PP=节点数起步
再按测量结果调整
```

---

## 21.3 Ray 路线：当前官方文档的标准做法

### 1. 启动集群

官方文档推荐使用：

- `examples/online_serving/run_cluster.sh`

仓库里还有一个更轻量的辅助脚本：

- `examples/online_serving/multi-node-serving.sh`

后者提供了：

- `leader`
- `worker`

两个子命令，本质上是在帮你包装 `ray start --head` / `ray start --address=...`。

### 2. 为每个节点设置正确的 `VLLM_HOST_IP`

这是当前多节点排障里最重要的变量之一。

官方文档明确强调：

- 每个节点上的 `VLLM_HOST_IP` 都应该是**该节点自己的可达 IP**
- 最好位于**私有网络**

原因有两个：

1. Ray / vLLM 需要用它来定位节点
2. 文档明确提醒，多节点内部流量是**未加密**的，不应暴露到不可信网络

### 3. 检查集群

进入容器后执行：

```bash
ray status
ray list nodes
```

确认：

- 节点数正确
- GPU 资源正确
- 每个节点的 IP 和你预期一致

### 4. 启动 vLLM

典型命令：

```bash
vllm serve /path/to/model \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray
```

这就是当前官方文档里最典型的多节点单副本部署方式。

---

## 21.4 multiprocessing 路线：当前仓库不能忽略的第二条主线

旧教程最容易漏掉这一点：当前 vLLM 文档已经明确给出了**multi-node multiprocessing** 的例子。

### 主节点

```bash
vllm serve /path/to/model \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr <HEAD_NODE_IP>
```

### 其他节点

```bash
vllm serve /path/to/model \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr <HEAD_NODE_IP> \
  --headless
```

### `--headless` 到底在干什么？

看 `vllm/vllm/entrypoints/cli/serve.py` 的 `run_headless(...)`，能更准确理解它：

- **不启动 API server**
- 只运行当前节点负责的 engine / worker
- 在非主节点上作为“只做执行、不接 HTTP”的节点存在

也就是说，`--headless` 不是一个“调试模式”，而是当前 multiprocessing 多节点部署的重要组成部分。

### `create_engine_config()` 做了哪些检查？

在 `vllm/vllm/engine/arg_utils.py` 里，多节点相关的校验包括：

- `world_size % nnodes == 0`
- `node_rank < nnodes`
- 根据 `nnodes` 计算 local world size

这些检查能帮你尽早发现拓扑参数写错。

---

## 21.5 多节点时最容易出问题的，不是模型，而是网络

### 问题 1：节点之间根本没走到正确网卡

官方排障文档 `docs/serving/distributed_troubleshooting.md` 明确建议：

- 通过集群启动脚本统一传环境变量
- 不要只在当前 shell 里临时 `export`

最常见变量是：

```bash
NCCL_SOCKET_IFNAME=eth0
```

如果你在本地 shell 设置，但 worker 节点没继承到，排障会非常痛苦。

### 问题 2：`No available node types can fulfill resource request`

这类错误并不一定代表 GPU 真不够，当前文档给出的常见根因是：

- 机器有多个 IP
- Ray 和 vLLM 选的不是同一个 IP

优先检查：

- `VLLM_HOST_IP`
- `ray status`
- `ray list nodes`

### 问题 3：跨节点 TP 性能差

官方文档建议：

- 尽量使用 InfiniBand / RDMA
- 需要时开启 GPUDirect RDMA
- 用 `NCCL_DEBUG=TRACE` 看实际走的是哪条链路

如果日志里出现：

- `NET/IB/GDRDMA`：说明走到了高性能路径
- `NET/Socket`：说明只是普通 TCP socket，跨节点 TP 通常会比较差

---

## 21.6 当前多节点文档已经不只讲“一个副本跨机器”

这是另一个需要校准的点。

如果继续沿着 `docs/serving/` 往下读，你会发现当前仓库的分布式地图已经明显扩大了：

- `data_parallel_deployment.md`
- `context_parallel_deployment.md`
- `expert_parallel_deployment.md`
- `distributed_troubleshooting.md`

也就是说，今天的“多节点 vLLM”不只是在讲：

```text
把一个 405B 模型拆到几台机器
```

还在讲：

- 多副本 DP
- 长上下文 DCP
- MoE 的 EP
- 多节点 headless / hybrid / external LB

所以本章聚焦的是**单副本多节点主线**，但你应该知道当前仓库的分布式能力已经远不止这一种形态。

---

## 21.7 多节点部署的当前源码锚点

| 主题 | 当前文件 |
|------|----------|
| 并行与扩缩容总说明 | `vllm/docs/serving/parallelism_scaling.md` |
| 多节点排障 | `vllm/docs/serving/distributed_troubleshooting.md` |
| multiprocessing 设计说明 | `vllm/docs/design/multiprocessing.md` |
| Ray 辅助脚本 | `vllm/examples/online_serving/multi-node-serving.sh` |
| 容器集群脚本 | `vllm/examples/online_serving/run_cluster.sh` |
| CLI headless 路径 | `vllm/vllm/entrypoints/cli/serve.py` |
| 多进程 executor | `vllm/vllm/v1/executor/multiproc_executor.py` |
| 参数与 world size 校验 | `vllm/vllm/engine/arg_utils.py` |
| DP / CP / EP 延展文档 | `vllm/docs/serving/*.md` |

---

## 本章小结

| 结论 | 当前仓库里的正确理解 |
|------|--------------------|
| 多节点主路径 | Ray 和 multiprocessing 两条路线都要会看 |
| 常见拓扑 | `TP=每节点 GPU 数`，`PP=节点数` |
| `--headless` 作用 | 非主节点只跑 engine / worker，不启动 API server |
| 关键前提 | 环境一致、模型路径一致、IP 选择正确、网络可达 |
| 最常见故障点 | 错网卡、错 IP、NCCL 走了普通 socket |

---

## 动手实验

### 实验 1：验证多节点 Ray 集群感知

在测试环境里搭一个两节点 Ray 集群，执行：

```bash
ray status
ray list nodes
```

记录：

- 看到的节点数
- 每个节点的 IP
- 总 GPU 数

确认它和你的部署规划一致。

### 实验 2：用 headless 路径跑一个最小双节点案例

找一个较小模型，在两台机器上分别执行：

- 主节点：正常 `vllm serve`
- 从节点：`vllm serve ... --headless`

确认：

- 从节点不暴露 HTTP 端口
- 主节点仍能对外提供统一服务

---

## 练习题

### 基础题

1. 当前 vLLM 的多节点单副本部署，至少有哪两条主路径？
2. `--headless` 在 multiprocessing 多节点路线里扮演什么角色？

### 思考题

3. 遇到 `No available node types can fulfill resource request` 时，你会先检查哪几个与 IP 相关的设置？
4. 如果跨节点 TP 性能很差，你会如何用文档里的方法判断它到底走的是 RDMA 还是普通 socket？
