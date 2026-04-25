# 第19章：Kubernetes for AI

> Kubernetes 是 AI 平台最常见的运行底座，但它解决的是“通用编排问题”，不是全部 AI 语义问题。

> **关联章节**：如果你还不熟悉镜像、运行时和 GPU 设备是怎样接进容器的，建议先看 [第18章](./18-containers-and-runtime.md)。第18章解释的是执行路径，本章解释的是这些路径如何被 Kubernetes 组织起来。

## 学习目标

完成本章学习后，你将能够：

1. 理解 Kubernetes 在 AI 平台中的合适定位
2. 区分 Pod、Job、Deployment、CRD / Operator 在 AI 场景中的用途
3. 理解 GPU 调度、存储挂载、网络与设备插件如何进入 K8s 运行模型
4. 识别 Kubernetes 能解决什么，不能解决什么
5. 为训练和推理任务写出最小 K8s 表达草图

---

## 正文内容

### 19.1 Kubernetes 是底座，不是完整 AI 平台

Kubernetes 擅长：

- 运行容器
- 声明资源
- 服务发现
- 滚动发布
- 健康检查
- 基础伸缩

但它不直接解决：

- 数据集版本
- 实验追踪
- 模型评测
- 发布门禁
- KV Cache 调度

所以一个成熟 AI 平台通常是：

```text
AI control plane
  on top of
Kubernetes runtime plane
```

### 19.2 AI 场景常见对象

### Pod

最小运行单元，适合：

- 单个训练 worker
- 单个推理实例

### Job

适合：

- 训练任务
- 评测任务
- 批处理任务

### Deployment

适合：

- 在线推理服务
- 网关和辅助服务

### CRD / Operator

适合：

- 更高层的训练或 serving 语义
- 多 worker 协调
- 生命周期管理

### 19.3 一个训练 Job 草图

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-reranker
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: trainer
          image: ai-train:cuda12.4
          resources:
            limits:
              nvidia.com/gpu: 4
          command: ["python", "train.py"]
          args: ["--config", "configs/reranker.yaml"]
```

真实平台通常还会补：

- PVC / 对象存储挂载
- 环境变量
- 调度约束
- 节点选择
- 日志采集
- 失败重试策略

### 19.4 一个在线推理 Deployment 草图

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving
spec:
  replicas: 4
  template:
    spec:
      containers:
        - name: server
          image: llm-serving:latest
          resources:
            limits:
              nvidia.com/gpu: 1
```

上线场景里，Deployment 更看重：

- 副本数
- readiness / liveness
- 灰度策略
- 扩缩容联动

### 19.5 GPU 在 K8s 里不是“普通资源”

Kubernetes 原生很擅长 CPU / 内存，但 GPU 有额外复杂性：

- 型号不同
- 显存差异大
- 多卡任务需要 gang scheduling
- 某些节点有本地 NVMe、RDMA、NVLink 等附加特征

因此实际平台常常需要：

- device plugin
- 节点标签
- 拓扑感知调度
- 更高层队列系统

这正是 [第18章](./18-containers-and-runtime.md) 里“镜像 -> runtime -> GPU”链路被平台化之后的样子：K8s 本身不替你消化复杂性，只是把复杂性暴露成可编排对象。

#### 19.5.1 AI 场景下的关键 K8s 扩展

为什么 AI 集群几乎都会装额外扩展？因为原生 K8s 主要理解“多少资源”，但 AI 任务更常需要“什么形状的资源、在哪些节点、一起何时启动”。

| 扩展 / 机制 | 主要作用 | 什么时候最有价值 |
|------|----------|------------------|
| GPU Device Plugin | 把 GPU 资源注册给 K8s，让 Pod 能声明 `nvidia.com/gpu` | 所有 GPU 集群的基础前提 |
| Topology-aware scheduling | 结合节点标签、NUMA / PCIe / NVLink 拓扑做更稳的放置 | 多卡训练、带 NVLink 的高带宽拓扑 |
| NVIDIA GPU Operator | 统一管理驱动、container toolkit、device plugin、DCGM | 希望把 GPU 节点运维做成标准化基线 |
| Volcano | 提供队列、gang scheduling、批任务优先级语义 | 多卡训练、批处理和资源竞争明显的集群 |

这里的边界要说清楚：这些扩展能让 K8s 更像 AI 运行底座，但它们仍然不替代模型评测、发布门禁和成本治理控制面。

#### 19.5.2 分布式训练在 K8s 上的编排

分布式训练为什么不能只靠“开多个 Pod”？因为很多训练作业要求所有 worker 同时拿到资源、同时启动、同时知道彼此的 rank。只要有一个 worker 没起来，整个训练就可能白等。

| 关键需求 | 为什么需要 | K8s 常见实现 |
|------|------------|---------------|
| Gang Scheduling | 所有 worker 一起拿到资源，避免部分启动、部分等待 | Volcano；或 Kueue 配合 JobSet / 队列准入策略做近似控制 |
| 训练 Operator | 把 `master/worker`、失败重试、环境注入做成高层语义 | Kubeflow Training Operator、TorchJob 等 |
| `completionMode: Indexed` | 让每个 Pod 有稳定索引，便于 rank 映射和 rendezvous | `Job` / `Indexed Job` 场景 |

平台工程上更稳的做法不是让用户自己拼 Pod，而是让用户提交“训练任务”，再由 Operator 或控制面翻译成底层对象。

### 19.6 存储和网络在 K8s 中如何体现

### 存储

训练通常需要：

- 数据集读取
- checkpoint 输出
- 模型仓库访问

这会体现为：

- PVC
- 对象存储 sidecar / SDK
- 本地盘缓存

### 网络

训练和推理都依赖网络，但关注点不同：

- 训练更关注带宽和节点间通信稳定性
- 推理更关注服务链路延迟和入口流量治理

### 19.7 Kubernetes 的边界

K8s 不知道：

- 你的模型是否已通过评测
- 你的数据集版本是否正确
- 你的 KV Cache 是否会爆显存
- 你的多租户配额是否合理

因此，K8s 解决的是“怎么运行”，而不是“为什么运行、是否该运行、运行得好不好”。

在多租户场景里，K8s 还能提供两类基础隔离，但它们也只是底线能力：

| 机制 | 解决什么 | 解决不了什么 |
|------|----------|--------------|
| RBAC | 控制谁能看、改、删 namespace 内对象 | 不理解模型版本、数据权限、发布门禁 |
| ResourceQuota | 给 namespace 设 CPU / 内存 / GPU 上限 | 不能表达跨队列公平性和业务优先级 |

所以 namespace 级别的 RBAC 和配额更像“治理底板”，真正的多租户策略仍要继续上收到平台控制面（也可对照 [第20章](./20-queues-quotas-and-autoscaling.md) 的配额与队列机制）。

### 19.8 工程建议

- 用 Kubernetes 承接通用运行语义
- 把训练 / 推理 / 发布 / 评测的 AI 语义放在更高层控制面
- 对 GPU 任务强制加入节点标签、资源画像与调度约束
- 不要把所有 AI 问题都强行塞回原生 K8s 资源对象

### 本章涉及的常见工具

| 概念 | 常见工具 / 命令 | 备注 |
|------|-----------------|------|
| GPU 节点基线 | NVIDIA GPU Operator | 统一管理驱动、device plugin 与监控组件 |
| 批任务调度 | Volcano、Kueue | 常用于 gang scheduling 和队列治理 |
| 分布式训练编排 | Kubeflow Training Operator、TorchJob | 把训练 job 抽象成高层语义 |
| 模型服务编排 | KServe | 在 K8s 上封装模型服务部署和扩缩容 |

---

## 本章小结

| 对象 | AI 场景典型用途 |
|------|----------------|
| Pod | 单个 worker / serving 实例 |
| Job | 训练、评测、批处理 |
| Deployment | 在线推理与长期运行服务 |
| CRD / Operator | 训练和 serving 的高层语义封装 |

---

## 练习题

1. 为什么说 Kubernetes 是 AI 平台底座，而不是 AI 平台本身？
2. 训练任务和推理服务分别更适合哪些 K8s 对象？
3. GPU 为什么在 K8s 里不能被简单当成“另一个 CPU”？
4. 请写出一个需要额外调度语义的 AI 场景，说明为什么原生资源对象不够。
