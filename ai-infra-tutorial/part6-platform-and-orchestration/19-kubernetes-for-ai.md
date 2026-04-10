# 第19章：Kubernetes for AI

> Kubernetes 是 AI 平台最常见的运行底座，但它解决的是“通用编排问题”，不是全部 AI 语义问题。

## 学习目标

完成本章学习后，你将能够：

1. 理解 Kubernetes 在 AI 平台中的合适定位
2. 区分 Pod、Job、Deployment、CRD / Operator 在 AI 场景中的用途
3. 理解 GPU 调度、存储挂载、网络与设备插件如何进入 K8s 运行模型
4. 识别 Kubernetes 能解决什么，不能解决什么
5. 为训练和推理任务写出最小 K8s 表达草图

---

## 正文内容

## 19.1 Kubernetes 是底座，不是完整 AI 平台

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

## 19.2 AI 场景常见对象

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

## 19.3 一个训练 Job 草图

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

## 19.4 一个在线推理 Deployment 草图

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

## 19.5 GPU 在 K8s 里不是“普通资源”

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

## 19.6 存储和网络在 K8s 中如何体现

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

## 19.7 Kubernetes 的边界

K8s 不知道：

- 你的模型是否已通过评测
- 你的数据集版本是否正确
- 你的 KV Cache 是否会爆显存
- 你的多租户配额是否合理

因此，K8s 解决的是“怎么运行”，而不是“为什么运行、是否该运行、运行得好不好”。

## 19.8 工程建议

- 用 Kubernetes 承接通用运行语义
- 把训练 / 推理 / 发布 / 评测的 AI 语义放在更高层控制面
- 对 GPU 任务强制加入节点标签、资源画像与调度约束
- 不要把所有 AI 问题都强行塞回原生 K8s 资源对象

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

