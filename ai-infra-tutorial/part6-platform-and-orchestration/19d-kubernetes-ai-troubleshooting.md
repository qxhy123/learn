# 第19d章：Kubernetes for AI 排障 SOP

> AI on Kubernetes 的排障目标不是“找到一行看起来像原因的日志”，而是建立证据链：对象状态、事件、调度条件、容器日志、节点状态、GPU/DCGM 指标、NCCL 日志、网络、存储和服务入口逐层闭环。本章把常见故障拆成可执行 SOP。

---

## 19d.1 第一性原理拆解 + 学习大纲

### 拆：不可化简的问题

Kubernetes 的一个症状往往不是一个根因。`Pod Pending` 可能是 GPU 不足、节点标签不匹配、污点没有容忍、PVC 没绑定、quota 超限、gang 调度未准入；`NCCL timeout` 可能是 rank 没齐、DNS 解析错、NetworkPolicy 阻断、RDMA 设备没注入、MTU 不一致、某个 rank OOM、GPU Xid 或跨拓扑通信退化。

AI 工作负载更难排，因为它横跨多个平面：

- Kubernetes 控制面：API Server、scheduler、controller、events、CRD status。
- 节点运行时：kubelet、container runtime、CNI、CSI、device plugin。
- 加速器：GPU driver、CUDA、NVIDIA container toolkit、MIG、DCGM。
- 分布式通信：NCCL、MPI、Ray、RDMA、RoCE/IB、DNS、Service。
- 存储路径：PVC、对象存储、并行文件系统、checkpoint、模型下载。
- 应用语义：训练 rank、driver、launcher、model server、readiness、batching。

不可化简的问题是：**如何把用户看到的症状，映射到具体平面的具体失败点，并形成可复现、可交接、可自动化的证据链。**

### 推：从问题推导 SOP

从“症状跨层”推出固定顺序：先看对象状态和 Events，再看容器日志，再看节点、GPU、网络、存储和应用指标。顺序错了会浪费时间，比如没确认 Pod 是否调度成功就去看 NCCL。

从“证据会消失”推出先采集再修复。`kubectl logs --previous`、events、节点日志、DCGM 异常窗口、NCCL debug 日志都可能被轮转或覆盖。生产排障前先留证据，再删除 Pod 或重启服务。

从“AI 故障常复发”推出标准 runbook。Pending、ImagePullBackOff、GPU 不可见、NCCL timeout、OOMKilled、readiness flapping、Service/Ingress、PVC、DNS 都应该有固定证据、判断和处理动作。

从“平台要规模化值班”推出结论格式：症状、影响范围、证据、根因、动作、后续预防。没有证据链的“可能是网络问题”不能作为事故结论。

### 学习大纲

本章按以下顺序展开：

1. 概念边界：Kubernetes 排障、AI 排障、应用排障、性能排障分别是什么。
2. 架构：控制路径、数据路径、观测路径和责任边界。
3. 原理：为什么要按状态机排障，Events、describe、logs、node、DCGM、NCCL logs 各自证明什么。
4. 工程化：生产 SOP、证据保全、版本矩阵、发布变更、观测与治理。
5. 方案设计：一套可执行的值班排障决策表和证据模板。
6. 专项 SOP：Pending、ImagePull、GPU、NCCL、OOM、readiness、Service/Ingress、PVC、DNS。
7. 反模式、Checklist、Worked Example 和练习题。

---

## 19d.2 概念先说清楚

### Kubernetes 排障是什么，不是什么

Kubernetes 排障是从 API 对象和控制器状态出发，判断系统为什么没有达到期望状态。它回答的是：

- Pod 为什么没有被调度？
- 容器为什么没有启动？
- Service 为什么没有 endpoint？
- Deployment 为什么 rollout 卡住？
- CRD 为什么 status 不更新？

它不直接回答“模型为什么收敛差”或“推理质量为什么下降”。那些属于模型、数据或应用逻辑问题，但 Kubernetes 排障可以先证明基础设施是否正常。

### AI 排障是什么，不是什么

AI 排障是在 Kubernetes 排障基础上加入 GPU、分布式通信、模型加载、checkpoint 和 AI runtime 语义。它要回答：

- GPU 是否被调度、注入、识别和正确使用？
- NCCL/MPI/Ray 通信是否所有 rank 齐、网络通、设备路径正确？
- 模型是否下载、加载、warmup 并进入 ready？
- OOM 是 cgroup memory、GPU memory、ephemeral storage 还是节点压力？
- 训练失败是框架错误、基础设施故障还是外部依赖故障？

AI 排障不是调参。遇到 NCCL timeout 先不要调 batch size；遇到 OOMKilled 先不要加 GPU；遇到 readiness 抖动先不要改 Ingress。先定位层级，再做动作。

### 证据链是什么

证据链是能支持结论的一组事实，至少包含：

| 证据 | 证明什么 | 常用命令 |
|------|----------|----------|
| 对象 status | 当前 Kubernetes 认为对象处于什么状态 | `kubectl get -o yaml` |
| Events | 控制器、scheduler、kubelet 记录的近期原因 | `kubectl describe`、`kubectl get events` |
| 容器日志 | 应用、runtime、框架的直接错误 | `kubectl logs` |
| previous logs | 上一次崩溃前的错误 | `kubectl logs --previous` |
| Node 状态 | 节点资源、压力、taint、allocatable | `kubectl describe node` |
| GPU/DCGM | GPU 健康、Xid、显存、温度、利用率、ECC | DCGM exporter、节点 `nvidia-smi` |
| NCCL logs | rank、网卡、RDMA、collective 过程 | `NCCL_DEBUG=INFO` |
| Service/Endpoint | 流量是否有 ready backend | `kubectl get svc,endpointslice` |
| PVC/CSI | 存储是否绑定、挂载、读写异常 | `kubectl describe pvc/pv/pod` |
| DNS | 名称解析和服务发现是否正常 | `nslookup`、`dig`、CoreDNS logs |

单条日志不是证据链。证据链应该能解释“为什么这个症状发生在这个对象、这个时间、这个范围”。

### SOP、Runbook、Postmortem 的边界

SOP 是当下排障步骤。Runbook 是可复用操作手册。Postmortem 是事故后复盘，关注根因、影响、修复和预防。值班时先按 SOP 采证和止血，事后再把新经验沉淀到 runbook。

---

## 19d.3 架构：排障视角下的路径和责任边界

### 控制路径

```text
kubectl / platform API
  -> API Server
  -> admission / CRD schema
  -> controller / operator
  -> scheduler / queue
  -> kubelet
  -> container runtime
  -> Pod status / events
```

控制路径上的失败通常表现为对象没有进入期望状态：Pending、ContainerCreating、ImagePullBackOff、CrashLoopBackOff、CRD status 卡住、rollout 超时。

### 数据路径

```text
训练：dataset / object storage / PVC -> Pod -> GPU -> NCCL/RDMA -> peer GPU -> checkpoint storage
推理：client -> DNS -> Ingress/Gateway -> Service -> EndpointSlice -> Pod -> model runtime -> GPU
```

数据路径上的失败通常表现为任务运行后卡住、timeout、吞吐下降、502/504、模型下载失败、checkpoint 写失败、readiness 抖动。

### 观测路径

```text
Kubernetes events/status
  + container logs
  + operator logs
  + node/kubelet/runtime logs
  + Prometheus metrics
  + DCGM exporter
  + NCCL/MPI/Ray logs
  + tracing / ingress logs
```

排障不是所有来源都看一遍，而是按症状选择最短证据路径。比如 Pending 首先看 scheduler events；NCCL timeout 首先看所有 rank 状态和 NCCL INIT/NET 日志；Service 不通首先看 EndpointSlice。

### 责任边界

| 责任方 | 负责 | 不负责 |
|--------|------|--------|
| 用户代码 | 训练逻辑、模型加载、应用日志、checkpoint 调用 | 节点 GPU 驱动和调度策略 |
| 平台 Operator | CRD 状态、底层对象、events、默认配置 | 容器内业务 bug |
| Scheduler/Queue | 资源准入、节点选择、gang、quota | 应用 readiness 语义 |
| GPU 平台 | driver、device plugin、runtime、DCGM | PyTorch 参数正确性 |
| 网络平台 | CNI、Service、Ingress、RDMA fabric、DNS | 模型自身延迟 |
| 存储平台 | PVC、CSI、对象存储、吞吐和可用性 | 训练 checkpoint 策略 |

清楚边界不是为了推责，而是为了让排障动作有针对性。

---

## 19d.4 原理：为什么这些证据能定位问题

### Pod 状态机

Pod 生命周期中的常见状态和排障含义：

| 状态/原因 | 意味着什么 | 优先看什么 |
|-----------|------------|------------|
| Pending | Pod 未绑定节点或依赖未满足 | `describe pod` Events、quota、PVC、node |
| ContainerCreating | 已调度，容器还没启动 | CNI、CSI mount、image unpack、runtime |
| ImagePullBackOff | 镜像拉取失败且进入退避 | Events、Secret、registry、节点网络 |
| CrashLoopBackOff | 容器启动后退出并反复重启 | current/previous logs、exit code、probe |
| OOMKilled | 容器超过 cgroup memory limit 被杀 | lastState、node memory、应用内存 |
| Running but NotReady | 容器运行但 readiness 未通过 | probe events、app logs、model load |
| Succeeded/Failed | Job 类任务完成或失败 | exit code、logs、Job/CRD status |

先确认状态机位置，再决定下一步。Pending 的 Pod 没有容器日志；ImagePullBackOff 的 Pod 没有应用日志；CrashLoopBackOff 要看 `--previous`。

### Events 的价值和限制

Events 是 scheduler、kubelet、controller 等组件写出的近期事实，适合定位调度、镜像、挂载、probe、驱逐等问题。

限制是：

- Events 会被 TTL 清理。
- Events 有聚合和限流，可能丢细节。
- Events 记录的是组件观察，不一定包含应用根因。

因此 Events 是入口，不是全部结论。

### Logs 的价值和限制

容器日志证明应用或 runtime 输出了什么。对于 AI 任务，必须区分：

- 主容器日志：训练脚本、模型服务 runtime。
- init container 日志：模型下载、依赖准备。
- sidecar 日志：model agent、mesh proxy、queue proxy。
- previous logs：崩溃前的最后输出。
- Operator logs：控制器为什么没有收敛。

没有打开 NCCL debug 时，训练日志可能只显示 timeout，不显示网卡选择和 rank 细节。生产镜像应支持通过环境变量打开详细日志。

### Node、GPU、DCGM 的价值

节点证据证明问题是否发生在单个节点、单类节点池或整个集群：

- Node condition：Ready、MemoryPressure、DiskPressure、PIDPressure、NetworkUnavailable。
- Allocatable：GPU、CPU、memory、ephemeral-storage 是否存在且足够。
- Taints：节点是否被标记不可调度或维护。
- DCGM：GPU Xid、ECC、温度、功耗、显存、利用率、NVLink 错误。
- `nvidia-smi`：驱动、CUDA 兼容、MIG、进程、拓扑。

GPU 不可见或 NCCL 超时不能只看 Pod。必须把 Pod 所在节点、GPU 设备和网络拓扑纳入证据链。

---

## 19d.5 工程化：生产排障体系

### 最小证据包

遇到 AI on K8s 故障，先采集最小证据包：

```bash
NS=<namespace>
POD=<pod>

kubectl get pod -n "$NS" "$POD" -o wide
kubectl describe pod -n "$NS" "$POD"
kubectl logs -n "$NS" "$POD" --all-containers --tail=300
kubectl logs -n "$NS" "$POD" --all-containers --previous --tail=300
kubectl get events -n "$NS" --sort-by=.lastTimestamp
```

如果 Pod 已调度到节点：

```bash
NODE=$(kubectl get pod -n "$NS" "$POD" -o jsonpath='{.spec.nodeName}')
kubectl describe node "$NODE"
kubectl get node "$NODE" -o yaml
```

如果是 CRD/Operator：

```bash
kubectl get <kind> -n "$NS" <name> -o yaml
kubectl describe <kind> -n "$NS" <name>
kubectl logs -n <operator-ns> deploy/<operator-deploy> --tail=300
```

如果是服务流量：

```bash
kubectl get svc,endpointslice,ingress -n "$NS"
kubectl describe svc -n "$NS" <svc>
kubectl get endpointslice -n "$NS" -l kubernetes.io/service-name=<svc> -o yaml
```

### 证据命名和交接格式

值班记录建议固定格式：

```text
时间：
影响范围：
对象：
当前症状：
关键证据：
排除项：
根因判断：
已执行动作：
风险和下一步：
```

“关键证据”要写具体事件、日志片段摘要、对象状态和时间，而不是“看起来像网络”。

### 版本矩阵

生产排障时必须知道当前版本组合：

| 维度 | 例子 | 常见问题 |
|------|------|----------|
| Kubernetes | 1.28/1.29/1.30 | API 行为、scheduler、kubelet bug |
| Container Runtime | containerd、CRI-O | 镜像拉取、runtime hook、cgroup |
| CNI | Calico、Cilium、VPC CNI | Pod 网络、NetworkPolicy、MTU |
| CSI/Storage | EBS、Ceph、Lustre、NFS、对象存储 | PVC 绑定、挂载、吞吐、锁 |
| GPU Stack | driver、CUDA、device plugin、GPU Operator | GPU 注入、MIG、兼容性 |
| NCCL/RDMA | NCCL、OFED、RoCE/IB、UCX | timeout、fallback、带宽 |
| AI Runtime | PyTorch、Ray、Triton、vLLM、KServe | 启动参数、readiness、batching |

很多“偶发问题”其实是变更问题。排障时要问：最近是否升级了镜像、驱动、Operator、CNI、CSI、KServe、Ingress 或模型版本。

### 观测与告警

推荐告警维度：

- Pod：Pending 超时、CrashLoopBackOff、ImagePullBackOff、OOMKilled、NotReady。
- Node：NotReady、MemoryPressure、DiskPressure、GPU allocatable 消失。
- GPU：Xid、ECC、DCGM exporter down、显存长期满、温度/功耗异常。
- NCCL/训练：rank timeout、job restart count、checkpoint failure。
- Serving：ready endpoint 数为 0、5xx、p99 延迟、模型加载失败、revision rollout 超时。
- Operator：reconcile error、workqueue depth、status update conflict、leader election 异常。
- 存储：PVC Pending、mount failure、object storage 5xx、checkpoint 写入失败。

告警必须能链接到 runbook。只报“服务不可用”不够，要能提示下一步看 EndpointSlice、readiness、Ingress controller 还是 backend logs。

---

## 19d.6 方案设计：AI K8s 值班决策表

### 快速分类决策表

| 用户症状 | 第一判断 | 第一命令 | 下一步 |
|----------|----------|----------|--------|
| 任务一直没开始 | Pod 是否 Pending | `kubectl get pod -o wide` | 看 describe events |
| 镜像拉不下来 | 是否 ImagePullBackOff | `kubectl describe pod` | 查 tag、secret、registry、CA |
| 代码说没有 GPU | Pod 是否请求 GPU 且节点 allocatable 正常 | `describe pod/node` | 查 device plugin、runtime、容器内 `nvidia-smi` |
| 分布式训练卡住 | 所有 rank 是否 Running | `kubectl get pod -l job=<job>` | 查 NCCL logs、DNS、网络、GPU Xid |
| 训练被杀 | 是否 OOMKilled | `get pod -o yaml` | 区分 cgroup memory、GPU OOM、ephemeral storage |
| 服务间歇 502 | Service 是否有 ready endpoint | `get endpointslice` | 查 readiness、Ingress、backend logs |
| 模型服务 rollout 卡住 | revision 是否 Ready | `get inferenceservice/revision` | 查 model download、runtime、probe |
| PVC 一直挂不上 | PVC 是否 Bound | `describe pvc/pod` | 查 StorageClass、CSI、access mode、容量 |
| 域名解析失败 | Pod 内 DNS 是否正常 | `nslookup` | 查 CoreDNS、Service、NetworkPolicy |

### 可执行 triage 流程

```text
1. 定位对象：namespace、Pod/Job/CRD/Service 名称。
2. 判断状态机位置：Pending、Pulling、Starting、Running、Ready、Terminating。
3. 采集最小证据包：get、describe、events、logs、previous logs。
4. 若已调度：补充 node、GPU、runtime、storage、network 证据。
5. 若是 CRD：补充 CRD status、operator logs、owned resources。
6. 若是服务流量：补充 Service、EndpointSlice、Ingress/Gateway、backend。
7. 做最小修复动作：不先删除证据，不做无关变更。
8. 记录根因和预防：告警、准入、默认值、runbook、测试。
```

---

## 19d.7 Pod Pending SOP

### 症状

Pod 长时间 `Pending`，训练任务未开始，CRD status 可能显示 `WaitingForResources`、`PodsPending` 或 `Admitted=False`。

### 证据

```bash
kubectl get pod -n <ns> <pod> -o wide
kubectl describe pod -n <ns> <pod>
kubectl get events -n <ns> --sort-by=.lastTimestamp
kubectl get resourcequota -n <ns>
kubectl get pvc -n <ns>
```

查看节点 GPU：

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu
kubectl describe node <node>
```

### 常见事件和根因

| Event 关键字 | 根因 | 处理动作 |
|--------------|------|----------|
| `Insufficient nvidia.com/gpu` | GPU 数量或形状不足 | 等待队列、扩容、降低请求、调整节点池 |
| `didn't match Pod's node affinity/selector` | nodeSelector/affinity 过窄或标签错误 | 修 Pod 约束或节点标签 |
| `had untolerated taint` | 节点有 taint，Pod 无 toleration | 加 toleration 或换节点池 |
| `persistentvolumeclaim is not bound` | PVC 未绑定 | 查 StorageClass、容量、accessMode |
| `exceeded quota` | namespace/queue quota 不足 | 释放资源或调整 quota |
| `pod has unbound immediate PersistentVolumeClaims` | Immediate binding 阻塞调度 | 改 WaitForFirstConsumer 或修 PV |
| `Preemption is not helpful` | 抢占也无法满足形状 | 需要扩容或改拓扑/资源请求 |

### AI 特有判断

GPU 总数足够不代表可调度。8 卡 worker 需要单节点 8 张可用 GPU；集群里 8 张碎片 GPU 分布在 8 台机器上没有用。多 worker 训练还要考虑 gang：只调度一半 worker 可能占住 GPU 但训练无法开始。

### 处理动作

1. 如果是资源不足：确认是否应该进入队列等待，而不是裸 Pending。
2. 如果是标签/污点：修正平台默认值，避免用户每次手写。
3. 如果是 PVC：先修存储绑定，不要删除训练 Pod 反复重试。
4. 如果是 gang 问题：接入 Kueue/Volcano/PodGroup，整体准入。
5. status 应写明 `reason`，例如 `InsufficientGPU`、`PVCNotBound`、`WaitingForGangAdmission`。

---

## 19d.8 ImagePullBackOff SOP

### 症状

Pod 状态为 `ErrImagePull` 或 `ImagePullBackOff`。容器尚未启动，因此没有应用日志。

### 证据

```bash
kubectl describe pod -n <ns> <pod>
kubectl get secret -n <ns>
kubectl get serviceaccount -n <ns> <sa> -o yaml
kubectl get pod -n <ns> <pod> -o jsonpath='{.spec.containers[*].image}'
```

### 常见根因

| Event/错误 | 根因 | 处理动作 |
|------------|------|----------|
| `manifest unknown` | tag 不存在或镜像名写错 | 修镜像 tag，发布不可变 digest |
| `unauthorized` | imagePullSecret 缺失或权限不足 | 修 Secret、ServiceAccount、registry 权限 |
| `x509: certificate signed by unknown authority` | 私有 CA 未下发到节点 runtime | 配置节点 CA，重启 container runtime |
| `i/o timeout` | 节点到 registry 网络不通 | 查节点网络、代理、防火墙、registry |
| `no matching manifest for linux/amd64` | 架构不匹配 | 构建正确架构镜像 |
| `toomanyrequests` | registry 限流 | 配置镜像缓存、私有 registry、凭据 |

### 处理动作

- 优先用 digest 或受治理的 tag，避免 `latest`。
- 镜像发布和训练任务提交之间要有准入检查。
- 私有 registry 应配置节点级缓存或预拉取，降低大镜像启动抖动。
- 不要通过删除 Pod 解决 ImagePullBackOff；根因不修复，重建只会重放失败。

---

## 19d.9 GPU 不可见 SOP

### 症状

- Pod 已 Running，但容器内 `nvidia-smi` 不存在或报错。
- PyTorch `torch.cuda.is_available()` 为 `False`。
- 应用报 `CUDA driver version is insufficient for CUDA runtime version`。
- Pod 请求了 GPU，但应用只看到 0 张卡或看到错误 MIG 设备。

### 证据

```bash
kubectl describe pod -n <ns> <pod>
kubectl get pod -n <ns> <pod> -o jsonpath='{.spec.nodeName}'
kubectl describe node <node> | grep -A10 -i nvidia
kubectl get ds -A | grep -i nvidia
kubectl logs -n gpu-operator-resources ds/nvidia-device-plugin-daemonset --tail=200
```

容器内验证：

```bash
kubectl exec -n <ns> <pod> -- nvidia-smi
kubectl exec -n <ns> <pod> -- python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### 分层判断

| 层 | 检查 | 常见根因 |
|----|------|----------|
| Pod spec | 是否请求 `nvidia.com/gpu` 或 MIG resource | 只设置 env，未请求 GPU |
| Scheduler | Pod 是否调到 GPU 节点 | selector/taint/queue 错 |
| Node allocatable | `allocatable.nvidia.com/gpu` 是否存在 | device plugin 未注册 |
| Device plugin | DaemonSet 是否 Ready，日志是否报错 | driver 不匹配、MIG 配置错 |
| Runtime | NVIDIA container toolkit / runtimeClass | 容器未注入设备 |
| Driver/CUDA | 节点 driver 与镜像 CUDA 是否兼容 | 镜像 CUDA 太新或驱动太旧 |
| MIG | resource name 是否正确 | 请求整卡但节点只暴露 MIG，或反之 |

### 处理动作

1. 先确认 Pod 是否真的请求 GPU。没有 resource request，Kubernetes 不会注入设备。
2. 节点 allocatable 没 GPU，查 device plugin 和 GPU Operator。
3. 节点有 GPU 但容器不可见，查 runtimeClass、container toolkit、runtime hook。
4. 容器可见 GPU 但框架不可用，查 CUDA、PyTorch、driver 兼容。
5. 单节点异常时先 cordon 节点并迁移任务；集群性异常再回滚 GPU 栈变更。

---

## 19d.10 NCCL Timeout SOP

### 症状

- PyTorch DDP/FSDP 卡在 rendezvous、init process group、allreduce。
- 日志出现 `NCCL timeout`、`socket timeout`、`unhandled system error`。
- 多机训练吞吐极低，NCCL fallback 到 Socket。
- 某些 rank 退出，其他 rank 等待直到超时。

### 证据

先确认所有 rank：

```bash
kubectl get pod -n <ns> -l <job-label> -o wide
kubectl logs -n <ns> <pod> --all-containers --tail=500
kubectl logs -n <ns> <pod> --all-containers --previous --tail=500
```

建议在训练任务中打开：

```text
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET,COLL
TORCH_DISTRIBUTED_DEBUG=DETAIL
```

如果使用 RDMA，还要确认设备和网络：

```bash
kubectl exec -n <ns> <pod> -- sh -c 'ls -l /dev/infiniband || true'
kubectl exec -n <ns> <pod> -- sh -c 'env | grep -E "NCCL|UCX|GLOO|MASTER"'
kubectl exec -n <ns> <pod> -- nvidia-smi topo -m
```

### 排查顺序

1. 所有 rank 是否 Running 且 Ready。一个 rank OOM 或 CrashLoop 会让其他 rank timeout。
2. `MASTER_ADDR`、`MASTER_PORT`、world size、rank 是否一致。
3. Pod 间 DNS 和端口是否互通，NetworkPolicy 是否阻断。
4. NCCL 选择的网卡是否符合预期，是否选错 `eth0`、docker bridge 或管理网。
5. RDMA 设备是否注入容器，库和驱动是否匹配。
6. MTU、PFC、ECN、GID index、RoCE/IB 配置是否一致。
7. GPU/NIC locality 是否异常，是否跨 socket、跨 rack、跨 AZ。
8. DCGM 是否有 Xid、NVLink 错误、ECC 错误或 GPU reset。

### 日志线索

| NCCL 日志线索 | 可能根因 | 处理动作 |
|---------------|----------|----------|
| 卡在 rendezvous | rank 未齐、master DNS/端口错 | 查 Pod 状态、Service/DNS、环境变量 |
| `NET/IB : No device found` | RDMA 设备未注入或库缺失 | 查 RDMA device plugin、镜像 OFED/UCX |
| fallback to Socket | IB/RDMA 不可用或 NCCL 选择错误 | 设置 `NCCL_SOCKET_IFNAME`/IB 参数，修设备 |
| `socketStartConnect` timeout | Pod 网络不通、NetworkPolicy、端口被挡 | 查 CNI、policy、Service |
| 某个 collective hang | 单 rank 慢、OOM、GPU Xid、网络丢包 | 对齐各 rank 日志和 DCGM 时间线 |

### 处理动作

- 先保证 rank 齐，再看网络通，再看 RDMA/NVLink locality。
- 不要只在一个 Pod 看日志。NCCL 问题必须按 rank 对齐时间线。
- 生产训练模板应支持一键打开 NCCL debug，并把 rank、node、GPU、NIC 信息写入日志。
- 对频繁出现 Xid 的节点先 cordon，避免反复污染训练任务。

---

## 19d.11 OOMKilled 与内存类故障 SOP

### 概念边界

`OOMKilled` 通常指容器超过 cgroup memory limit，被 Linux 内核杀死。它不等同于 CUDA out of memory。GPU OOM 多数出现在应用日志里，容器可能退出、被框架捕获，也可能继续运行。

还要区分 ephemeral storage 耗尽。它会导致 Pod 被驱逐或写文件失败，但不是内存 OOM。

### 证据

```bash
kubectl get pod -n <ns> <pod> -o yaml
kubectl describe pod -n <ns> <pod>
kubectl logs -n <ns> <pod> --previous --tail=300
kubectl top pod -n <ns> <pod>
kubectl describe node <node>
```

关注字段：

```yaml
lastState:
  terminated:
    reason: OOMKilled
    exitCode: 137
```

### 类型判断

| 类型 | 证据 | 常见根因 | 处理动作 |
|------|------|----------|----------|
| cgroup memory OOM | `reason=OOMKilled`、exit 137 | dataloader、tokenizer、checkpoint 序列化、内存 limit 太低 | 提高 memory limit，降低 dataloader workers，分片 checkpoint |
| CUDA OOM | 应用日志 `CUDA out of memory` | batch/context/KV cache 太大，显存碎片 | 降 batch/context，gradient checkpointing，调 serving cache |
| Node MemoryPressure | node condition、eviction events | 节点整体内存压力 | 降低 overcommit，迁移任务，修 requests/limits |
| Ephemeral storage | events 提到 `ephemeral-storage` | 缓存、日志、模型文件、临时 checkpoint | 设置 ephemeral request/limit，清理缓存，挂载持久卷 |
| GPU Xid/reset | DCGM、节点日志、应用异常 | GPU 硬件/驱动问题 | cordon 节点，迁移任务，检查硬件 |

### 处理动作

- 看到 `OOMKilled` 先查 cgroup memory，不要直接加 GPU。
- 训练任务给 dataloader、tokenization、checkpoint 留足 CPU memory。
- 推理服务要把模型权重、KV cache、batching 峰值和 health probe 都纳入容量估算。
- 对 recurring OOM 建立基线：输入长度、batch、worker 数、checkpoint 周期、内存曲线。

---

## 19d.12 Readiness Flapping SOP

### 症状

Pod 反复 Ready/NotReady，EndpointSlice 抖动，Ingress 或 Gateway 间歇 502/503。KServe/Knative 场景中可能表现为 revision 一直不 Ready 或流量切换失败。

### 证据

```bash
kubectl describe pod -n <ns> <pod>
kubectl get endpointslice -n <ns> -l kubernetes.io/service-name=<svc> -w
kubectl logs -n <ns> <pod> --all-containers --tail=300
kubectl get deploy,rs,pod -n <ns> -l app=<app>
```

如果是 KServe：

```bash
kubectl get inferenceservice -n <ns> <name> -o yaml
kubectl describe inferenceservice -n <ns> <name>
kubectl get revision -n <ns>
```

### 常见根因

| 现象 | 根因 | 处理动作 |
|------|------|----------|
| 启动后很久 NotReady | 模型下载/加载/warmup 慢 | 增大 initialDelay 或 startupProbe，优化模型缓存 |
| 负载高时 NotReady | readiness probe 太重或与推理争资源 | probe 改轻量本地检查，设置超时和并发隔离 |
| 间歇 502 | endpoint 短暂为 0 或 proxy 超时 | 看 EndpointSlice 时间线和 Ingress logs |
| 新 revision 不 Ready | storageUri、Secret、runtime、镜像问题 | 查 init/model agent/runtime logs |
| mesh/queue proxy 失败 | sidecar readiness 失败 | 查 sidecar logs 和端口配置 |

### 处理动作

- 使用 startupProbe 保护冷启动，readinessProbe 只表示“现在能否接流量”。
- readiness 不要做真实大模型推理；可做轻量健康检查和模型加载标志检查。
- 模型服务发布时设置最小 ready endpoint 或灰度比例，避免 endpoint 清零。
- 记录 model load time、warmup time、ready transition 次数。

---

## 19d.13 Service / Ingress / Gateway SOP

### 症状

Pod Running/Ready，但客户端访问失败、超时、502/503/504，或集群内服务名无法访问。

### 证据

```bash
kubectl get svc,endpointslice,ingress -n <ns>
kubectl describe svc -n <ns> <svc>
kubectl get endpointslice -n <ns> -l kubernetes.io/service-name=<svc> -o yaml
kubectl describe ingress -n <ns> <ingress>
kubectl logs -n <ingress-ns> deploy/<ingress-controller> --tail=300
```

集群内连通性：

```bash
kubectl run -n <ns> netshoot --rm -it --image=nicolaka/netshoot -- bash
curl -v http://<svc>.<ns>.svc.cluster.local:<port>/health
```

### 分层判断

| 层 | 检查 | 根因 |
|----|------|------|
| Service selector | selector 是否匹配 Pod labels | selector 错导致无 endpoint |
| EndpointSlice | 是否有 ready endpoint、端口是否正确 | readiness 失败、targetPort 错 |
| Pod port | containerPort/listen port 是否一致 | 应用监听错端口或只监听 localhost |
| NetworkPolicy | client 到 backend 是否允许 | policy 阻断 |
| Ingress/Gateway | host/path/TLS/backend 配置 | 路由错、证书错、timeout |
| Backend app | 应用是否处理路径和超时 | 模型慢、队列满、崩溃 |

### 处理动作

- Service 不通先看 EndpointSlice。没有 endpoint 时不要先改 Ingress。
- Endpoint 存在但请求失败，再查端口、NetworkPolicy 和应用监听地址。
- Ingress 5xx 要区分 502、503、504：无后端、后端不可用、后端超时的处理不同。
- 大模型推理常需要调大 gateway/proxy timeout，但先确认 backend 队列和 readiness 没问题。

---

## 19d.14 PVC / 存储 SOP

### 症状

Pod Pending、ContainerCreating 卡住、模型下载慢、checkpoint 写失败、训练中 I/O timeout。

### 证据

```bash
kubectl get pvc,pv -n <ns>
kubectl describe pvc -n <ns> <pvc>
kubectl describe pod -n <ns> <pod>
kubectl get storageclass
kubectl get events -n <ns> --sort-by=.lastTimestamp
```

如果是 CSI 问题：

```bash
kubectl get pod -A | grep -i csi
kubectl logs -n <csi-ns> deploy/<csi-controller> --tail=300
kubectl logs -n <csi-ns> ds/<csi-node> --tail=300
```

### 常见根因

| 现象 | 根因 | 处理动作 |
|------|------|----------|
| PVC Pending | StorageClass 不存在、容量不足、accessMode 不支持 | 修 StorageClass、容量、访问模式 |
| Pod 等 PVC | Immediate binding 与节点拓扑冲突 | 使用 WaitForFirstConsumer 或匹配 zone |
| Mount 失败 | CSI node plugin 异常、权限、网络 | 查 CSI logs、节点、Secret |
| Checkpoint 慢 | 单卷吞吐不足、小文件过多、对象存储限流 | 分片 checkpoint、并发控制、提高吞吐 |
| 模型下载失败 | Secret、CA、对象路径、网络 | 查 init/model agent logs、对象存储 |
| Ephemeral 耗尽 | 模型缓存或临时文件写满节点盘 | 挂载缓存卷、设置清理策略和 limit |

### AI 特有判断

训练 checkpoint 会产生周期性 I/O 峰值，可能导致 GPU 等待、NCCL 超时或容器 OOM。推理模型冷启动会受模型大小、对象存储吞吐、节点缓存命中率影响。存储问题不一定只表现为 PVC Pending，也可能表现为 readiness 慢或吞吐抖动。

---

## 19d.15 DNS SOP

### 症状

应用报无法解析 Service、master 地址、对象存储域名或模型 registry 域名。分布式训练 rank 找不到 master；推理服务依赖调用间歇失败。

### 证据

Pod 内测试：

```bash
kubectl exec -n <ns> <pod> -- nslookup kubernetes.default.svc.cluster.local
kubectl exec -n <ns> <pod> -- nslookup <svc>.<ns>.svc.cluster.local
kubectl exec -n <ns> <pod> -- cat /etc/resolv.conf
```

CoreDNS：

```bash
kubectl get pod -n kube-system -l k8s-app=kube-dns -o wide
kubectl logs -n kube-system deploy/coredns --tail=300
kubectl describe configmap -n kube-system coredns
```

### 常见根因

| 现象 | 根因 | 处理动作 |
|------|------|----------|
| Service 名解析失败 | Service 不存在、namespace 错、search path 误解 | 使用 FQDN，确认 Service |
| 外部域名慢或失败 | CoreDNS upstream、网络、限流 | 查 CoreDNS logs、上游 DNS、NodeLocal DNSCache |
| 大量超时 | DNS QPS 高、ndots 过高、连接跟踪压力 | 启用 NodeLocal DNSCache，优化 ndots |
| 只有某 namespace 失败 | NetworkPolicy 或 DNS policy | 查 egress 到 kube-dns |
| master 地址解析错 | headless Service 或 selector 错 | 查 EndpointSlice 和 Pod labels |

### 处理动作

- 分布式训练建议使用稳定 Service DNS 或平台注入的 master 地址。
- 高 QPS 推理服务要监控 DNS 延迟和 CoreDNS 饱和。
- 排查 NCCL/rendezvous 时，把 DNS 解析结果纳入证据链。

---

## 19d.16 故障排除汇总表

| 症状 | 关键证据 | 高概率根因 | 动作 |
|------|----------|------------|------|
| Pod Pending | `describe pod` Events | GPU/标签/污点/quota/PVC/gang | 修资源、约束、队列或存储 |
| ImagePullBackOff | Events、Secret、SA | tag、权限、CA、网络、架构 | 修镜像发布和拉取凭据 |
| GPU 不可见 | Pod spec、node allocatable、device plugin logs | 未请求 GPU、plugin/runtime/driver/CUDA | 分层修 GPU 栈 |
| NCCL timeout | rank 状态、NCCL logs、DCGM、网络 | rank 不齐、DNS、NetworkPolicy、RDMA、Xid | 对齐 rank 时间线，修网络/GPU |
| OOMKilled | lastState、previous logs、metrics | cgroup memory、GPU OOM、临时盘 | 区分类型后调资源或代码 |
| Readiness flapping | Pod events、EndpointSlice、runtime logs | probe、模型加载、依赖、显存压力 | 调 probe、缓存、warmup、发布策略 |
| Service 不通 | EndpointSlice、Service selector、Ingress logs | selector、端口、readiness、policy、gateway | 从 endpoint 到 gateway 逐层查 |
| PVC 卡住 | PVC/PV events、CSI logs | StorageClass、容量、accessMode、zone | 修存储类、绑定模式、CSI |
| DNS 失败 | Pod nslookup、CoreDNS logs | Service 名、upstream、QPS、NetworkPolicy | 修 FQDN、CoreDNS、NodeLocal DNS |

---

## 19d.17 反模式与 Checklist

### 反模式

| 反模式 | 后果 | 修正 |
|--------|------|------|
| 先删除 Pod 再看证据 | previous logs、events、现场消失 | 先采集最小证据包 |
| Pending 时看应用日志 | Pod 根本没启动，浪费时间 | 先看 scheduler events |
| ImagePullBackOff 时重启 Pod | 根因不变，退避重放 | 修 tag、Secret、registry、CA |
| GPU 不可见就改代码 | 可能是 device plugin/runtime/driver | 先查 Pod spec、node、plugin、runtime |
| NCCL timeout 只看一个 rank | 分布式问题需要多 rank 对齐 | 收集所有 rank 日志和状态 |
| OOMKilled 直接加 GPU | 可能是 CPU memory 或临时盘 | 区分 cgroup、CUDA、ephemeral |
| 502 先改 Ingress | 可能没有 ready endpoint | 先查 EndpointSlice |
| 不记录版本和变更 | 无法判断回归 | 记录镜像、驱动、Operator、CNI、CSI、模型版本 |
| 结论没有证据 | 事故复盘不可审计 | 写症状、证据、根因、动作 |

### 值班 Checklist

- 是否确认 namespace、对象名、时间窗口和影响范围？
- 是否采集 `get/describe/events/logs/previous logs`？
- Pod 处于哪个状态机位置：Pending、Pulling、Starting、Running、Ready、Terminating？
- 若 Pending，是否检查 GPU、quota、PVC、affinity、taint、gang？
- 若 Running，是否检查 node、GPU、runtime、CNI、CSI？
- 若分布式训练，是否收集所有 rank 状态和 NCCL logs？
- 若服务失败，是否从 EndpointSlice 到 Ingress/Gateway 逐层验证？
- 是否区分 cgroup OOM、CUDA OOM 和 ephemeral storage？
- 是否检查最近变更：镜像、模型、驱动、Operator、CNI、CSI、Ingress？
- 修复前是否保留证据，修复后是否补充预防动作？

---

## 19d.18 Worked Example：推理服务间歇 502

### 场景

一个 KServe InferenceService `reranker` 发布新模型后，用户反馈每隔几分钟出现 502。Pod 看起来大部分时间 Running，CPU/GPU 利用率不高。

### 采证

先看服务和 endpoint：

```bash
kubectl get inferenceservice -n serving reranker -o yaml
kubectl get svc,endpointslice,ingress -n serving
kubectl get endpointslice -n serving -l kubernetes.io/service-name=reranker-predictor -w
```

发现 EndpointSlice 在 0 和 2 个 ready endpoint 之间反复切换。

再看 Pod：

```bash
kubectl describe pod -n serving <pod>
kubectl logs -n serving <pod> --all-containers --tail=300
```

Events 显示 readiness probe 间歇 timeout。runtime 日志显示 probe 路径会触发一次轻量推理，但新模型冷启动后 KV cache 预热和首批请求竞争 GPU，probe 超过 1 秒超时。

### 根因

Ingress 502 是结果，不是根因。真正根因是 readiness probe 太重且 timeout 过短，导致 ready endpoint 反复清零。EndpointSlice 抖动把问题传递到 Ingress，客户端看到 502。

### 处理

短期止血：

1. 把 readiness probe 改成只检查本地模型加载标志和 runtime event loop。
2. 增加 `timeoutSeconds` 和 `failureThreshold`，增加 startupProbe 覆盖模型加载和 warmup。
3. 灰度发布时保持旧 revision 部分流量，避免新 revision endpoint 清零影响全量请求。

长期预防：

- 记录 model load time、warmup time、ready transition count。
- 发布前压测新模型冷启动和 readiness。
- 告警增加 `ready endpoints == 0` 和 readiness flapping 次数。

### 结论格式

```text
症状：reranker 新版本间歇 502。
证据：EndpointSlice ready endpoint 在 0/2 间抖动；Pod events 显示 readiness timeout；runtime logs 显示 probe 与 warmup 竞争 GPU。
根因：readiness probe 过重且 timeout 过短。
动作：改轻量 readiness，增加 startupProbe，灰度保留旧 revision。
预防：增加 endpoint 清零告警和冷启动压测。
```

---

## 本章小结

Kubernetes for AI 排障的核心是分层和证据链。先判断对象处于哪个状态机位置，再选择对应证据：Pending 看 Events 和调度条件，ImagePull 看镜像和凭据，GPU 看 device plugin/runtime/driver，NCCL 看 rank、网络、RDMA 和 DCGM，服务失败看 EndpointSlice、readiness 和入口日志，存储和 DNS 要纳入数据路径。

好的 SOP 能把一次救火变成可复用的工程资产：固定采证、明确根因、最小动作、记录预防。

## 练习题

1. 一个 Pod Pending，Events 同时出现 PVC 未绑定和 GPU 不足。你会先处理哪个，为什么？
2. PyTorch 训练报 NCCL timeout。请列出你会收集的 6 类证据，并说明每类证据证明什么。
3. 如何区分 `OOMKilled`、CUDA OOM 和 ephemeral storage 耗尽？
4. 推理服务 503，但 Pod 全部 Running。为什么第一步应该看 EndpointSlice？
5. 设计一份值班记录模板，用于交接一次 GPU 不可见故障。
