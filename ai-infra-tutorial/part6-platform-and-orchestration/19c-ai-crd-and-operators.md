# 第19c章：AI CRD 与 Operator：TorchJob、RayJob、MPIJob 与 KServe

> Pod、Job、Deployment 是 Kubernetes 的通用抽象；AI 平台还需要表达分布式训练、Ray 集群、MPI launcher、模型服务灰度、弹性伸缩和失败恢复。本章从第一性原理出发，讲清 CRD/Operator 如何把这些 AI 语义变成 Kubernetes API 中可观察、可治理、可恢复的一等对象。

---

## 19c.1 第一性原理拆解 + 学习大纲

### 拆：不可化简的问题

AI 平台上的高层任务不是“一组容器”这么简单。一个 64 卡 PyTorch 训练任务至少包含这些约束：

- 角色：master、worker、可能还有 evaluator、parameter server、launcher。
- 拓扑：每个 worker 几张 GPU，是否要求整机，是否要求同 rack 或同 IB fabric。
- 协议：rendezvous 地址、rank、world size、NCCL 网卡、端口。
- 生命周期：初始化、等待所有 rank、训练、checkpoint、失败重启、成功归档。
- 失败语义：单个 worker 失败时是重启全局训练、弹性缩容，还是直接失败。
- 观测语义：用户需要看到的是“等待 8 卡 gang 调度”，不是“某个 Pod Pending”。

同理，RayJob 不是一个普通 Job，MPIJob 不是一堆 worker Pod，KServe InferenceService 也不是一个 Deployment。它们都有各自的控制面语义。如果把这些语义分散在脚本、CI 模板、平台后端和人工 runbook 里，系统会变得不可恢复、不可解释、不可审计。

不可化简的问题是：**如何把 AI 任务的高层语义声明为稳定 API，并持续把真实集群状态推进到期望状态。**

### 推：从问题推导机制

从“用户应该提交高层语义”推出 CRD。CRD 让 Kubernetes API 增加 `PyTorchJob`、`RayJob`、`MPIJob`、`InferenceService` 等资源类型。

从“真实世界持续变化”推出 Operator。节点会重启，Pod 会失败，镜像会拉取失败，PVC 会等待绑定，用户会修改副本数，服务 revision 会切流量。一次性生成 YAML 不能处理这些变化，必须有控制器持续 reconcile。

从“期望状态和真实状态必须分离”推出 `spec` 与 `status`。`spec` 是用户或平台声明的目标；`status` 是控制器观察到的结果。控制器不应该偷偷修改 `spec` 来表达运行状态。

从“删除也有语义”推出 finalizer。删除训练任务时可能要清理临时 Service、外部 Ray cluster、云端 endpoint、模型缓存或 checkpoint lease。Kubernetes 的普通垃圾回收只认识集群内对象，不认识外部资源。

从“底层对象必须有归属”推出 ownerReference 和 label selector。Operator 创建的 Pod、Service、ConfigMap、Job、Revision 必须能被反查、清理和聚合状态。

从“AI 任务有阶段”推出状态机与 conditions。单个 `phase=Running` 不够，平台还要表达 `Admitted`、`PodsCreated`、`RendezvousReady`、`ModelLoaded`、`TrafficReady`、`CheckpointRestored` 等细粒度条件。

### 学习大纲

本章按以下顺序展开：

1. 概念边界：CRD、Custom Resource、Controller、Operator、Webhook、Helm、Scheduler 分别是什么。
2. 架构：API Server、etcd、watch cache、controller-runtime、workqueue、ownerReference、finalizer 的协作路径。
3. 原理：reconciliation loop、幂等性、乐观并发、状态机、条件、失败恢复。
4. AI CRD：TorchJob、RayJob、MPIJob、KServe InferenceService 的核心语义和底层对象。
5. 工程化：配置、版本矩阵、发布、观测、RBAC、安全治理和兼容性。
6. 方案设计：如何为训练和推理选择 CRD，以及如何设计一个可上线的 Operator。
7. 排障与反模式：从症状、证据、根因到处理动作。
8. Worked Example：把 8 节点 PyTorch 训练做成可恢复、可排障的 CRD 工作负载。

---

## 19c.2 概念先说清楚

### CRD 是什么，不是什么

CRD 的全称是 CustomResourceDefinition。它向 Kubernetes API 注册一种新的资源类型。注册后，用户可以像操作 Pod 一样操作自定义对象：

```bash
kubectl get pytorchjobs -n train
kubectl describe rayjob -n ml batch-embed-v3
kubectl get inferenceservice -n serving reranker -o yaml
```

CRD 是 API 扩展机制，不是调度器，不是工作流引擎，也不是模板语言。CRD 本身只负责让 API Server 接受、校验、存储和返回这种对象。没有控制器时，一个 CRD 对象只是 etcd 里的声明，不会自动创建 Pod。

### Custom Resource 是什么，不是什么

Custom Resource 是 CRD 的实例。例如 `kind: PyTorchJob` 的 `train-reranker` 对象就是一个自定义资源。它通常包含三类字段：

| 字段 | 归属 | 含义 |
|------|------|------|
| `metadata` | Kubernetes | name、namespace、labels、annotations、finalizers、ownerReferences |
| `spec` | 用户或平台 | 期望状态，例如副本数、镜像、资源、模型路径、重启策略 |
| `status` | 控制器 | 观察状态，例如 phase、conditions、replica status、revision、失败原因 |

`spec` 是输入，`status` 是输出。用户改 `spec`，控制器改 `status`。把运行时状态写进 `spec` 会破坏声明式 API 的边界。

### Controller 与 Operator 的边界

Controller 是 Kubernetes 控制循环的通用名字：监听对象变化，比较期望状态和真实状态，执行修正动作。

Operator 是面向某个领域的控制器加运维知识。AI Operator 不只是创建 Pod，还要知道训练角色、rank、launcher、checkpoint、模型服务 revision、流量切分、自动扩缩容和失败恢复策略。

可以这样区分：

| 名称 | 关注点 | 例子 |
|------|--------|------|
| Controller | 通用控制循环 | Deployment controller、Job controller |
| Operator | 带领域语义的控制器 | Kubeflow Training Operator、KubeRay Operator、KServe controller |
| Admission Webhook | 创建或更新前的校验/默认值/变更 | 校验 GPU 请求、注入默认 runtime |
| Scheduler | 为 Pod 选择节点 | default-scheduler、Volcano、Kueue 集成调度 |
| Helm/Kustomize | 渲染和发布 YAML | 安装 Operator、生成默认 CRD 实例 |

Operator 不是“高级 Helm”。Helm 渲染发生在提交前，Operator 控制发生在运行中。真正的 Operator 必须面对失败、重试、并发、删除和状态漂移。

### Reconciliation Loop 是什么

Reconciliation loop 是 Operator 的核心。它反复执行：

```text
观察对象变化
  -> 读取期望状态：CRD spec
  -> 读取真实状态：Pod、Service、Job、Revision、PVC、外部系统
  -> 计算差异
  -> 创建、更新、删除或等待
  -> 写 status、记录 event、导出 metric
  -> 决定是否重新入队
```

这个循环必须是幂等的。同一个对象可能被重复 reconcile；一次 reconcile 也可能执行到一半失败。正确实现的控制器不依赖“刚才那一步一定成功”，而是每次都从 API Server 重新读取事实。

### Finalizer 与 OwnerReference 的边界

`ownerReference` 用于 Kubernetes 内部对象的级联删除。PyTorchJob 创建的 worker Pod 带上 ownerReference 后，删除 PyTorchJob 时，垃圾回收器可以删除这些 Pod。

`finalizer` 用于删除前置清理。它会阻止对象立刻从 API Server 消失，直到控制器完成清理并移除 finalizer。适合清理 Kubernetes 垃圾回收器看不到的东西：外部负载均衡器、云端 endpoint、对象存储临时目录、checkpoint lock、Ray runtime env 缓存等。

不要用 finalizer 做普通状态流转，也不要把所有底层对象都用 finalizer 管起来。finalizer 是删除协议，不是生命周期状态机。

---

## 19c.3 架构：组件、路径与责任边界

### 关键组件

| 组件 | 责任 |
|------|------|
| API Server | 接收 CRD 和自定义资源请求，执行认证、授权、准入、校验和持久化 |
| etcd | 存储 CRD schema、自定义资源和底层对象状态 |
| CRD Schema | 定义字段、类型、默认值、版本、status subresource、conversion |
| Admission Webhook | 做默认值、校验、变更、策略拦截 |
| Operator Controller | watch 目标对象，执行 reconciliation loop |
| Workqueue | 对变化对象去重、限速、重试、延迟处理 |
| Cache/Informer | 本地缓存 watch 到的对象，降低 API Server 压力 |
| Owned Resources | Pod、Job、Service、ConfigMap、PVC、Revision、HPA 等底层对象 |
| Scheduler/Queue | 为 Pod 或 Workload 做准入和调度，处理 GPU、gang、quota、priority |
| Observability | events、status conditions、logs、metrics、traces、dashboards |

### 控制路径

以 PyTorchJob 为例，控制路径如下：

```text
用户提交 PyTorchJob
  -> API Server 校验 CRD schema 和 admission policy
  -> etcd 持久化对象
  -> Training Operator watch 到变化
  -> Operator 读取 spec，生成 master/worker Pod、Service、ConfigMap
  -> Scheduler 为 Pod 绑定节点
  -> kubelet 启动容器并注入 GPU
  -> Operator watch Pod 状态，更新 PyTorchJob status
  -> 用户通过 kubectl/API/平台 UI 看到训练状态
```

这里有两个重要边界：

- Operator 不直接启动容器，它创建 Kubernetes 对象，由 kube-scheduler 和 kubelet 完成调度与运行。
- Scheduler 不理解 PyTorchJob 的完整业务语义，除非 Operator 把 gang、quota、priority 等信息通过 PodGroup、Workload 或 labels 传给调度层。

### 数据路径

训练或推理的数据路径不经过 Operator。Operator 只是控制面。

```text
训练数据：对象存储 / PVC / 并行文件系统 -> worker Pod -> GPU
参数同步：GPU -> NCCL / MPI / Ray object store -> GPU
推理请求：Client -> Gateway / Ingress -> Service -> Pod -> GPU
模型加载：Model registry / object storage -> init/model agent -> runtime
```

这意味着 Operator 状态正常不代表数据路径正常。InferenceService `Ready=True` 也不等于模型延迟达标；TorchJob `Running=True` 也不等于 NCCL 带宽正常。

### 责任边界

| 层 | 负责什么 | 不负责什么 |
|----|----------|------------|
| CRD Schema | API 字段、类型、版本、默认值边界 | 运行时恢复和业务决策 |
| Operator | 期望状态到真实状态的收敛、status、events | 容器内训练代码正确性 |
| Admission | 创建/更新前校验、默认值、准入策略 | 运行中的失败恢复 |
| Scheduler/Queue | 资源准入、节点选择、gang、公平性 | rank 协议和 checkpoint |
| Device Plugin/Runtime | GPU 资源注册和容器设备注入 | AI 任务状态机 |
| Serving Runtime | 模型加载、推理协议、batching | K8s 对象生命周期 |

清晰的责任边界能避免“所有问题都怪 Operator”。Operator 应该暴露证据，但不应该吞掉底层失败。

---

## 19c.4 原理：Reconciliation、状态机与对象语义

### 为什么需要 Reconciliation

声明式系统的核心不是“创建一次”，而是“持续收敛”。真实状态会因为以下原因漂移：

- Pod 被 kubelet 重启或被节点故障驱逐。
- 用户修改了 CRD 的副本数、镜像、资源或流量比例。
- Scheduler 长时间无法满足 GPU、PVC、affinity 或 gang 约束。
- 其他控制器修改了 Service、EndpointSlice、HPA、Revision。
- Operator 自己重启，丢失内存中的中间状态。
- API Server 返回冲突，status 更新失败。

因此 Operator 的正确姿势是每次都重新观察对象，而不是依赖本地变量：

```text
desired = read(cr.spec)
actual = list_owned_resources(cr)
plan = diff(desired, actual)
apply(plan)
patch_status(observed(actual))
```

### 幂等性与乐观并发

幂等性意味着重复执行同一个 reconcile 不会产生额外副作用。常见做法：

- 底层对象使用确定性名称，例如 `<job-name>-master-0`、`<job-name>-worker-3`。
- 用 labels 和 ownerReference 查找已有对象，而不是盲目创建。
- 使用 server-side apply 或 patch，只管理自己负责的字段。
- status 更新使用 `resourceVersion` 或 patch，遇到 conflict 重新读取。
- 外部资源用幂等 API，例如 create-or-get、delete-if-exists。

Operator 不能假设“我上次创建的对象还在”。它必须从 API Server 中重建事实。

### Spec、Status、Conditions

一个有工程质量的 CRD status 应该回答三个问题：

1. 当前阶段是什么。
2. 卡住或失败的直接原因是什么。
3. 系统下一步是否会自动处理。

示例：

```yaml
status:
  observedGeneration: 7
  phase: Pending
  conditions:
    - type: Admitted
      status: "False"
      reason: InsufficientGPU
      message: "queue train-prod needs 64 nvidia.com/gpu, currently 48 fit the topology"
      lastTransitionTime: "2026-05-04T10:20:00Z"
    - type: PodsCreated
      status: "False"
      reason: WaitingForAdmission
  replicaStatuses:
    Worker:
      desired: 8
      active: 0
      failed: 0
```

`observedGeneration` 很关键。它表示 status 对应的 spec generation。如果用户刚修改 spec，而 controller 还没处理，`metadata.generation` 会大于 `status.observedGeneration`。平台 UI 和自动化系统应据此判断 status 是否新鲜。

### 状态机

AI Operator 常见状态机如下：

| Phase | 触发条件 | 控制器动作 |
|-------|----------|------------|
| Created | CRD 已保存，controller 第一次观察到 | 加 finalizer，创建或修正底层对象 |
| Admitting | 等待队列、quota、gang 或策略准入 | 创建 Workload/PodGroup，写等待原因 |
| Pending | 底层对象已创建但关键 Pod 未就绪 | 汇总调度、PVC、镜像、准入事件 |
| Initializing | Pod 启动，init、模型加载、rendezvous 或 runtime env 准备中 | 观察日志信号、readiness、sidecar 状态 |
| Running | 关键角色进入运行态 | 监控 replica、driver、launcher、revision、metrics |
| Restarting | 发生可恢复失败 | 按策略重建、全局重启、resume checkpoint |
| Succeeded | 任务达到成功条件 | 写完成时间，清理临时资源 |
| Failed | 不可恢复或超过重试预算 | 写失败原因，保留证据 |
| Terminating | 用户删除对象 | finalizer 清理外部资源，移除 finalizer |

状态机不是 UI 标签，而是控制器决策的依据。对于非 elastic PyTorch 训练，任意 worker 失败可能意味着全局 restart；对于 elastic training，worker 短暂丢失可能只是缩容和重新 rendezvous。

### OwnerReference、Label 与 Selector

Operator 创建底层对象时应同时使用 ownerReference 和稳定 labels：

```yaml
metadata:
  labels:
    app.kubernetes.io/name: pytorchjob
    training.kubeflow.org/job-name: train-reranker
    training.kubeflow.org/replica-type: worker
  ownerReferences:
    - apiVersion: kubeflow.org/v1
      kind: PyTorchJob
      name: train-reranker
      uid: 1f4...
      controller: true
      blockOwnerDeletion: true
```

ownerReference 让 Kubernetes 知道归属关系；labels 让 controller、用户、Prometheus、日志系统、成本系统能查询和聚合。不要只依赖名字前缀。

### Finalizer

删除带 finalizer 的对象时，API Server 会设置 `metadata.deletionTimestamp`，但对象仍存在。Operator 看到 deletionTimestamp 后进入清理逻辑：

```text
if deletionTimestamp is set:
  delete external endpoint / cloud load balancer / temporary model cache
  release checkpoint lease or queue admission
  remove finalizer
  return
```

finalizer 要快、幂等、可重试。清理失败时要写 event 和 condition，避免对象永久卡在 Terminating 且没有解释。

---

## 19c.5 常见 AI CRD 的语义

### TorchJob / PyTorchJob

TorchJob 表达 PyTorch 分布式训练。不同发行版中 kind 可能叫 `PyTorchJob`，用户口语常称 TorchJob。核心不是“创建多个 Pod”，而是表达 PyTorch 分布式语义。

关键字段通常包括：

- `pytorchReplicaSpecs`：Master、Worker 等角色。
- 每个角色的 `replicas`、Pod template、资源请求。
- `runPolicy`：cleanPodPolicy、backoffLimit、ttlSecondsAfterFinished、suspend。
- restart policy：ExitCode、OnFailure、Never 等实现依发行版而定。
- 环境变量和启动命令：master address、rank、world size、backend、NCCL 参数。

示例：

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: train-reranker
  namespace: train
spec:
  runPolicy:
    cleanPodPolicy: Running
    backoffLimit: 3
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: registry.local/ml/reranker-train:cuda12.4-v42
              command: ["torchrun"]
              args:
                - "--nnodes=8"
                - "--nproc_per_node=8"
                - "--rdzv_backend=c10d"
                - "--rdzv_endpoint=$(MASTER_ADDR):29400"
                - "train.py"
              resources:
                limits:
                  nvidia.com/gpu: 8
    Worker:
      replicas: 7
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: registry.local/ml/reranker-train:cuda12.4-v42
              resources:
                limits:
                  nvidia.com/gpu: 8
```

生产中，TorchJob 往往需要和 Kueue、Volcano 或其他 gang scheduler 集成。否则 8 个 worker 可能只启动 5 个，训练卡在 rendezvous。

### RayJob

RayJob 表达“提交一个 Ray driver，并在 RayCluster 上运行”。常见对象关系是：

```text
RayJob
  -> RayCluster
       -> head Pod + head Service
       -> worker Pods
  -> submitter / driver Job
```

RayJob 的关键语义：

- RayCluster 生命周期：由 RayJob 创建、引用已有集群，还是任务结束后清理。
- Head/Worker 规格：CPU、GPU、内存、对象存储内存、节点选择。
- Runtime env：Python 包、working directory、环境变量。
- Driver 状态：提交成功、运行中、失败、成功。
- Ray 内部任务状态和 Kubernetes Pod 状态之间的映射。

RayJob 的排障不能只看 Kubernetes Pod。Pod Running 只是 Ray runtime 在跑，不代表 driver 成功，也不代表 Ray task 没有失败。Operator 应把 driver 状态、RayCluster 状态和 Kubernetes 状态都写到 status。

### MPIJob

MPIJob 表达 launcher + worker slot 的启动模式，常用于 Horovod、OpenMPI、传统 HPC 风格训练。

典型结构：

```text
MPIJob
  -> launcher Pod
       -> mpirun / mpiexec
       -> hostfile / ssh or kubexec protocol
  -> worker Pods
       -> slots / GPU resources
```

MPIJob 的核心语义：

- launcher 是控制入口，worker 是计算资源。
- slot 数必须和 worker 副本、每 Pod GPU/CPU 数一致。
- hostfile 或等价发现机制必须准确反映 worker 地址。
- worker 未齐时，launcher 不应过早启动或应能重试。

MPIJob 常见失败是 launcher 先启动，worker 未齐，随后连接超时。成熟 Operator 会通过 init、状态检查或调度集成降低这类问题。

### InferenceService / KServe

KServe 的 InferenceService 表达在线模型服务。它通常把一个服务拆成 predictor、transformer、explainer，并通过 Knative、Deployment、Service、Ingress/Gateway、autoscaler 等底层对象运行。

简化示例：

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: reranker
  namespace: serving
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
      runtime: vllm-runtime
      storageUri: s3://model-registry/reranker/v42
      resources:
        limits:
          nvidia.com/gpu: 1
```

生产语义包括：

- 模型加载：storageUri、model agent、local cache、Secret、CA。
- Runtime：Triton、TorchServe、vLLM、自定义 runtime。
- Revision：每次模型或配置变化生成新 revision。
- Traffic split：灰度、金丝雀、回滚。
- Autoscaling：基于并发、QPS、延迟、GPU 指标或队列长度。
- Readiness：模型是否加载完成、是否可以接流量。

InferenceService 不是普通 Deployment 的薄包装。它的价值在于把模型版本、流量、readiness、runtime 和弹性策略变成模型服务控制面。

### JobSet 与其他组合对象

JobSet、LeaderWorkerSet、XGBoostJob、TFJob 等对象也常出现在 AI 平台。它们的共同点是表达多角色、多 Job 或框架特定语义。选择 CRD 时要看业务语义，而不是看哪个 YAML 最短。

#### LeaderWorkerSet (LWS)：大模型多机推理的主流 CRD

LWS（[Kubernetes SIGs LeaderWorkerSet](https://github.com/kubernetes-sigs/lws)）专为"一组 Pod 共同服务一个推理副本"的场景设计——典型如 vLLM / SGLang / TRT-LLM 把一个 70B-405B 模型用 TP+PP 切到多机多卡上推理。每个推理副本由 1 个 leader Pod + N-1 个 worker Pod 组成，所有 Pod 必须 gang 启动、共享 NCCL/RDMA bootstrap、一起 ready 才能接流量。这套语义用裸 Deployment + Service 表达极其复杂，LWS 把它做成原生 CRD：

| 字段 | 含义 | 与 vLLM 对应 |
|---|---|---|
| `replicas` | 推理副本数（每个副本是一组 leader+worker） | vLLM 副本数 |
| `leaderWorkerTemplate.size` | 单副本内 Pod 总数（leader + worker） | TP × PP × DP / GPUs_per_pod |
| `leaderWorkerTemplate.leaderTemplate` | leader Pod 模板（处理 HTTP 入口 + Ray head 等） | vLLM serve `--tensor-parallel-size` 入口节点 |
| `leaderWorkerTemplate.workerTemplate` | worker Pod 模板（仅参与计算） | vLLM Ray worker |
| `leaderWorkerTemplate.restartPolicy` | `RecreateGroupOnPodRestart` 任一 Pod 重启则整组重启 | 必选，否则 NCCL 状态不一致 |
| `rolloutStrategy` | 副本级滚动升级策略 | 不能像 Deployment 一样按 Pod 滚 |
| `startupPolicy` | `LeaderCreated` / `LeaderReady` 控制 worker 何时开始拉起 | leader 先把模型加载完再起 worker |

**为什么需要 LWS（vs JobSet / StatefulSet / 自研）**：

| 替代方案 | 不适合大模型多机推理的原因 |
|---|---|
| Deployment + Service | 没有 gang 语义；Pod 个体重启会让 NCCL 链路死锁 |
| StatefulSet | 提供有序启动，但缺少"组级 ready"和"组级重启"语义 |
| Job / JobSet | Job 假设跑完即结束；推理是长服务，不该用 Job 模型 |
| Ray Operator + RayCluster | 强依赖 Ray runtime，不适合 vLLM 之外的引擎 |
| **LWS** | **专为"长生命周期 + 组级语义 + 滚动升级"设计，已是 vLLM/TRT-LLM/SGLang 多机部署的官方推荐路径** |

**典型 LWS YAML（vLLM 70B TP=8 跨 2 机）**：

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: vllm-llama-70b
spec:
  replicas: 2                # 2 个推理副本
  leaderWorkerTemplate:
    size: 2                  # 每副本 2 个 Pod（leader + 1 worker）
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      spec:
        containers:
        - name: vllm-leader
          image: vllm/vllm-openai:latest
          command: ["/bin/sh","-c"]
          args:
          - |
            python -m vllm.entrypoints.openai.api_server \
              --model meta-llama/Llama-3-70B \
              --tensor-parallel-size 8 \
              --pipeline-parallel-size 2 \
              --distributed-executor-backend ray
          resources:
            limits:
              nvidia.com/gpu: 8
              rdma/hca: 1
    workerTemplate:
      spec:
        containers:
        - name: vllm-worker
          image: vllm/vllm-openai:latest
          resources:
            limits:
              nvidia.com/gpu: 8
              rdma/hca: 1
  rolloutStrategy:
    type: RollingUpdate
    rollingUpdateConfiguration:
      maxUnavailable: 1
      maxSurge: 1
```

> [!NOTE]
> **LWS 已进入 Kubernetes SIGs 官方路径**（kubernetes-sigs/lws v0.4+），并被 NVIDIA / vLLM / Anyscale / Google 官方文档作为大模型多机推理推荐 CRD。如果你在裸 K8s 上做 70B+ 模型多机推理且没用 LWS，意味着你正在自己重新实现 gang 启动、组级重启、组级滚动这些已经被解决的问题。

> [!WARNING]
> **LWS + Volcano/Kueue 的叠加**：调度层（Volcano PodGroup / Kueue Workload）和工作负载层（LWS）解决不同问题：调度层确保副本"一组 Pod 同时拿到 GPU"，LWS 确保"已分配的 Pod 一起启动且组级管理"。两者通常一起用而不是替代关系。

---

## 19c.6 工程化：生产落地、版本、发布、观测与治理

### 配置设计

CRD spec 应该表达稳定的业务意图，不要把所有 Pod 字段原样暴露给用户。推荐分层：

| 层 | 示例字段 | 管理者 |
|----|----------|--------|
| 用户意图 | `replicas`、`modelRef`、`datasetRef`、`framework`、`entrypoint` | 用户或训练平台 |
| 资源策略 | GPU 数、节点池、priority、queue、gang、checkpointPolicy | 平台 |
| Pod 模板 | image、env、volume、securityContext | 平台和高级用户 |
| 默认值 | runtimeClass、tolerations、labels、sidecar、probe | admission/operator |
| 运行状态 | phase、conditions、replicaStatuses、revision | operator status |

字段越靠近业务语义，越适合作为 CRD 顶层字段；字段越靠近容器细节，越适合放在 template 或平台默认值里。

### 版本矩阵

AI Operator 的版本兼容不是单一维度。生产发布前至少维护这张矩阵：

| 维度 | 示例 | 兼容风险 |
|------|------|----------|
| Kubernetes | 1.28、1.29、1.30 | API 版本、admission、CEL 校验、Server-Side Apply |
| CRD API | `v1alpha1`、`v1beta1`、`v1` | 字段变更、status 语义、conversion |
| Operator | Training Operator、KubeRay、KServe 版本 | reconcile 行为、RBAC、底层对象变化 |
| Scheduler/Queue | default、Volcano、Kueue | gang、quota、priority、preemption |
| GPU 栈 | driver、CUDA、device plugin、GPU Operator | resource name、MIG、runtimeClass |
| Serving 栈 | KServe、Knative、Istio/Envoy、runtime | revision、autoscaling、probe、网关路由 |
| 框架 | PyTorch、Ray、MPI、Triton、vLLM | 启动参数、环境变量、协议 |

不要只写“支持 Kubernetes 1.29”。对 AI 平台而言，更重要的是“PyTorchJob v1 + Kueue x.y + H100 CUDA 12.4 + NCCL 2.x + driver branch + operator 版本”的组合是否验证过。

### CRD 升级与兼容性

CRD 一旦成为平台 API，就要像产品 API 一样治理。

关键规则：

- 新增字段优先做可选字段，并提供默认值。
- 不要轻易改变字段语义；语义变化要新增字段或新版本。
- 删除字段前先标记 deprecated，保留兼容窗口。
- status 字段也会被平台 UI、告警、自动化依赖，不能随意改名。
- 多版本 CRD 要提供 conversion webhook 或清晰迁移路径。
- storage version 迁移要测试回滚场景。

典型版本策略：

| 阶段 | API 版本 | 策略 |
|------|----------|------|
| 实验 | `v1alpha1` | 快速迭代，不承诺长期兼容 |
| 试生产 | `v1beta1` | 字段基本稳定，允许小幅调整 |
| 生产稳定 | `v1` | 强兼容，变更需要迁移和公告 |

升级 Operator 时要先确认旧对象能被新控制器 reconcile，新对象不会被旧控制器误处理，回滚时不会因为 CRD schema 或 conversion 导致对象不可读。

### 发布策略

推荐发布顺序：

1. 发布 CRD schema，但不立即启用新行为。
2. 发布 admission webhook，先以 audit 或 warn 模式验证策略。
3. 滚动升级 Operator，观察 reconcile error、workqueue depth、status update conflict。
4. 对少量 namespace 或 queue 启用新字段。
5. 扩大灰度，更新平台 UI 和文档。
6. 固化版本矩阵和回滚手册。

Operator 本身要支持 leader election，避免多副本同时控制同一对象。多副本 controller 提高可用性，但只有 leader 执行写操作。

### 观测

Operator 观测至少包含：

- 日志：reconcile 开始/结束、对象 namespace/name、generation、错误、重试原因。
- Events：面向用户的关键状态变化，例如 `WaitingForGPU`、`ImagePullFailed`、`RevisionReady`。
- Metrics：reconcile duration、error count、queue depth、requeue count、status update conflict、managed objects。
- Status：phase、conditions、observedGeneration、replica status、failure reason。
- Trace 或关联 ID：把平台提交、CRD、Operator、底层 Pod 日志串起来。

对于 AI CRD，还要暴露领域指标：训练 job 运行时长、失败重启次数、checkpoint 恢复次数、Ray driver 状态、InferenceService revision ready 延迟、模型加载耗时、流量切换状态。

### RBAC 与安全治理

Operator 权限要最小化。常见权限边界：

- 只能 watch/list/get 自己管理的 CRD 和底层对象。
- 只能 create/update/delete 自己 namespace 或目标 namespace 下的对象。
- status 更新使用 `/status` 子资源权限。
- finalizer 更新需要更新主资源 metadata 的权限。
- 不要给 cluster-admin 作为默认安装方式。

治理策略还包括：

- Admission 校验镜像来源、GPU 请求、queue、priority、hostPath、privileged。
- 限制用户直接修改 Operator 管理的底层对象。
- 对 CRD spec 做审计，记录谁提交了什么模型、数据和资源。
- 为不同团队划分 namespace、queue、quota 和 service account。

---

## 19c.7 方案设计：AI CRD 选择与 Operator 设计

### CRD 选择决策表

| 需求 | 推荐对象 | 关键判断 |
|------|----------|----------|
| 单机训练、一次性脚本 | Job | 不需要分布式角色和框架语义 |
| PyTorch DDP/FSDP 多机训练 | PyTorchJob + queue/gang | 需要 rank、world size、worker 状态、全局失败恢复 |
| Ray 数据处理、RL、批推理 | RayJob | 需要 RayCluster、driver、runtime env |
| Horovod/OpenMPI 训练 | MPIJob | 需要 launcher、hostfile、slot |
| 在线模型服务 | InferenceService | 需要模型加载、revision、autoscaling、traffic split |
| 多角色复杂批任务 | JobSet 或自定义 CRD | 需要多个 Job 的整体状态和准入 |
| 公司内部统一训练平台 | 自定义 TrainingJob CRD | 需要屏蔽框架差异、统一治理 |

### 设计方案：公司内部 TrainingJob CRD

假设平台要支持 PyTorch 和 Ray 两类训练，同时接入 Kueue 做队列准入，接入对象存储做 checkpoint。可以设计一个上层 `TrainingJob`，由内部 Operator 翻译为 PyTorchJob 或 RayJob。

设计目标：

- 用户提交统一对象，不直接接触框架 Operator 的复杂字段。
- 平台统一治理镜像、资源、queue、checkpoint、日志和成本标签。
- 底层继续复用成熟 Operator，避免重写 PyTorch/Ray 细节。

示例：

```yaml
apiVersion: platform.example.com/v1beta1
kind: TrainingJob
metadata:
  name: reranker-v42
  namespace: train
spec:
  framework: pytorch
  queue: h100-prod
  image: registry.local/ml/reranker-train:cuda12.4-v42
  entrypoint:
    command: ["torchrun"]
    args: ["--nnodes=8", "--nproc_per_node=8", "train.py"]
  replicas:
    workers: 8
    gpuPerWorker: 8
  checkpoint:
    uri: s3://checkpoints/reranker/v42
    resumePolicy: Latest
  failurePolicy:
    maxRestarts: 3
    restartScope: AllWorkers
```

内部 Operator 的责任：

```text
TrainingJob
  -> validate queue, image, checkpoint policy
  -> create Kueue Workload / PodGroup
  -> create PyTorchJob or RayJob
  -> copy important status back to TrainingJob
  -> emit platform-level events and metrics
```

责任边界：

| 组件 | 责任 |
|------|------|
| TrainingJob Operator | 统一 API、治理、准入、状态聚合 |
| PyTorchJob/RayJob Operator | 框架级底层对象和运行状态 |
| Kueue/Volcano | 资源队列、gang、priority、quota |
| GPU Operator | 驱动、device plugin、runtime |
| 用户代码 | 训练逻辑、checkpoint 读写、框架参数 |

这个方案的风险是状态转译复杂。内部 Operator 不应把底层 CRD 的所有字段重新发明一遍，而应只抽象平台真正需要稳定承诺的字段。

---

## 19c.8 故障排除：症状、证据、根因、动作

| 症状 | 证据 | 常见根因 | 处理动作 |
|------|------|----------|----------|
| CRD 创建后没有底层 Pod | `kubectl describe <crd>`、Operator logs、events | Operator 未运行、RBAC 不足、watch namespace 错、admission 拒绝 | 修复部署、权限、namespace selector 或 webhook |
| status 长时间不更新 | `status.observedGeneration`、Operator logs | controller 卡住、status 子资源无权限、update conflict、leader election 异常 | 检查 `/status` RBAC、reconcile error、leader |
| 删除对象卡在 Terminating | `metadata.finalizers`、Operator logs | finalizer 清理失败、外部资源删除失败、Operator 离线 | 恢复 Operator，修复外部 API；确认无泄漏后才手工移除 finalizer |
| TorchJob worker 部分 Running | Pod events、queue status、CRD conditions | 无 gang 调度、GPU 碎片、quota 不足 | 接入 Kueue/Volcano，调整队列或节点池 |
| TorchJob 反复重启 | Pod previous logs、checkpoint logs、status restart count | 非 elastic 训练单 rank 失败、checkpoint 不可恢复、应用异常 | 区分应用 bug 与节点故障，调整 failurePolicy |
| RayJob Pod Running 但任务失败 | RayJob status、driver logs、Ray dashboard | runtime env 失败、依赖缺失、driver 异常 | 看 driver 而不只看 Pod，修 runtime env 或入口命令 |
| MPIJob launcher 超时 | launcher logs、worker Pod 状态、hostfile | worker 未齐、hostfile 错、端口/NetworkPolicy 阻断 | 延后 launcher、修发现机制、检查网络策略 |
| InferenceService Ready=False | InferenceService status、revision、pod logs、endpoint | 模型下载失败、runtime 启动失败、readiness 失败 | 查 storage secret、runtime、probe、revision |
| 新版本无法回滚 | CRD versions、storedVersions、conversion logs | CRD schema 不兼容、conversion webhook 失败 | 固定 storage version，测试降级，修 conversion |

常用命令：

```bash
kubectl get crd | grep -E 'pytorch|ray|mpi|inference'
kubectl get pytorchjob -n train train-reranker -o yaml
kubectl describe pytorchjob -n train train-reranker
kubectl get pod -n train -l training.kubeflow.org/job-name=train-reranker -o wide
kubectl logs -n kubeflow deploy/training-operator --tail=200
kubectl get events -n train --sort-by=.lastTimestamp
```

排障时先看 CRD status，再看底层对象，再看 Operator 日志。只看 Pod 会丢失高层语义；只看 CRD 会丢失底层失败证据。

---

## 19c.9 反模式与 Checklist

### 反模式

| 反模式 | 后果 | 修正 |
|--------|------|------|
| 把 Operator 写成 YAML 渲染器 | Pod 漂移、失败、删除后无法恢复 | 实现真正 reconcile，按实际状态收敛 |
| status 只有 Running/Failed | 用户不知道卡在调度、镜像、模型加载还是应用 | 使用 conditions、reason、message、observedGeneration |
| 控制器覆盖用户和其他控制器字段 | 破坏手工修复、HPA、service mesh、平台策略 | 用 patch/SSA 管理字段边界 |
| 不设置 ownerReference 和稳定 labels | 无法清理、查询、聚合成本和日志 | 所有 managed objects 统一标签和 owner |
| finalizer 不幂等 | 删除对象永久卡住 | finalizer 只做删除清理，失败可重试并写事件 |
| 失败恢复不分类 | 镜像错误、代码错误、节点故障都反复重启 | 区分调度、镜像、应用、节点、外部依赖 |
| CRD 字段直接暴露所有 Pod 细节 | API 难以治理，升级兼容困难 | 顶层表达业务语义，template 留给高级场景 |
| 忽略版本矩阵 | 升级后旧任务不可读或行为变化 | 维护 CRD、Operator、K8s、GPU、调度、框架矩阵 |

### Checklist

- CRD 顶层字段是否表达 AI 语义，而不是简单复制 Pod template？
- `spec` 与 `status` 的边界是否清楚？
- 是否启用了 status subresource？
- status 是否包含 `observedGeneration`、conditions、reason、message？
- reconcile 是否幂等，是否能处理重复事件和中途失败？
- 底层对象是否有 ownerReference、稳定 labels 和可查询 selector？
- finalizer 是否只处理删除清理，且幂等、可超时、可重试？
- 失败策略是否区分可恢复、不可恢复和需要人工介入？
- 是否接入 queue/gang/priority/quota，而不是让多 worker 训练裸跑？
- Operator 是否有最小 RBAC、leader election、metrics、events 和告警？
- CRD 升级是否测试旧对象、新对象、回滚和 conversion？
- 平台 UI 是否基于 conditions 展示原因，而不是只展示 phase？

---

## 19c.10 Worked Example：8 节点 PyTorch 训练的控制面设计

### 场景

团队要训练一个 reranker，需求如下：

- 8 个节点，每节点 8 张 H100，总计 64 GPU。
- 必须所有 worker 同时启动，否则 rendezvous 会超时。
- 允许节点故障后从最近 checkpoint 全局重启，最多 3 次。
- 训练结束后保留失败 Pod 证据，成功后清理运行中 Pod。
- 平台要在 UI 上展示“等待队列准入”“等待 GPU”“训练中”“从 checkpoint 恢复”等状态。

### 设计

选择对象：

| 需求 | 设计 |
|------|------|
| PyTorch 分布式语义 | 使用 PyTorchJob |
| 64 GPU 整体准入 | 接入 Kueue Workload 或 Volcano PodGroup |
| 全局失败恢复 | failurePolicy 设置为 AllWorkers restart，配合 checkpoint |
| 状态展示 | Operator 写 conditions，平台聚合 Pod events |
| 删除清理 | ownerReference 清理 Pod，finalizer 释放 queue admission 和 checkpoint lease |

控制路径：

```text
TrainingJob/PyTorchJob submitted
  -> admission injects queue, labels, tolerations, runtimeClass
  -> Operator creates Workload/PodGroup
  -> Queue admits only when 8x8 H100 fit
  -> Operator creates master/worker Pods and rendezvous Service
  -> Scheduler binds all Pods
  -> Pods start torchrun
  -> Operator observes replica status and writes Running
  -> failure triggers global restart and checkpoint resume
```

### 状态设计

```yaml
status:
  observedGeneration: 3
  phase: Admitting
  conditions:
    - type: QueueAdmitted
      status: "False"
      reason: WaitingForGangResources
      message: "requires 8 nodes with 8 H100 each in queue h100-prod"
    - type: PodsReady
      status: "False"
      reason: NotCreated
    - type: CheckpointReady
      status: "True"
      reason: LatestCheckpointFound
      message: "s3://checkpoints/reranker/v42/step-18400"
```

### 故障演练

故障：5 个 worker Running，3 个 worker Pending，训练卡在 rendezvous。

证据：

```bash
kubectl get pod -n train -l training.kubeflow.org/job-name=reranker-v42 -o wide
kubectl describe pytorchjob -n train reranker-v42
kubectl get events -n train --sort-by=.lastTimestamp
```

根因：没有 gang 准入，调度器先绑定了部分 Pod，剩余 3 个 Pod 因 GPU 形状不足 Pending。对 PyTorch 非 elastic 训练而言，部分启动没有业务意义。

处理：

1. 为任务创建 Workload/PodGroup，要求 8 个 8-GPU worker 整体准入。
2. 未准入前不创建训练 Pod，或让 Pod 处于 gated 状态。
3. status 写 `QueueAdmitted=False`，reason 为 `WaitingForGangResources`。
4. 释放已占用但无法成组的 Pod，避免 GPU 被半个任务长时间占住。

这个例子说明，AI Operator 的职责不是少写 YAML，而是把训练语义传递给调度、状态、恢复和治理系统。

---

## 本章小结

CRD 让 AI 语义进入 Kubernetes API，Operator 通过 reconciliation loop 把这些语义持续落实到底层对象。高质量的 AI Operator 必须处理 spec/status 边界、幂等控制、状态机、ownerReference、finalizer、失败恢复、版本兼容、观测和治理。

TorchJob、RayJob、MPIJob、InferenceService 的核心价值不是包装 Pod，而是把 rank、driver、launcher、revision、traffic、checkpoint 和恢复策略变成平台可解释、可审计、可自动化的控制面能力。

## 练习题

1. 为什么说 Operator 不是 Helm 的替代品？请从运行时漂移和失败恢复两个角度解释。
2. 设计一个 `TrainingJob` status，至少包含 5 个 conditions，并说明每个 condition 对用户有什么价值。
3. 一个 PyTorchJob 删除后卡在 Terminating，你会按什么顺序检查 finalizer、Operator 日志和外部资源？
4. RayJob 的 Pod 全部 Running，但任务失败。为什么不能只根据 Pod 状态判断 RayJob 成功？
5. 选择一个你熟悉的模型服务场景，说明 InferenceService 比 Deployment 多表达了哪些生产语义。
