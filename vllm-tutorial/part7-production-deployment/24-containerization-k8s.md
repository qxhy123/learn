# 第24章：容器化与 Kubernetes 部署

> 从"本机跑通"到"生产上线"，容器化和 Kubernetes 是绕不开的一步。本章教你如何把 vLLM 包装成可靠的生产服务。

---

## 学习目标

学完本章，你将能够：

1. 构建 vLLM 的 Docker 镜像
2. 编写 Kubernetes Deployment 和 Service 清单
3. 配置 GPU 调度和资源限制
4. 实现健康检查和自动扩缩容
5. 管理模型存储和加载

---

## 24.1 Docker 部署

### 使用官方镜像

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
```

### 自定义 Dockerfile

```dockerfile
FROM vllm/vllm-openai:latest

# 预下载模型（避免运行时下载）
RUN pip install huggingface-hub && \
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir /models/qwen2.5-7b

# 设置默认启动参数
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "/models/qwen2.5-7b", \
     "--port", "8000", \
     "--gpu-memory-utilization", "0.9"]
```

### 健康检查

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

---

## 24.2 Kubernetes 部署

### Deployment 清单

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  labels:
    app: vllm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - "--model"
            - "Qwen/Qwen2.5-7B-Instruct"
            - "--port"
            - "8000"
            - "--gpu-memory-utilization"
            - "0.9"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: 1
            requests:
              cpu: "4"
              memory: "32Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 10
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-pvc
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

### 多 GPU 部署

```yaml
# 2 GPU 张量并行
resources:
  limits:
    nvidia.com/gpu: 2

args:
  - "--model"
  - "meta-llama/Llama-3.1-70B-Instruct"
  - "--tensor-parallel-size"
  - "2"
```

---

## 24.3 模型存储

### PVC 持久化

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 200Gi
  storageClassName: fast-ssd
```

### 初始化容器下载模型

```yaml
initContainers:
  - name: download-model
    image: python:3.11-slim
    command: ["sh", "-c"]
    args:
      - |
        pip install huggingface-hub &&
        huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
          --local-dir /models/qwen
    volumeMounts:
      - name: model-cache
        mountPath: /models
```

---

## 24.4 自动扩缩容

### HPA（基于 CPU/自定义指标）

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 1
  maxReplicas: 4
  metrics:
    - type: Pods
      pods:
        metric:
          name: vllm_num_requests_waiting
        target:
          type: AverageValue
          averageValue: "20"
```

### KEDA（事件驱动扩缩容）

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
spec:
  scaleTargetRef:
    name: vllm-server
  minReplicaCount: 1
  maxReplicaCount: 8
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: vllm_requests_waiting
        query: avg(vllm:num_requests_waiting)
        threshold: "30"
```

---

## 24.5 生产检查清单

```
部署前:
  ☐ 镜像中已包含或可访问模型权重
  ☐ GPU 资源限制正确设置
  ☐ /dev/shm 已挂载（共享内存）
  ☐ 健康检查已配置（初始延迟足够长）
  ☐ 模型加载超时合理设置

运行时:
  ☐ Prometheus 指标已接入监控
  ☐ 关键告警已配置
  ☐ 日志已收集（ELK/Loki）
  ☐ HPA/KEDA 扩缩容已测试

安全:
  ☐ API 认证已启用
  ☐ 网络策略限制了访问范围
  ☐ 镜像来源可信
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| Docker | 官方镜像 + 自定义配置 |
| K8s | Deployment + Service + PVC |
| GPU 调度 | `nvidia.com/gpu` 资源限制 |
| 健康检查 | `/health` 端点 |
| 扩缩容 | HPA/KEDA 基于请求队列深度 |

---

## 练习题

### 实践题

1. 使用 Docker 部署 vLLM 并通过健康检查验证。
2. 编写完整的 K8s 部署清单（Deployment + Service + PVC）。

### 思考题

3. 为什么 `initialDelaySeconds` 需要设置得比较长（如 120s）？
4. GPU 上的 HPA 与 CPU 上的 HPA 有什么不同的考虑？
