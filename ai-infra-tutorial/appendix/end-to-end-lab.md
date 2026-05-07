# 附录K：端到端可运行实验

## 从数据瓶颈到上线回滚的一条实验主线

这个附录不是新的理论章节，而是一条可以在实验环境里反复跑的 lab spine。它把数据加载、单机训练、DDP 通信、checkpoint、vLLM serving、量化、发布回滚和可观测性串成同一条证据链。真实集群里的镜像名、模型名、对象存储路径和监控系统会不同，下面命令应按本地环境替换后执行；不要直接在生产命名空间里运行故障注入。

建议先准备一个小模型和一个小数据集，例如 100MB 到 5GB 的 tokenized shard、一个 100M 到 1B 参数量模型，或团队内部的 smoke-test checkpoint。目标不是追求绝对性能，而是让每一阶段都有基线、证据和回滚动作。

## 实验目录约定

```bash
export LAB_ID=ai-infra-e2e-$(date +%Y%m%d-%H%M%S)
export LAB_ROOT=$PWD/runs/$LAB_ID
mkdir -p "$LAB_ROOT"/{data,profiles,checkpoints,serving,release,observability}
```

必须收集的公共证据：

| 证据 | 用途 |
|------|------|
| `run.env` | 记录镜像、驱动、CUDA、PyTorch、NCCL、vLLM、模型和数据版本 |
| `config.yaml` | 记录 batch、seq_len、并行策略、checkpoint 周期、serving 参数 |
| `metrics.jsonl` | 每阶段统一写入 tokens/s、step time、P95/P99、错误率、GPU/CPU/IO 指标 |
| `events.log` | 记录人工动作、故障注入、回滚、重跑时间点 |

```bash
{
  date
  nvidia-smi
  python - <<'PY'
import torch, sys
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu_count", torch.cuda.device_count())
PY
} | tee "$LAB_ROOT/run.env"
```

通过标准：任意阶段失败时，能从 `run.env`、配置、profile、日志和指标里解释失败点；不是只留下“跑过一次”的口头结论。

## 1. 数据加载瓶颈

目标：先判断 GPU 是否被数据链路饿住。不要一上来改模型，先把 dataset、tokenizer、DataLoader worker、pin memory、远程读取和本地缓存拆开测。

示例命令：

```bash
python tools/bench_dataloader.py \
  --dataset "$LAB_ROOT/data/train" \
  --batch-size 8 \
  --seq-len 2048 \
  --num-workers 0,2,4,8 \
  --pin-memory true \
  --output "$LAB_ROOT/data/dataloader-bench.jsonl"
```

如果仓库或项目没有现成脚本，可以用等价脚本记录每秒样本数、每秒 token 数、batch 构造时间、H2D copy 时间和 worker CPU 使用率。

预期信号：

- `num_workers` 从 0 增加到合理值后，batch ready time 明显下降。
- GPU 侧训练 step 中的 data wait 占比下降，而不是只看到 CPU 更忙。
- 远程对象存储读取时，首轮慢、缓存命中后变快；如果两轮都慢，要查 shard 大小、并发和网络。

故障注入：

```bash
# 限制本地读取带宽，模拟远程存储或共享盘抖动。只在实验机执行。
sudo tc qdisc add dev eth0 root tbf rate 200mbit burst 32kbit latency 400ms
# 恢复
sudo tc qdisc del dev eth0 root
```

应收集证据：DataLoader benchmark、`iostat -x 1`、`pidstat -urd 1`、缓存冷热两轮对比、坏样本或慢 shard 列表。

通过标准：固定 batch 和 seq_len 后，数据链路能稳定提供训练所需 token/s 的 1.2 倍以上；如果达不到，必须明确瓶颈在 CPU preprocess、tokenizer、磁盘、网络还是 worker 调度。

## 2. 单机训练 profiling

目标：在单节点建立可解释的 step time baseline，再决定是否扩到多机。单机不清楚，多机只会把问题放大。

示例命令：

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --config configs/lab-single-node.yaml \
  --max-steps 200 \
  --log-interval 10 \
  --profile-steps 20:60 \
  --output-dir "$LAB_ROOT/profiles/single"
```

也可以先跑 1 GPU，再跑单机 2/4/8 GPU，分开记录吞吐和显存。

预期信号：

- warmup 后 step time 收敛，P50/P95 差距可解释。
- profiler 里 GPU kernel、H2D、DataLoader wait、optimizer step 的占比清楚。
- 显存峰值与配置估算接近；OOM 时能区分 activation、optimizer state、KV/缓存或 fragmentation。

故障注入：

```bash
# 故意关闭 pin_memory 或把 num_workers 设为 0，验证 profile 是否能看到 data wait 上升。
torchrun --standalone --nproc_per_node=1 train.py \
  --config configs/lab-single-node.yaml \
  --data.num_workers 0 \
  --data.pin_memory false \
  --max-steps 80
```

应收集证据：PyTorch profiler trace、`nvidia-smi dmon`、GPU memory summary、step time breakdown、loss 曲线前 200 step。

通过标准：单机 baseline 有固定配置、固定输入、固定随机种子和可复现吞吐；至少能回答“瓶颈是数据、计算、显存还是优化器/框架开销”。

## 3. DDP 通信与扩展效率

目标：从单机扩到 DDP/FSDP 时，先验证通信路径和拓扑，再看训练吞吐。NCCL fallback、跨 socket 放置和慢节点都会让扩展效率失真。

预检命令：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
torchrun --nnodes=2 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:29500" \
  train.py --config configs/lab-ddp.yaml \
  --max-steps 100 \
  --output-dir "$LAB_ROOT/profiles/ddp"
```

如果有 `nccl-tests`，先跑 all-reduce 基准：

```bash
./build/all_reduce_perf -b 8M -e 4G -f 2 -g 8 | tee "$LAB_ROOT/profiles/nccl-allreduce.log"
```

预期信号：

- NCCL 日志显示使用预期的 IB/RDMA 或 NVLink/NVSwitch 路径，而不是 `NET/Socket` fallback。
- GPU 数翻倍后，global tokens/s 提升接近预期；至少能计算 scaling efficiency。
- all-reduce 时间、backward compute 时间和 data wait 时间口径分开。

故障注入：

```bash
# 模拟错误网卡选择，验证 NCCL 日志和吞吐是否能发现 fallback。
NCCL_SOCKET_IFNAME=lo NCCL_DEBUG=INFO \
torchrun --nnodes=2 --nproc_per_node=8 train.py \
  --config configs/lab-ddp.yaml --max-steps 30
```

应收集证据：NCCL debug 日志、all-reduce benchmark、每 rank step time、straggler gap、节点拓扑、训练吞吐对比表。

通过标准：通信路径可被日志证明，扩展效率有数字结论；若效率低于预期，能定位到网络、拓扑、batch 太小、梯度 bucket、慢 rank 或数据不均衡中的一类。

## 4. Checkpoint 与恢复演练

目标：证明训练不是只能“顺利跑完”，而是能在中断后恢复到明确状态。checkpoint 必须包含权重、optimizer、scheduler、RNG、数据游标和 manifest。

示例命令：

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/lab-checkpoint.yaml \
  --max-steps 300 \
  --save-every 100 \
  --output-dir "$LAB_ROOT/checkpoints/run-a"

python tools/inspect_checkpoint.py \
  --path "$LAB_ROOT/checkpoints/run-a/step_000200" \
  --verify-checksum
```

故障注入：

```bash
# 在 checkpoint 写入附近终止 rank，验证 manifest 原子性和恢复行为。
pkill -f "train.py.*lab-checkpoint"

torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/lab-checkpoint.yaml \
  --resume "$LAB_ROOT/checkpoints/run-a/latest" \
  --max-steps 360 \
  --output-dir "$LAB_ROOT/checkpoints/run-a-resume"
```

预期信号：

- 未完成 checkpoint 不会被 `latest` manifest 指向。
- resume 后 loss、lr、global step、数据游标连续。
- RTO、RPO、checkpoint 写入时间和存储带宽有记录。

应收集证据：checkpoint manifest、checksum、对象数量、写入耗时、恢复耗时、resume 前后 loss delta、清理过期 checkpoint 的记录。

通过标准：任意一次人为中断后能从最近完整 checkpoint 恢复；恢复后的前 20 个 step 没有重复样本、学习率跳变或 loss 异常尖峰。

## 5. vLLM / Serving 基线

目标：把训练产物转成可服务模型，明确 TTFT、ITL、吞吐、KV Cache、批处理和冷启动边界。

示例命令：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model "$LAB_ROOT/release/model" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  2>&1 | tee "$LAB_ROOT/serving/vllm.log"
```

压测命令：

```bash
python tools/bench_serving.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model lab-model \
  --concurrency 1,4,16,32 \
  --prompt-len 512 \
  --output-len 128 \
  --duration 300 \
  --output "$LAB_ROOT/serving/bench.jsonl"
```

预期信号：

- 单并发时 TTFT、ITL 和输出 token/s 稳定。
- 并发上升时，continuous batching 提升吞吐，但 P99 不应无限恶化。
- vLLM 日志和 metrics 能看到 KV Cache 使用率、eviction、OOM 或 scheduler backlog。

故障注入：

```bash
# 用长 prompt 压 KV Cache，验证 OOM、限流或拒绝策略。
python tools/bench_serving.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model lab-model \
  --concurrency 16 \
  --prompt-len 3500 \
  --output-len 512 \
  --duration 120
```

应收集证据：启动日志、冷启动时间、压测原始结果、P50/P95/P99、TTFT/ITL、GPU memory、KV Cache 指标、失败请求样本。

通过标准：服务在目标并发下达到实验 SLO；若失败，能判断是 prefill、decode、KV Cache、路由、冷启动还是模型加载问题。

## 6. 量化实验

目标：验证量化是否真的降低成本或延迟，而不是只改变模型文件大小。量化必须同时看性能、质量和兼容性。

示例命令：

```bash
python tools/quantize_model.py \
  --model "$LAB_ROOT/release/model" \
  --method awq \
  --calib-data "$LAB_ROOT/data/calib.jsonl" \
  --output "$LAB_ROOT/release/model-awq"

python tools/eval_golden.py \
  --model "$LAB_ROOT/release/model-awq" \
  --golden "$LAB_ROOT/data/golden-prompts.jsonl" \
  --output "$LAB_ROOT/release/quant-eval.json"
```

预期信号：

- 显存占用下降，单副本可承载更大 batch、更多并发或更长上下文。
- TTFT/ITL 至少一项改善，或者单位请求成本下降。
- golden prompt、离线评测和人工抽检没有越过质量退化阈值。

故障注入：

```bash
# 故意使用不匹配的校准集，观察质量门禁是否能拦截。
python tools/quantize_model.py \
  --model "$LAB_ROOT/release/model" \
  --method awq \
  --calib-data "$LAB_ROOT/data/wrong-domain-calib.jsonl" \
  --output "$LAB_ROOT/release/model-awq-wrong-calib"
```

应收集证据：量化配置、校准集 hash、模型大小、显存峰值、serving benchmark、golden diff、失败样本、兼容性说明。

通过标准：量化模型同时通过性能门禁和质量门禁；如果只降低显存但质量或 P99 不达标，不进入发布阶段。

## 7. 发布、灰度与回滚

目标：把模型、tokenizer、prompt、serving 参数和量化策略作为一个发布单元管理。回滚不能只回权重。

示例发布记录：

```bash
cat > "$LAB_ROOT/release/release.json" <<'JSON'
{
  "model": "lab-model",
  "version": "2026.05.07-lab",
  "artifact": "model-awq",
  "tokenizer": "tokenizer-v1",
  "serving_config": "vllm-awq-4096.yaml",
  "canary_percent": 5,
  "rollback_to": "2026.05.07-baseline"
}
JSON
```

示例灰度动作：

```bash
kubectl -n ai-serving set image deploy/lab-model-canary \
  server=registry.example.com/lab-model:2026.05.07-lab

kubectl -n ai-serving annotate deploy/lab-model-canary \
  release.ai.example.com/id="$LAB_ID" \
  release.ai.example.com/rollback-to="2026.05.07-baseline"
```

预期信号：

- canary 流量、错误率、P99、质量采样和成本/request 与 baseline 分开看。
- 发布记录能定位到完整制品组合，而不是一个模糊 tag。
- 回滚动作有预演，回滚后指标回到 baseline 区间。

故障注入：

```bash
# 模拟错误 tokenizer 或错误 serving 参数的 canary，验证门禁能否阻断扩大流量。
kubectl -n ai-serving set env deploy/lab-model-canary TOKENIZER_ID=tokenizer-wrong
```

回滚命令：

```bash
kubectl -n ai-serving rollout undo deploy/lab-model-canary
kubectl -n ai-serving rollout status deploy/lab-model-canary --timeout=5m
```

应收集证据：release record、canary dashboard、golden prompt 在线采样、错误样本、rollout event、rollback event、发布前后 diff。

通过标准：canary 达标才扩大流量；故障注入能触发阻断或回滚；回滚后模型、tokenizer、prompt、serving config 全部回到同一 baseline 版本。

## 8. 可观测性与事故闭环

目标：把前面所有实验的证据接入一个统一观测面。训练看 step、通信、checkpoint；推理看请求、队列、KV Cache、质量和成本；发布看版本和回滚事件。

最小指标清单：

| 层次 | 指标 |
|------|------|
| 数据 | samples/s、tokens/s、bad shard count、DataLoader wait、cache hit |
| 训练 | step time、loss、MFU/HFU、HBM peak、all-reduce time、straggler gap |
| Checkpoint | write duration、restore duration、RPO、RTO、manifest failure |
| Serving | request/s、TTFT、ITL、P95/P99、queue wait、KV usage、OOM、error rate |
| 发布 | release version、canary percent、rollback count、quality sample fail rate |
| 成本 | GPU hours、cost/request、idle GPU、tenant quota usage |

示例查询：

```promql
histogram_quantile(0.99, sum(rate(llm_request_latency_seconds_bucket[5m])) by (le, model, version))
sum(rate(vllm_generation_tokens_total[5m])) by (model, version)
max(training_step_time_seconds) by (job, rank)
sum(rate(checkpoint_write_failures_total[10m])) by (job)
```

故障注入：

```bash
# 让 canary 承担过高流量，验证 burn rate 告警、限流和回滚链路。
kubectl -n ai-serving annotate ingress/lab-model \
  traffic.ai.example.com/canary-percent="80" --overwrite
```

应收集证据：dashboard snapshot、告警事件、trace 样本、应用日志、Kubernetes event、release diff、incident timeline。

通过标准：一次故障注入能形成完整闭环：告警触发 -> 定位版本或资源层 -> 缓解或回滚 -> 指标恢复 -> 复盘项落到配置、门禁或容量规则。

## 最终验收表

| 阶段 | 必须回答的问题 | 通过证据 |
|------|----------------|----------|
| 数据加载 | GPU 是否在等数据，瓶颈在哪一层？ | dataloader bench、IO/CPU 指标、冷热缓存对比 |
| 单机训练 | 单机 step time 能否解释？ | profiler trace、显存预算、loss 曲线 |
| DDP 通信 | 多机慢是通信、拓扑还是 batch 问题？ | NCCL 日志、all-reduce benchmark、scaling efficiency |
| Checkpoint | 中断后能否恢复到明确状态？ | manifest、checksum、resume loss、RTO/RPO |
| vLLM serving | 延迟和吞吐瓶颈在哪里？ | TTFT/ITL/P99、KV Cache、压测日志 |
| 量化 | 是否同时改善成本并守住质量？ | 量化配置、显存/延迟对比、golden eval |
| 发布回滚 | 回滚是否覆盖完整发布单元？ | release record、canary 指标、rollback event |
| 可观测性 | 事故能否从告警走到复盘？ | dashboard、trace/log、timeline、改进项 |

如果八个阶段都能给出证据，而不是只给出结论，这个实验才算通过。此时你已经具备一条可迁移的 AI Infra 生产演练骨架：换模型、换集群、换 serving engine 时，只需要替换命令和阈值，不需要重写排障思路。
