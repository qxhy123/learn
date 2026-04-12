# 第18章：性能、基准与调优

> 高阶文件系统性能分析的核心，不是“跑个 benchmark 看吞吐”，而是先定义工作负载、识别层次、拆开元数据与数据路径、再区分 cache、文件系统和设备各自贡献了什么。

## 学习目标

完成本章后，你将能够：

1. 区分吞吐、IOPS、平均延迟、尾延迟、放大因子等关键指标
2. 理解顺序/随机、读/写、同步/异步、元数据密集/数据密集工作负载的差异
3. 识别 page cache、readahead、copy-up、journaling、flush 对 benchmark 的扭曲方式
4. 建立从应用模式到设备层的性能定位框架
5. 用更接近工程实战的方式理解“为什么慢、慢在哪、怎么改”
6. 认识 benchmark 设计矩阵、容器/虚拟化偏差和按层观测证据的必要性

---

## 正文内容

### 18.1 文件系统性能从来不是一个数字

你看到的“快/慢”至少可能来自这些完全不同的路径：

- 顺序大文件读取
- 小文件随机写入
- 目录扫描和 `stat` 风暴
- 高频 `fsync`
- overlayfs 首次 copy-up
- snapshot/reflink 带来的写时复制
- NFS/远端缓存导致的额外往返

所以一个 benchmark 的结论如果不带负载描述，几乎没有解释价值。

### 18.2 先把工作负载分类

至少要先把负载放进这些维度：

| 维度 | 典型选项 |
|------|----------|
| 数据模式 | 顺序读、顺序写、随机读、随机写、混合 |
| 数据粒度 | 大块、4K、小文件、目录项、元数据密集 |
| 持久化要求 | buffer write、`fsync`、`fdatasync`、同步事务 |
| 访问路径 | page cache、mmap、O_DIRECT、FUSE、远端协议 |
| 并发形态 | 单线程、多线程、多进程、多租户 |
| 文件系统类型 | ext4、xfs、btrfs、overlayfs、NFS 等 |

只有先做这一步，后面的“调优建议”才不会变成空话。

### 18.3 吞吐高不代表延迟低，平均值低也不代表尾巴没炸

很多线上故障不是平均值变坏，而是：

- 99p/99.9p 尾延迟暴涨
- 一小部分请求卡在 flush / checkpoint / copy-up
- 大多数请求被 cache 命中掩盖了慢路径

因此至少要看：

- 吞吐
- 平均延迟
- p95/p99/p99.9
- IOPS
- 写放大 / 读放大
- cache hit / miss 直觉

### 18.4 元数据负载为什么经常比数据负载更难看出来

很多文件系统性能问题根本不在“读写数据块”，而在：

- 路径查找
- dentry/inode cache miss
- 大目录索引
- 创建/删除海量小文件
- rename / link / unlink
- overlayfs copy-up 和 whiteout 处理

如果你只盯设备吞吐，很可能会错过真正热点：CPU、锁、VFS、目录索引、journal 元数据路径。

### 18.5 page cache 如何欺骗 benchmark

一个很典型的误判链：

1. 第一次读大文件，触发 page-in 和 readahead
2. 第二次读，全部命中 cache
3. 得出结论：“磁盘速度超级快”

另一个典型误判链：

1. 连续 `write()` 大量数据
2. 只看写系统调用返回速度
3. 不测 `fsync`
4. 得出结论：“这个文件系统写入特别快”

实际上你测到的往往只是 dirty page 进入 cache 的速度，不是稳定介质吞吐。

### 18.6 journaling、flush、copy-up 会制造慢路径

即使大部分请求很快，也可能有一些操作天然更慢，因为它们跨越了更重的语义边界：

- `fsync` 需要推动日志提交和设备 flush
- overlayfs 首次写需要 copy-up
- 大目录 rename 可能需要更多元数据操作
- btrfs/zfs 一类 CoW 路径可能触发更多元数据分叉
- NFS 可能在 close/open 边界上承担额外同步成本

这些慢路径不一定吞掉平均值，但会明显拉高尾延迟。

### 18.7 一个从上到下的定位模型

可以按下面顺序定位：

1. **应用层**：访问模式是什么？是不是自己制造了小同步写或 `stat` 风暴？
2. **VFS/路径层**：卡在路径查找、目录遍历、锁竞争，还是对象生命周期？
3. **cache/writeback 层**：是不是 page cache、dirty throttling、回写压力在作怪？
4. **文件系统层**：是不是 journaling、extent 分配、copy-up、snapshot/CoW 路径代价高？
5. **块层/设备层**：是不是 flush、队列深度、随机写、TRIM、设备尾延迟？

只要这五层没拆开，“调优”往往只是碰运气改参数。

### 18.8 benchmark 设计前，先写实验协议

真正有意义的 benchmark 至少要说明：

- 工作集是否大于内存
- 是否清 cache / 预热
- 是否包含 `fsync` 或 flush
- 是否走 page cache 还是 O_DIRECT
- 文件大小与文件数量
- 并发度
- 文件系统类型和挂载参数
- 设备类型和虚拟化环境

更进一步，最好在实验开始前就写出：

- 你要验证的假设是什么
- 你预计会落在哪一层成为瓶颈
- 什么结果会推翻当前判断
- 哪些指标是主指标，哪些只是辅助观测

否则你测到的数字很可能无法复现，也不能迁移到生产环境。

### 18.9 容器、cgroup、NUMA 和虚拟化会污染结论

很多 benchmark 在裸机上和在线容器里差异很大，不是因为文件系统突然“变差了”，而是因为：

- cgroup I/O 限流改变了排队形态
- memory limit 改变了 page cache 命中率和回收压力
- NUMA 远端内存访问放大了缺页和拷贝代价
- 虚拟化/云盘把 flush、队列深度、突发带宽包装成另一套行为

如果不把这些环境因素写清楚，你做的可能不是“文件系统 benchmark”，而是“当前运行环境综合 benchmark”。

### 18.10 把观测命令嵌进 benchmark，而不是事后补猜

一个更可信的实验不会只留下 `fio` 或应用程序的最终数字，还应同时收集证据：

| 想回答的问题 | 典型证据 |
|--------------|----------|
| 是 CPU/锁还是设备在卡 | `perf stat`、`perf record/report` |
| 是 cache 在掩盖还是设备真快 | `vmstat 1`、`/proc/meminfo`、冷热缓存对照 |
| 是回写/flush 在拖尾巴 | `iostat -xz 1`、`pidstat -d 1`、`fsync` 对照实验 |
| 是元数据风暴还是数据路径 | 小文件/目录操作与大文件顺序 I/O 分开测 |
| 是 overlayfs / 远端语义在作怪 | `findmnt`、`mountinfo`、NFS mountstats、copy-up 场景对照 |

没有这条证据链，你很容易只得到“它慢了”，却不知道“它为什么慢”。

### 18.11 一个更可信的对照实验矩阵

如果你真的要比较两种文件系统或两组参数，至少做出这些对照：

1. 冷缓存 vs 热缓存
2. 不含 `fsync` vs 包含 `fsync`
3. 大文件顺序 I/O vs 小文件元数据密集操作
4. 单线程 vs 多线程/多进程并发
5. 本地文件系统 vs overlayfs / 远端挂载层

这样你最后得到的不是一个数字，而是一张行为地图：系统在哪些语义和负载下便宜，在哪些条件下突然变贵。

### 18.12 调优不是“换更快磁盘”那么简单

常见有效调优方向包括：

- 改应用写入协议，减少高频同步点
- 合并小文件，降低目录和 inode 压力
- 重新设计目录层级，避免超大目录热点
- 控制 cache 污染，区分热数据与冷扫描
- 选择更匹配负载的文件系统和挂载参数
- 避免容器层无意义 copy-up
- 对远端文件系统重新设计同步协议而不是照搬本地假设

但更高阶的一点是：很多最有效的调优根本不在挂载参数，而在协议与数据组织方式。例如：

- 把“每条记录都 `fsync`”改成批量提交
- 把“上百万小文件”改成分层目录或对象打包
- 把“用 mtime 当同步协议”改成显式版本号和 manifest

### 18.13 一个高级判断标准

不要问“这个文件系统是不是更快”，而要问：

- 对我这种负载，它的快路径是什么？
- 它的慢路径是什么？
- 最贵的是元数据、数据、flush、copy-up 还是网络往返？
- 我能不能通过协议设计绕开最贵路径？

这才是高阶性能分析，而不是比较宣传页数字。

### 18.14 性能调优的最终目标不是“更快”，而是“更可预测”

很多工程事故里，真正致命的不是平均吞吐低，而是：

- 偶发 `fsync` 卡顿
- 首次写 overlay copy-up 尾巴极长
- 元数据路径在高峰期突然恶化
- 远端挂载在抖动时把应用重试风暴放大

因此高阶调优目标往往是：

- 控制尾延迟
- 缩小语义昂贵路径的出现频率
- 让提交、回写、目录操作和缓存行为更可解释

这和“跑出更漂亮的单次 benchmark”不是同一件事。

### 18.15 什么时候该停下 benchmark，转向协议重设计

如果你已经确认下面这些现象反复出现，继续刷 benchmark 数字通常收益不大：

- 每次瓶颈都落在 `fsync` / flush，而应用却坚持超高频同步点
- 元数据风暴来自目录结构设计，而不是文件系统实现 bug
- overlayfs copy-up 成本来自部署形态，而不是某个神秘参数
- 远端存储问题本质是可见性和锁语义错位，而不是吞吐不足

这时更有效的动作常是：

- 重新设计提交批次
- 重新组织目录与文件布局
- 把状态同步从 `mtime`/目录扫描切到显式 manifest/WAL
- 把共享写工作负载从语义较弱的远端挂载迁回本地或专门的存储协议

---

### 18.16 `fio` 完整使用指南：从工作负载到可信结论

`fio`（Flexible I/O Tester）是文件系统性能测试的标准工具：

```bash
# 安装
apt install fio     # Debian/Ubuntu
yum install fio     # CentOS/RHEL

# 基础用法：4KB 随机读，测试 page cache 热/冷差异
fio --name=randread_hot \
    --ioengine=libaio \     # 异步 I/O 引擎（模拟数据库 I/O 模式）
    --iodepth=32 \          # 队列深度 32（并发未完成的 I/O 数）
    --rw=randread \         # 随机读
    --bs=4k \               # 块大小 4KB
    --size=4G \             # 测试文件大小 4GB
    --numjobs=4 \           # 4 个并发工作进程
    --runtime=30 \          # 运行 30 秒
    --time_based \          # 按时间而非完成量停止
    --filename=/tmp/fio_test \
    --output-format=json \  # JSON 格式输出（便于解析）
    --output=/tmp/fio_hot.json \
    --group_reporting

# 先清 page cache，测冷读
sync && echo 3 > /proc/sys/vm/drop_caches
fio --name=randread_cold \
    --ioengine=libaio --iodepth=32 --rw=randread --bs=4k \
    --size=4G --numjobs=4 --runtime=30 --time_based \
    --filename=/tmp/fio_test \
    --output-format=json --output=/tmp/fio_cold.json \
    --group_reporting

# 解析结果：比较热/冷 IOPS 和延迟
python3 - <<'EOF'
import json

for label, path in [("HOT", "/tmp/fio_hot.json"), ("COLD", "/tmp/fio_cold.json")]:
    with open(path) as f:
        data = json.load(f)
    job = data['jobs'][0]
    read = job['read']
    print(f"\n{label} cache:")
    print(f"  IOPS:    {read['iops']:,.0f}")
    print(f"  BW:      {read['bw_bytes']/1024/1024:.1f} MB/s")
    print(f"  lat avg: {read['lat_ns']['mean']/1000:.1f} μs")
    print(f"  lat p99: {read['clat_ns']['percentile']['99.000000']/1000:.1f} μs")
    print(f"  lat p99.9:{read['clat_ns']['percentile']['99.900000']/1000:.1f} μs")
EOF
# 典型 NVMe SSD 结果：
# HOT cache:
#   IOPS:    2,456,789     ← page cache 命中，极高 IOPS
#   BW:      9,596.8 MB/s
#   lat avg: 52.3 μs
#   lat p99: 134.5 μs
#
# COLD cache:
#   IOPS:    456,789       ← 直接读设备，受限于 NVMe 延迟
#   BW:      1,784.0 MB/s
#   lat avg: 278.4 μs
#   lat p99: 512.6 μs
```

**关键工作负载的标准 fio 配方**：

```ini
# 配方 1: 数据库 OLTP 模拟（随机读写混合，含 fsync）
# 文件：db_oltp.fio
[global]
ioengine=libaio
iodepth=32
bs=4k
direct=1          # O_DIRECT（数据库自管 buffer pool）
size=10G
numjobs=8
runtime=120
time_based
group_reporting
filename=/tmp/fio_db_test

[read]
rw=randread
rate_iops=5000    # 限制读 IOPS，模拟混合比例

[write]
rw=randwrite
rate_iops=1000    # 限制写 IOPS
fsync=1           # 每次写后 fsync（数据库事务提交场景）
```

```bash
# 配方 2: 日志写入（顺序追加写，含 fdatasync）
fio --name=log_write \
    --ioengine=sync \
    --rw=write \
    --bs=128k \          # 日志通常大块顺序写
    --size=20G \
    --numjobs=1 \        # 单线程（WAL 通常单写线程）
    --fdatasync=1 \      # 每次写后 fdatasync
    --filename=/var/log/fio_log_test \
    --group_reporting

# 配方 3: 元数据密集（海量小文件创建/删除）
fio --name=metadata_storm \
    --ioengine=sync \
    --rw=write \
    --bs=4k \
    --size=4k \          # 每个文件只有 4KB
    --numjobs=16 \
    --nrfiles=10000 \    # 每个 job 创建 10000 个文件
    --openfiles=100 \    # 同时打开 100 个
    --directory=/tmp/fio_small_files/ \
    --group_reporting

# 配方 4: overlayfs copy-up 热点模拟
# 在 overlayfs 上进行，每次写都触发 copy-up（文件来自 lower 层）
mount -t overlay overlay \
    -o lowerdir=/tmp/lower,upperdir=/tmp/upper,workdir=/tmp/work \
    /tmp/overlay_test

# 先在 lower 层创建大文件
dd if=/dev/zero of=/tmp/lower/testfile bs=1M count=1024

fio --name=copyup_bench \
    --ioengine=sync --rw=write --bs=4k \
    --size=1G --numjobs=1 \
    --filename=/tmp/overlay_test/testfile \
    --group_reporting
# 第一次 write → copy-up 触发 → 延迟极高
# 之后写入直接到 upper → 正常延迟
```

---

### 18.17 `perf` 文件系统性能分析：从火焰图到函数级延迟

```bash
# 1. 采集文件系统操作的 CPU 热点火焰图
# 目标：找到哪个内核函数消耗最多 CPU

# 运行工作负载时同时采集
perf record -F 999 -a -g --call-graph=dwarf \
    --filter 'ip > 0xffffffff80000000' \  # 只采集内核栈（可选）
    -- fio --name=test --ioengine=sync --rw=randwrite --bs=4k \
            --size=1G --runtime=30 --time_based --filename=/tmp/perf_test

# 生成 SVG 火焰图
perf script | stackcollapse-perf.pl | flamegraph.pl > fs_flame.svg
# 打开 fs_flame.svg，点击各函数块查看详情

# 常见热点函数含义：
# ext4_da_write_begin      ← 写操作的延迟分配开始（元数据准备）
# jbd2_journal_dirty_metadata ← journal 元数据标记（事务相关）
# __d_lookup_rcu           ← dcache 查找（路径解析热点）
# shrink_slab              ← slab 内存回收（内存压力时出现）
# writeback_inodes_wb      ← 脏页回写（writeback 线程）

# 2. perf stat 精确统计特定工作负载的事件
perf stat -e \
    cache-references,cache-misses,\
    L1-dcache-loads,L1-dcache-load-misses,\
    LLC-loads,LLC-load-misses,\
    page-faults,major-faults,\
    context-switches \
    -- fio --name=test --ioengine=sync --rw=randread \
            --bs=4k --size=4G --runtime=30 --time_based --filename=/tmp/perf_test

# 输出示例：
#  12,345,678      cache-misses              #    8.92 % of all cache refs
#      45,678      page-faults
#         123      major-faults              # ← 几乎为 0 说明全是 page cache 命中
#      12,345      context-switches

# 3. 追踪特定函数的调用延迟（perf probe）
# 为 ext4_sync_file 加入动态探针
perf probe --add 'ext4_sync_file entry'
perf probe --add 'ext4_sync_file%return latency=\$retval'

# 采集
perf record -e 'probe:ext4_sync_file*' -a -- sleep 30

# 分析
perf script | awk '/ext4_sync_file return/ {print $NF}' | \
    sort -n | awk 'NR%10==0' | head -20
# → 打印 fsync 的延迟分布（每 10 个打一个）

# 4. 用 perf-trace 分析系统调用层面的延迟（轻量版 strace）
perf trace --call-graph=dwarf -e 'read,write,fsync,fdatasync' \
    -p $(pgrep mysqld) -- sleep 30 2>&1 | \
    awk '/fsync/{split($0,a,/[()]/); if(a[2]+0 > 10) print}' | \
    sort -t'<' -k2 -rn | head -20
# → 找出超过 10ms 的 fsync 调用及其内核调用栈

# 5. 用 BPF 精确分析每个 I/O 操作的延迟分布
bpftrace -e '
kprobe:ext4_file_write_iter { @start[tid] = nsecs; }
kretprobe:ext4_file_write_iter /@start[tid]/ {
    $lat_us = (nsecs - @start[tid]) / 1000;
    @write_us = hist($lat_us);
    if ($lat_us > 50000) {  // 超过 50ms 的写打印堆栈
        printf("SLOW WRITE: %s(%d) lat=%d us\n", comm, pid, $lat_us);
        print(kstack);
    }
    delete(@start[tid]);
}
interval:s:30 { print(@write_us); }
' -- sleep 60
```

---

### 18.18 基准测试协议模板

设计可信 benchmark 的标准协议：

```bash
#!/bin/bash
# 文件系统性能测试标准协议模板

DEVICE="/dev/nvme0n1"
FS_TYPE="ext4"
MOUNT_POINT="/mnt/benchmark"
TEST_DIR="$MOUNT_POINT/fio_test"
RESULTS_DIR="/tmp/benchmark_results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"

# === 环境记录（必须！）===
{
    echo "=== System Info ==="
    uname -r
    lscpu | grep "Model name"
    free -h
    echo ""

    echo "=== Storage Info ==="
    lsblk -o NAME,TYPE,SIZE,ROTA,SCHED,MODEL "$DEVICE"
    cat /sys/block/$(basename $DEVICE)/queue/scheduler
    echo ""

    echo "=== Filesystem Info ==="
    tune2fs -l "$DEVICE" 2>/dev/null | grep -E "Block size|Filesystem features|Mount options"
    mount | grep "$MOUNT_POINT"
    echo ""

    echo "=== Kernel Params ==="
    sysctl vm.dirty_ratio vm.dirty_background_ratio
    echo ""
} > "$RESULTS_DIR/environment.txt"

# === 准备阶段 ===
# 挂载文件系统
mkfs.ext4 -q "$DEVICE"
mount "$DEVICE" "$MOUNT_POINT"
mkdir -p "$TEST_DIR"

# 预分配测试文件（排除文件创建时间）
fio --name=prepare --ioengine=libaio --rw=write --bs=1M \
    --size=20G --numjobs=1 --iodepth=16 \
    --filename="$TEST_DIR/testfile" > /dev/null

# === 测试 1: 顺序读（冷缓存 vs 热缓存）===
echo "Test 1: Sequential Read"

# 冷缓存
sync && echo 3 > /proc/sys/vm/drop_caches
fio --name=seq_read_cold --ioengine=libaio --rw=read --bs=1M \
    --size=10G --numjobs=1 --iodepth=16 --runtime=60 --time_based \
    --filename="$TEST_DIR/testfile" \
    --output-format=json --output="$RESULTS_DIR/seq_read_cold.json"

# 热缓存（不清 cache）
fio --name=seq_read_hot --ioengine=libaio --rw=read --bs=1M \
    --size=10G --numjobs=1 --iodepth=16 --runtime=60 --time_based \
    --filename="$TEST_DIR/testfile" \
    --output-format=json --output="$RESULTS_DIR/seq_read_hot.json"

# === 测试 2: 随机读写（IOPS 测试）===
echo "Test 2: Random 4K Read/Write"

sync && echo 3 > /proc/sys/vm/drop_caches
fio --name=rand_rw --ioengine=libaio --rw=randrw --rwmixread=70 --bs=4k \
    --size=10G --numjobs=8 --iodepth=64 --runtime=60 --time_based \
    --direct=1 \        # 绕过 page cache（纯设备 IOPS）
    --filename="$TEST_DIR/testfile" \
    --output-format=json --output="$RESULTS_DIR/rand_rw.json" \
    --group_reporting

# === 测试 3: 同步写入（持久化延迟）===
echo "Test 3: Sync Write (fsync latency)"

fio --name=sync_write --ioengine=sync --rw=randwrite --bs=4k \
    --size=4G --numjobs=1 --fsync=1 \
    --filename="$TEST_DIR/testfile" \
    --output-format=json --output="$RESULTS_DIR/sync_write.json"

# === 测试 4: 元数据操作（小文件风暴）===
echo "Test 4: Metadata Storm (small files)"

mkdir -p "$TEST_DIR/small_files"
fio --name=small_files --ioengine=sync --rw=write --bs=4k \
    --size=4k --numjobs=16 --nrfiles=5000 \
    --directory="$TEST_DIR/small_files" \
    --output-format=json --output="$RESULTS_DIR/metadata.json" \
    --group_reporting

# === 收集证据（与 benchmark 同期）===
# 启动背景监控
(while true; do
    date "+%H:%M:%S" >> "$RESULTS_DIR/iostat.txt"
    iostat -xz 1 1 "$DEVICE" >> "$RESULTS_DIR/iostat.txt"
    grep -E "Dirty|Writeback|Cached" /proc/meminfo >> "$RESULTS_DIR/meminfo.txt"
    sleep 5
done) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# === 结果汇总 ===
python3 - "$RESULTS_DIR" <<'PYEOF'
import json, sys, os, glob

results_dir = sys.argv[1]
print(f"\n{'='*60}")
print(f"Benchmark Results: {results_dir}")
print(f"{'='*60}")

for json_file in sorted(glob.glob(f"{results_dir}/*.json")):
    name = os.path.basename(json_file).replace('.json', '')
    try:
        with open(json_file) as f:
            data = json.load(f)
        job = data['jobs'][0]
        print(f"\n{name}:")
        for io_type in ['read', 'write']:
            d = job.get(io_type)
            if d and d.get('iops', 0) > 0:
                print(f"  {io_type}: IOPS={d['iops']:>8,.0f}  "
                      f"BW={d['bw_bytes']/1024/1024:>7.1f}MB/s  "
                      f"avg_lat={d['lat_ns']['mean']/1000:>7.1f}μs  "
                      f"p99_lat={d['clat_ns']['percentile']['99.000000']/1000:>8.1f}μs")
    except Exception as e:
        print(f"  Error parsing {name}: {e}")
PYEOF
```

---

## 本章小结

| 主题 | 结论 |
|------|------|
| 性能 | 必须和工作负载一起描述 |
| 指标 | 吞吐、IOPS、平均延迟、尾延迟都要看 |
| 元数据路径 | 经常比数据路径更容易成为热点 |
| page cache | 是 benchmark 最大的扭曲源之一 |
| 慢路径 | `fsync`、copy-up、远端同步会放大尾延迟 |
| 实验协议 | 必须先写假设、变量和证据链 |
| 调优 | 先拆层次，再改协议/结构/参数/介质 |
| 可预测性 | 常比峰值吞吐更接近真实生产目标 |

---

## 练习题

1. 设计一个 benchmark，用来区分“元数据瓶颈”和“设备吞吐瓶颈”。
2. 为什么平均延迟很漂亮时，线上用户仍可能感到系统很卡？
3. overlayfs copy-up 为什么特别容易出现在尾延迟里？
4. 为什么不带工作负载描述的 benchmark 数字几乎没用？
5. 如果一个系统 `write()` 很快但 `fsync()` 很慢，你会如何拆层分析？
6. 为什么 cgroup 和内存限制会让同一文件系统在容器里表现出另一种“性格”？
7. 什么情况下你应该停止“继续压 benchmark”，转而重做应用协议或目录布局？
