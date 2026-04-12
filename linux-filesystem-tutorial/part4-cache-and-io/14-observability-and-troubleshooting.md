# 第14章：观测与排障

> 高阶排障不是背更多命令，而是先知道自己到底在查名字层、对象层、缓存层、文件系统层，还是设备层；命令只是把这个模型落地的工具。

## 学习目标

完成本章后，你将能够：

1. 用分层模型定位路径、权限、缓存、writeback、设备和远端语义问题
2. 理解 `df`、`du`、`stat`、`findmnt`、`lsof`、`filefrag`、`vmstat`、`iostat`、`perf` 各自回答什么问题
3. 说明 deleted-but-open、缓存污染、元数据瓶颈、回写拥塞是如何留下观测痕迹的
4. 认识 `/proc`、`tracepoint`、`eBPF`、`perf` 在高阶定位中的价值
5. 建立从现象到层次、再到证据链的排障顺序

---

## 正文内容

### 14.1 第一步永远不是跑命令，而是判断层次

文件系统问题通常至少落在这几层之一：

- **名字层**：路径、软链接、挂载视图、namespace
- **对象层**：inode、链接数、open file、权限元数据
- **缓存层**：page cache、readahead、dirty pages、writeback
- **文件系统层**：目录索引、journal、extent、quota、copy-up
- **设备层**：队列深度、flush、介质错误、尾延迟
- **远端层**：属性缓存、close-to-open、网络抖动、锁服务

如果这一步不做，你几乎必然会在错误层上看一堆“正确但没用”的命令输出。

### 14.2 `df`、`du`、`stat`、`ls -li` 为什么经常吵架

它们看的不是同一件事：

- `df`：文件系统实例层剩余空间/ inode
- `du`：目录树可见路径累计占用
- `stat`：单个对象的元数据和文件系统实例信息
- `ls -li`：名字层 + inode 号 + 权限一瞥

所以：

- `df` 和 `du` 对不上，不一定谁错了
- deleted-but-open、快照、reflink、保留空间都可能制造“矛盾”
- `stat` 往往比 `ls -l` 更接近对象层真相

### 14.3 `findmnt` 解决的是视图问题，不是空间问题

很多“文件怎么没了”的问题，不是对象消失，而是路径视图变了。`findmnt` 的价值在于：

- 看到当前路径处在哪个挂载点下
- 理解 bind mount、overlay、volume mount 是否遮住了原内容
- 分辨容器/宿主机是不是在看同一路径视图

如果不先看挂载树，很多路径问题根本查不到对象层去。

### 14.4 `lsof +L1` 是 deleted-but-open 排障利器

当 `df` 很大、`du` 却找不到时，最常见原因之一是 deleted-but-open。`lsof +L1` 直接从 open file 关系切入，而不是从目录树切入，这正好击中这种问题的本质。

这也再次说明：路径树和对象生命周期是两套不同证据链。

### 14.5 `filefrag`、`fiemap`、`stat -f` 解决的是空间形态问题

如果问题不是“有没有空间”，而是“空间形态怎么了”，你需要的不是 `df`，而是：

- `filefrag -v`：看 extent 碎裂、区间分布
- `fiemap`：看逻辑到物理映射
- `stat -f`：看文件系统实例层信息

这些工具更接近布局和分配器层，而不是名字层。

### 14.6 `/proc/meminfo`、`vmstat`、`iostat` 让你看到缓存和设备在打什么架

- `/proc/meminfo` 里的 `Dirty`、`Writeback` 让你看到脏页和回写规模
- `vmstat` 让你看到阻塞、回收、回写节奏
- `iostat -xz` 让你看到设备利用率、队列、await、svctm 一类指标

如果应用“偶发性卡住”，却又不是路径或权限问题，下一步常常就是看这里。

### 14.7 `perf`、tracepoint、eBPF 解决的是“到底慢在哪条路径”

高阶性能问题常常不再是“磁盘快不快”，而是：

- 卡在路径解析？
- 卡在目录锁？
- 卡在 page fault？
- 卡在 writeback 提交？
- 卡在 ext4 journal commit？
- 卡在 FUSE 用户态往返？

这时 `perf`、tracepoint 和 eBPF 的价值在于：你终于能看到“慢的函数和事件”而不是猜。

### 14.8 一个系统化排障顺序

可以按这条顺序查：

1. 路径是否真的指向你以为的对象？
2. 对象元数据、权限、链接、打开引用是否合理？
3. page cache / writeback 是否在影响可见性和延迟？
4. 文件系统内部结构（目录索引、extent、quota、copy-up）是否出现问题？
5. 底层设备或远端服务是否拖慢或改变语义？

这个顺序的价值在于：它迫使你先建立证据链，再解释现象。

### 14.9 三类典型错判

1. **把缓存命中当成设备性能**
2. **把路径树可见性当成对象生命周期真相**
3. **把某层指标的正常，误当成所有层都正常**

例如：

- `iostat` 不忙，不等于路径解析不慢
- `df` 正常，不等于 deleted-but-open 不存在
- `ls` 可见，不等于持久化已经完成

### 14.10 观测不是目的，证据链才是目的

真正好的排障结论应该长这样：

```text
路径视图没问题
-> 对象还被某进程打开
-> dentry 和 inode 正常
-> page cache / writeback 正常
-> deleted-but-open 导致 df/du 对不上
```

而不是：

```text
我跑了 8 个命令，看着都差不多，猜测是磁盘问题
```

---

### 14.11 `/proc/diskstats` 与 `iostat` 字段精解

`iostat` 的数据来源于 `/proc/diskstats`，理解每个字段是正确分析的前提：

```bash
# /proc/diskstats 的原始格式（每行一个块设备）
cat /proc/diskstats
# 8       0 sda 12345 678 901234 56789 2345 678 345678 12345 0 34567 69134 0 0 0 0
# ↑主设备号 ↑次设备号 ↑设备名
#   字段1-11（从"12345"开始）：

# 字段解释（依次）：
# 1.  reads_completed:      完成的读请求次数
# 2.  reads_merged:         被合并的读请求数（内核合并相邻读 → 减少实际 I/O）
# 3.  sectors_read:         读取的扇区数（* 512 = 字节数）
# 4.  time_reading_ms:      读操作累计耗时（毫秒）
# 5.  writes_completed:     完成的写请求次数
# 6.  writes_merged:        被合并的写请求数
# 7.  sectors_written:      写入的扇区数
# 8.  time_writing_ms:      写操作累计耗时（毫秒）
# 9.  io_in_progress:       当前进行中的 I/O 数（瞬时值）
# 10. time_io_ms:           有 I/O 进行时的累计时间（毫秒）— 设备忙时间
# 11. weighted_time_io_ms:  加权 I/O 时间（= 各请求等待时间之和）— iostat 计算 await 用

# iostat -xz 输出字段解释
iostat -xz 1 3 /dev/sda
# Device  r/s    rKB/s  rrqm/s  %rrqm  r_await  rareq-sz  w/s  wKB/s  wrqm/s  %wrqm  w_await  wareq-sz  aqu-sz  %util

# r/s:       每秒完成的读请求数（= reads_completed 差值 / 时间间隔）
# rKB/s:     每秒读取的 KB 数
# rrqm/s:    每秒被合并的读请求数（内核合并减少实际 I/O）
# %rrqm:     读合并率（高 → 顺序读，内核在帮你合并）
# r_await:   读请求平均等待时间（ms）= time_reading_ms / reads_completed
#            包含：设备队列等待时间 + 实际服务时间
# rareq-sz:  平均读请求大小（KB）= rKB/s / r/s
# w/s w...:  对应写的各项指标
# aqu-sz:    设备队列平均长度（= weighted_time_io_ms / 1000 / 时间间隔）
#            < 1 → 设备基本不忙; > 32 → 设备严重排队
# %util:     设备忙碌时间占比（= time_io_ms 差值 / 时间间隔 / 10）
#            注意：%util=100% 不一定代表瓶颈（NVMe 并行能力很强，100% 仍可接受更多 I/O）

# 诊断示例：
# 1. aqu-sz 高 + r_await 高 + %util 低 → 软件层面的等待（如 journal 锁）
# 2. aqu-sz 高 + r_await 高 + %util ≈ 100% → 设备饱和
# 3. rrqm/s 高 → 顺序读，内核合并良好
# 4. w_await 远高于 r_await → 写路径有问题（可能 writeback 拥塞）
```

**`/proc/diskstats` 自定义分析脚本**：

```bash
# 实时计算设备延迟分布（无需额外工具）
python3 - <<'EOF'
import time, re

def read_diskstats():
    result = {}
    with open('/proc/diskstats') as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 14 and fields[2] in ('sda', 'nvme0n1', 'vda'):
                dev = fields[2]
                result[dev] = {
                    'reads': int(fields[3]),
                    'read_ms': int(fields[6]),
                    'writes': int(fields[7]),
                    'write_ms': int(fields[10]),
                    'io_ms': int(fields[12]),
                    'weighted_ms': int(fields[13]),
                }
    return result

prev = read_diskstats()
time.sleep(1)

for _ in range(10):
    curr = read_diskstats()
    for dev in curr:
        if dev not in prev:
            continue
        p, c = prev[dev], curr[dev]
        reads = c['reads'] - p['reads']
        writes = c['writes'] - p['writes']
        r_await = (c['read_ms'] - p['read_ms']) / max(reads, 1)
        w_await = (c['write_ms'] - p['write_ms']) / max(writes, 1)
        util = (c['io_ms'] - p['io_ms']) / 10  # 百分比
        aqu = (c['weighted_ms'] - p['weighted_ms']) / 1000

        print(f"{dev}: r/s={reads:5d}  r_await={r_await:6.1f}ms  "
              f"w/s={writes:5d}  w_await={w_await:6.1f}ms  "
              f"aqu={aqu:5.1f}  util={util:5.1f}%")
    prev = curr
    time.sleep(1)
EOF
```

---

### 14.12 `blktrace` / `blkparse` 深度观察块 I/O

`blktrace` 捕获块层每个事件，是分析 I/O 路径和延迟的最底层工具：

```bash
# 捕获 /dev/sda 的块 I/O 事件（30秒）
blktrace -d /dev/sda -w 30 -o /tmp/sda_trace

# 解析原始二进制 trace
blkparse /tmp/sda_trace.blktrace.* -o /tmp/sda_parsed.txt

# 关键事件类型（blkparse 输出中的字母）：
# Q  = queued（I/O 请求进入队列）
# G  = get request（从 request pool 分配 request）
# I  = inserted（插入设备队列）
# D  = issued（下发给驱动，真正开始 I/O）
# C  = completed（驱动报告完成）
# M  = merged（请求被合并到已有请求）
# P  = plug（调度器插入软屏障）
# U  = unplug（释放软屏障，批量下发）
# S  = sleep（没有请求了，调度器休眠）
# A  = remap（请求被重映射，如 dm 层）
# X  = split（大请求被拆分为小请求）
# F  = flush（FLUSH cache 命令）

# 典型输出：
cat /tmp/sda_parsed.txt | head -20
# 8,0    3   1     0.000000000 12345  Q   W 2097152+8 [ext4-io-0]
# ↑major ↑minor ↑序号 ↑时间戳(纳秒) ↑进程PID ↑事件 ↑I/O类型 ↑起始扇区+长度 ↑进程名
#
# I/O 类型：R=读, W=写, D=丢弃(discard), F=flush, S=同步
# 加后缀 S 表示同步（如 WS = 同步写）, N 表示不重要

# 分析 Q→C 延迟（请求入队到完成的全链路延迟）
blkparse /tmp/sda_trace.blktrace.* -d /tmp/sda.bin
btt -i /tmp/sda.bin
# ==================== All Devices ====================
# ALL       MIN      AVG      MAX    N
# Q2Q       0.000001 0.001234 0.234567 12345   ← 相邻请求间隔
# Q2D       0.000001 0.000567 0.123456 12345   ← 队列到下发（调度器延迟）
# D2C       0.000010 0.000789 0.156789 12345   ← 下发到完成（设备延迟）
# Q2C       0.000011 0.001356 0.280245 12345   ← 全链路延迟

# 分析写合并率
blkparse /tmp/sda_trace.blktrace.* | grep " M " | wc -l   # 合并的请求数
blkparse /tmp/sda_trace.blktrace.* | grep " D " | wc -l   # 实际下发数
# 合并率 = 合并数 / (合并数 + 下发数)

# 实时 blktrace（更方便的 btop 方式）
# 用 bpftrace 替代（不需要写文件）：
bpftrace -e '
tracepoint:block:block_rq_issue {
    @io_type[args->rwbs] = count();
    @size_hist = hist(args->bytes);
}
tracepoint:block:block_rq_complete {
    @latency_us = hist((nsecs - @start[args->sector]) / 1000);
    delete(@start[args->sector]);
}
tracepoint:block:block_rq_insert {
    @start[args->sector] = nsecs;
}
interval:s:10 { print(@io_type); print(@size_hist); print(@latency_us); clear(@); }'
```

---

### 14.13 eBPF / bpftrace 文件系统追踪工具集

**BCC 工具集（需要安装 bcc-tools）**：

```bash
# 1. opensnoop：追踪所有 open() 调用
opensnoop -p $(pgrep postgres)
# PID    COMM     FD ERR PATH
# 1234   postgres  3   0 /var/lib/postgresql/data/base/16384/1259
# 1234   postgres  4   0 /var/lib/postgresql/data/pg_wal/000000010000000000000001

# 2. filelife：追踪文件的创建和删除，测量文件生命周期
filelife
# TIME     PID    COMM        AGE(s) FILE
# 12:34:56 1234   nginx       0.23   /tmp/fastcgi.sock
# 12:34:57 5678   python3     1.45   /tmp/tmpXXXXXX

# 3. filetop：类似 top，按文件 I/O 实时排序
filetop -C 30   # 每 30 秒刷新
# TID    COMM             READS  WRITES R_Kb    W_Kb    T FILE
# 1234   postgres         120    45     480     360     R pg_wal/000001
# 5678   mysqld           89     234    356     1872    R ibdata1

# 4. ext4slower：追踪 ext4 慢操作（超过阈值才记录）
ext4slower 10   # 超过 10ms 的操作
# TIME     COMM    PID   T BYTES   OFF_KB  LAT(ms) FILENAME
# 12:34:56 python3 1234  R 4096    0       15.234  data.csv
# 12:34:57 mysqld  5678  W 16384   4096    23.456  ibdata1

# 5. cachestat：page cache 命中率
cachestat 1
# HITS   MISSES  DIRTIES  HITRATIO  BUFFERS_MB  CACHED_MB
# 4096   128     64       96.97%    128         4096

# 6. cachetop：按进程的 page cache 统计
cachetop 5   # 每 5 秒刷新
# 12:34:56 Buffers MB: 128 / Cached MB: 4096 / Sort: HITS / Order: descending
# PID      UID      CMD              HITS     MISSES   DIRTIES  READ_HIT%  WRITE_HIT%
# 1234     postgres postgres        2048     16       32       99.2%      99.8%

# 7. biotop：按进程的块 I/O 统计
biotop -C 30
# PID    COMM             D MAJ MIN DISK       I/O  Kbytes  AVGms
# 1234   mysqld           W   8   1 sda        234  45678   2.34
# 5678   postgres         R   8   1 sda        123  12345   1.23
```

**自定义 bpftrace 脚本**：

```bash
# 脚本 1: 追踪文件系统延迟热点（VFS 层）
bpftrace - <<'EOF'
kprobe:vfs_read {
    @start_read[tid] = nsecs;
    @files_read[str(((struct file *)arg0)->f_path.dentry->d_name.name)] = count();
}
kretprobe:vfs_read / @start_read[tid] / {
    @read_lat_us = hist((nsecs - @start_read[tid]) / 1000);
    delete(@start_read[tid]);
}
kprobe:vfs_write {
    @start_write[tid] = nsecs;
}
kretprobe:vfs_write / @start_write[tid] / {
    @write_lat_us = hist((nsecs - @start_write[tid]) / 1000);
    delete(@start_write[tid]);
}
interval:s:10 {
    printf("\n=== Top files read ===\n"); print(@files_read, 10);
    printf("\n=== Read latency ===\n");  print(@read_lat_us);
    printf("\n=== Write latency ===\n"); print(@write_lat_us);
    clear(@files_read); clear(@read_lat_us); clear(@write_lat_us);
}
EOF

# 脚本 2: 追踪 page fault 的来源（哪些文件导致 major fault）
bpftrace - <<'EOF'
kprobe:do_read_fault {
    $vmf = (struct vm_fault *)arg0;
    $vma = $vmf->vma;
    if ($vma->vm_file) {
        @major_faults[
            comm,
            str($vma->vm_file->f_path.dentry->d_name.name)
        ] = count();
    }
}
interval:s:5 {
    print(@major_faults);
    clear(@major_faults);
}
EOF

# 脚本 3: 追踪 fsync/fdatasync 延迟（哪些进程在等持久化）
bpftrace - <<'EOF'
tracepoint:syscalls:sys_enter_fsync,
tracepoint:syscalls:sys_enter_fdatasync {
    @start[tid] = nsecs;
    @calls[comm] = count();
}
tracepoint:syscalls:sys_exit_fsync,
tracepoint:syscalls:sys_exit_fdatasync {
    if (@start[tid]) {
        $lat_ms = (nsecs - @start[tid]) / 1000000;
        @lat_hist[comm] = hist($lat_ms);
        if ($lat_ms > 100) {
            printf("SLOW FSYNC: comm=%s pid=%d lat=%dms\n", comm, pid, $lat_ms);
        }
        delete(@start[tid]);
    }
}
interval:s:30 {
    printf("=== fsync call counts ===\n"); print(@calls);
    printf("=== fsync latency (ms) ===\n"); print(@lat_hist);
}
EOF

# 脚本 4: 追踪 dentry 缓存失效（path walk 性能退化指示器）
bpftrace - <<'EOF'
kprobe:d_invalidate {
    @invalidations[comm] = count();
    @inval_paths[str(((struct dentry *)arg0)->d_name.name)] = count();
}
kprobe:unlazy_walk {
    @rcu_fallbacks[comm] = count();  // RCU path walk 回退到 ref-walk
}
interval:s:10 {
    print(@invalidations); print(@inval_paths); print(@rcu_fallbacks);
    clear(@invalidations); clear(@inval_paths); clear(@rcu_fallbacks);
}
EOF
```

---

### 14.14 deleted-but-open 的系统化检测

`df` 空间满但 `du` 找不到文件的经典场景，需要从 open file 角度切入：

```bash
# 方法 1: lsof +L1（最直接）
lsof +L1
# COMMAND  PID  USER  FD  TYPE  DEVICE SIZE/OFF NLINK NODE NAME
# nginx   1234  www   12u  REG   8,1   52428800     0 12345 /var/log/nginx/access.log (deleted)
# ↑ NLINK=0 表示目录项已删除，但 fd 仍保持文件打开
# ↑ 文件名后跟 "(deleted)"

# 方法 2: 通过 /proc/pid/fd 手动检查
for pid in /proc/[0-9]*/; do
    pid_num=$(basename $pid)
    for fd in $pid/fd/*; do
        if [ -L "$fd" ]; then
            target=$(readlink -f "$fd" 2>/dev/null)
            if [[ "$target" == *"(deleted)"* ]] || \
               [ "$(stat -L --format=%h "$fd" 2>/dev/null)" = "0" ]; then
                size=$(stat -L --format=%s "$fd" 2>/dev/null || echo 0)
                comm=$(cat "$pid/comm" 2>/dev/null || echo "unknown")
                echo "PID=$pid_num COMM=$comm FD=$(basename $fd) SIZE=$size TARGET=$target"
            fi
        fi
    done
done 2>/dev/null | sort -t= -k5 -rn | head -20  # 按大小降序排列

# 方法 3: 通过 /proc/pid/fdinfo 获取 deleted 文件的精确大小
python3 - <<'EOF'
import os, glob

total_deleted = 0
results = []

for pid_dir in glob.glob('/proc/[0-9]*/fd'):
    pid = pid_dir.split('/')[2]
    try:
        comm = open(f'/proc/{pid}/comm').read().strip()
    except:
        continue

    for fd_path in glob.glob(f'{pid_dir}/*'):
        try:
            target = os.readlink(fd_path)
            if '(deleted)' in target:
                fd = fd_path.split('/')[-1]
                stat = os.stat(fd_path)  # stat 通过 fd 获取实际大小
                size = stat.st_size
                total_deleted += size
                results.append((size, pid, comm, fd, target))
        except (PermissionError, FileNotFoundError):
            continue

results.sort(reverse=True)
for size, pid, comm, fd, target in results[:20]:
    print(f"{size/1024/1024:8.1f} MB  pid={pid:6s}  comm={comm:20s}  fd={fd:4s}  {target}")

print(f"\nTotal deleted-but-open: {total_deleted/1024/1024/1024:.2f} GB")
EOF

# 方法 4: 如何释放空间（不杀进程）
# 对于日志文件，可以 truncate 而不删除（保持 fd 有效但清空内容）
# 找到对应 fd：
PID=1234; FD=12   # 从 lsof +L1 获取
# 清空文件（通过 /proc/pid/fd/N 访问，不需要原路径）
> /proc/$PID/fd/$FD   # 等价于 truncate -s 0

# 方法 5: eBPF 监控文件删除但仍有 fd 的情况
bpftrace -e '
kprobe:vfs_unlink {
    $dentry = (struct dentry *)arg1;
    $inode = $dentry->d_inode;
    if ($inode->i_nlink == 1) {  // 删除后将变为 0（有 open 的 fd 时不释放）
        printf("UNLINK with open fds: %s (nlink=%d, fds=%d)\n",
               str($dentry->d_name.name),
               (int32)$inode->i_nlink,
               (int32)$inode->__i_nlink);
    }
}'
```

---

### 14.15 `perf` 文件系统性能分析

`perf` 能精准定位文件系统路径上的 CPU 热点：

```bash
# 1. 火焰图分析文件系统 CPU 热点
# 采样 30 秒，记录所有内核栈
perf record -F 99 -a -g --call-graph dwarf -e cpu-clock -- sleep 30

# 生成火焰图（需要 FlameGraph 工具）
perf script | stackcollapse-perf.pl | \
    grep -E "ext4|jbd2|vfs|fs_|path|dentry|inode" | \
    flamegraph.pl --title "Filesystem CPU hotspot" > fs_flame.svg

# 2. perf stat 统计文件系统相关 PMU 事件
perf stat -e \
    ext4:ext4_da_write_begin,\
    ext4:ext4_da_write_end,\
    ext4:ext4_writepages,\
    ext4:ext4_sync_file_enter,\
    jbd2:jbd2_commit_locking,\
    jbd2:jbd2_commit_flushing,\
    jbd2:jbd2_start_commit \
    -p $(pgrep mysqld) -- sleep 30

# 3. perf ftrace：函数级延迟追踪（无需 bpftrace）
# 追踪 ext4_sync_file 的调用和延迟
perf ftrace -G 'ext4_sync_file' -p $(pgrep mysqld) -- sleep 10
# 输出示例：
# ext4_sync_file() {
#   filemap_write_and_wait() {
#     __writeback_single_inode() {
#       ...
#     }; /* 5.234 ms */
#   }; /* 5.456 ms */
#   jbd2_complete_transaction(); /* 1.234 ms */
# }; /* 6.890 ms */

# 4. perf trace（strace 的高性能替代）
# 追踪特定进程的文件系统系统调用
perf trace -e openat,read,write,fsync,fdatasync,close \
    -p $(pgrep nginx) -- sleep 10 2>&1 | \
    awk '{print $NF, $0}' | sort -rn | head -20  # 按耗时降序

# 5. tracepoint：观察内核 ext4 事件
# 列出所有可用的 ext4 tracepoint
ls /sys/kernel/debug/tracing/events/ext4/
# ext4_alloc_da_blocks  ext4_da_write_begin  ext4_punch_hole  ...

# 启用特定 tracepoint
echo 1 > /sys/kernel/debug/tracing/events/ext4/ext4_da_write_begin/enable
echo 1 > /sys/kernel/debug/tracing/events/jbd2/jbd2_commit_locking/enable
cat /sys/kernel/debug/tracing/trace | head -30
# <pid>-<tid> [cpu] flags timestamp: event_name: field1=val1 ...
# mysqld-1234 [003] .... 12345.678901: ext4_da_write_begin: dev 8,1 ino 1234 pos 0 len 4096 flags 0

# 6. 定位元数据瓶颈（目录操作慢）
# 使用 perf 追踪目录锁竞争
perf record -e 'ext4:*' -g -- ls /large-directory/ > /dev/null
perf report --stdio | grep -A5 "ext4_htree_next_level"
# 若 htree 函数占比高 → 目录 B 树在扫描大目录

# 验证：大目录 vs 小目录的 opendir/readdir 延迟
python3 -c "
import os, time

dirs = [('/tmp', 'small'), ('/usr/lib', 'medium'), ('/usr/lib/python3', 'large')]
for path, label in dirs:
    t0 = time.monotonic()
    try:
        entries = os.listdir(path)
        elapsed_ms = (time.monotonic() - t0) * 1000
        print(f'{label:8s}: {len(entries):6d} entries  {elapsed_ms:.2f}ms')
    except:
        pass
"
```

---

### 14.16 `dmesg` 文件系统错误模式识别

内核在文件系统层面遇到问题时会输出特征性的 `dmesg` 消息，识别这些模式能快速定位问题：

```bash
# 过滤文件系统相关的 dmesg 消息
dmesg -T | grep -E "EXT4|JBD2|XFS|BTRFS|NFS|EIO|ESTALE|filesystem|quota|inode"

# 常见错误模式及其含义：

# 模式 1: ext4 日志错误 → 磁盘故障或文件系统损坏
# EXT4-fs error (device sda1): ext4_find_entry:1455: inode #1234: comm mysqld: reading directory lblock 0
# → 读取目录块失败（EIO），通常是磁盘故障

# 模式 2: JBD2 abort → 日志层遇到不可恢复错误
# JBD2: Detected aborted journal
# JBD2: Error -5 detected when updating journal superblock for sda1-8.
# → -5 = EIO，磁盘写入失败导致 journal 中止

# 模式 3: ext4 remount read-only → 遇到严重错误自动降级保护
# EXT4-fs (sda1): ext4_abort called.
# EXT4-fs error (device sda1): ext4_journal_check_start:57: Detected aborted journal
# EXT4-fs (sda1): Remounting filesystem read-only
# → errors=remount-ro 挂载选项触发，文件系统已只读

# 模式 4: SCSI/NVMe 错误 → 设备层问题
# blk_update_request: I/O error, dev sda, sector 12345678
# sd 0:0:0:0: [sda] tag#1 FAILED Result: hostbyte=DID_ERROR driverbyte=DRIVER_OK
# → 物理磁盘 I/O 失败，检查 S.M.A.R.T

# 模式 5: 配额超限
# EXT4-fs warning (device sda1): ext4_update_inode_size:5456: inode #1234: comm mysqld: quota exceeded

# 监控脚本：实时过滤文件系统告警
dmesg -w | grep --line-buffered -E \
    "EXT4-fs error|JBD2.*Error|Remounting.*read-only|I/O error.*dev|SCSI error|quota exceed" | \
while read line; do
    echo "$(date '+%Y-%m-%d %T') ALERT: $line"
    # 可以在这里触发告警（发 Slack/PagerDuty）
done &

# 检查磁盘健康
smartctl -a /dev/sda | grep -E "Reallocated|Pending|Offline|Command_Timeout"
# 5 Reallocated_Sector_Ct   0x0033   090   090   036    Pre-fail  Always       -       100
# 196 Reallocated_Event_Count 0x0032   090   090   000    Old_age   Always       -       100
# 197 Current_Pending_Sector  0x0022   100   100   000    Old_age   Always       -       0
# 198 Offline_Uncorrectable   0x0030   100   100   000    Old_age   Offline      -       0
# Reallocated_Sector_Ct > 0 → 有坏扇区被重映射，警惕磁盘即将故障

# 检查文件系统状态
dumpe2fs -h /dev/sda1 | grep -E "Filesystem state|Last mount|Last checked|Mount count|Maximum mount"
# Filesystem state:         clean         ← clean=正常, with errors=有错误
# Last mount time:          Mon Apr 12 10:23:45 2026
# Mount count:              47
# Maximum mount count:      -1           ← -1 = 不强制 fsck（推荐改为 100）
# Last checked:             Mon Jan  1 00:00:00 2024
```

---

### 14.17 综合排障案例：从现象到证据链

**案例 1: `df` 显示满，`du` 找不到原因**

```bash
# 步骤 1: 区分 block 空间和 inode 空间
df -h /    # 检查 block 空间
df -i /    # 检查 inode 空间
# 若 inode 用满（Use% = 100%）但 block 有剩余 → 小文件过多耗尽 inode

# 步骤 2: 检查 deleted-but-open
lsof +L1 | sort -k7 -rn | head -10  # 按文件大小排序

# 步骤 3: 检查快照 / btrfs 子卷占用
# btrfs: btrfs subvolume list / btrfs qgroup show
# lvm: lvs 查看所有 snapshot

# 步骤 4: 检查挂载点遮盖（路径下面有别的挂载）
findmnt --list | grep /mountpoint
# → bind mount 或 overlay 可能让 du 遍历了错误的层

# 步骤 5: 量化 page cache vs 实际已用
echo "Page cache:" $(cat /proc/meminfo | grep "^Cached:" | awk '{print $2/1024"MB"}')
echo "Dirty pages:" $(cat /proc/meminfo | grep "^Dirty:" | awk '{print $2"KB"}')
```

**案例 2: 写入延迟突然增大（偶发 100ms+ 写操作）**

```bash
# 步骤 1: 确认是写入路径还是 fsync 路径
strace -T -e trace=write,pwrite64,writev,fsync,fdatasync \
    -p $(pgrep your_app) 2>&1 | \
    awk '/fsync|fdatasync/{if($NF+0 > 0.1) print "SLOW:", $0}'
# → 若 fsync 慢 → 进入 journal commit 路径分析

# 步骤 2: 观察脏页积累
watch -n 0.5 'grep -E "Dirty|Writeback|nr_dirty" /proc/meminfo'
# → 若 Dirty 持续超过 dirty_ratio（默认 20% 内存）→ dirty throttling 在生效

# 步骤 3: 检查 journal commit 频率
perf stat -e jbd2:jbd2_start_commit,jbd2:jbd2_commit_locking -p PID -- sleep 10
# → commit_locking 时间高 → 多个线程竞争同一个 transaction

# 步骤 4: blktrace 定位到设备层
blktrace -d /dev/sda -w 10 -o /tmp/trace
btt -i /tmp/trace.bin
# → D2C 延迟高 → 设备问题；Q2D 延迟高 → 软件/调度器问题

# 步骤 5: 检查是否有 FLUSH 命令（每次 journal commit 都会发 FLUSH）
blkparse /tmp/trace.blktrace.* | grep " F " | wc -l  # FLUSH 次数
# → FLUSH 次数 ≈ journal commit 次数，每次 FLUSH 都会让队列排队
```

---

## 本章小结

| 主题 | 结论 |
|------|------|
| 排障起点 | 先判断层次，再选工具 |
| `df` / `du` / `stat` | 分别对应实例层、路径树层、对象层 |
| `findmnt` | 用于理解名字空间和挂载视图 |
| `lsof +L1` | 用于命中 deleted-but-open 生命周期问题 |
| `/proc` / `vmstat` / `iostat` | 用于观察缓存、回写、设备层冲突 |
| `perf` / eBPF | 用于回答“到底哪条路径慢” |

---

## 练习题

1. 为什么排障第一步应该是“判断层次”而不是“先跑命令”？
2. `df` 和 `du` 对不上时，最先该怀疑哪几类问题？
3. `findmnt` 为什么能解决很多看起来像“文件不见了”的问题？
4. 为什么 deleted-but-open 必须从 open file 关系而不是目录树切入？
5. 怎样才算一条完整的文件系统排障证据链？
