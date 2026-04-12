# 第16章：overlayfs 与容器镜像分层

> overlayfs 的难点不在“有上下层”，而在于如何让多个层拼出一个看似自然的 POSIX 视图，同时把 copy-up、whiteout、目录遮蔽、缓存和性能代价都藏在背后。

## 学习目标

完成本章后，你将能够：

1. 理解 lower/upper/work/merged 四层对象在联合视图中的职责
2. 解释 copy-up、whiteout、opaque dir、redirect_dir、metacopy 各自解决什么问题
3. 说明为什么 overlayfs 让“读取很快、第一次写突然变慢”成为常见现象
4. 理解 overlayfs 与 page cache、容器镜像层、volume 覆盖之间的关系
5. 用联合视图模型解释容器根文件系统中的路径行为与性能差异

---

## 正文内容

### 16.1 overlayfs 的目标不是“多层存储”，而是“多层视图”

容器镜像通常有大量共享只读层，运行中的容器只需要记录自己的修改。overlayfs 的核心价值是：

- 不预复制整棵根文件系统
- 把只读镜像层和可写层合成一个 merged 视图
- 让进程感觉自己在一个完整目录树上工作

所以 overlayfs 的本质是视图拼接，而不只是存储压缩。

### 16.2 lower、upper、work、merged 各自干什么

- **lower**：镜像只读层，可有多层堆叠
- **upper**：当前容器可写层
- **work**：内部协调和元数据转换需要的工作区
- **merged**：对进程可见的最终视图

进程大多只看到 merged，但真正的语义判断必须回到 lower/upper/work 的分工上。

### 16.3 copy-up 为什么是 overlayfs 的成本核心

如果一个文件原本只存在于 lower，当容器第一次修改它时，overlayfs 通常要：

1. 在 upper 创建对应对象
2. 复制必要元数据或内容
3. 让后续写入落到 upper

这就是 copy-up。它的意义是保持 lower 共享不被破坏，但代价包括：

- 第一次写入突然变慢
- 大量小修改会触发很多隐式复制
- 元数据操作也可能触发 copy-up，而不仅是大块数据写入

### 16.4 whiteout 和 opaque dir 解决的是遮蔽语义

如果 lower 中有 `foo`，而 upper 想让 merged 视图里看不到它，就不能简单“什么都不做”，否则 lower 仍会透出来。于是需要 whiteout。

如果整个目录要被上层完整遮蔽，就需要 opaque directory 一类语义。

这说明 overlayfs 不是“把两个目录合并显示”这么简单，而是要明确定义：

- 哪些下层对象继续可见
- 哪些对象被上层隐藏
- 哪些目录被上层整体替代

### 16.5 redirect_dir、metacopy、index 这些高级特性在补什么洞

在复杂目录操作和性能优化中，overlayfs 还会引入一些更细机制，例如：

- **redirect_dir**：帮助目录重命名/移动后的路径重定向
- **metacopy**：某些场景先只复制元数据，而不是立刻复制完整数据
- **index**：帮助更稳定地跟踪对象关系和避免某些别名/一致性问题

你不必把每个实现细节背成内核手册，但要知道：这些特性存在，恰恰说明“联合视图”背后的真实协调工作并不轻。

### 16.6 page cache 与 overlayfs 的交互为什么会让性能判断更难

读 merged 视图的文件时，真实数据可能来自 lower，也可能来自 upper。第一次写后又会切换到 upper。于是性能判断会受到：

- lower 所在文件系统特性
- upper 所在文件系统特性
- page cache 是否已命中原 lower 数据
- copy-up 是否刚发生
- volume mount 是否覆盖了某个子路径

所以“容器里读写这个路径很慢”并不只是一个 overlayfs 问题，而是联合视图、cache 和底层文件系统共同作用的结果。

### 16.7 为什么容器里的同一条路径可能在不同阶段语义不同

一个路径在容器启动早期、第一次写入后、volume mount 覆盖后、热更新后，可能分别处于：

- lower 透出
- upper copy-up 后接管
- bind mount 覆盖
- 被 whiteout 隐藏

这意味着“同一路径”的对象来源和性能特征可以随阶段变化。很多线上“偶发慢”“只第一次慢”“容器重启后恢复”的问题，都与这个阶段转换有关。

### 16.8 overlayfs 的边界在哪里

overlayfs 很适合：

- 镜像分层
- 启动快、共享多、修改相对局部的场景

但它不是天然适合：

- 极度频繁的小写和目录重构
- 严格依赖单层本地文件系统细微语义的工作负载
- 你根本不想承担 copy-up 和联合视图复杂性的场景

有时直接挂 volume、用本地文件系统、或在应用层分离热写数据会更合理。

---

### 16.9 overlayfs 内核数据结构与挂载参数

**overlayfs 的挂载语法**：

```bash
# 基础用法
mount -t overlay overlay \
    -o lowerdir=/lower1:/lower2:/lower3,upperdir=/upper,workdir=/work \
    /merged

# 参数说明：
# lowerdir: 冒号分隔的多层只读层（从右到左优先级降低，最左优先级最高）
# upperdir: 可写层（修改写入这里）
# workdir:  工作目录（必须与 upperdir 在同一文件系统，用于原子 copy-up）
# merged:   合并视图挂载点

# overlayfs 挂载选项（Linux 5.x）：
# redirect_dir=on/off/follow/nofollow  ← 目录重命名的跨层重定向
# index=on/off                          ← 索引层（避免 hardlink 和 rename 问题）
# metacopy=on/off                       ← 只复制元数据，推迟数据 copy-up
# xino=on/off                           ← 扩展 inode 号（多层时避免 inode 号冲突）
# nfs_export=on/off                     ← 允许 overlayfs 上做 NFS export

# Docker 的实际挂载方式（在宿主上观察）
docker run --rm -d --name demo alpine sleep 60
docker_pid=$(docker inspect --format='{{.State.Pid}}' demo)
# 查看容器的 overlayfs 挂载
cat /proc/$docker_pid/mountinfo | grep overlay
# 163 162 0:52 / / rw,relatime - overlay overlay \
#   rw,lowerdir=/var/lib/docker/overlay2/abc.../diff:\
#              /var/lib/docker/overlay2/def.../diff,\
#   upperdir=/var/lib/docker/overlay2/xyz.../diff,\
#   workdir=/var/lib/docker/overlay2/xyz.../work

# 查看 Docker image 各层
docker image inspect alpine | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('Layers:')
for i, layer in enumerate(data[0]['RootFS']['Layers']):
    print(f'  [{i}] {layer[:20]}...')
"

# 查看 overlay2 的物理布局
ls /var/lib/docker/overlay2/ | head -5
# 每个目录对应一个镜像层或容器层
ls /var/lib/docker/overlay2/abc.../
# diff/     ← 该层实际存储的文件增量（lowerdir 的内容）
# link      ← 短链接名（用于缩短 lowerdir 参数长度，避免超过内核限制）
# lower     ← 此层依赖的下层 link 列表
# work/     ← workdir（仅容器层有，镜像层无）
# merged/   ← merged 视图（仅容器层有）
docker stop demo
```

**overlayfs 核心内核结构**：

```c
/* fs/overlayfs/ovl_entry.h */

/* overlayfs 的 inode 私有数据 */
struct ovl_inode {
    /* 真实 inode 指针（指向 upper 或 lower 层的实际 inode）*/
    struct inode        *__upperdentry;   /* upper 层的 dentry（若已 copy-up）*/
    struct inode        *lower;           /* lower 层的 inode（用于读操作）*/
    /* lowerdata: 分离 metacopy 和 data 的情况 */
    struct inode        *lowerdata;

    /* inode 标志 */
    unsigned long        flags;
    /* OVL_F_UPPER_ALIAS: inode 有 upper 别名 */
    /* OVL_F_IMPURE:      目录内有 upper 文件（用于 rename 一致性）*/
    /* OVL_F_METACOPY:    只有元数据被 copy-up，数据仍在 lower */

    /* 引用计数保护 */
    struct mutex         lock;

    /* 旧 dentry（用于 rename 期间的原子性）*/
    const char          *redirect;       /* redirect_dir 的重定向路径 */

    /* 指向 overlay fs 超级块中的层信息 */
    u64                  version;        /* inode 版本（检测并发修改）*/
    unsigned int         nlink;          /* 虚拟链接数（跨层聚合）*/
};

/* 每个目录项的 overlay 状态 */
struct ovl_entry {
    struct ovl_path {
        struct vfsmount *layer;          /* 该文件所在层的挂载 */
        struct dentry   *dentry;         /* 该层中的实际 dentry */
    } lowerstack[];                      /* lower 层栈（可能有多层）*/
};

/* overlayfs 超级块私有数据 */
struct ovl_fs {
    struct vfsmount     *upper_mnt;      /* upper 层挂载 */
    unsigned int         numlower;       /* lower 层数量 */
    struct ovl_layer    *lower_layers;   /* lower 层数组（从上到下排列）*/
    struct ovl_layer     upper_layer;    /* upper 层 */
    struct dentry       *workdir;        /* work 目录 dentry */
    long                 namelen;        /* 最长文件名长度 */
    const struct ovl_config config;      /* 挂载时的配置选项 */
    /* redirect, index, metacopy, xino... */
};
```

---

### 16.10 copy-up 的内核实现路径

```
copy-up 触发条件：
  1. 写操作（write, truncate, setxattr, chmod, chown...）
  2. 目标文件/目录只在 lower 层存在（upper 中无对应项）
  3. → 内核必须先把文件"复制"到 upper，再进行修改

copy-up 执行路径（fs/overlayfs/copy_up.c）：

ovl_write_iter() 或 ovl_setattr()
  └─ ovl_copy_up(dentry)
        └─ ovl_copy_up_one(parent, dentry, flags)
              ├─ 1. 确保父目录也已 copy-up（递归）
              │      → 若 /a/b/c 要 copy-up，先确保 /a/b 已在 upper 中
              │
              ├─ 2. 在 upper 创建目标对象：
              │      文件:      ovl_copy_up_inode()
              │                   → vfs_create(upper_parent, name, mode)
              │                   → 复制 xattr（SELinux 标签等）
              │                   → 复制 ownership（uid/gid）
              │                   → 复制 timestamps（atime/mtime/ctime）
              │      目录:      vfs_mkdir(upper_parent, name, mode)
              │      符号链接:  vfs_symlink(upper_parent, name, target)
              │
              ├─ 3. 复制文件内容（仅普通文件，目录不需要）：
              │      ovl_copy_up_data()
              │        → ovl_do_copy_up_file_range()
              │            → vfs_copy_file_range(lower_file, 0,
              │                                  upper_file, 0, file_size)
              │            → 利用 copy_file_range() 系统调用（内核内 zero-copy）
              │            → 在支持 reflink 的文件系统上可以使用 reflink
              │              （Docker overlay2 + btrfs: copy-up 接近 O(1)！）
              │
              ├─ 4. 原子替换（利用 workdir）：
              │      → 先写到 work/copy_up_tmp_XXXXX（避免不完整的 copy-up 可见）
              │      → 完成后 rename 到 upper（原子操作）
              │      → 类似 WAL/原子写入协议
              │
              └─ 5. 更新 overlayfs 的 inode 映射：
                     → ovl_inode_set_upper(inode, upper_dentry)
                     → 后续操作直接走 upper 路径
```

**量化 copy-up 开销**：

```bash
# 测量 copy-up 延迟（在 overlayfs 上的第一次写）
python3 - <<'EOF'
import os, time, statistics, subprocess, tempfile

# 创建测试 overlayfs
lower = '/tmp/ovl_lower'
upper = '/tmp/ovl_upper'
work  = '/tmp/ovl_work'
merged = '/tmp/ovl_merged'

for d in [lower, upper, work, merged]:
    os.makedirs(d, exist_ok=True)

# 在 lower 中创建不同大小的测试文件
sizes = [4096, 64*1024, 1024*1024, 10*1024*1024]  # 4KB, 64KB, 1MB, 10MB
for size in sizes:
    path = f'{lower}/file_{size}'
    with open(path, 'wb') as f:
        f.write(b'x' * size)

# 挂载 overlayfs
subprocess.run(['mount', '-t', 'overlay', 'overlay',
    '-o', f'lowerdir={lower},upperdir={upper},workdir={work}',
    merged], check=True)

results = {}
for size in sizes:
    path = f'{merged}/file_{size}'

    # 确保 upper 中没有该文件（重置）
    upper_path = f'{upper}/file_{size}'
    if os.path.exists(upper_path):
        os.unlink(upper_path)

    # 测量 copy-up 时间（第一次写操作）
    trials = []
    for _ in range(5):
        # 重置 upper
        if os.path.exists(upper_path):
            os.unlink(upper_path)

        t0 = time.monotonic()
        with open(path, 'r+b') as f:
            f.write(b'y')    # 第一次写 → 触发 copy-up
            os.fsync(f.fileno())  # 确保 copy-up 完成
        elapsed_ms = (time.monotonic() - t0) * 1000
        trials.append(elapsed_ms)

    results[size] = trials

for size, times in results.items():
    print(f"文件大小={size//1024:6d}KB  copy-up: "
          f"avg={statistics.mean(times):6.1f}ms  "
          f"min={min(times):5.1f}ms  max={max(times):5.1f}ms")

subprocess.run(['umount', merged])
EOF

# 典型结果（ext4 upperdir，NVMe SSD）：
# 文件大小=     4KB  copy-up: avg=  0.8ms  min=  0.6ms  max=  1.2ms
# 文件大小=    64KB  copy-up: avg=  1.5ms  min=  1.2ms  max=  2.0ms
# 文件大小=  1024KB  copy-up: avg= 12.3ms  min= 11.8ms  max= 13.1ms
# 文件大小= 10240KB  copy-up: avg=108.7ms  min=105.2ms  max=112.3ms
# → copy-up 延迟与文件大小线性正相关
# → 大文件（如数据库文件）的第一次写特别慢
```

---

### 16.11 whiteout 与 opaque directory 的磁盘表示

```bash
# whiteout 是一个字符设备文件（主设备号=0, 次设备号=0）
# 当 merged 视图中删除 lower 层中的文件时，overlayfs 在 upper 中创建 whiteout

# 演示 whiteout 创建
mkdir -p /tmp/demo/{lower,upper,work,merged}
echo "original" > /tmp/demo/lower/file.txt

mount -t overlay overlay \
    -o lowerdir=/tmp/demo/lower,upperdir=/tmp/demo/upper,workdir=/tmp/demo/work \
    /tmp/demo/merged

# 在 merged 中删除文件
rm /tmp/demo/merged/file.txt

# 查看 upper 中的 whiteout
ls -la /tmp/demo/upper/
# c---------. 1 root root 0, 0 Apr 12 12:00 file.txt
#                              ↑ 字符设备，主次设备号均为 0 = whiteout

file /tmp/demo/upper/file.txt
# /tmp/demo/upper/file.txt: character special (0/0)

stat /tmp/demo/upper/file.txt
# File: /tmp/demo/upper/file.txt
# Size: 0               Blocks: 0          IO Block: 4096   character special file
# Device: 802h/2050d    Inode: 1234        Links: 1
# Device type: 0,0      ← 主次设备号均为 0，这是 whiteout 的标志

# opaque directory（目录被上层整体替代）
# 通过 xattr trusted.overlay.opaque=y 标记
mkdir /tmp/demo/upper/newdir
setfattr -n trusted.overlay.opaque -v y /tmp/demo/upper/newdir
# 现在 merged/newdir 不再透出 lower/newdir 的内容

getfattr -n trusted.overlay.opaque /tmp/demo/upper/newdir
# # file: /tmp/demo/upper/newdir
# trusted.overlay.opaque="y"

# redirect_dir xattr（记录目录重命名后的路径）
# 当目录从 lower 层移动到 upper 时
mv /tmp/demo/merged/lower_dir /tmp/demo/merged/upper_dir  # 重命名
getfattr -n trusted.overlay.redirect /tmp/demo/upper/upper_dir
# trusted.overlay.redirect="/lower_dir"   ← 记录原始路径

# 清理
umount /tmp/demo/merged
```

---

### 16.12 overlayfs 的 page cache 共享机制

overlayfs 的 page cache 有一个重要特性：**lower 层的 page cache 被合并视图和 lower 层共享**，不需要重复缓存：

```
page cache 共享模型：

lower 文件系统（如 ext4 上的镜像层）：
  /lower/file.txt → inode A → address_space A → page cache 页 [0..N]

overlayfs merged 视图：
  /merged/file.txt → ovl_inode（指向 lower inode A）
                   → 读操作转发到 lower inode A 的 address_space
                   → 命中同一批 page cache 页 [0..N]

效果：
  多个运行中的容器共享同一镜像层 → page cache 中只有一份文件数据
  → 100 个容器运行同一个 nginx 镜像 → /usr/sbin/nginx 的 page cache 只有一份

copy-up 后的 page cache 切换：
  write() 触发 copy-up → upper 层创建新文件
  → ovl_inode 的 upper_dentry 指向新文件
  → 后续读写操作切换到 upper 的 address_space（新的 page cache）
  → lower 层的旧 page cache 页逐渐被回收（无引用）
```

**观察 overlayfs 的 page cache 共享**：

```bash
# 实验：多个容器共享同一镜像层的 page cache
docker pull nginx:alpine

# 启动 3 个 nginx 容器
for i in 1 2 3; do
    docker run -d --name nginx_$i nginx:alpine
done

# 查看内存使用（PSS 考虑共享页的按比例分摊）
for pid in $(docker inspect --format='{{.State.Pid}}' nginx_1 nginx_2 nginx_3); do
    rss=$(awk '/Rss:/{sum+=$2}END{print sum}' /proc/$pid/smaps)
    pss=$(awk '/Pss:/{sum+=$2}END{print sum}' /proc/$pid/smaps)
    echo "pid=$pid  RSS=${rss}KB  PSS=${pss}KB  shared=$((rss-pss))KB"
done
# pid=1234  RSS=45678KB  PSS=12345KB  shared=33333KB
# pid=1235  RSS=45678KB  PSS=12345KB  shared=33333KB
# pid=1236  RSS=45678KB  PSS=12345KB  shared=33333KB
# → RSS 看起来三倍，但 PSS 揭示了大量页被共享（只计一份）

# 用 vmtouch 确认镜像层被缓存
vmtouch /var/lib/docker/overlay2/$(ls /var/lib/docker/overlay2/ | head -1)/diff/usr/sbin/nginx
# Files: 1
# Resident Pages: 512/512  2M/2M  100%    ← 完全在 page cache 中

docker stop nginx_1 nginx_2 nginx_3
docker rm nginx_1 nginx_2 nginx_3
```

---

### 16.13 overlayfs 性能分析与对比实验

**实验一：量化 overlayfs 相对于原生文件系统的写延迟**

```bash
# 在同一台机器上对比三种场景的写性能
# 场景 A：直接写 ext4（裸盘）
# 场景 B：写 overlayfs 的 upper layer（已 copy-up 的文件）
# 场景 C：写 overlayfs 中 lower 层的文件（首次写，触发 copy-up）

python3 - << 'EOF'
import os, time, statistics, subprocess, tempfile

def measure_write_latency(path, size=4096, n=500, trigger_copyup=False):
    """测量写延迟（μs），可选是否先触发 copy-up"""
    latencies = []
    for i in range(n):
        if trigger_copyup:
            # 每次写前先确保文件在 lower（删除 upper 中的版本）
            # 这在真实场景中不会发生，仅用于测量首次 copy-up
            pass

        t0 = time.monotonic_ns()
        with open(path, 'r+b') as f:
            f.write(b'\x00' * size)
            os.fsync(f.fileno())
        latencies.append((time.monotonic_ns() - t0) / 1000)

    return {
        'avg':  statistics.mean(latencies),
        'p50':  sorted(latencies)[n//2],
        'p99':  sorted(latencies)[int(n*0.99)],
        'max':  max(latencies),
    }

# 测试路径（需要预先设置）
scenarios = {
    'ext4 直接写':     '/tmp/ext4_direct/testfile',
    'overlayfs upper': '/tmp/ovl/merged/testfile_upper',  # 已在 upper 中
}

for label, path in scenarios.items():
    if os.path.exists(path):
        stats = measure_write_latency(path)
        print(f"\n{label}:")
        print(f"  avg={stats['avg']:.1f}μs  p50={stats['p50']:.1f}μs  "
              f"p99={stats['p99']:.1f}μs  max={stats['max']:.1f}μs")
    else:
        print(f"\n{label}: 路径不存在，跳过")

EOF
```

**实验二：overlayfs 层数对查找性能的影响**

```bash
# 多层 overlayfs 的查找开销（每一层都需要检查）
python3 - << 'EOF'
import os, subprocess, time, tempfile

def create_multilayer_overlay(num_layers, base_dir):
    """创建指定层数的 overlayfs（通过嵌套实现）"""
    layers = []
    for i in range(num_layers):
        layer_dir = os.path.join(base_dir, f"layer_{i}")
        os.makedirs(layer_dir, exist_ok=True)
        # 每层放一些文件
        for j in range(10):
            with open(os.path.join(layer_dir, f"file_{j}.txt"), 'w') as f:
                f.write(f"layer {i}, file {j}")
        layers.append(layer_dir)
    return layers

def benchmark_file_lookup(mount_point, n=10000):
    """测量文件查找速度"""
    target = os.path.join(mount_point, "file_9.txt")
    if not os.path.exists(target):
        target = os.path.join(mount_point, "file_0.txt")

    t0 = time.monotonic()
    for _ in range(n):
        try:
            os.stat(target)
        except FileNotFoundError:
            pass
    elapsed = time.monotonic() - t0
    return n / elapsed  # ops/second

# 实验：1层 vs 5层 vs 15层 lower
base = tempfile.mkdtemp()
for num_layers in [1, 5, 10, 15]:
    # 准备目录
    lower_base = os.path.join(base, f"lower_{num_layers}")
    upper = os.path.join(base, f"upper_{num_layers}")
    work  = os.path.join(base, f"work_{num_layers}")
    merged = os.path.join(base, f"merged_{num_layers}")
    for d in [upper, work, merged]:
        os.makedirs(d, exist_ok=True)

    layers = create_multilayer_overlay(num_layers, lower_base)
    lowerdir = ":".join(reversed(layers))  # 最新层在最左边

    # 挂载
    ret = subprocess.run(
        ['mount', '-t', 'overlay', 'overlay',
         '-o', f'lowerdir={lowerdir},upperdir={upper},workdir={work}',
         merged],
        capture_output=True
    )
    if ret.returncode != 0:
        print(f"  {num_layers}层: 挂载失败（需要 root）")
        continue

    ops = benchmark_file_lookup(merged)
    subprocess.run(['umount', merged], capture_output=True)
    print(f"  {num_layers}层 lower: {ops:,.0f} stat ops/s")

# 典型结果：
# 1层 lower:  450,000 stat ops/s
# 5层 lower:  380,000 stat ops/s  ← 层数增加，查找稍慢
# 10层 lower: 310,000 stat ops/s
# 15层 lower: 255,000 stat ops/s  ← 镜像层多时有明显影响
EOF
```

**实验三：理解 copy-up 尾延迟的影响**

```bash
# 模拟真实应用在 overlayfs 上的 p99 延迟
# 应用以 4K 写为主，但偶发 copy-up 会造成尾延迟突刺

python3 - << 'EOF'
import os, time, random, statistics, subprocess, tempfile

def run_mixed_workload(lower_dir, upper_dir, work_dir, merged_dir,
                       n_files=100, n_writes=2000):
    """
    混合工作负载：
    - 随机选择文件写入
    - 部分文件在 lower（首次写触发 copy-up）
    - 部分文件在 upper（正常写）
    """
    # 先在 lower 创建文件
    for i in range(n_files):
        path = os.path.join(lower_dir, f"file_{i}.dat")
        with open(path, 'wb') as f:
            f.write(os.urandom(64 * 1024))  # 64KB 文件

    # 挂载 overlayfs
    subprocess.run([
        'mount', '-t', 'overlay', 'overlay',
        '-o', f'lowerdir={lower_dir},upperdir={upper_dir},workdir={work_dir}',
        merged_dir
    ], check=True)

    latencies = []
    for _ in range(n_writes):
        file_idx = random.randint(0, n_files - 1)
        path = os.path.join(merged_dir, f"file_{file_idx}.dat")

        t0 = time.monotonic_ns()
        with open(path, 'r+b') as f:
            f.seek(random.randint(0, 60*1024))
            f.write(b'\x00' * 4096)
        latencies.append((time.monotonic_ns() - t0) / 1000)

    subprocess.run(['umount', merged_dir])
    return latencies

# 运行实验
base = tempfile.mkdtemp()
dirs = {
    'lower': os.path.join(base, 'lower'),
    'upper': os.path.join(base, 'upper'),
    'work':  os.path.join(base, 'work'),
    'merged': os.path.join(base, 'merged'),
}
for d in dirs.values():
    os.makedirs(d)

try:
    lats = run_mixed_workload(**dirs)
    s = sorted(lats)
    n = len(s)
    print(f"overlayfs 混合工作负载 ({n} 次写入):")
    print(f"  avg={statistics.mean(lats):.1f}μs")
    print(f"  p50={s[n//2]:.1f}μs")
    print(f"  p95={s[int(n*0.95)]:.1f}μs")
    print(f"  p99={s[int(n*0.99)]:.1f}μs   ← copy-up 造成的尾延迟")
    print(f"  max={s[-1]:.1f}μs")

    # 分析延迟分布
    thresholds = [100, 500, 1000, 5000, 10000]
    print("\n延迟分布:")
    for t in thresholds:
        pct = sum(1 for l in lats if l > t) / n * 100
        print(f"  >{t}μs: {pct:.1f}%")
except PermissionError:
    print("需要 root 权限才能挂载 overlayfs")
EOF
```

### 16.14 生产环境 overlayfs 问题排查

```bash
# 常见问题 1：容器启动慢（大量 copy-up 触发）

# 诊断：追踪 copy-up 事件
bpftrace -e '
kprobe:ovl_copy_up_one {
    @copy_up_count = count();
    @copy_up_start[tid] = nsecs;
}
kretprobe:ovl_copy_up_one /@copy_up_start[tid]/ {
    $lat_ms = (nsecs - @copy_up_start[tid]) / 1000000;
    @copy_up_lat_ms = hist($lat_ms);
    if ($lat_ms > 100) {
        printf("SLOW copy-up: %dms comm=%s pid=%d\n", $lat_ms, comm, pid);
    }
    delete(@copy_up_start[tid]);
}
interval:s:10 {
    print(@copy_up_count);
    print(@copy_up_lat_ms);
}' -- sleep 60 &

# 同时启动容器
docker run --rm python:3.11-slim python -c "import numpy"  # 会触发大量 copy-up

# 常见问题 2：容器磁盘空间占用大

# 分析每个容器的 upper layer 大小
docker ps -q | while read cid; do
    name=$(docker inspect --format '{{.Name}}' $cid)
    upper=$(docker inspect --format '{{.GraphDriver.Data.UpperDir}}' $cid)
    size=$(du -sh "$upper" 2>/dev/null | cut -f1)
    echo "$size  $name  ($upper)"
done | sort -h

# 查看哪些文件被修改（在 upper 层）
CONTAINER="my_container"
UPPER=$(docker inspect --format '{{.GraphDriver.Data.UpperDir}}' $CONTAINER)
echo "=== 容器修改的文件（upper layer）==="
find "$UPPER" -type f ! -name "*.wh.*" -printf "%s\t%p\n" | \
    sort -rn | head -20 | numfmt --to=iec-i --field=1

echo "=== 容器删除的文件（whiteout）==="
find "$UPPER" -name ".wh.*" | while read wh; do
    original=$(dirname $wh)/$(basename $wh | sed 's/^\.wh\.//')
    echo "  deleted: $original"
done

# 常见问题 3：不同容器看到相同文件内容不一致

# 诊断：确认文件在 lower 还是 upper
check_file_layer() {
    local container=$1
    local file_path=$2
    local pid=$(docker inspect --format '{{.State.Pid}}' $container)
    local upper=$(docker inspect --format '{{.GraphDriver.Data.UpperDir}}' $container)

    # 检查 upper 中是否存在
    local rel_path="${file_path#/}"  # 去掉开头的 /
    if [ -f "$upper/$rel_path" ]; then
        echo "$file_path is in UPPER (container-specific, has been modified)"
        md5sum "$upper/$rel_path"
    else
        echo "$file_path is in LOWER (shared read-only layer)"
        # 找到实际所在的 lower 层
        local lower=$(docker inspect --format \
            '{{.GraphDriver.Data.LowerDir}}' $container)
        IFS=':' read -ra LAYERS <<< "$lower"
        for layer in "${LAYERS[@]}"; do
            if [ -f "$layer/$rel_path" ]; then
                echo "  Found in layer: $layer"
                md5sum "$layer/$rel_path"
                break
            fi
        done
    fi
}

check_file_layer my_container /etc/nginx/nginx.conf

# 常见问题 4：tmpfs 优化（绕过 overlayfs 的写路径）

# 把频繁写入的目录挂载为 tmpfs，避免 copy-up
docker run -d \
    --tmpfs /tmp:rw,size=100m,exec \
    --tmpfs /var/cache:rw,size=200m \
    my_app:latest
# → /tmp 和 /var/cache 的写操作不经过 overlayfs，无 copy-up 开销
# → 容器重启后这些目录内容丢失（tmpfs 是内存文件系统）
```

---

## 本章小结

| 主题 | 结论 |
|------|------|
| overlayfs | 提供多层对象树的联合视图 |
| copy-up | 是写路径成本和隔离能力的关键交换 |
| whiteout / opaque | 负责表达“下层对象被上层隐藏或替代” |
| redirect_dir / metacopy | 说明联合视图需要复杂协调机制 |
| 性能 | 受 lower/upper、copy-up、cache、volume 共同影响 |
| 边界 | 它适合镜像层共享，不是所有工作负载的最佳写路径 |

---

## 练习题

1. overlayfs 的核心目标为什么是“联合视图”而不是“多层存储”？
2. copy-up 解决了什么问题，又引入了什么成本？
3. whiteout 和 opaque dir 分别在表达哪种遮蔽语义？
4. 为什么同一路径在容器里可能随着阶段变化而语义不同？
5. 哪些场景不适合把 overlayfs 当作主要写入路径？
