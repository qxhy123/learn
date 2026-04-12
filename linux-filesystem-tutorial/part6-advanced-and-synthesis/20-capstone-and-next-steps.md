# 第20章：综合实践与下一步

> 真正会用文件系统知识的人，不是能背术语的人，而是遇到故障、性能问题、容器异常、远端语义错位时，能把问题快速拆回正确层次，并设计出一条能验证、能恢复、能解释的协议路径。

## 学习目标

完成本章后，你将能够：

1. 用统一模型复述路径、对象、缓存、挂载、一致性、远端语义的关系
2. 用协议和失败边界思维分析真实工程问题
3. 把教程内容转化为一套排障与设计清单
4. 识别哪些问题该交给文件系统，哪些必须由应用协议承担
5. 为继续深入内核、存储、容器、分布式存储建立明确路线
6. 能把“现象 -> 证据 -> 协议 -> 演练”固定成设计评审和事故复盘动作

---

## 正文内容

### 20.1 重新看整条主线：Linux 文件系统到底在回答什么

本教程从“路径是什么”一路走到：

- 路径、目录项、inode、file 的对象拆分
- block group、extent、journal、writeback 的布局与更新机制
- VFS、mount、namespace、overlayfs 的视图协议
- `write`、`fsync`、`rename`、`fsync(dir)` 的一致性边界
- NFS/FUSE/对象存储桥接下的远端语义折中

如果要压缩成一句话，可以说：

**Linux 文件系统不是只在回答“数据放哪”，而是在回答“谁通过什么名字、以什么缓存和持久化边界、在什么失败模型下，看到什么对象”。**

### 20.2 一个统一的分析框架：八问

以后再遇到问题，可以先问这八个问题：

1. 这是路径层、对象层、cache 层、文件系统层、块层、设备层，还是远端语义层的问题？
2. 当前路径处在什么 mount namespace 和挂载视图里？
3. 看到的是名字问题，还是对象生命周期问题？
4. 相关 cache 是谁：dcache、page cache、属性缓存、客户端缓存？
5. 当前真正的同步/提交边界是什么：`write`、`fsync(file)`、`rename`、`fsync(dir)`、close-to-open，还是应用 WAL？
6. 崩溃/失败发生在哪里：进程、内核、设备、网络、远端节点？
7. 应用是否在依赖某种未被底层真正保证的语义？
8. 证据来自哪里：`strace`、`lsof`、`findmnt`、`vmstat`、`iostat`、`perf`、日志还是代码审查？

能把这八问问清，很多“玄学问题”会立刻失去神秘感。

### 20.3 三个完整工程案例

#### 案例一：配置文件可靠更新

目标：更新配置文件，要求断电后要么旧版本，要么完整新版本。

需要考虑：

- 新内容先写到临时文件
- `fsync(temp)` 确保文件数据边界
- `rename(temp, target)` 做名字切换
- `fsync(parent dir)` 保证目录项关系
- 应用层最好带版本号/校验和，避免读到逻辑上损坏的“完整文件”

这里真正可靠的不是某个单独 API，而是一条提交协议。做排障或代码审查时，至少还要补问：

- 临时文件是否和目标文件位于同一文件系统实例内？
- 是否有人只做了 `rename`，却忘了目录持久化边界？
- 读取方是否验证版本号、长度、校验和，而不是只要“能打开就算成功”？

#### 案例二：容器里卷挂载后行为怪异

现象：容器里的 `/app` 看起来和宿主机不一致，安装软件后性能异常。

可能原因：

- bind mount 覆盖了镜像里的原路径
- overlayfs merged 视图隐藏了 lower/upper 细节
- propagation 导致挂载事件串台
- 首次写大量触发 copy-up
- volume 后端其实是远端文件系统而不是本地 ext4

这类问题如果只在目录树上看，很容易误判。必须回到挂载视图和联合层协议。排查顺序通常应是：

1. 先看 `mountinfo` / `findmnt`，确认你实际站在哪棵挂载树里。
2. 再分清 lower、upper、bind mount、tmpfs、volume plugin 分别是谁。
3. 最后才讨论权限、容量、性能，因为这些都依赖上面两步的现实。

#### 案例三：本地能跑的程序迁到 NFS 后偶发错乱

可能暴露的隐藏假设包括：

- 默认 close 后所有客户端都立刻看到更新
- 用 mtime 作为同步协议
- 假设 rename 语义和本地完全一样
- 认为文件锁一定像单机一样工作
- 把 `fsync` 理解成统一强保证

真正要做的是把应用协议重新审视一遍，而不是只怪“网络慢”。如果迁移后出现偶发问题，最有效的动作通常不是继续盯应用日志，而是同时补三类证据：

- 路径/挂载证据：`findmnt`、`mountinfo`、挂载参数
- 缓存/可见性证据：mtime、属性缓存、close/open 边界
- 锁/失败证据：锁是否依赖单机语义，客户端或服务端重连后行为是否变化

### 20.4 一份可复用的事故复盘模板

文件系统问题复盘如果只写“磁盘满了”“NFS 不稳定”“容器挂载有问题”，价值非常低。更高质量的复盘至少应包含：

1. **现象层**：用户看到了什么，错误发生在路径、容量、权限、性能还是一致性上？
2. **层次定位**：问题主要落在名字空间、VFS/cache、具体文件系统、块层、设备层还是远端语义层？
3. **失败边界**：真正出问题的是断电、进程崩溃、节点重启、网络分区，还是运维操作？
4. **错误假设**：团队此前把什么当成理所当然？例如“rename 之后一定万无一失”“容器里路径就是宿主路径”“对象挂载等于 POSIX”。
5. **修复协议**：最终是改应用协议、改文件系统选择、改挂载参数，还是补观测与演练？

如果一份复盘没有写清这五点，它大概率无法帮助下一次同类问题。

### 20.5 六个综合实践题

如果你想把整套教程真正吃透，建议至少完成其中 3 个：

| 题目 | 目标能力 |
|------|----------|
| 解释一次 deleted-but-open 空间异常 | 把目录项、inode、fd、容量观察拆开 |
| 设计一个可靠配置更新协议 | 把 `write`、`fsync`、`rename`、目录持久化串成协议 |
| 给容器卷异常设计排查流程 | 把 namespace、bind mount、overlayfs、volume backend 串起来 |
| 重写一个 benchmark 计划 | 把工作负载、缓存状态、证据链、对照组写完整 |
| 评审一次“迁移到远端共享存储”的方案 | 把语义债、锁、可见性、恢复路径显式化 |
| 选择一个源码入口并画对象图 | 把抽象概念落到 `fs/namei.c`、`mm/filemap.c`、`fs/namespace.c` 等代码路径 |

### 20.6 文件系统与应用协议的职责边界

一个高阶工程师必须清楚：哪些事能交给文件系统，哪些不能。

文件系统擅长提供：

- 名字空间和对象抽象
- 一定程度的元数据事务和崩溃恢复
- 基本的权限和共享语义
- 面向块设备或远端文件接口的统一访问层

应用仍然必须自己负责：

- 业务事务边界
- 跨多个文件的版本一致性
- checksum/版本号/WAL/幂等恢复
- 远端语义弱化后的补偿协议

把应用问题硬压给文件系统，是很多“看起来合理但长期不稳”的设计根源。

### 20.7 如果继续深入，下一步该怎么学

你可以按方向拆：

- **内核/VFS**：path walk、dcache、RCU path lookup、inode/file 操作表
- **本地文件系统实现**：ext4、xfs、btrfs、zfs 的布局和更新协议对比
- **块层/设备**：NVMe、flush、FUA、RAID、TRIM、写放大、虚拟块设备
- **容器存储**：overlayfs、snapshotter、containerd image layer、CSI volume
- **远端存储**：NFS、CephFS、SMB、对象存储网关、FUSE 协议桥接
- **协议设计**：可靠写入、manifest/metadata 原子更新、WAL、恢复演练

如果你希望从“读教程”进入“读代码/做实验”，可以直接结合[附录 A4：高阶实验与源码阅读路线](../appendix/advanced-labs-and-source-roadmap.md)推进：

- 先做 2-3 个实验，建立现象和证据链
- 再选一个源码入口，把对象关系图画出来
- 最后回到自己的业务系统，重写一次设计假设或事故复盘

### 20.8 最后给你的一个判断标准

以后再学任何一个“文件系统特性”时，都不要先问它“高级不高级”，而先问：

- 它工作在哪一层？
- 它替我补了哪个语义缺口？
- 它的代价是什么？
- 它的失败边界是什么？
- 如果它失效，应用能否自己恢复？

如果这五个问题你都能答出来，才算真的理解了这个机制。

### 20.9 把文件系统问题写进设计评审，而不是留给事故再解释

如果你正在评审一个依赖文件系统的系统，建议显式写出下面这张清单：

1. 数据通过什么路径提交：buffered I/O、`mmap`、O_DIRECT、远端挂载，还是对象网关？
2. 真实提交点在哪里：`fsync(file)`、`rename`、`fsync(dir)`、`syncfs()`，还是应用 WAL？
3. 失败后谁恢复：文件系统 replay、平台快照、应用 checksum/manifest，还是人工介入？
4. 运行时视图是否稳定：mount namespace、bind mount、overlayfs、远端缓存会不会改变路径含义？
5. 性能上最贵的语义是什么：高频 `fsync`、元数据风暴、copy-up、锁撤销，还是网络往返？

只要这五项没写清，系统往往只是“暂时能跑”，还没有形成可审计、可恢复、可演练的协议。

### 20.10 这门教程真正的毕业标准

你不需要把所有内核源码背下来，但至少应该能独立完成下面四件事：

- 看到异常时，先定位层次，而不是先怪磁盘或网络
- 找到能证明/推翻假设的命令和证据，而不是只靠经验猜
- 写出一个明确的提交或恢复协议，而不是说“理论上应该没问题”
- 把一个具体问题延伸成实验、源码入口和事故复盘

做到这里，教程内容才真正从知识点变成工程能力。

---

### 20.11 完整实现：可靠配置文件更新协议

把第 20.3 节的案例一做成可运行的生产级代码：

```python
#!/usr/bin/env python3
"""
可靠配置文件更新协议
目标：断电后要么看到完整旧版本，要么看到完整新版本，不会有中间状态
"""
import os, json, hashlib, time, tempfile, struct
from pathlib import Path
from typing import Any

class ReliableConfigWriter:
    """
    使用 write-to-temp + fsync + rename + fsync(dir) 协议的安全写入器
    
    持久化保证：
      1. 数据先写到临时文件，与目标同目录（保证 rename 是同文件系统）
      2. fsync(temp) 确保新内容到达存储介质
      3. rename(temp, target) 原子替换目录项（POSIX 保证）
      4. fsync(dir_fd) 确保 rename 对目录项的修改也持久化
    
    崩溃安全分析：
      - 崩溃在步骤 1/2 之间：临时文件不完整，目标完好
      - 崩溃在步骤 2/3 之间：临时文件完整但未替换，目标完好  
      - 崩溃在步骤 3/4 之间：rename 已原子完成，目标是新版本
                             （但目录项 rename 可能在重启后重放）
      - 崩溃在步骤 4 之后：完全一致
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.dir = self.path.parent

    def write(self, data: Any, version: int | None = None) -> None:
        """原子写入配置数据（dict/list/str 均可）"""
        # 序列化内容
        if isinstance(data, (dict, list)):
            content = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            content = str(data)
        content_bytes = content.encode('utf-8')

        # 构造带版本和校验和的 envelope
        envelope = {
            'version': version or int(time.time_ns()),
            'checksum': hashlib.sha256(content_bytes).hexdigest(),
            'size': len(content_bytes),
            'content': content,
        }
        final_bytes = json.dumps(envelope, ensure_ascii=False).encode('utf-8')

        # 步骤 1：写到同目录的临时文件
        # 注意：必须与目标在同一文件系统（保证 rename 是原子的，不是 copy+delete）
        fd, tmp_path = tempfile.mkstemp(
            dir=self.dir,
            prefix=f'.{self.path.name}.tmp.',
            suffix='.partial'
        )
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(final_bytes)
                # 步骤 2：fsync 临时文件（确保内容到达存储介质）
                f.flush()
                os.fsync(f.fileno())

            # 步骤 3：原子 rename（POSIX 保证目录项替换是原子的）
            os.rename(tmp_path, self.path)
            tmp_path = None  # rename 成功，临时文件已消失

        except Exception:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)  # 清理残留临时文件
            raise

        finally:
            # 步骤 4：fsync 目录（确保 rename 对目录项的修改也持久化）
            # 如果跳过这步，重启后可能找不到新文件（目录项丢失）
            dir_fd = os.open(str(self.dir), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    def read(self) -> Any:
        """读取配置，验证校验和"""
        try:
            with open(self.path, 'rb') as f:
                envelope = json.loads(f.read().decode('utf-8'))

            content_bytes = envelope['content'].encode('utf-8')
            expected = envelope['checksum']
            actual = hashlib.sha256(content_bytes).hexdigest()

            if actual != expected:
                raise ValueError(
                    f"Config corrupted! expected={expected[:16]}... "
                    f"actual={actual[:16]}..."
                )

            return json.loads(envelope['content'])

        except FileNotFoundError:
            return None

    def atomic_update(self, updater_fn) -> None:
        """原子地读取-修改-写入（避免读写竞争）"""
        current = self.read() or {}
        updated = updater_fn(current)
        self.write(updated)


# 演示使用
if __name__ == '__main__':
    import shutil

    config_path = '/tmp/demo_config.json'
    writer = ReliableConfigWriter(config_path)

    # 初始写入
    writer.write({'version': 1, 'db_url': 'postgres://localhost/mydb',
                  'max_connections': 10})
    print("初始配置:", writer.read())

    # 原子更新（只改一个字段）
    writer.atomic_update(lambda c: {**c, 'max_connections': 50, 'version': 2})
    print("更新后:", writer.read())

    # 验证目录中是否有残留临时文件（不应该有）
    tmp_files = list(Path('/tmp').glob(f'.demo_config.json.tmp.*'))
    assert not tmp_files, f"发现残留临时文件: {tmp_files}"
    print("✓ 无残留临时文件")

    # 模拟崩溃恢复：如果存在旧临时文件，启动时清理
    def cleanup_partial_writes(config_path: str):
        p = Path(config_path)
        for tmp in p.parent.glob(f'.{p.name}.tmp.*.partial'):
            print(f"清理未完成的临时文件: {tmp}")
            tmp.unlink()

    cleanup_partial_writes(config_path)
    print("✓ 崩溃恢复检查完成")
```

### 20.12 内核源码入口地图

按教程章节对应的内核源码位置，按需深入：

```
Linux 内核文件系统源码关键入口（fs/ 目录）
══════════════════════════════════════════════════════════════

路径解析（第 2 章 + VFS 基础）
  fs/namei.c
    path_lookupat()          ← 路径查找主入口
    link_path_walk()         ← 逐分量解析循环
    do_sys_openat2()         ← openat2() 系统调用
    __do_sys_rename()        ← rename() 系统调用

VFS 对象模型（第 1 章 + 部分第 4 章）
  fs/inode.c
    inode_init_once()        ← inode 对象初始化
    iget_locked()            ← 获取/创建 inode 缓存条目
    evict_inode()            ← inode 回收（nlink=0 且无 fd 时触发）
    mark_inode_dirty()       ← 标记脏 inode（会触发 writeback）
  
  fs/dcache.c
    d_alloc()                ← 分配 dentry
    __d_lookup_rcu()         ← RCU path walk 的快速查找
    d_splice_alias()         ← 把 inode 挂载到 dentry 上

  fs/file.c
    alloc_file()             ← 分配 struct file 对象
    fd_install()             ← 把 file 注册到进程 fd 表

page cache 与 I/O（第 7-9 章）
  mm/filemap.c
    filemap_read()           ← 读操作通过 page cache 的路径
    filemap_fault()          ← mmap page fault 处理
    __filemap_fdatawrite_range() ← writeback 触发路径

  mm/readahead.c
    page_cache_async_ra()    ← 异步预读触发
    ondemand_readahead()     ← 按需预读逻辑

  mm/page-writeback.c
    balance_dirty_pages()    ← 脏页流控（超 dirty_ratio 时阻塞写）
    wb_writeback()           ← writeback 工作线程主循环

挂载与命名空间（第 11 章 + 第 15 章）
  fs/namespace.c
    do_mount()               ← mount() 系统调用核心
    do_loopback()            ← bind mount 实现
    pivot_root()             ← 容器根切换

  fs/mount.h
    struct mount             ← 每个挂载点的完整描述
    struct mnt_namespace     ← mount namespace 内核对象

overlayfs（第 16 章）
  fs/overlayfs/
    ovl_open()               ← open 操作（决定走 upper 还是 lower）
    ovl_copy_up()            ← copy-up 主函数
    ovl_lookup()             ← 目录项查找（合并多层结果）

ext4 实现（第 12-13 章）
  fs/ext4/
    ext4_file_write_iter()   ← write() 入口
    ext4_sync_file()         ← fsync() 实现
    ext4_mb_new_blocks()     ← block 分配（mballoc）
    ext4_ext_map_blocks()    ← extent 树查找

NFS 客户端（第 17 章）
  fs/nfs/
    nfs_file_release()       ← close-to-open 实现（flush 脏页）
    nfs_revalidate_inode()   ← 属性缓存验证
    nfs_file_read()          ← read 操作（含 cache 命中）
```

```bash
# 快速定位源码工具（需要内核源码）
# 下载内核源码
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.8.tar.xz
tar xf linux-6.8.tar.xz && cd linux-6.8

# 搜索特定函数
grep -rn "link_path_walk" fs/namei.c | head -5
# 或用 cscope/ctags
make cscope && cscope -d -f cscope.out

# 用 bpftrace 验证内核路径（不需要编译，直接在运行内核上）
bpftrace -e '
kprobe:link_path_walk {
    printf("path walk: %s (pid=%d comm=%s)\n",
           str(arg1), pid, comm);
}' -- sleep 5 &
ls /etc/hosts  # 触发路径解析
```

### 20.13 从实验到深入：一条循序渐进的学习路线

**阶段 1：建立观察能力（1-2 周）**

```bash
# 实验 1.1：观察一次 open() 的完整内核路径
strace -e trace=openat,read,write,close,stat \
    python3 -c "open('/etc/hosts').read()" 2>&1 | head -20

# 实验 1.2：观察 page cache 的工作
# 第一次读（冷缓存）
sync && echo 3 > /proc/sys/vm/drop_caches
time cat /dev/null < /bin/python3  # 触发 page-in

# 第二次读（热缓存）
time cat /dev/null < /bin/python3  # 几乎瞬间完成

# 查看 python3 的 page cache 状态
vmtouch /usr/bin/python3
# Files: 1
# Resident Pages: 800/800  3.1M/3.1M  100%  ← 全部在 cache 里

# 实验 1.3：观察 deleted-but-open
exec 9>/tmp/alive.txt
echo "I live on" >&9
rm /tmp/alive.txt
ls -la /proc/$$/fd/9   # (deleted) 但仍存在
lsof +L1 | grep alive  # 找到它
exec 9>&-              # 真正释放

# 实验 1.4：观察 fsync 延迟分布
bpftrace -e '
kprobe:vfs_fsync_range { @start[tid] = nsecs; }
kretprobe:vfs_fsync_range /@start[tid]/ {
    $us = (nsecs - @start[tid]) / 1000;
    @lat_us = hist($us);
    if ($us > 10000) {
        printf("SLOW fsync: %dms comm=%s\n", $us/1000, comm);
    }
    delete(@start[tid]);
}
interval:s:5 { print(@lat_us); clear(@lat_us); }' -- sleep 30 &

# 触发一些 fsync
python3 -c "
import os, tempfile
f = tempfile.NamedTemporaryFile()
for i in range(100):
    f.write(b'x'*4096)
    f.flush()
    os.fsync(f.fileno())
"
```

**阶段 2：理解层次结构（2-4 周）**

```bash
# 实验 2.1：用 perf 生成 VFS 操作火焰图
perf record -F 99 -a -g -- sleep 10 &  # 后台采集
# 同时运行工作负载
fio --name=test --ioengine=sync --rw=randrw --bs=4k \
    --size=1G --runtime=8 --time_based --filename=/tmp/fio_test

# 生成火焰图（需要 FlameGraph 工具）
git clone https://github.com/brendangregg/FlameGraph
perf script | ./FlameGraph/stackcollapse-perf.pl | \
    ./FlameGraph/flamegraph.pl > vfs_flame.svg

# 实验 2.2：理解 writeback 流控
# 设置较小的 dirty 比例触发流控
sysctl vm.dirty_ratio=5
sysctl vm.dirty_background_ratio=1

# 同时监控
watch -n 0.5 'cat /proc/meminfo | grep -E "Dirty|Writeback"'

# 写入数据并观察脏页累积和回写
dd if=/dev/zero of=/tmp/bigfile bs=1M count=1000

# 实验 2.3：mount namespace 隔离
# 在隔离的 namespace 中做挂载，不影响宿主
unshare --mount bash << 'INNER'
    mount -t tmpfs tmpfs /mnt
    echo "inside namespace" > /mnt/test.txt
    cat /mnt/test.txt
    # 退出后 /mnt 在宿主上不变
INNER
ls /mnt  # 宿主 /mnt 未受影响

# 实验 2.4：overlayfs copy-up 的实际延迟
mkdir -p /tmp/ovl/{lower,upper,work,merged}
# 创建不同大小的文件在 lower
for size in 4K 64K 1M 10M; do
    dd if=/dev/zero of=/tmp/ovl/lower/file_$size bs=$size count=1 2>/dev/null
done
mount -t overlay overlay \
    -o lowerdir=/tmp/ovl/lower,upperdir=/tmp/ovl/upper,workdir=/tmp/ovl/work \
    /tmp/ovl/merged

# 测量 copy-up 延迟
for f in /tmp/ovl/merged/file_*; do
    rm -f /tmp/ovl/upper/$(basename $f)
    start=$(date +%s%N)
    echo "x" >> $f  # 触发 copy-up
    end=$(date +%s%N)
    echo "$(basename $f): $((($end-$start)/1000)) μs"
done

umount /tmp/ovl/merged
```

**阶段 3：读内核源码（4-8 周）**

```bash
# 推荐的源码阅读顺序（从简单到复杂）

# 入口 1：tmpfs（最简单的文件系统，纯内存）
# mm/shmem.c
# 关键函数：shmem_file_read_iter, shmem_write_begin, shmem_evict_inode
# 好处：没有磁盘 I/O，逻辑清晰，适合初次阅读

# 入口 2：procfs（虚拟文件系统，展示 VFS 灵活性）
# fs/proc/
# 关键：proc_file_operations, proc_read_iter
# 好处：了解"文件系统 = 接口，不必是磁盘"的直觉

# 入口 3：ext4（最常用的磁盘文件系统）
# fs/ext4/
# 推荐顺序：
#   inode.c (inode 基本操作) →
#   extents.c (extent 树) →
#   mballoc.c (block 分配) →
#   namei.c (目录操作) →
#   super.c (挂载/卸载)

# 用 lxr 或 elixir 在线阅读（无需下载）
# https://elixir.bootlin.com/linux/latest/source/fs/namei.c#L3600
# 在线搜索函数定义和调用关系

# 本地用 cscope/gtags 导航
cd linux-6.8
make cscope
cscope -d
# 在 cscope 中：
#   Ctrl+\ s  查找符号定义
#   Ctrl+\ c  查找调用者
#   Ctrl+\ f  查找文件
```

**阶段 4：构建系统级排障能力（持续）**

```bash
# 一个完整的"文件系统健康巡检"脚本
cat << 'HEALTHCHECK' > /usr/local/bin/fs-health-check
#!/bin/bash
# 文件系统健康巡检

echo "=== $(date) ==="

echo -e "\n[1] 磁盘空间"
df -h | grep -v tmpfs | grep -v devtmpfs

echo -e "\n[2] inode 使用率"
df -i | grep -v tmpfs | grep -v devtmpfs

echo -e "\n[3] 脏页状态"
grep -E "^(Dirty|Writeback|NFS_Unstable):" /proc/meminfo

echo -e "\n[4] I/O 等待进程"
ps aux | awk '$8 == "D" {print}' | head -10

echo -e "\n[5] 大文件（deleted but open）"
lsof +L1 2>/dev/null | awk '$7+0 > 104857600' | head -5  # >100MB

echo -e "\n[6] 文件系统错误（最近 1 小时）"
dmesg --since "1 hour ago" | grep -E "EXT4-fs error|XFS.*error|BTRFS.*error|I/O error"

echo -e "\n[7] 慢 I/O（iostat）"
iostat -xz 1 1 | awk 'NR>3 && $NF+0 > 50'  # util > 50%

echo -e "\n[8] NFS 重传统计"
nfsstat -c 2>/dev/null | head -5

echo "=== 巡检完成 ==="
HEALTHCHECK
chmod +x /usr/local/bin/fs-health-check
```

### 20.14 综合案例：一次完整的生产故障分析

**场景**：应用日志服务器磁盘空间告警，但 `du` 和 `df` 不一致。

```bash
# 第一步：确认问题层次（是哪一层的问题）

# df 显示使用量
df -h /var/log
# Filesystem  Size  Used Avail Use%  Mounted on
# /dev/sda1   100G   95G    5G  95%  /var/log

# du 显示使用量（不一致！）
du -sh /var/log
# 23G    /var/log   ← 和 df 差了 72GB！

# 第二步：找证据 — 是 deleted-but-open
lsof +L1 /var/log
# COMMAND  PID   USER  FD  TYPE DEV     SIZE/OFF NLINK  NODE  NAME
# rsyslog  1234  root  7w  REG  8,1  73456789012     0  5678  /var/log/syslog (deleted)
# ↑ rsyslog 日志文件被 logrotate 删除，但 rsyslog 仍持有 fd，72GB 未释放！

# 第三步：不重启服务回收空间
# 方法 1：让 rsyslog 重新打开日志文件（发 SIGHUP 触发 reload）
kill -HUP 1234  # rsyslog 会关闭旧 fd，打开新文件
# 验证
lsof +L1 /var/log  # 应该清空了

# 方法 2：如果服务不响应 SIGHUP，直接截断 fd
truncate -s 0 /proc/1234/fd/7  # 立即释放磁盘空间

# 第四步：验证空间已回收
df -h /var/log
# 现在 Used 应该和 du 一致了

# 第五步：预防措施
# 在 logrotate 配置中添加 postrotate 脚本
cat >> /etc/logrotate.d/rsyslog << 'EOF'
postrotate
    systemctl kill --signal=SIGHUP rsyslog.service 2>/dev/null || true
endscript
EOF

# 第六步：建立监控告警
cat << 'MONITOR' > /etc/cron.d/check-deleted-open
# 每 15 分钟检查是否有 deleted-but-open 大文件
*/15 * * * * root lsof +L1 | awk '$7+0 > 1073741824' | \
    mail -s "Alert: deleted-but-open files >1GB on $(hostname)" admin@example.com
MONITOR

# 记录本次故障的教训（事故复盘）
cat << 'POSTMORTEM'
事故复盘模板：

1. 现象层：df 显示 95% 使用率，但 du 只显示 23GB，差 72GB
2. 层次定位：对象生命周期层（inode 引用计数问题）
3. 失败边界：运维操作（logrotate 删除文件，但未通知服务重新打开）
4. 错误假设：认为 logrotate postrotate 脚本已配置（实际未配置）
5. 修复协议：
   - 短期：truncate fd 回收空间
   - 长期：logrotate 配置 postrotate 发 SIGHUP，加监控告警
POSTMORTEM
```

---

## 本章小结

| 主题 | 结论 |
|------|------|
| 全书主线 | 文件系统是在定义对象、视图、缓存、持久化和失败边界 |
| 八问框架 | 能把大多数文件系统问题拆回正确层次 |
| 工程案例 | 可靠写入、容器卷、远端语义是三类经典实战问题 |
| 综合实践 | 实验、源码阅读和事故复盘应形成闭环 |
| 职责边界 | 文件系统提供基础语义，应用仍要负责业务事务与恢复 |
| 深入路线 | 可向内核、存储、容器、远端协议、应用恢复继续深入 |
| 毕业标准 | 能把问题稳定地写成证据链、协议和复盘，而不只是解释名词 |

---

## 练习题

1. 用“八问框架”分析一次 deleted-but-open 导致的空间异常。
2. 为一个配置文件更新系统设计完整提交协议，并解释为什么每一步都必要。
3. 为什么“本地能跑”完全不能证明“远端也没问题”？
4. 在容器场景里，哪些问题应交给 overlayfs/挂载视图，哪些应交给应用协议？
5. 选择一个你熟悉的生产故障，用本章框架重写一次事后分析。
6. 设计一个“从实验到源码再到事故复盘”的个人学习计划，并说明每一步验证什么假设。
7. 按照 20.9 的清单，评审一个你熟悉的系统，并指出它最危险的语义债在哪里。
