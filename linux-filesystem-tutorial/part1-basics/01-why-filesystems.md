# 第1章：为什么需要文件系统

> **本章导读**：文件系统的核心价值不是”存字节”，而是把字节组织成可命名、可授权、可共享、可恢复、可推理的系统对象。本章建立全局视图——从裸设备出发，理解文件系统同时解决命名、定位、授权、并发和恢复这五类问题，以及它为什么是理解内核、存储、容器和分布式的主线。

**前置知识**：无（入门章节）
**预计学习时间**：30 分钟

---

## 学习目标

完成本章后，你将能够：

1. 区分裸设备、块设备、文件系统、数据库和对象存储的职责边界
2. 理解文件系统为什么首先是对象与名字空间系统，而不仅是存储格式
3. 说明文件系统必须同时处理命名、定位、授权、并发和恢复
4. 建立“文件系统是操作系统对象接口层”的视角
5. 为后续 inode、VFS、缓存和一致性章节搭好地图

---

## 正文内容

### 1.1 如果只有一块裸设备

假设你拿到一块裸磁盘或一个裸块设备。它能提供的大致只是：

- 某些逻辑地址可以读
- 某些逻辑地址可以写
- 写入可能有顺序、对齐、缓存、失败边界等限制

它并不知道：

- 哪些地址属于某个“文件”
- 哪些字节属于目录、日志、配置还是数据库页
- 谁对这些字节有访问权
- 哪些对象还被引用
- 崩溃后该如何解释中间状态

所以裸设备只解决“能不能存放比特”，不解决“系统如何理解这些比特”。

### 1.2 文件系统解决的不是一个问题，而是一组问题

文件系统至少要同时回答五类问题：

1. **命名问题**：人和程序如何稳定地指向某个对象？
2. **定位问题**：名字如何映射到元数据，再映射到真实数据块？
3. **授权问题**：谁能读、谁能写、谁能遍历、谁能执行？
4. **生命周期问题**：重命名、链接、打开、删除后，对象何时真正消失？
5. **恢复问题**：中途崩溃后，哪些状态算已提交，哪些状态应回滚或重放？

如果少掉其中任何一类，系统都还能“存数据”，但已经不像一个成熟文件系统了。

### 1.3 文件系统为什么首先是名字空间系统

很多初学者把文件系统理解为“磁盘上的文件盒子”。这个比喻太弱，因为它忽略了名字空间的重要性。

文件系统真正提供的是：

- 层级目录树
- 路径解析规则
- 挂载点与视图拼接
- 不同进程/容器看到的名字空间差异

例如：

```text
/etc/hosts
/var/log/nginx/access.log
/home/alice/project/.git/config
```

这些路径不只是“位置描述”，也是系统组织知识的方式。没有名字空间，你只能记住偏移量；有了名字空间，你才能组织权限、隔离、多用户和应用约定。

### 1.4 文件系统是对象接口，不只是磁盘格式

Linux 里通过文件系统接口暴露的不只有普通文件：

- 普通文件和目录
- 设备节点
- 套接字和管道对应的 fd 语义
- `procfs` / `sysfs` 里的内核对象
- 容器根文件系统和挂载视图

所以文件系统不只是“块设备上的布局”，也是操作系统的一种统一对象接口。`open/read/write/stat/mmap` 这套接口之所以重要，不是因为底层都一样，而是因为系统把大量不同对象都放进了这套交互模型里。

### 1.5 文件系统与数据库、对象存储的边界

虽然它们都在“保存数据”，但职责很不同：

| 系统 | 主要解决什么 | 典型抽象 |
|------|--------------|----------|
| 文件系统 | 名字空间、对象管理、权限、部分 POSIX 语义 | 路径、目录、inode、fd |
| 数据库 | 结构化查询、事务、索引、业务一致性 | 表、行、索引、事务日志 |
| 对象存储 | 海量对象、分布式扩展、简单键接口 | bucket、key、object |

数据库通常假设底层已经有文件系统；对象存储通常不提供传统路径、硬链接、目录 `fsync` 之类语义。把这些抽象混在一起，会直接导致系统设计误判。

### 1.6 为什么文件系统必须关心并发

只讨论“一个程序写一个文件”还远远不够。现实系统里会出现：

- 多进程同时 `open` 同一路径
- 一个进程 `rename`，另一个进程正在读取旧名字
- 某个日志文件已经 `unlink`，服务仍持有 fd
- 多个线程共享 open file description
- 容器、宿主机、sidecar 同时操作同一路径视图

因此文件系统不仅是静态布局，还在定义并发下哪些语义成立，哪些语义不成立。

### 1.7 为什么文件系统必须关心失败

如果只讨论正常路径，文件系统听起来像“目录 + 元数据 + 数据块”。真正让它变复杂的是失败：

- 写一半断电怎么办？
- 目录项写了，数据没写怎么办？
- 数据写了，名字还没出现怎么办？
- 写请求已经离开 page cache，但设备缓存没 flush 怎么办？

所以一个文件系统真正成熟的标志，不是“平时写得快”，而是“失败后仍能解释、恢复、或至少保持结构不自相矛盾”。

### 1.8 为什么说文件系统在定义“系统如何看世界”

当你理解文件系统时，你其实也在理解这些更大的问题：

- 操作系统如何把对象组织成统一接口
- 为什么路径解析和权限检查会成为安全边界
- 为什么容器和挂载视图会改变“相同路径”的含义
- 为什么持久化语义不能只看 `write()` 返回值

所以文件系统不是某个“操作系统基础章节”的配角，而是把内核、存储、权限、容器、分布式和运维全部串起来的一根主线。

---

### 1.9 VFS 对象模型：open() 调用后内核里发生了什么

上面五类问题在内核里有非常具体的对应结构。一次 `open(“/etc/hosts”, O_RDONLY)` 在内核里大致经历：

```
用户态: open(“/etc/hosts”, O_RDONLY)
          │
          ▼ 系统调用入口
sys_openat(AT_FDCWD, “/etc/hosts”, O_RDONLY, 0)
          │
          ▼ 路径解析 (path walk)
path_openat()
  → do_filp_open()
      → path_init()          ← 确定起点（根 / 或 cwd）
      → link_path_walk()     ← 逐分量解析 “etc” → “hosts”
      → 每步从 dentry cache (dcache) 查找 → 未命中则调用 inode_operations.lookup()
          │
          ▼ 找到目标 inode
do_open()
  → vfs_open()
      → inode_operations.permission()   ← 权限检查
      → file_operations.open()          ← 文件系统特定的 open 实现
          │
          ▼ 分配 file 对象
alloc_file()  → 获得 struct file *fp
fd_install()  → 把 fp 注册到进程的 files_struct 里
          │
          ▼ 返回用户态
返回文件描述符整数（fd）
```

涉及的四大 VFS 对象：

```c
/* 四大对象是 VFS 的骨架，每种文件系统都要实现这套接口 */

/* 1. 超级块 — 代表一个挂载的文件系统实例 */
struct super_block {
    dev_t           s_dev;          /* 设备号 */
    unsigned long   s_blocksize;    /* 块大小（字节）*/
    loff_t          s_maxbytes;     /* 最大文件大小 */
    struct file_system_type *s_type;/* 文件系统类型（如 ext4）*/
    const struct super_operations *s_op; /* 操作函数表 */
    struct dentry   *s_root;        /* 根目录 dentry */
    unsigned long   s_flags;        /* 挂载标志（MS_RDONLY 等）*/
    void            *s_fs_info;     /* 文件系统私有数据（如 ext4_sb_info）*/
    /* ...还有 journal、配额、inode 缓存等 */
};

/* 2. inode — 代表一个文件对象（与路径/名字无关）*/
struct inode {
    umode_t         i_mode;         /* 文件类型 + 权限 (S_IFREG, 0644 等)*/
    unsigned short  i_opflags;
    kuid_t          i_uid;          /* 所有者 uid */
    kgid_t          i_gid;          /* 所有者 gid */
    unsigned int    i_flags;        /* 文件系统标志 */
    const struct inode_operations *i_op; /* inode 操作（lookup, create 等）*/
    struct super_block *i_sb;       /* 所属超级块 */
    struct address_space *i_mapping;/* page cache 的 address_space */
    unsigned long   i_ino;          /* inode 号 */
    loff_t          i_size;         /* 文件大小（字节）*/
    struct timespec64 i_atime;      /* 最近访问时间 */
    struct timespec64 i_mtime;      /* 最近内容修改时间 */
    struct timespec64 i_ctime;      /* 最近 inode 状态变化时间 */
    unsigned int    i_nlink;        /* 硬链接计数 */
    const struct file_operations *i_fop; /* 默认 file 操作 */
    void            *i_private;     /* 文件系统私有数据 */
};

/* 3. dentry — 代表一个路径分量（名字到 inode 的映射缓存）*/
struct dentry {
    unsigned int    d_flags;        /* DCACHE_VALID, DCACHE_OP_HASH 等 */
    struct inode    *d_inode;       /* 对应的 inode（negative dentry 时为 NULL）*/
    struct dentry   *d_parent;      /* 父目录 dentry */
    struct qstr     d_name;         /* 文件名（含 hash 和长度）*/
    struct list_head d_child;       /* 父目录的子 dentry 链表 */
    struct list_head d_subdirs;     /* 子 dentry 链表 */
    const struct dentry_operations *d_op; /* dentry 操作（可选）*/
    struct super_block *d_sb;       /* 所属超级块 */
    void            *d_fsdata;      /* 文件系统私有数据 */
};

/* 4. file — 代表一个已打开的文件会话（与进程、fd 绑定）*/
struct file {
    struct path     f_path;         /* 包含 dentry + vfsmount（路径快照）*/
    struct inode    *f_inode;       /* 对应 inode（f_path.dentry->d_inode 的缓存）*/
    const struct file_operations *f_op; /* 操作函数表 */
    spinlock_t      f_lock;
    atomic_long_t   f_count;        /* 引用计数 */
    unsigned int    f_flags;        /* O_RDONLY, O_NONBLOCK 等 */
    fmode_t         f_mode;         /* FMODE_READ, FMODE_WRITE 等 */
    loff_t          f_pos;          /* 当前读写偏移（文件位置指针）*/
    struct fown_struct f_owner;     /* 异步 I/O 通知相关 */
    void            *private_data;  /* 文件系统私有状态（如 ext4 的加密上下文）*/
};
```

这四个对象的关系可以这样理解：

```
进程的 fd 表
  fd=3 ──→ struct file (f_pos=0, f_flags=O_RDONLY)
                │
                ▼ f_path
           struct dentry (d_name=”hosts”, d_inode=→)
                │                     │
                ▼ d_parent            ▼ d_inode
           struct dentry          struct inode (i_ino=1234, i_size=217)
           (d_name=”etc”)              │
                │                     ▼ i_mapping
                ▼ d_parent       address_space (page cache)
           struct dentry
           (d_name=”/”, 根 dentry)
```

### 1.10 /proc 入口：用观测代替猜测

不需要读内核源码，也可以通过 `/proc` 直接观察文件系统的运行状态：

```bash
# 1. 查看当前内核支持的文件系统类型（已注册到 VFS）
cat /proc/filesystems
# nodev   sysfs         ← nodev = 不需要块设备（虚拟文件系统）
# nodev   tmpfs
# nodev   bpf
#         ext4          ← 真实文件系统，需要块设备
#         xfs
#         btrfs

# 2. 查看当前挂载的所有文件系统
cat /proc/mounts
# 或更详细版本（含挂载 ID、传播信息）
cat /proc/self/mountinfo
# 36 35 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw,errors=remount-ro
# ↑  ↑  ↑  ↑ ↑ ↑            ↑         ↑    ↑         ↑
# ID 父 设备 根 挂载点 挂载选项  传播      类型 源设备    超级块选项

# 3. 查看 dentry cache 和 inode cache 的当前使用情况
cat /proc/slabinfo | grep -E “dentry|inode_cache|ext4_inode”
# dentry           85000  90000    192   21    1 : tunables    0    0    0 ...
# ↑名字             ↑活跃  ↑总数   ↑大小

# 4. 查看当前进程打开的所有 fd 及其对应路径
ls -la /proc/$$/fd/
# lrwx------ 1 user user 64 Apr 12 /proc/1234/fd/0 -> /dev/pts/0
# lrwx------ 1 user user 64 Apr 12 /proc/1234/fd/1 -> /dev/pts/0
# lr-x------ 1 user user 64 Apr 12 /proc/1234/fd/3 -> /etc/hosts

# 5. 查看进程打开的 file description（含 pos, flags）
cat /proc/$$/fdinfo/3
# pos:    0        ← 当前文件位置指针
# flags:  0100000  ← O_RDONLY (八进制)
# mnt_id: 36       ← 对应 mountinfo 中的挂载 ID

# 6. 统计 VFS 对象缓存命中率
cat /proc/sys/fs/dentry-state
# 91000 85000 45 0 0 0
# ↑总数  ↑使用中 ↑回收期间 ...

# 7. 查看 inode 统计
cat /proc/sys/fs/inode-state
# 123456 45678 0 0 0 0 0
# ↑总数  ↑空闲数 ...
```

### 1.11 strace 实战：一次文件保存的完整系统调用序列

编辑器（如 vim）保存文件时的实际系统调用序列：

```bash
# 追踪 vim 保存文件的系统调用（过滤掉信号和 epoll）
strace -e trace=openat,read,write,rename,fsync,close,unlink,stat -o /tmp/vim_save.txt \
    vim /tmp/test.txt

# 分析关键调用序列（vim 的”安全写入”策略）：
```

```
# vim 保存 /tmp/test.txt 的典型系统调用序列：

# 1. 先写一个临时文件（同目录下），防止保存到一半崩溃
openat(AT_FDCWD, “/tmp/test.txt.swp”, O_RDWR|O_CREAT, 0600) = 4

# 2. 把新内容写入临时文件
write(4, “new content...”, 1024) = 1024

# 3. 把临时文件内容 fsync（强制落盘）
fsync(4) = 0

# 4. 原子替换（rename 是原子操作）
rename(“/tmp/test.txt~”, “/tmp/test.txt”) = 0
# ← 这一刻，/tmp/test.txt 变成新内容；同时旧内容还在 test.txt~ 里

# 5. 关闭临时文件 fd
close(4) = 0

# 6. 一些编辑器还会再 fsync 目录，确保目录项落盘
openat(AT_FDCWD, “/tmp”, O_RDONLY|O_DIRECTORY) = 5
fsync(5) = 0   # 确保 rename 对目录项的修改也落盘
close(5) = 0
```

这个序列体现了 1.7 节讲的”失败处理”：
- **写临时文件**：保护原文件不被截断到一半
- **fsync 临时文件**：确保新内容到达存储介质
- **rename 原子替换**：目录项切换是原子的，观察者要么看到旧版，要么看到新版
- **fsync 目录**：确保 rename 的目录项变更也持久化（否则重启后可能两个文件都丢）

```bash
# 如果只是 cat >> 追加，没有这些保护，会看到更简单的序列：
strace -e write,openat,close cat >> /tmp/simple.txt << EOF
new line
EOF
# openat(AT_FDCWD, “/tmp/simple.txt”, O_WRONLY|O_CREAT|O_APPEND, 0666) = 3
# write(3, “new line\n”, 9) = 9
# close(3) = 0
# ← 没有 fsync！系统崩溃的话这条内容可能会丢
```

### 1.12 文件系统的两个视角：用户 API 层 vs 内核实现层

这套 API 统一了所有不同底层：

```
用户 API 层（POSIX 接口）
─────────────────────────────────────────
open() read() write() stat() mkdir() rename() unlink() mmap() fsync()
─────────────────────────────────────────
VFS 层（内核抽象）
  struct file_operations：每种文件系统实现这套接口
  struct inode_operations：处理元数据操作
  struct address_space_operations：处理页缓存 I/O
─────────────────────────────────────────
具体文件系统层
  ext4:  日志（JBD2）、extent 树、块分配器
  xfs:   B+ 树 inode 索引、allocation group
  btrfs: CoW B-tree、extent tree、校验和
  tmpfs: 纯内存，无磁盘布局
  proc:  内核对象的虚拟文件接口
─────────────────────────────────────────
块层 / VFS bio 层
  page cache、readahead、writeback
─────────────────────────────────────────
设备层
  NVMe / SATA / iSCSI / NFS over TCP ...
```

VFS 保证你的应用代码不需要知道底层是哪种文件系统，但**并不保证所有语义在所有文件系统上完全相同**——这是后续章节一再出现的核心张力。

---

## 例子：一次”保存文件”背后其实发生了什么

假设编辑器保存 `notes.txt`，可能发生：

- 用户态 buffer 写入内核
- page cache 更新
- inode 大小变化
- extent 映射变化
- 目录项可能切换到临时文件新版本
- journal 事务提交
- 设备缓存 flush

所以“保存文件”从来不是一个动作，而是一串跨层协议。你在后面章节会看到：只要这串协议中某一步的语义被误解，就会出现“文件明明保存了，为什么重启后没了”这类问题。

---

## 常见误区

- 误以为磁盘天然知道“文件”是什么
- 误以为文件系统只解决存储，不解决命名和授权
- 误以为文件系统和数据库/对象存储只是叫法不同
- 误以为正常路径上的读写速度就是文件系统的全部价值
- 误以为“保存成功”天然包含崩溃恢复语义

---

## 本章小结

| 主题 | 结论 |
|------|------|
| 裸设备 | 只提供地址化存储，不提供对象语义 |
| 文件系统 | 同时解决名字、定位、授权、生命周期和恢复 |
| 名字空间 | 是文件系统最重要的抽象之一 |
| 文件系统接口 | 也是操作系统组织对象的统一入口 |
| 并发与失败 | 让文件系统从“格式”变成“协议” |
| 全局视角 | 文件系统是理解内核、存储、容器和分布式的主线之一 |

---

## 练习题

### 基础题

**1.1** 为什么说块设备本身并不知道”文件”是什么？列举块设备能提供的能力和它无法提供的能力。

**1.2** 文件系统至少需要同时解决哪五类问题？对每类问题各给出一个真实场景示例。

### 中级题

**1.3** 为什么名字空间比”把数据放哪”更接近文件系统的核心？用”容器隔离”场景具体说明。

**1.4** 文件系统、数据库、对象存储的职责边界分别是什么？当三者混用时会产生什么误判？

### 提高题

**1.5** 以”编辑器保存文件”为例，梳理从用户态 buffer 到磁盘的完整路径，标出哪些步骤涉及并发语义、哪些涉及失败处理，并分析在哪个步骤中断会导致什么后果（提示：page cache、journal commit、设备 flush 各自保证什么）。

---

## 练习答案

**1.1** 块设备提供：按逻辑地址读写块、顺序与对齐约束。不提供：文件名、目录结构、访问控制、对象生命周期管理、崩溃恢复语义——这些都需要文件系统叠加在上面实现。

**1.2** 五类问题：①命名（稳定指向对象，如路径 `/etc/hosts`）；②定位（名字→inode→数据块，如 ext4 extent 树）；③授权（`rwx` + ACL，如 `/tmp` 目录 sticky bit）；④生命周期（`unlink` 后 inode 何时回收，如 deleted-but-open）；⑤恢复（journal commit 后崩溃可重放，如 ext4 journaling）。

**1.3** 同一物理数据在不同 mount namespace 下可以以不同路径出现，权限隔离、多租户、容器根文件系统视图都依赖名字空间。仅靠”把数据放哪”（即物理布局）无法表达”进程 A 看到 `/app`，进程 B 看到 `/data`，底层实际是同一 inode”这类语义。

**1.4** 文件系统：路径 + POSIX 语义；数据库：结构化查询 + 事务 + 索引；对象存储：海量 key + 分布式扩展 + 简单接口。混用误判：用对象存储期望 POSIX `fsync` 一致性（对象存储不保证）；用文件系统存百万小对象期望水平扩展（inode 和目录是瓶颈）。

**1.5** 路径：`write()` → page cache 标记脏页 → writeback 线程异步写回 → 写入 journal → journal commit → 数据块落盘 → 设备 flush。并发：page lock 保护同一页的并发修改；失败分析：断电在 page cache 后、journal commit 前→恢复时重放 journal；断电在 journal commit 后、数据块落盘前→journal 重放补写；断电在设备 buffer 未 flush→仅设备缓存丢失（取决于硬件保证）。

---

## 延伸阅读

1. **Arpaci-Dusseau**. *Operating Systems: Three Easy Pieces*, Ch.39-42 — 文件与目录、文件系统实现基础（开放获取）
2. **Tanenbaum**. *Modern Operating Systems*, Ch.4 — 文件系统综述
3. **Linux 内核文档**. `Documentation/filesystems/vfs.rst` — VFS 对象模型官方说明
4. **McKusick et al.**. *The Design and Implementation of the FreeBSD Operating System*, Ch.8 — 文件系统设计经典分析

---

[下一章：路径、文件与目录 →](./02-paths-files-and-directories.md)

[返回目录](../README.md)
