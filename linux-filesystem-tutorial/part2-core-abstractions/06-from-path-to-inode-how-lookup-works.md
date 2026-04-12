# 第6章：从路径到 inode：查找过程

> **本章导读**：路径解析不是一次”字符串匹配”，而是一层层沿目录树和挂载树行走；真正难的地方在于，它还夹杂着权限、缓存、软链接、命名空间和竞争条件。本章梳理 path walk 的逐段过程、RCU 快速路径与 ref-walk 回退机制，以及 `openat2()` 如何把路径约束变成内核可执行的安全边界。

**前置知识**：第4-5章（inode/dentry 对象模型、open file description）
**预计学习时间**：55 分钟

---

## 学习目标

完成本章后，你将能够：

1. 理解路径解析的逐段查找过程
2. 说明权限检查与查找的关系
3. 认识符号链接与挂载点如何影响解析
4. 建立“路径遍历”而非“一次命中”的直觉
5. 理解路径查找为何依赖缓存与名字空间语义
6. 认识 RCU path walk、revalidation、`openat2()` 这些更高阶的查找与安全边界

---

## 正文内容

### 6.1 路径是分段解析的

以 `/var/log/nginx/access.log` 为例，系统不会把整串字符一次性直接映射为对象，而是大致按如下顺序：

1. 从起点（根目录或当前工作目录）开始
2. 查找 `var`
3. 再查找 `log`
4. 再查找 `nginx`
5. 最后查找 `access.log`

每一步都可能触发权限检查、缓存命中或底层文件系统操作。

### 6.2 起点并不总是 `/`

路径查找的起点取决于上下文：

- 绝对路径通常从当前命名空间的根开始
- 相对路径从进程当前工作目录开始
- `openat()` / `openat2()` 这类接口可以从某个目录 fd 开始

这意味着“同一段路径字符串”能否解析成功，除了依赖目录树本身，也依赖调用者所在的命名空间、工作目录和 API 选择。

### 6.3 目录的执行权限为什么重要

目录上的 `x` 权限通常表示“可遍历”。如果没有它，即使你知道某个名字，也未必能继续向下解析路径。

这说明：**路径访问不仅依赖目标文件权限，也依赖沿途目录的可遍历性**。

更高阶一点看，路径解析中的权限检查并不是只在“最终命中目标文件时”才做，而是贯穿整个 path walk。

### 6.4 dcache 与负 dentry 为什么关键

路径解析极其高频，因此 Linux 会大量依赖 dcache。它不仅缓存“这个名字存在并且指向谁”，还会缓存“这个名字不存在”。后者常被叫作**负 dentry**。

负 dentry 的价值在于：

- 避免频繁重复查找一个根本不存在的名字
- 降低目录扫描成本
- 提升大量失败查找场景的性能（例如 Web 服务探测多个候选路径）

这也提醒我们：路径查找性能并不只由磁盘决定，名字缓存本身就是一层核心加速结构。

### 6.5 符号链接会插入新的解析步骤

如果某段路径是软链接，内核需要把该链接内容解释为新的路径片段，再继续解析。这也是为什么软链接可能带来额外成本和复杂性。

高级一点看，还要考虑：

- 链接目标是绝对路径还是相对路径
- 链接链过长时的循环保护
- 安全敏感场景下是否要限制跟随软链接

很多安全问题，本质上就是“程序以为自己打开的是 A，实际上 path walk 在中途被符号链接重定向到了 B”。

### 6.6 挂载点会改变“继续往下走”的对象树

当某个目录是挂载点时，路径继续向下解析的对象树，可能已经切换到另一个文件系统实例上。

因此路径解析既要沿目录树走，也要沿挂载树跳转。

进一步看，mount namespace 会让不同进程看到不同的挂载树。于是：

- 相同路径字符串
- 在不同 namespace 中
- 可能落到完全不同的 superblock / inode / dentry 组合上

### 6.7 `.`、`..`、chroot 与路径边界

路径里还可能出现 `.`、`..`。它们看似简单，实际上会牵扯目录层级回退与“能否越过边界”的问题。

例如：

- `chroot` 改变的是进程看到的根边界
- mount namespace 改变的是挂载树视图
- 安全场景要避免路径遍历逃逸（如 `../../..`）

所以“路径规范化”不是字符串替换这么简单，它和命名空间、目录边界、安全策略绑在一起。

### 6.8 路径查找中的并发与实现模式

Linux 为了让路径查找足够快，会在不少场景下走更轻量的查找路径，例如基于缓存和只读观察的快速遍历；必要时再退回更重的引用和锁路径。你不必记住所有内核实现细节，但要知道：

- path walk 不是始终都一样重
- 缓存命中与否、是否遇到软链接、是否遇到挂载穿越、是否需要权限检查，都会改变开销
- 路径查找既是语义问题，也是性能热点

### 6.9 RCU path walk、ref-walk 与“为什么查找会突然变重”

更进一步看，Linux 路径查找并不总是用同一种“重量级别”完成。一个很重要的实现思路是：

- **RCU path walk**：尽量在不拿重锁、不增加过多引用成本的前提下快速穿过稳定缓存对象
- **ref-walk**：一旦遇到需要更强一致性确认的场景，就退回更重的引用与检查路径

常见触发回退的情况包括：

- 遇到需要 revalidation 的 dentry
- 遇到符号链接，需要插入新的解析过程
- 遇到权限检查、挂载穿越、重命名竞争
- 底层文件系统本身无法完全相信缓存结果，尤其是远端或可失效场景

这背后的工程含义非常重要：

- “路径查找很快”常常是建立在缓存稳定和对象没有竞争变化的前提上
- 一旦进入回退路径，同一条路径字符串的成本可能陡增
- 所谓元数据热点，很多时候不是磁盘慢，而是 path walk 的快路径不断失效

### 6.10 revalidation、ESTALE 与“缓存不是永远可信”

在本地 ext4 这类场景下，dcache 往往足够稳定；但在 NFS、FUSE、overlayfs 等场景里，内核经常需要额外确认缓存的目录项是否仍然有效。

这会带来几个高阶现象：

- 同样的路径查找，在本地文件系统和远端文件系统上成本模型可能完全不同
- dentry 命中不一定等于“可以直接相信”，还可能触发额外 revalidation
- 当远端对象已经变化或句柄失效时，应用可能看到 `ESTALE` 这类错误

所以负 dentry、正 dentry、revalidation 三者要连起来理解：

- dcache 负责让“多数情况”更快
- revalidation 负责在“不再稳定”的情况下补正确性
- 远端语义越复杂，路径查找就越像一次缓存协商，而不是单机上的局部查找

### 6.11 `openat2()` 为什么不是语法糖，而是安全边界工具

只要程序需要在“不可信路径输入”下安全地打开文件，传统“先拼路径再 `open()`”往往不够。`openat()` 已经允许你从目录 fd 出发，而 `openat2()` 更进一步，把一些原本需要应用自己小心兜底的约束显式交给内核：

- 限制不得逃出某个目录边界
- 限制不得跟随符号链接
- 限制不得跨挂载点
- 限制解析过程必须留在某个受控根内

这类能力的重要性在于：

- 它让路径安全边界从“应用自己猜测字符串是否安全”转向“内核按真实 path walk 规则执行限制”
- 它能显著降低 symlink race、`../../` 逃逸、bind mount 绕过等问题
- 它提醒我们：路径安全从来不是字符串过滤问题，而是名字空间解释问题

如果你要读源码或做安全工程，第6章真正的升级版理解应该是：

**路径解析既是性能关键路径，也是安全关键路径。**

---

### 6.12 nameidata：路径查找的内核工作结构

Linux 路径查找的状态在 `struct nameidata`（`fs/namei.c`，内部结构）中维护。它贯穿整个 path walk 过程，记录当前解析位置和约束。

```c
/* fs/namei.c（内核内部，不暴露给用户态）*/
struct nameidata {
    struct path     path;           /* 当前解析位置：path.mnt + path.dentry */
    struct qstr     last;           /* 当前待查找的路径组件（名字+哈希）*/
    struct path     root;           /* 路径查找的根（chroot 或 / 或 openat2 的 anchor）*/
    struct inode    *inode;         /* path.dentry->d_inode 的缓存（避免每次解引用）*/
    unsigned int    flags;          /* LOOKUP_FOLLOW（跟随 symlink）、LOOKUP_DIRECTORY 等 */
    unsigned        seq;            /* RCU 序列号（检测 dentry 是否在 RCU 期间被修改）*/
    unsigned        m_seq;          /* mount 序列号（检测 vfsmount 是否被修改）*/
    unsigned        r_seq;          /* root 的 RCU 序列号 */
    int             last_type;      /* LAST_NORM/LAST_DOT/LAST_DOTDOT（路径组件类型）*/
    unsigned        depth;          /* 符号链接嵌套深度（防止无限循环，上限 40）*/
    int             total_link_count; /* 总 symlink 跟随次数（上限 MAXSYMLINKS=40）*/
    struct saved {
        struct path     link;       /* 保存的符号链接 path（进入 symlink 前的状态）*/
        struct delayed_call done;   /* symlink 目标字符串释放回调 */
        const char *    name;       /* 符号链接目标字符串 */
        unsigned        seq;        /* 保存时的 RCU 序列号 */
    } *stack, internal[EMBEDDED_LEVELS]; /* 符号链接嵌套栈 */
    struct filename *name;          /* 从用户态传入的路径字符串（包含内核副本）*/
    struct nameidata *saved;        /* 外层 nameidata（嵌套调用场景）*/
    unsigned        root_seq;
    int             dfd;            /* openat 的目录 fd（AT_FDCWD 或 fd 整数）*/
    kuid_t          dir_uid;        /* 最终目录的 UID（部分权限检查用）*/
    umode_t         dir_mode;       /* 最终目录的 mode */
};
```

**LOOKUP flags 的实际含义**：

```c
/* 常见 LOOKUP flags（影响 path walk 行为）*/
#define LOOKUP_FOLLOW      0x0001  /* 跟随最终组件的 symlink（open 默认开启）*/
#define LOOKUP_DIRECTORY   0x0002  /* 最终结果必须是目录（open(O_DIRECTORY) 用）*/
#define LOOKUP_AUTOMOUNT   0x0004  /* 允许触发 automount */
#define LOOKUP_EMPTY       0x4000  /* 允许空路径（AT_EMPTY_PATH 场景，如 fstatat(fd, "")）*/
#define LOOKUP_DOWN        0x8000  /* 从 dfd 向下（openat2 的 RESOLVE_BENEATH 用）*/
#define LOOKUP_MOUNTPOINT  0x0080  /* 查找挂载点本身（umount 用）*/
#define LOOKUP_REVAL       0x0020  /* 强制 revalidate（NFS 等远端文件系统用）*/
#define LOOKUP_RCU         0x0040  /* 当前处于 RCU 快路径 */
#define LOOKUP_NO_SYMLINKS 0x010000 /* 禁止跟随 symlink（openat2 RESOLVE_NO_SYMLINKS）*/
#define LOOKUP_NO_MAGICLINKS 0x020000 /* 禁止 magic symlink（如 /proc/self/exe）*/
#define LOOKUP_NO_XDEV     0x040000 /* 禁止跨挂载点（openat2 RESOLVE_NO_XDEV）*/
#define LOOKUP_BENEATH     0x080000 /* 必须在 dfd 下方（openat2 RESOLVE_BENEATH）*/
#define LOOKUP_IN_ROOT     0x100000 /* 把 dfd 当作根（openat2 RESOLVE_IN_ROOT）*/
```

---

### 6.13 RCU path walk 与 ref-walk 的内核实现

#### RCU 快路径（LOOKUP_RCU 模式）

RCU path walk 是 Linux 路径查找的默认快路径，由 Al Viro 在 Linux 2.6.38 引入。其核心思路：

```
RCU path walk 的不变量：
  1. 不拿任何 spinlock 或 semaphore
  2. 不增加 dentry/vfsmount 的引用计数
  3. 通过 seqcount（d_seq）检测 dentry 是否在读取期间被修改

代码路径（fs/namei.c）：
  path_init()
    → 初始化 nameidata，从 root 或 cwd 出发
    → 设置 nd->flags |= LOOKUP_RCU
    → rcu_read_lock()（进入 RCU 读临界区）

  link_path_walk() 的每一步：
    → walk_component(nd, &path, WALK_FOLLOW)
          → lookup_fast(nd, &path)
                → __d_lookup_rcu(parent, name, seqp)
                      → 在 dentry 哈希表中查找
                      → 读取 dentry 并记录 d_seq（无锁）
                → 读取 dentry->d_inode
                → read_seqcount_retry(&dentry->d_seq, seq)
                      → 若 seq 变化 → dentry 在读取期间被修改 → 切换到 ref-walk
                → inode_permission_rcu(inode, MAY_EXEC)
                      → 无锁检查权限位
          → 如果是挂载点：__follow_mount_rcu()
                → 读取挂载树，检查 m_seq
  
  退出 RCU 快路径：unlazy_walk() 或 unlazy_child()
    → rcu_read_unlock()
    → 对当前 path 增加引用计数，转入 ref-walk 继续
```

#### 触发回退到 ref-walk 的场景

```
触发 unlazy_walk()（切换到 ref-walk）的场景：
  1. dentry 不在 dcache 中（需要调用 ->lookup() 获取锁）
  2. d_seq 验证失败（dentry 在 RCU 窗口内被修改，如 rename/unlink）
  3. m_seq 验证失败（mount 树在 RCU 窗口内变化）
  4. 遇到符号链接（需要读 symlink 内容，可能分配内存）
  5. dentry 需要 revalidate（d_op->d_revalidate != NULL 且未命中缓存）
  6. LOOKUP_RCU 不支持的操作（如 mkdir、create）
```

#### ref-walk 路径

```
ref-walk 相对于 RCU 的差异：
  → 进入每个 dentry 时增加引用计数（dget）
  → 进入每个 mount 时增加引用计数（mntget）
  → 离开时减少引用计数（dput/mntput）
  → 查找时可能拿 d_lock spinlock

性能对比：
  RCU walk（dcache 全命中，无竞争）：~200ns per path component
  ref-walk（同条件）：~400ns per path component（额外引用计数开销）
  ref-walk with revalidation（NFS）：~1-5ms per path component（网络往返）
```

**用 ftrace 观察 path walk**：

```bash
# 追踪路径查找相关函数
echo 'lookup_fast unlazy_walk link_path_walk' > /sys/kernel/debug/tracing/set_ftrace_filter
echo function > /sys/kernel/debug/tracing/current_tracer
echo 1 > /sys/kernel/debug/tracing/tracing_on

# 触发一次路径查找
stat /var/log/nginx/access.log

echo 0 > /sys/kernel/debug/tracing/tracing_on
cat /sys/kernel/debug/tracing/trace | head -30
# bash-1234  [002] .  1234.567890: lookup_fast <-walk_component
# bash-1234  [002] .  1234.567891: lookup_fast <-walk_component
# bash-1234  [002] .  1234.567892: unlazy_walk <-walk_component  ← RCU → ref-walk 切换
```

---

### 6.14 link_path_walk 逐段执行流程

下面是一次 `/var/log/nginx/access.log` 的路径查找执行流程（简化版）：

```
path_init("/var/log/nginx/access.log", LOOKUP_FOLLOW, nd)
  │
  ├── 绝对路径 → nd->path = current->fs->root（进程根目录）
  ├── 设置 LOOKUP_RCU，rcu_read_lock()
  │
  └── link_path_walk("/var/log/nginx/access.log", nd)
        │
        ├── 跳过开头的 '/'
        │
        ├── ① 处理 "var"（名字哈希：0xABCD）
        │     → may_lookup(nd)：检查当前目录(/)的 MAY_EXEC 权限
        │     → lookup_fast(nd, "var")
        │           → __d_lookup_rcu(/, "var", 0xABCD) → 命中 dcache
        │           → read_seqcount_retry → OK（seq 未变）
        │           → inode_permission_rcu(var_inode, MAY_EXEC) → 通过
        │     → nd->path = {mnt=rootmnt, dentry=var_dentry}
        │
        ├── ② 处理 "log"（名字哈希：0x5678）
        │     → lookup_fast(nd, "log") → 命中 dcache
        │     → nd->path = {mnt=rootmnt, dentry=log_dentry}
        │
        ├── ③ 处理 "nginx"（名字哈希：0x1234）
        │     → lookup_fast(nd, "nginx")
        │     → 假设未命中 dcache（nginx 目录刚创建）：
        │           → unlazy_walk()  ← 切换到 ref-walk
        │           → lookup_slow(nd, "nginx")
        │                 → mutex_lock(log_inode->i_mutex)
        │                 → ext4_lookup(log_inode, "nginx", nd)
        │                       → 读取 log 目录的数据块
        │                       → 在 HTree 中搜索 "nginx"
        │                       → 找到 → iget(inode_num) → 加载 nginx_inode
        │                 → d_add(nginx_dentry, nginx_inode)
        │                 → mutex_unlock(log_inode->i_mutex)
        │
        └── ④ 此时 "access.log" 是最后一段（由调用者的 do_last() 处理）
              → 不在 link_path_walk 中，而在 do_filp_open() 中
              → 处理 O_CREAT、O_EXCL、最终权限检查等
```

**用 strace 验证逐段查找**：

```bash
# strace 观察路径查找引发的系统调用
strace -e trace=openat,stat,getdents64 ls /var/log/nginx/ 2>&1 | head -15
# openat(AT_FDCWD, "/var/log/nginx/", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
# getdents64(3, ..., 32768) = 128    ← 读取目录内容

# 对比 dcache 冷（第一次）和热（第二次）的系统调用延迟
strace -T stat /var/log/nginx/access.log 2>&1 | grep "newfstatat"
# 第一次：newfstatat(AT_FDCWD, "/var/log/nginx/access.log", ...) = 0 <0.000876>
echo 2 > /proc/sys/vm/drop_caches
strace -T stat /var/log/nginx/access.log 2>&1 | grep "newfstatat"
# drop_caches 后：newfstatat(AT_FDCWD, "/var/log/nginx/access.log", ...) = 0 <0.003421>
# → 约 4 倍延迟差异（dcache miss 需要磁盘 I/O）
```

---

### 6.15 openat2() 的安全边界：RESOLVE flags 实现

`openat2()` 是 Linux 5.6 引入的系统调用（`man 2 openat2`），通过 `struct open_how` 传递额外约束：

```c
/* 用户态接口 */
struct open_how {
    __u64 flags;    /* O_RDONLY/O_WRONLY/O_RDWR 等，同 open()*/
    __u64 mode;     /* 创建时的权限位 */
    __u64 resolve;  /* 路径解析约束（RESOLVE_xxx 标志）*/
};

/* RESOLVE flags */
#define RESOLVE_NO_XDEV     0x01  /* 禁止跨挂载点（包括 bind mount）*/
#define RESOLVE_NO_MAGICLINKS 0x02 /* 禁止 magic symlink（/proc/pid/exe 等）*/
#define RESOLVE_NO_SYMLINKS  0x04  /* 禁止所有 symlink 跟随 */
#define RESOLVE_BENEATH      0x08  /* 路径必须在 dfd 目录之下 */
#define RESOLVE_IN_ROOT      0x10  /* 把 dfd 当作根（类似 chroot）*/
#define RESOLVE_CACHED       0x20  /* 只允许命中 dcache，不触发磁盘 I/O */
```

**各 flag 防御的攻击向量**：

```bash
# 场景：Web 服务器提供静态文件服务
# 目录结构：
# /var/www/static/     ← 我们的基目录
#   index.html
#   ../../../etc/passwd ← 恶意路径
#   secret_link -> /etc/shadow  ← 恶意 symlink

# 攻击 1：路径穿越（../../../etc/passwd）
# 纯字符串过滤的问题：攻击者可能绕过（URL 编码、双重编码等）

# 正确做法：使用 openat2() + RESOLVE_BENEATH
python3 -c "
import os, ctypes, ctypes.util

libc = ctypes.CDLL('libc.so.6', use_errno=True)

class OpenHow(ctypes.Structure):
    _fields_ = [
        ('flags', ctypes.c_uint64),
        ('mode', ctypes.c_uint64),
        ('resolve', ctypes.c_uint64),
    ]

SYS_openat2 = 437  # x86_64
RESOLVE_BENEATH = 0x08
RESOLVE_NO_SYMLINKS = 0x04

base_fd = os.open('/var/www/static', os.O_RDONLY | os.O_PATH | os.O_DIRECTORY)

# 测试 1：正常路径
how = OpenHow(flags=os.O_RDONLY, mode=0, resolve=RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)
fd = libc.syscall(SYS_openat2, base_fd, b'index.html', ctypes.byref(how), ctypes.sizeof(how))
print(f'正常路径: fd={fd}')  # 正数 fd

# 测试 2：路径穿越
how = OpenHow(flags=os.O_RDONLY, mode=0, resolve=RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)
fd = libc.syscall(SYS_openat2, base_fd, b'../../../etc/passwd', ctypes.byref(how), ctypes.sizeof(how))
err = ctypes.get_errno()
import errno
print(f'路径穿越: fd={fd} errno={errno.errorcode.get(err, err)}')  # EXDEV 或 EACCES

# 测试 3：恶意 symlink
fd = libc.syscall(SYS_openat2, base_fd, b'secret_link', ctypes.byref(how), ctypes.sizeof(how))
err = ctypes.get_errno()
print(f'恶意 symlink: fd={fd} errno={errno.errorcode.get(err, err)}')  # ELOOP
"

# strace 验证 openat2 与 openat 的系统调用差异
strace -e trace=openat,openat2 python3 -c "
import os
os.open('/tmp/test', os.O_RDONLY)  # 传统 openat
"
# openat(AT_FDCWD, \"/tmp/test\", O_RDONLY) = 3

strace -e trace=openat,openat2 python3 -c "
import ctypes, ctypes.util, os
# 使用 openat2
libc = ctypes.CDLL('libc.so.6')
# ... (同上)
" 2>&1 | grep openat
# openat2(AT_FDCWD, \"index.html\", {flags=O_RDONLY, mode=0, resolve=RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS}, 24) = 3
# openat2(AT_FDCWD, \"../../../etc/passwd\", ...) = -1 EXDEV (Invalid cross-device link)
```

**RESOLVE_BENEATH 内核实现原理**：

```
内核如何实现 RESOLVE_BENEATH：

path_init() 时：
  → 若设置 RESOLVE_BENEATH → 记录基目录的 nd->root = {dfd_dentry, dfd_mount}

link_path_walk() 处理 ".." 时：
  → follow_dotdot_rcu(nd)
        → 普通情况：nd->path = nd->path.dentry->d_parent
        → RESOLVE_BENEATH：若 nd->path.dentry == nd->root.dentry
                            → 返回 -EXDEV（禁止跨越基目录边界）

处理绝对路径 symlink 时：
  → nd->last_type == LAST_ROOT
        → RESOLVE_BENEATH：返回 -EXDEV（绝对路径会逃出 beneath 边界）
        → RESOLVE_IN_ROOT：使用 dfd 作为根，继续解析
```

---

## 常见误区

- 误以为路径解析只看最终文件权限
- 误以为挂载只影响“存储位置”，不影响路径解释
- 误以为软链接只是显示层的别名，实际上它参与解析流程
- 误以为“路径不存在”就一定要落到磁盘查一遍，忽略了负 dentry 的缓存意义

---

## 本章小结

| 主题 | 结论 |
|------|------|
| 路径解析 | 是逐段遍历目录树的过程 |
| 起点 | 可能来自根目录、当前工作目录或目录 fd |
| 目录权限 | 会影响路径能否继续向下解析 |
| 软链接 | 会插入新的路径解释逻辑 |
| 挂载点 | 会改变后续路径所在的文件系统树 |
| dcache / 负 dentry | 是高频查找性能和失败查找性能的关键 |
| RCU path walk / ref-walk | 解释了为什么路径查找有时极轻，有时突然很重 |
| `openat2()` | 把路径约束变成内核可执行的安全边界 |

---

## 练习题

### 基础题

**6.1** 为什么路径解析需要按段逐步进行，而不能一次性把完整路径字符串映射到对象？列举在单次逐段查找中可能触发的三类内核操作。

**6.2** 目录的执行权限（`x` 位）在路径解析中起什么作用？解释为什么”目标文件权限正确但仍然 `EACCES`”的根因往往在中间目录。

### 中级题

**6.3** 比较 RCU path walk 和 ref-walk 的适用场景与触发条件：哪些情况会让 path walk 从快速路径回退到 ref-walk？这对元数据密集型工作负载有什么性能影响？

**6.4** 挂载点和 mount namespace 如何分别影响路径解析？举例说明同一路径字符串在两个不同 mount namespace 中解析到不同 inode 的场景（提示：容器 `/proc` 挂载）。

### 提高题

**6.5** 分析 `openat2()` 相对于”应用层字符串过滤”的安全边界优势：①用 `strace` 观察 `openat2` 与 `open` 在内核调用层面的差异；②描述 `RESOLVE_BENEATH`、`RESOLVE_NO_SYMLINKS`、`RESOLVE_NO_XDEV` 各自防御的攻击向量；③设计一个 Web 服务器静态文件服务场景，说明如何结合这三个 flag 防御路径穿越攻击，并指出字符串过滤方案为何在 symlink race 下仍然脆弱。

---

## 练习答案

**6.1** 按段查找原因：每段组件都是独立的名字-inode 映射查找，中途可能遇到挂载点切换（不同文件系统）、symlink 插入（新路径重新解析）、权限检查（目录 `x` 位）。单次查找可能触发：①dcache 查找（命中则返回 dentry）；②底层文件系统 `->d_revalidate` / `->lookup`（缓存未命中或需验证）；③挂载穿越（`__follow_mount`）。

**6.2** 目录 `x` 权限控制该目录是否可被 path walk 穿越（`inode_permission(dir, MAY_EXEC)`）。若中间某目录 `x` 位为 0，path walk 在该层停止，返回 `EACCES`，与目标文件权限无关。典型场景：`/data` 权限 `750`，other 用户访问 `/data/pub/file.txt`，`/data` 无 `x` for other，解析止步于此。

**6.3** RCU path walk：无锁读取稳定 dcache 对象，适合缓存命中且目录树稳定的高频查找（大多数情况）。触发回退到 ref-walk：遇到 symlink（需重新解析）、dentry 需要 revalidation（NFS/FUSE）、遇到 mount crossing、rename/unlink 竞争导致 dentry 无效。性能影响：元数据热点（大量文件创建/删除/重命名）使 dcache 频繁失效，RCU 快路径不断回退到 ref-walk，延迟陡增，这就是元数据 IOPS 瓶颈的典型根因。

**6.4** 挂载点：path walk 到达挂载点时调用 `__follow_mount`，切换到该挂载点对应的 superblock/root dentry，后续路径在新文件系统上解析。mount namespace：不同 namespace 维护不同挂载树（`struct mnt_namespace`），同一路径字符串在 namespace A 中可能解析到宿主 `/proc`（显示宿主 PID），在容器 namespace B 中解析到容器自己的 `procfs`（显示容器 PID），完全不同的 inode。

**6.5** ①`strace`：`open()` 调用 `openat(AT_FDCWD, path, flags)`；`openat2()` 传入 `struct open_how`，内核在 `do_filp_open()` 中按 `resolve` 字段约束 path walk 行为。②`RESOLVE_BENEATH`：防止路径通过 `..` 或绝对 symlink 逃出基目录；`RESOLVE_NO_SYMLINKS`：拒绝任何 symlink 跟随，防御 symlink race；`RESOLVE_NO_XDEV`：禁止跨挂载点，防止 bind mount 绑过边界。③Web 服务设计：`base_fd = open(“/var/www/static”, O_PATH|O_DIRECTORY)`；对每个请求 `openat2(base_fd, user_path, &{.flags=O_RDONLY, .resolve=RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS})`。字符串过滤脆弱性：攻击者可先建一个合法路径的 symlink，在服务 stat 检查（TOCTOU 窗口）和 open 之间切换目标，字符串过滤完全无法检测。

---

## 延伸阅读

1. **Linux 内核源码**. `fs/namei.c` — `path_lookupat()`、`link_path_walk()`、`do_filp_open()` 实现
2. **man 2 openat2** — `struct open_how` 及 `RESOLVE_*` flag 完整文档
3. **LWN.net**. *Pathname lookup in Linux* (Al Viro) — RCU path walk 设计深度解析
4. **Corbet, Rubini & Kroah-Hartman**. *Linux Device Drivers*, Ch.14 — VFS 与文件系统接口

---

[← 上一章：文件描述符与 open file](./05-file-descriptors-and-open-files.md)

[下一章：块、超级块与空间分配 →](../part3-layout-and-mount/07-blocks-superblocks-and-allocation.md)

[返回目录](../README.md)
