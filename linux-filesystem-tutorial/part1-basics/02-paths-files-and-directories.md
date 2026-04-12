# 第2章：路径、文件与目录

> **本章导读**：路径不是文件本体，而是一次名字解析协议；目录也不是”盒子”，而是可被挂载、缓存、权限和命名空间共同作用的名字映射层。本章梳理路径起点、目录本质、路径与对象的多对多关系，以及 TOCTOU 和 symlink race 等安全边界问题。

**前置知识**：第1章（文件系统基本概念）
**预计学习时间**：40 分钟

---

## 学习目标

完成本章后，你将能够：

1. 区分绝对路径、相对路径、当前工作目录、dirfd 起点等不同解析上下文
2. 理解目录本质上是名字到对象标识的映射结构
3. 说明为什么同一对象可以有多个名字，而同一路径在不同视图下也可能得到不同结果
4. 认识 `.`、`..`、路径规范化、路径遍历与安全边界的关系
5. 为后续 path walk、mount namespace、symlink race 章节打下语义基础

---

## 正文内容

### 2.1 路径是“如何到达对象”的描述

路径是从某个起点到目标对象的一串名字。例如：

```text
/etc/hosts
./notes/today.md
../bin/python
```

这几条字符串共同点是：它们都不是文件本体，而是告诉系统“你该如何走到那里”。

所以路径至少包含两层信息：

- 从哪里开始走
- 中途经过哪些名字组件

只要这两层中任意一层变了，最终结果都可能不同。

### 2.2 绝对路径、相对路径和 dirfd 起点

常见起点有三类：

- **绝对路径**：从当前名字空间根目录开始
- **相对路径**：从当前工作目录开始
- **基于目录 fd 的路径**：从某个已打开目录开始，例如 `openat()` 族系统调用

这第三类很关键，因为它说明路径不只是“字符串”，还是一个**调用上下文问题**。很多安全方案和容器工具都刻意使用 dirfd 来减少路径竞争和遍历逃逸问题。

### 2.3 当前工作目录是进程状态，不是磁盘状态

执行：

```bash
pwd
cd /var/log
```

改变的是进程后续相对路径的解释方式，而不是磁盘结构本身。所以当前工作目录属于：

- 进程运行时上下文
- 路径解析起点
- 与名字空间、`chroot`、mount namespace 一起作用的视图状态

把当前工作目录误当成“磁盘上有个东西变了”，会让人很难理解容器和多进程路径行为。

### 2.4 目录首先是名字映射，不是内容容器

目录不是“装文件的盒子”。更准确地说，目录是一个**名字到对象标识的映射表**。这意味着：

- 目录项描述的是“在这个父目录下，这个名字指向谁”
- 文件对象本体通常由 inode 等元数据结构表示
- 删除目录项不等于对象立刻消失
- 重命名多数时候主要改的是目录项关系，而不是文件内容

这个视角能解释很多看似反直觉的现象，比如同 inode 多硬链接、rename 原子替换、deleted-but-open 等。

### 2.5 为什么“文件”和“路径”不是一回事

一个对象可能有多个名字入口：

- 硬链接让多个目录项指向同一 inode
- bind mount 让一棵对象树在多个路径出现
- overlayfs merged 视图可能让下层对象通过新路径呈现

反过来，同一个路径字符串在不同环境下也可能落到不同对象：

- 当前工作目录不同
- mount namespace 不同
- 中途某个组件是符号链接
- 某个目录是挂载点

所以“路径”与“对象”是多对多关系，而不是一一对应关系。

### 2.6 `.`、`..` 与路径规范化为什么不是字符串小题目

`./a`、`../b` 看起来像简单字符串，但涉及：

- 当前目录语义
- 父目录边界
- 根目录截断规则
- mount namespace 与 `chroot` 边界
- 符号链接插入后的重新解释

安全上，路径规范化尤其敏感。因为“把字符串里的 `../` 去掉”不等于真的理解了内核如何走路径。很多路径穿越漏洞，本质上就是把字符串处理误当成了真实路径解析。

### 2.7 目录权限为什么会改变路径结果

即使目标文件权限看起来没问题，路径中间的目录如果没有执行/遍历权限，查找也可能失败。因此路径权限检查是**沿途发生**的，而不是等走到终点才看。

这也是为什么：

- `Permission denied` 常常不是目标文件权限错了
- 目录 `x` 权限对路径可达性极其关键
- 安全边界经常建立在目录遍历能力上，而不只是文件读写位上

### 2.8 路径解析与安全：TOCTOU 和 symlink race

如果程序先 `stat(path)` 再 `open(path)`，中间别的进程可能已经把路径替换成另一个对象。这就是典型的 TOCTOU（time-of-check to time-of-use）问题。

符号链接竞争也是类似问题：

- 程序以为打开的是安全目录里的文件
- 实际上路径中某个组件被替换成了 symlink
- 最终访问了不该访问的位置

所以高阶路径处理常常会用：

- `openat()` / `openat2()`
- 目录 fd 作为锚点
- 不跟随符号链接的 flag
- 更严格的解析约束

### 2.9 路径和挂载树一起决定最终结果

路径不仅沿目录树走，还会在挂载点处切换到另一个文件系统实例。于是同一个字符串路径，真正结果取决于：

1. 当前工作目录 / 根目录
2. 中间目录项映射
3. 挂载点切换
4. 符号链接重新解释
5. 当前名字空间视图

这也是为什么单纯看字符串，很难准确推断路径最终落点。

---

### 2.10 `struct nameidata`：内核路径解析的核心状态机

路径解析在内核里由 `path_lookupat()` / `link_path_walk()` 驱动，核心状态保存在 `struct nameidata` 中：

```c
/* fs/namei.c — 路径解析上下文（简化版，实际字段更多）*/
struct nameidata {
    struct path     path;           /* 当前解析到的位置（dentry + vfsmount）*/
    struct qstr     last;           /* 最后一个路径分量（名字 + hash）*/
    struct path     root;           /* 解析起点（根目录或 dirfd 对应目录）*/
    struct inode    *inode;         /* path.dentry->d_inode 的缓存 */
    unsigned int    flags;          /* LOOKUP_* 标志集合 */
    unsigned        seq;            /* RCU 序列号（用于 RCU path walk）*/
    unsigned        m_seq;          /* mount 树序列号 */
    int             last_type;      /* LAST_NORM / LAST_ROOT / LAST_DOT / LAST_DOTDOT */
    unsigned        depth;          /* 符号链接嵌套深度（最深 40 层）*/
    int             total_link_count; /* 全局符号链接跟随计数（防止循环）*/
    struct saved    *stack;         /* 符号链接状态保存栈 */
    /* ... */
};

/* LOOKUP 标志决定解析行为 */
#define LOOKUP_FOLLOW       0x0001  /* 跟随最终分量的符号链接 */
#define LOOKUP_DIRECTORY    0x0002  /* 要求目标必须是目录 */
#define LOOKUP_AUTOMOUNT    0x0004  /* 触发 autofs 自动挂载 */
#define LOOKUP_EMPTY        0x4000  /* 允许空路径（O_PATH 等场景）*/
#define LOOKUP_NO_SYMLINKS  0x0020  /* 拒绝跟随任何符号链接 */
#define LOOKUP_BENEATH      0x0080  /* 不允许解析超出起点目录 */
#define LOOKUP_IN_ROOT      0x0100  /* 把解析起点当作进程的 / */
```

**路径解析主循环（简化版）**：

```
link_path_walk("/etc/hosts", nd):

  1. 解析绝对路径起点
     path_init(nd) → nd->path = 当前 namespace 的根 dentry + vfsmount

  2. 循环逐分量处理（直到路径耗尽）:
     每次迭代：从 nd->path 出发，查找下一个分量

     分量 = "etc":
       a. 计算名字 hash → qstr
       b. __d_lookup_rcu(nd->path.dentry, &last)
          → 在 dcache hash 表中查找（RCU 无锁路径）
          → 命中 → 验证 dentry 仍有效（seqcount 检查）
          → 未命中 → 降级到 ref-walk（获锁，调用 .lookup()）
       c. 检查 nd->path 是否是挂载点
          → 如果是 → 切换到新文件系统的根 dentry

     分量 = "hosts":
       a. 同上，查找 dentry
       b. last_type = LAST_NORM（普通最终分量）
       c. 如果是 O_CREAT → 执行 create 路径
       d. 如果是只读 open → 直接获取 inode

  3. 解析完成，nd->path 指向目标 dentry
```

### 2.11 RCU path walk：为什么路径解析可以不加锁

早期内核每个路径分量解析都需要加锁，高并发时 dcache 锁是性能瓶颈。Linux 2.6.38+ 引入了 RCU-walk（无锁路径解析），原理：

```
RCU path walk 的核心：seqlock + RCU 引用

dcache 的每个 dentry 有 d_seq（sequence counter）：
  写操作（rename, invalidate）：先 write_seqcount_begin() 递增 seq 的奇数部分
  读操作（path walk）：记录 seq 值，操作完成后 read_seqcount_retry() 检查

路径解析步骤：
  1. rcu_read_lock()           ← 进入 RCU 临界区（禁止 GC 回收 dentry）
  2. 记录 nd->seq = dentry->d_seq
  3. 查找子 dentry（__d_lookup_rcu）
  4. 使用找到的 dentry（读取 d_inode、d_name 等）
  5. 检查 nd->seq 是否和 dentry->d_seq 匹配
     → 匹配：读取有效，继续下一分量
     → 不匹配：有并发写（rename/invalidate 在进行），降级到 ref-walk

降级条件（从 RCU-walk 退出到 ref-walk）：
  - dentry 未在 RCU 期间保持有效
  - 遇到符号链接（跟随时需要 refcount）
  - 遇到特殊挂载点
  - 遇到需要调用 .permission() 的情况（某些 MAC 实现）
```

```bash
# 用 perf 观察路径解析的 RCU 效率
# 低比例的 ref-walk = RCU walk 命中率高
perf stat -e \
    'fs:do_sys_openat2,fs:lookup_fast,fs:lookup_slow' \
    -- find /usr/lib -name "*.so" -count 2>/dev/null

# 用 bpftrace 实时追踪路径解析降级（RCU → ref-walk）
bpftrace -e '
kprobe:unlazy_walk {
    @unlazy_count = count();  # 每次从 RCU 降级到 ref-walk
}
interval:s:5 {
    print(@unlazy_count);
    clear(@unlazy_count);
}' -- sleep 30
```

### 2.12 `openat2()`：比 `open()` 更安全的路径解析

Linux 5.6 引入 `openat2()`，通过 `struct open_how` 提供精细的路径解析约束：

```c
/* include/uapi/linux/openat2.h */
struct open_how {
    __u64 flags;      /* O_RDONLY, O_CREAT, O_NOFOLLOW 等（与 open 相同）*/
    __u64 mode;       /* 创建模式（只有 O_CREAT/O_TMPFILE 时有效）*/
    __u64 resolve;    /* 新增：路径解析约束（RESOLVE_* 标志集合）*/
};

/* resolve 标志 — 这是 openat2 的核心创新 */
#define RESOLVE_NO_XDEV      0x01  /* 禁止跨挂载点（不允许路径跨越文件系统边界）*/
#define RESOLVE_NO_MAGICLINKS 0x02 /* 禁止跟随 /proc/pid/fd/N 类 magic symlink */
#define RESOLVE_NO_SYMLINKS  0x04  /* 禁止跟随任何符号链接 */
#define RESOLVE_BENEATH      0x08  /* 解析结果必须在 dirfd 所在目录树内 */
#define RESOLVE_IN_ROOT      0x10  /* 把 dirfd 当作整个解析的根（类似 chroot）*/
#define RESOLVE_CACHED       0x20  /* 只使用 dcache，不触发慢路径（磁盘 I/O）*/
```

**完整的 `openat2()` 使用示例**：

```c
/* 安全地打开用户提供的相对路径，限制在指定目录内 */
#include <linux/openat2.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* openat2 没有 glibc 封装，需要直接调用 */
static int openat2_syscall(int dirfd, const char *pathname,
                            struct open_how *how, size_t size) {
    return syscall(__NR_openat2, dirfd, pathname, how, size);
}

int safe_open_beneath(int base_dirfd, const char *user_path) {
    struct open_how how = {
        .flags   = O_RDONLY | O_CLOEXEC,
        .mode    = 0,
        .resolve = RESOLVE_BENEATH       /* 不允许逃出 base_dirfd 目录树 */
                 | RESOLVE_NO_SYMLINKS   /* 不跟随任何符号链接 */
                 | RESOLVE_NO_MAGICLINKS /* 不跟随 /proc/xxx 类魔法链接 */
                 | RESOLVE_NO_XDEV,     /* 不允许跨挂载点 */
    };

    int fd = openat2_syscall(base_dirfd, user_path, &how, sizeof(how));
    if (fd < 0) {
        if (errno == EXDEV) {
            /* EXDEV: 路径逃出了 base_dirfd 范围，或跨越了挂载点 */
            fprintf(stderr, "Path escape attempt detected: %s\n", user_path);
        } else {
            perror("openat2 failed");
        }
        return -1;
    }
    return fd;
}

int main(void) {
    /* 打开受信任的基础目录 */
    int base_fd = open("/var/www/static", O_PATH | O_DIRECTORY);
    if (base_fd < 0) { perror("open base_dir"); return 1; }

    /* 尝试各种攻击路径 */
    const char *paths[] = {
        "index.html",           /* 正常 → 成功 */
        "../../../etc/passwd",  /* 目录穿越 → EXDEV */
        "link_to_etc",          /* 符号链接 → ELOOP（RESOLVE_NO_SYMLINKS）*/
        "subdir/../../secret",  /* 相对逃逸 → EXDEV */
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        int fd = safe_open_beneath(base_fd, paths[i]);
        if (fd >= 0) {
            printf("Opened: %s (fd=%d)\n", paths[i], fd);
            close(fd);
        }
    }

    close(base_fd);
    return 0;
}

/* 编译运行:
   gcc -o safe_open safe_open.c && ./safe_open
   输出:
   Opened: index.html (fd=5)
   Path escape attempt detected: ../../../etc/passwd
   openat2 failed: Too many levels of symbolic links   (ELOOP)
   Path escape attempt detected: subdir/../../secret
*/
```

**Python 封装（适合脚本场景）**：

```python
import os, ctypes, struct, errno

# openat2 系统调用号（x86_64 = 437）
__NR_openat2 = 437

# open_how 结构体布局
class OpenHow(ctypes.Structure):
    _fields_ = [
        ("flags",   ctypes.c_uint64),
        ("mode",    ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]

RESOLVE_NO_XDEV      = 0x01
RESOLVE_NO_MAGICLINKS= 0x02
RESOLVE_NO_SYMLINKS  = 0x04
RESOLVE_BENEATH      = 0x08
RESOLVE_IN_ROOT      = 0x10

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long

def openat2(dirfd: int, path: str, flags: int,
            resolve: int = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS) -> int:
    how = OpenHow(flags=flags, mode=0, resolve=resolve)
    fd = libc.syscall(__NR_openat2,
                      ctypes.c_int(dirfd),
                      path.encode(),
                      ctypes.byref(how),
                      ctypes.c_size_t(ctypes.sizeof(how)))
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), path)
    return fd

# 演示
base_fd = os.open("/tmp", os.O_PATH | os.O_DIRECTORY)

for test_path in ["safe_file.txt", "../etc/passwd", "link_to_root"]:
    try:
        fd = openat2(base_fd, test_path, os.O_RDONLY)
        print(f"OK:    {test_path} -> fd={fd}")
        os.close(fd)
    except OSError as e:
        print(f"BLOCK: {test_path} -> {e.strerror} (errno={e.errno})")

os.close(base_fd)
```

### 2.13 TOCTOU 实战：从竞争条件到利用

TOCTOU（time-of-check to time-of-use）是路径解析中最常见的安全漏洞类型：

```bash
# 演示 TOCTOU 竞争窗口（仅用于理解，不要在生产系统上运行）

# 场景：程序先 access() 检查，再 open()，中间窗口可被利用
# 有漏洞的伪代码：
cat << 'EOF' > /tmp/vuln_open.py
import os, time

def check_and_open(path):
    # 检查：用 access() 验证路径是否安全
    if os.access(path, os.R_OK):
        time.sleep(0.001)  # 这段延迟模拟真实程序的处理时间
        # ← 攻击者在这 1ms 里把 path 换成 /etc/shadow 的符号链接！
        with open(path) as f:  # 实际打开：此时 path 已经被替换
            return f.read()
    return None

result = check_and_open("/tmp/userfile.txt")
print(result[:100] if result else "Failed")
EOF

# 正确做法：用 openat2 的 RESOLVE_BENEATH 消除 TOCTOU
# 路径从 dirfd 出发，一次解析到底，不分两步
cat << 'EOF' > /tmp/safe_open.py
import os

def safe_check_and_open(base_fd, rel_path):
    """用 openat2 一步完成检查和打开，不存在 TOCTOU 窗口"""
    try:
        # 单次原子操作：path walk + 权限检查 + open
        fd = openat2(base_fd, rel_path, os.O_RDONLY,
                     resolve=RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)
        with os.fdopen(fd) as f:
            return f.read()
    except OSError:
        return None
EOF
```

```bash
# 在真实系统上用 strace 观察 path walk 的分量解析过程
strace -e trace=openat,stat,getdents64 -f ls /etc 2>&1 | head -20
# openat(AT_FDCWD, "/etc", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
# getdents64(3, /* 67 entries */, 32768) = 2312
# ← /etc 目录的内容被批量读取（不是一个 entry 一个 openat）

# 观察真实路径解析的逐分量过程（内核态 tracepoint）
bpftrace -e '
tracepoint:fs:do_sys_openat2 {
    printf("openat2: %s (pid=%d)\n", str(args->filename), pid);
}' -- sleep 5 &

# 同时在另一个终端做一些文件操作
ls /var/log/syslog
cat /etc/hostname
```

---

## 实践观察

可以在终端观察：

```bash
pwd
ls -a
readlink /proc/self/cwd
findmnt .
```

这些命令可以帮助你同时感受：

- 当前工作目录
- 当前路径对应对象
- 当前路径处于哪种挂载视图

---

## 常见误区

- 误以为路径就是文件本身
- 误以为目录只是“内容容器”，不是名字映射层
- 误以为绝对路径在所有名字空间里都一定一样
- 误以为路径规范化只是字符串处理
- 误以为安全问题只发生在文件权限，而不是路径解析过程本身

---

## 本章小结

| 主题 | 结论 |
|------|------|
| 路径 | 是名字解析请求，不是对象本体 |
| 起点 | 可能来自根目录、工作目录或目录 fd |
| 目录 | 首先是名字映射层 |
| 路径与对象 | 不是一一对应关系 |
| 路径规范化 | 涉及真实解析与安全边界，不只是字符串处理 |
| 路径结果 | 同时受目录树、挂载树、符号链接和名字空间影响 |

---

## 练习题

### 基础题

**2.1** 为什么路径应被理解为”解析协议”而不是”对象身份证”？举例说明同一路径字符串在哪些情况下会解析到不同对象。

**2.2** `open(path)` 和 `openat(dirfd, path)` 的语义差别在哪里？在什么场景下后者更安全？

### 中级题

**2.3** 为什么目录不是”装文件的盒子”？用 `rename`、硬链接、deleted-but-open 各举一例说明”名字映射层”的含义。

**2.4** 为什么单纯字符串规范化（如去掉 `../`）无法彻底解决路径遍历问题？举出内核路径解析中至少两种字符串处理无法模拟的行为。

### 提高题

**2.5** 设计一个安全的”受限文件访问”方案：用户提交相对路径，程序只允许访问指定目录树内的文件。要求：不依赖字符串过滤，说明如何利用 `openat2()` 的 `RESOLVE_BENEATH` flag 以及目录 fd 锚点来实现，并分析该方案能防御哪些攻击向量（TOCTOU、symlink race、mount point 穿越）。

---

## 练习答案

**2.1** 路径是”如何到达对象”的描述，包含起点（根目录、cwd、dirfd）和路径组件序列，任一层变化都可能得到不同对象。典型情形：①不同 mount namespace 下 `/proc/1/fd` 指向不同文件；②`chroot` 后绝对路径被截断；③路径中某段是 symlink 时跟随目标改变。

**2.2** `open(path)` 以进程 cwd 或根目录为起点，存在 TOCTOU 窗口；`openat(dirfd, path)` 以已打开目录 fd 为锚点，路径解析相对于该 fd，避免 cwd 改变或 symlink race 影响。场景：Web 服务器限制文件访问在 `static/` 目录下，用 `openat(static_fd, user_path, ...)` 替代 `open(“/var/www/static/” + user_path)`。

**2.3** 名字映射层示例：①`rename(“a”,”b”)` 只改目录项中的名字-inode 映射，inode 和数据块不动；②硬链接使两个目录项指向同一 inode，删一个名字对象仍存在；③`unlink` 后进程仍持有 fd（deleted-but-open），目录项消失但 inode 引用计数 > 0。

**2.4** 字符串 `../` 去掉只是字面替换，无法处理：①symlink——路径中某段是 symlink，跟随后可能逃出预期目录；②mount point——目录是挂载点，文件系统树在此切换；③RCU path walk 并发下目录树的实时变化；④`..` 在根目录下的截断语义（不能越过 `chroot` 边界）。

**2.5** 方案：用 `open(base_dir, O_PATH|O_DIRECTORY)` 获取 base_fd，再调用 `openat2(base_fd, user_path, &how, sizeof(how))`，其中 `how.resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS`。防御：①TOCTOU——fd 锚点固定，期间其他进程修改 cwd 无效；②symlink race——`RESOLVE_NO_SYMLINKS` 拒绝跟随 symlink；③mount point 穿越——`RESOLVE_BENEATH` 限制不得越过 base_fd 所在目录树根。

---

## 延伸阅读

1. **man 2 openat2** — Linux 5.6+ 新增的受限路径解析 API
2. **LWN.net**. *Filesystem path lookup* series — 路径查找机制深度解析
3. **Kerrisk**. *The Linux Programming Interface*, Ch.18 — 目录与链接
4. **CWE-22** — Path Traversal 漏洞分类与防御模式

---

[← 上一章：为什么需要文件系统](./01-why-filesystems.md)

[下一章：权限、元数据与链接 →](./03-permissions-metadata-and-links.md)

[返回目录](../README.md)
