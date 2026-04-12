# 第5章：文件描述符与 open file

> **本章导读**：文件描述符不是文件本身，甚至也不是”打开文件”的全部；真正的关键对象是 open file description，它把路径解析结果、运行时状态和后续 I/O 语义接了起来。本章梳理 fd / open file description / inode 三层的共享关系，以及 `dup`、`fork`、`CLOEXEC`、`O_APPEND` 等操作背后的语义根因。

**前置知识**：第4章（inode、dentry 对象模型）
**预计学习时间**：45 分钟

---

## 学习目标

完成本章后，你将能够：

1. 区分 fd、进程 fd 表、open file description、inode 各自处于哪一层
2. 解释 `open`、`dup`、`fork`、`exec` 如何改变共享关系
3. 理解偏移量、flag、`CLOEXEC`、文件锁等为什么属于 open file 层
4. 说明 deleted-but-open、日志轮转、`O_PATH`、`openat2` 等真实场景的语义根因
5. 建立“路径解析完成后，I/O 主要围绕 open file 展开”的视角

---

## 正文内容

### 5.1 fd 只是进程局部索引

文件描述符（fd）首先是进程文件描述符表里的一个整数槽位。它告诉进程：“去这个槽位看，你当前持有哪个 open file description 引用。”

所以 fd 本身不保存：

- 文件内容
- inode 元数据
- 路径字符串
- 偏移量本体

它只是本进程局部的一层索引。

### 5.2 open file description 才是运行时语义核心

执行 `open()` 后，内核通常会：

1. 解析路径，拿到目标对象
2. 创建或引用一个 open file description
3. 把这个 open file description 的引用挂到进程 fd 表某个槽位

这个 open file description 常常保存：

- 当前文件偏移量
- 打开标志（如 `O_APPEND`、`O_NONBLOCK`）
- 与具体 file operations 的绑定
- 锁、通知、异步 I/O 上下文
- 对底层 inode / mount / path 的引用关系

所以很多语义问题真正应问的是：**这些 fd 是不是共享同一个 open file description？**

### 5.3 `open`、`dup`、`fork` 的共享关系完全不同

| 操作 | 会发生什么 | 偏移量是否共享 |
|------|------------|----------------|
| 再次 `open(path)` | 通常创建新的 open file description | 通常不共享 |
| `dup(fd)` / `dup2(fd)` | 新 fd 指向同一个 open file description | 共享 |
| `fork()` | 子进程继承父进程 fd 表中的引用 | 通常共享 |

这解释了为什么：

- 同一路径两次 `open` 的读写偏移可能互不影响
- `dup` 后一个 fd `lseek`，另一个也受影响
- 父子进程可能“莫名其妙共享文件位置”

### 5.4 `exec` 与 `CLOEXEC` 为什么关键

`fork` 之后如果紧接着 `exec`，新程序默认可能继续继承原来的 fd。这既可能是功能需要，也可能是严重泄漏。

`O_CLOEXEC` / `FD_CLOEXEC` 的意义就在于：

- 让 fd 在 `exec` 时自动关闭
- 避免子进程意外继承不该看到的文件、socket、管道、目录 fd
- 减少安全问题和资源泄漏

这说明文件描述符不只是 I/O 句柄，也是跨进程边界的重要资源对象。

### 5.5 偏移量属于 open file description，不属于 inode

很多人第一次知道这一点时会突然理解很多现象。偏移量通常不是存在 inode 里，而是在 open file description 里。

因此：

- 同 inode 的不同 open file description 可以有不同偏移
- 共享同一个 open file description 的 fd 会看到同一个偏移变化
- `pread` / `pwrite` 之类接口的重要性就在于它们显式给偏移，不依赖共享状态

### 5.6 `O_APPEND`、`O_PATH`、`O_DIRECT` 为什么属于打开语义

同一个 inode，可以因为打开 flag 不同而表现出不同运行时语义：

- `O_APPEND`：写入前总是定位到当前文件尾
- `O_PATH`：只拿路径引用，不拿普通读写语义
- `O_DIRECT`：尝试绕过 page cache
- `O_NONBLOCK`：对某些对象影响阻塞行为

这说明“打开方式”本身就是语义的一部分，而不只是对象属性。

### 5.7 deleted-but-open 的根因到底是什么

执行 `unlink` 时，删掉的是目录项名字关系。只要 open file description 还活着，对象就仍然可能：

- 被继续读写
- 占据空间
- 在 `df` 里体现占用
- 在目录树里却已经不可见

这就是 deleted-but-open。它不是怪现象，而是“名字层”和“打开层”分离的正常结果。

### 5.8 日志轮转为什么经常踩这个坑

日志轮转的经典坑：

1. 服务启动时 `open("app.log")`
2. 轮转程序把 `app.log` rename 到旧名字
3. 再创建新的 `app.log`
4. 服务没重开日志 fd，继续往旧 inode 写

于是：

- 新文件不增长
- deleted-but-open 占空间
- 你在目录树里找不到真正正在增长的日志对象

这说明运维问题往往不是“文件系统坏了”，而是 open file 语义没被理解。

### 5.9 `openat2` 这类接口为什么重要

现代 Linux 提供 `openat2` 等更严格的路径打开接口，可以约束：

- 是否允许跟随 symlink
- 是否允许跨越某些边界
- 是否必须在某个目录锚点之下解析

它的意义在于：把“路径解析安全策略”纳入打开协议，而不只是用字符串处理来赌。

### 5.10 file locks 为什么也要区分层次

文件锁并不总是“锁 inode”。有些语义更接近：

- 锁某个 open file description
- 锁某个进程持有的 fd 关系
- 锁某个 inode 范围

不同锁模型的释放时机、`dup`/`fork` 继承语义、跨进程行为都会不同。这再次说明 open file description 是理解运行时行为的关键层。

### 5.11 进程 fd 表的内核实现：struct files_struct 与 fdtable

进程的文件描述符表不是一个简单数组，而是一套支持动态扩展和 RCU 并发访问的结构。

```c
/* include/linux/fdtable.h */
struct fdtable {
    unsigned int max_fds;           /* 当前表的最大 fd 数 */
    struct file __rcu **fd;         /* fd → file 指针数组（RCU 保护）*/
    unsigned long *close_on_exec;   /* 位图：哪些 fd 设置了 CLOEXEC */
    unsigned long *open_fds;        /* 位图：哪些 fd 槽位已分配 */
    unsigned long *full_fds_bits;   /* 加速查找空槽的二级位图 */
    struct rcu_head rcu;
};

struct files_struct {
    atomic_t count;                 /* 引用计数（fork 时共享 files_struct）*/
    bool resize_in_progress;
    wait_queue_head_t resize_wait;
    struct fdtable __rcu *fdt;      /* 当前使用的 fdtable（RCU 保护）*/
    struct fdtable fdtab;           /* 内嵌的初始 fdtable（进程初始有 NR_OPEN_DEFAULT=64 个槽位）*/
    spinlock_t file_lock;           /* 保护 fd 分配/释放 */
    unsigned int next_fd;           /* 下次分配从这里开始找空槽（加速 open() 的 fd 分配）*/
    unsigned long close_on_exec_init[1];    /* 初始 CLOEXEC 位图 */
    unsigned long open_fds_init[1];         /* 初始已分配位图 */
    unsigned long full_fds_bits_init[1];
    struct file __rcu *fd_array[NR_OPEN_DEFAULT];  /* 初始 fd 指针数组（64个槽）*/
};
```

**工程含义**：

- 进程启动时有内嵌的 64 个 fd 槽位（`fd_array`），不需要额外分配，避免堆分配开销。
- fd 数超过 64 时，内核分配更大的 `fdtable`，用 RCU 发布新表，旧表在 grace period 后释放。
- `open_fds` 位图 + `full_fds_bits` 二级位图让 `open()` 找到空槽只需 O(log N) 时间，而不是线性扫描。
- `close_on_exec` 位图记录哪些 fd 有 `O_CLOEXEC` 标志，`execve()` 时只需扫描此位图批量关闭，无需逐个检查每个 fd。

**查看进程的 fd 表统计**：

```bash
PID=<目标进程 PID>

# 查看进程打开的所有 fd
ls -la /proc/$PID/fd/ | head -20

# 查看 fd 上限
cat /proc/$PID/limits | grep "open files"
# Limit                     Soft Limit  Hard Limit  Units
# Max open files            1048576     1048576     files

# 查看当前 fd 数量
ls /proc/$PID/fd/ | wc -l

# 全系统当前打开 file 数
cat /proc/sys/fs/file-nr
# 例：14848  0  9223372036854775807
# 字段：已分配 file 数 / 未使用（始终为0）/ 最大允许 file 数

# 修改单进程 fd 上限
ulimit -n 65536      # 只影响当前 shell 及其子进程
# 或永久设置：/etc/security/limits.conf
```

**fork 时 fd 表的共享机制**：

```c
/* fork() 后的两种模式（由 CLONE_FILES flag 决定）*/

/* 模式1：默认 fork()（不设 CLONE_FILES）*/
/* → 复制 files_struct，父子各有独立 fd 表 */
/* → 但 fd 表里的 file 指针相同（f_count 增加）*/
/* → 父子共享同一组 open file description（偏移共享！）*/

/* 模式2：线程（设 CLONE_FILES）*/
/* → 父子共享同一 files_struct（count 增加）*/
/* → 一方 open/close fd 对另一方立即可见 */
```

所以 `fork()` 后父子进程拥有**独立的 fd 表**（各自可以 open/close 不影响对方），但共享**同一组 open file description**（偏移和 flag 共享）——这是很多"fork 后父子同时写同一文件"产生竞争的根本原因。

---

### 5.12 O_CLOEXEC 的实现机制与竞争窗口

`O_CLOEXEC` 的工程背景来自一个经典 TOCTOU（time-of-check-time-of-use）漏洞场景。

**传统双步操作的竞争窗口**：

```
进程A：open("socket")  → 返回 fd=5（无 CLOEXEC）
  ↓ 此时另一个线程调用 fork+exec
  ↓ exec 前的 close-on-exec 检查：fd=5 未设标志 → 不关闭
进程A的子进程：意外继承了 fd=5（socket），造成 fd 泄漏
进程A：后来才调用 fcntl(fd, F_SETFD, FD_CLOEXEC)  ← 已经太晚
```

**正确做法：原子设置 CLOEXEC**：

```bash
# 方式1：open() 时直接带 O_CLOEXEC（原子）
open("file", O_RDONLY | O_CLOEXEC)

# 方式2：socket() 时带 SOCK_CLOEXEC（原子）
socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0)

# 方式3：accept4() 带 SOCK_CLOEXEC（原子）
accept4(listen_fd, addr, addrlen, SOCK_CLOEXEC)

# 不推荐：两步操作（有竞争窗口）
fd = open("file", O_RDONLY)
fcntl(fd, F_SETFD, FD_CLOEXEC)   ← 两步之间有窗口
```

**内核实现**：`exec()` 执行时调用 `do_close_on_exec()`，遍历 `close_on_exec` 位图，批量关闭所有标记了 CLOEXEC 的 fd。时间复杂度 O(max_fd / 64)，位图操作效率高。

**用 strace 验证**：

```bash
# 观察 shell 启动子命令时的 fd 处理
strace -f -e trace=execve,close,openat bash -c 'ls' 2>&1 | grep -E "execve|close|O_CLOEXEC"
```

---

### 5.13 用 /proc 和 lsof 深入观察 open file description

open file description 是内核对象，用户态无法直接访问，但可以通过 `/proc` 伪文件系统的多个接口间接观察。

**`/proc/<pid>/fdinfo/<fd>`**：

```bash
cat /proc/self/fdinfo/3
# pos:    0             ← 当前偏移量（来自 file->f_pos）
# flags:  0100002       ← O_RDWR | O_LARGEFILE（来自 file->f_flags，八进制）
# mnt_id: 22            ← 所在挂载点 ID（对应 /proc/self/mountinfo 的 mnt_id）
# ino:    1234567       ← inode 号
```

**`/proc/<pid>/fd/<fd>`**：符号链接，指向文件路径（deleted-but-open 时显示 `path (deleted)`）

**`/proc/<pid>/maps`**：观察 mmap 打开的 file（mmap 区域里可以看到对应文件路径）

**`lsof` 深度使用**：

```bash
# 找到所有 deleted-but-open 文件（可能在消耗磁盘空间）
lsof +L1 | grep DEL
# 输出格式：COMMAND  PID USER FD TYPE DEVICE SIZE/OFF NLINK NAME
# NLINK 为 0 表示名字已删除但 fd 仍持有

# 查看某个 fd 的详细信息（包括偏移和打开标志）
lsof -p $PID -a -d 3  # 只看 fd=3
# FD 字段：3r（只读），3w（只写），3u（读写），3r mem（mmap 只读）

# 查看哪些进程打开了同一个 file 对象（共享 open file description）
# （dup/fork 创建的 fd 共享同一底层 file）
lsof -a -p $PID | awk '{print $4, $7, $9}' | sort | uniq -d

# 用文件 offset 观察大文件传输进度（替代 pv）
while sleep 1; do
  lsof -p $PID -a -d 3 | awk 'NR>1 {print $7}'
done
# 每秒打印一次偏移量，通过增量推算传输速度
```

**验证 dup 共享 open file description**：

```bash
python3 - <<'EOF'
import os, sys

# 打开文件
fd1 = os.open("/tmp/test_dup", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
os.write(fd1, b"hello world\n")
os.lseek(fd1, 0, 0)  # 回到开头

# dup 创建 fd2，共享同一 open file description
fd2 = os.dup(fd1)

# 通过 fd1 读 5 字节，偏移移动到 5
data1 = os.read(fd1, 5)

# 通过 fd2 读，偏移从 5 继续（不是 0！）
data2 = os.read(fd2, 5)

print(f"fd1 读到: {data1}")  # b'hello'
print(f"fd2 读到: {data2}")  # b' worl'（共享偏移！）

# 对比：再次 open 得到独立偏移
fd3 = os.open("/tmp/test_dup", os.O_RDONLY)
data3 = os.read(fd3, 5)
print(f"fd3 读到: {data3}")  # b'hello'（独立偏移，从 0 开始）

# 观察 fdinfo
pid = os.getpid()
for fd in [fd1, fd2, fd3]:
    with open(f"/proc/{pid}/fdinfo/{fd}") as f:
        pos_line = [l for l in f if l.startswith("pos:")][0]
    print(f"fd{fd} pos: {pos_line.strip()}")
# fd1 pos: pos: 10  （两次读共5+5=10）
# fd2 pos: pos: 10  （与 fd1 相同，共享 open file description）
# fd3 pos: pos: 5   （独立，只读了5字节）

os.close(fd1); os.close(fd2); os.close(fd3)
os.unlink("/tmp/test_dup")
EOF
```

---

### 5.14 文件锁的层次：flock 与 fcntl 的关键区别

文件锁不是一个统一概念，不同的锁接口工作在不同层次，继承和释放语义完全不同。

**`flock(2)`：与 open file description 绑定**

```c
/* flock 的语义：锁与 open file description 绑定，而不是与 fd 绑定 */
int fd1 = open("file", O_RDWR);
int fd2 = dup(fd1);            /* fd1 和 fd2 共享同一 open file description */

flock(fd1, LOCK_EX);           /* 排它锁 */
flock(fd2, LOCK_UN);           /* 解锁！因为 fd2 共享同一 file，同一把锁 */
/* 此时 fd1 的锁也被解了！ */

/* fork 场景 */
if (fork() == 0) {
    /* 子进程继承 fd，共享同一 open file description，共享同一把 flock 锁 */
    flock(fd1, LOCK_UN);  /* 子进程解锁，父进程的锁也没了 */
}
```

**`fcntl(2)` / POSIX 字节范围锁：与进程绑定**

```c
struct flock lk = {.l_type=F_WRLCK, .l_whence=SEEK_SET, .l_start=0, .l_len=1024};
fcntl(fd1, F_SETLKW, &lk);    /* 加写锁，锁住文件前 1024 字节 */

/* 关键差异：fcntl 锁与进程绑定，不与 fd 或 open file description 绑定 */
int fd2 = open("file", O_RDWR);  /* 同一文件再 open 一次 */
struct flock lk2 = {.l_type=F_UNLCK, .l_whence=SEEK_SET, .l_start=0, .l_len=1024};
fcntl(fd2, F_SETLK, &lk2);    /* 解锁！虽然是不同 fd，但同一进程，同一锁范围 */
```

**`OFD 锁`（Linux 3.15+）：与 open file description 绑定，解决线程锁问题**：

```c
/* F_OFD_SETLK：Open File Description Locks */
/* 行为：锁与 open file description 绑定（不与进程/线程绑定）*/
/* 适合多线程场景：不同线程用不同 fd（dup/fork 共享）各持一把独立锁 */
struct flock lk = {.l_type=F_WRLCK, ...};
fcntl(fd, F_OFD_SETLK, &lk);  /* 此线程的 open file description 持有锁 */
/* 另一个线程用 dup 得到的 fd2 调用 F_OFD_SETLK，拿到独立的锁 */
```

**锁选型对照**：

| 接口 | 绑定层次 | fork 继承 | exec 继承 | 跨线程 | 字节范围 |
|------|---------|----------|----------|--------|---------|
| `flock` | open file description | 继承（共享） | 继承（共享） | 无效（共享） | 不支持 |
| `fcntl` POSIX | 进程 | 不继承 | 不继承 | 不区分线程 | 支持 |
| `fcntl` OFD | open file description | 继承（独立） | 继承（独立） | 区分 | 支持 |

---

### 5.15 高阶场景：O_PATH、io_uring 与 fd 的现代演进

**`O_PATH`：只拿引用，不拿 I/O 权限**

```bash
# O_PATH 打开文件：只获得路径引用，不检查读写权限，不触发 open() 的完整 VFS 流程
# 用途：safe 路径替换（避免 TOCTOU）、获得 fd 用于 fstatat/faccessat 等 AT_EMPTY_PATH 调用

# C 示例：
# fd = open("/proc/1/exe", O_PATH);     // 不需要读权限，只需 execute 权限
# fstat(fd, &st);                        // 通过 fd 获取 inode 信息
# openat(fd, "relative_path", O_RDONLY); // 从这个路径 fd 出发打开相对路径
```

`O_PATH` 创建的 fd：`f_mode` 为 `FMODE_PATH`，不能用于 `read()`/`write()`，但可以用于 `fstat`、`fstatfs`、`readlink`、`openat`、`linkat` 等，实现"路径即句柄"的安全编程模式。

**`io_uring`：fd 在异步 I/O 中的新生命周期**

```bash
# io_uring 引入了两个 fd 管理新概念：
# 1. Fixed files（注册 fd）：把 fd 注册到 io_uring，内核持有引用，减少每次操作的 fd 查表开销
# 2. Buffer ring：固定内存 buffer，避免用户态每次 read/write 的 copy

# 注册 fd（避免每次 SQE 都做 fget/fput 的引用计数操作）
io_uring_register(ring_fd, IORING_REGISTER_FILES, &fds, num_fds)

# 好处：高频 I/O 场景（如网络 proxy）减少 ~15% 的 fget/fput 开销
```

**`epoll` 与 fd 的 close 时机问题**

```bash
# 经典 epoll TOCTOU：
fd = open(...)
epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev)  # 把 fd 加入监听
dup2(newfd, fd)  # fd 重定向到别处！
# 此时 epoll 内部仍持有原 open file description 的引用（通过 file->f_count）
# epoll 监听的是 file 对象，不是 fd 号！
# 即使 fd 号被 dup2 覆盖，epoll 仍在监听原来的 file

# 正确理解：epoll 通过 file 指针（而非 fd 号）追踪 I/O 就绪状态
# 同一 file 的所有 fd（dup 出来的）共享 epoll 注册
```

---

### 5.16 完整对象链模型

可以用下面这个模型理解一次 I/O：

```text
+ 进程 fd 表
-> fd 槽位
-> open file description
-> file operations / mount / inode
-> page cache / extent / 设备写入路径
```

路径解析阶段结束后，后续大多数普通 I/O 都不再依赖原路径字符串，而是依赖这条对象链。

---

## 例子：为什么 `lsof +L1` 能找到“目录里看不到但还占空间”的文件

因为 `lsof` 看的不是目录树，而是进程持有的 open file 关系。deleted-but-open 文件虽然没有目录项名字，但仍有 open file description 被进程引用，所以它仍能被枚举出来。

---

## 常见误区

- 误以为 fd 就是文件本身
- 误以为再次 `open` 和 `dup` 差不多
- 误以为 `unlink` 后对象必然立刻消失
- 误以为 I/O 语义只由 inode 决定，忽略 open file description 和 flag
- 误以为路径始终参与后续 I/O，忽略打开后是对象链在工作

---

## 本章小结

| 主题 | 结论 |
|------|------|
| fd | 进程局部索引，不是文件本体 |
| open file description | 运行时 I/O 语义核心层 |
| `open` / `dup` / `fork` / `exec` | 决定共享关系和继承边界 |
| 偏移量 | 通常属于 open file description |
| flag | 属于打开语义，不只是对象属性 |
| deleted-but-open | 是名字层与打开层分离的正常结果 |

---

## 练习题

### 基础题

**5.1** 为什么 fd 不能等同于文件对象？描述从 fd 到最终数据的完整对象链，指出每层分别解决什么问题。

**5.2** `dup(fd)` 与对同一路径再次 `open()` 的根本区别是什么？它们在偏移量共享、flag 继承、关闭行为上各有什么差异？

### 中级题

**5.3** 为什么 `O_CLOEXEC` 是跨进程安全边界的重要工具？描述一个不加 `CLOEXEC` 导致文件描述符泄漏的真实场景，以及如何用 `lsof` 或 `/proc/<pid>/fd` 检测。

**5.4** deleted-but-open 现象说明了哪几层被拆开了？结合日志轮转场景，解释为什么"文件已被 rotate 但磁盘空间未释放"，以及正确的修复思路。

### 提高题

**5.5** 分析以下多进程场景的语义：父进程 `open("log", O_WRONLY|O_APPEND)` 后 `fork()`，子进程和父进程同时调用 `write()`。请回答：①两个进程的写入是否安全（不互相覆盖）？②为什么 `O_APPEND` 能提供这个保证（提示：`lseek+write` 非原子，`O_APPEND` 的内核实现路径）？③如果不用 `O_APPEND` 而改用文件锁，需要锁住哪个层次的对象？

---

## 练习答案

**5.1** fd 是进程 fd 表中的整数槽位，指向 open file description；open file description 保存偏移量、flag、file operations 引用；file operations 通过 inode 访问 address_space；address_space 关联 page cache；page cache 按需从磁盘读取数据块。每层职责：fd（进程局部索引）→ open file description（运行时 I/O 语义）→ inode（对象元数据与数据映射）→ page cache（内存中的数据视图）。

**5.2** `dup` 创建新 fd 但共享同一个 open file description，因此偏移量共享、`O_NONBLOCK` 等 flag 共享、任一关闭不影响另一个继续使用；再次 `open` 创建新的 open file description，偏移量独立（从 0 开始），flag 可以不同，两者互不影响，各自关闭互不干扰。

**5.3** 场景：Web 服务器用 `fork+exec` 启动 CGI 脚本，若监听 socket fd 没有 `CLOEXEC`，子进程执行 exec 后继承该 socket，导致服务器 socket 泄漏给不可信程序。检测：`ls -la /proc/<pid>/fd/` 查看子进程继承的 fd；`lsof -p <pid>` 列出所有打开的文件/socket。现代做法：`open(..., O_CLOEXEC)`、`socket(..., SOCK_CLOEXEC)`、`accept4(..., SOCK_CLOEXEC)`。

**5.4** 被拆开的层：名字层（目录项已删除，路径不可见）vs 打开层（open file description 仍存活，fd 可继续读写）vs 对象层（inode 引用计数 > 0，空间未回收）。日志轮转场景：rotate 程序 rename 旧日志但服务仍持有旧 fd，继续写旧 inode；新文件虽建立但服务未重开 fd 指向它。修复：向服务发 `SIGHUP` 让其重新打开日志文件，或用 `truncate -s 0 /proc/<pid>/fd/<n>` 清空（立即释放磁盘空间，不中断服务）。

**5.5** ①`O_APPEND` 写入安全，不互相覆盖。②原因：`O_APPEND` 在内核 `write()` 路径中原子地先定位到文件末尾再写入（持有 inode 写锁），避免了两次 `lseek+write` 之间被其他进程插入的竞争窗口；而 `lseek(SEEK_END)+write` 是两步操作，fork 后共享同一 open file description，偏移量竞争会导致内容覆盖。③文件锁方案：需要锁 inode 级别的字节范围锁（`fcntl(F_SETLKW)`），但 fd 级锁（`flock`）会随 fd 继承，需注意 `fork` 后子进程持锁语义。

---

## 延伸阅读

1. **Kerrisk**. *The Linux Programming Interface*, Ch.5 — 文件 I/O 深度解析（open file description、dup、O_APPEND）
2. **man 2 open** — `O_CLOEXEC`、`O_APPEND`、`O_PATH` 语义说明
3. **LWN.net**. *Close-on-exec: the hard way* — CLOEXEC 历史与工程实践
4. **Stevens & Rago**. *Advanced Programming in the UNIX Environment*, Ch.3 — 文件 I/O

---

[← 上一章：inode、dentry 与目录项](./04-inodes-dentries-and-directory-entries.md)

[下一章：从路径到 inode：查找过程 →](./06-from-path-to-inode-how-lookup-works.md)

[返回目录](../README.md)
