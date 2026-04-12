# 附录 A4：高阶实验与源码阅读路线

这份附录面向已经读完主线章节、希望把“知道概念”提升为“能做实验、能追源码、能复盘故障”的读者。

它不试图再讲一遍概念，而是回答三个更高阶的问题：

- 你应该做哪些实验，才能真正看见 page cache、rename、overlayfs、NFS 这些语义边界？
- 你应该从哪些源码入口切进去，才不会一上来就淹死在 Linux 内核里？
- 你应该怎样把一次现象观察，升级成可以复用的排障或设计方法？

---

## 一、建议先具备哪些基础

开始这份附录前，最好已经读过：

- 第6章：从路径到 inode
- 第9章：VFS、挂载与文件系统家族
- 第10章：崩溃一致性与恢复
- 第11章：page cache 与读写路径
- 第12章：mmap、direct I/O 与 fsync
- 第16章：overlayfs 与容器镜像分层
- 第17章：网络文件系统与分布式存储
- 第18章：性能、基准与调优

如果这些章节还不熟，先回去打底，再做实验会更有效。

---

## 二、六个高阶实验

建议先准备一套最小观测命令：

```bash
findmnt -T <path>
namei -om <path>
lsof +L1
strace -yy -e openat,openat2,rename,fsync,fdatasync <cmd>
vmstat 1
iostat -xz 1
cat /proc/self/mountinfo | head -n 20
```

这样每个实验都不只是“看现象”，而是能立刻补对象、挂载和回写证据。

### 实验 1：deleted-but-open 与空间不回收

目标：亲眼确认“名字消失”和“对象生命周期结束”不是一回事。

建议步骤：

1. 用一个进程持续写日志文件。
2. 在另一个终端 `rm` 掉该文件。
3. 观察 `ls` 看不到它，但 `df -h` 空间没有明显恢复。
4. 用 `lsof +L1` 找到仍持有 fd 的进程。
5. 结束进程或关闭 fd，再观察空间回收。

最小证据集：

- `ls -li` 看名字和 inode
- `lsof +L1` 看 deleted-but-open
- `df -h` / `du -sh` 看不同层次的容量视角

你要看到的关键现象：

- 目录项消失后，inode/打开文件状态仍可能存在。
- `du`、`df`、`ls` 看到的是不同层次的现实。

对应章节：

- 第3章
- 第5章
- 第14章

### 实验 2：可靠文件替换协议

目标：区分 `write`、`fsync(file)`、`rename`、`fsync(dir)` 各自补的语义缺口。

建议步骤：

1. 写一个小程序或脚本，先把新内容写到临时文件。
2. 做一版只 `rename` 不 `fsync` 的流程。
3. 做一版 `fsync(temp)` + `rename` 但不 `fsync(parent)` 的流程。
4. 做一版完整协议：`fsync(temp)` -> `rename` -> `fsync(parent dir)`。
5. 比较三种方案分别在什么语义上更强，哪些边界仍依赖底层实现。

最小证据集：

- `strace -yy -e openat,rename,fsync,fdatasync <cmd>` 确认程序是否真的发出了预期系统调用
- `findmnt -T <target-dir>` 确认临时文件和目标文件是否在同一文件系统实例内

你要看到的关键现象：

- `rename` 主要解决名字切换的原子性，不自动等于目录项持久化。
- 可靠更新是协议，不是某个单独 API。

对应章节：

- 第10章
- 第12章
- 第20章

### 实验 3：page cache 如何扭曲性能判断

目标：区分“缓存命中很快”和“设备真的很快”。

建议步骤：

1. 准备一个明显大于可用内存，至少也要大于 page cache 可长期容纳规模的数据集。
2. 做冷缓存与热缓存对照；如果要手动清缓存，只应在可控实验环境里进行，而不要在生产环境操作。
3. 连续读取两次，比较第一次和第二次延迟差异。
4. 将随机读、小文件读、顺序读分开测。
5. 同时观察 `vmstat 1`、`/proc/meminfo` 中的 `Dirty`/`Writeback`。
6. 对比包含 `fsync` 和不包含 `fsync` 的写入测试。

最小证据集：

- `vmstat 1` 观察缓存与回写节奏
- `iostat -xz 1` 观察设备层是否真的在忙
- `/proc/meminfo` 中 `Dirty`/`Writeback` 的变化

你要看到的关键现象：

- 热 cache benchmark 很容易夸大设备能力。
- `write()` 快不代表持久化快。

对应章节：

- 第11章
- 第18章

### 实验 4：overlayfs 的 copy-up 成本

目标：确认容器写路径为什么常在首次写时突然变慢。

建议步骤：

1. 准备一个 lower 层有大量文件的 overlayfs 环境。
2. 对同一路径做首次写和后续写，对比延迟差异。
3. 分别观察文件内容修改、元数据修改、目录重命名的成本差异。
4. 结合 `mount`/`findmnt` 观察 upper、lower、work、merged。

最小证据集：

- `findmnt -T <merged-path>` 看当前路径真正挂到哪里
- `mount | grep overlay` 或 `cat /proc/self/mountinfo` 看 overlay 参数

你要看到的关键现象：

- overlayfs 的成本不只来自“多一层”，而是来自 copy-up、whiteout 和路径解析。
- 同一条路径在镜像层和可写层之间可能经历不同语义阶段。

对应章节：

- 第15章
- 第16章
- 第18章

### 实验 5：元数据风暴与大目录热点

目标：理解为什么很多“磁盘不忙但系统很慢”的问题其实是 VFS/目录/元数据路径问题。

建议步骤：

1. 构造大量小文件创建、删除、`stat`、`rename` 操作。
2. 把大目录热点和分层目录结构做对照。
3. 同时观察 `pidstat -d 1`、`perf stat` 或 `perf record/report`。
4. 对比“少量大文件顺序 I/O”和“大量小文件元数据操作”。

最小证据集：

- `perf stat` 看 CPU、context switch、fault 与 cache miss
- `pidstat -d 1` 看是否真在打 I/O
- `iostat -xz 1` 对照设备层是否同步饱和

你要看到的关键现象：

- 元数据路径常常先于设备吞吐成为瓶颈。
- “iowait 不高”不代表文件系统路径没问题。

对应章节：

- 第6章
- 第7章
- 第14章
- 第18章

### 实验 6：本地协议迁到远端后的语义错位

目标：识别应用对本地 POSIX 的隐式依赖。

建议步骤：

1. 选择一个依赖小文件更新、mtime 或 rename 的简单程序。
2. 在本地文件系统上运行，记录行为。
3. 把工作目录迁到 NFS、CephFS 或对象存储挂载层，再观察差异。
4. 重点检查可见性、mtime、锁、close-to-open 和属性缓存效果。

最小证据集：

- `findmnt` / `mountstats` / `nfsstat -m` 看挂载类型和参数
- 用两个客户端或两个挂载上下文交叉验证可见性，不要只看单侧日志

你要看到的关键现象：

- 远端问题常是语义错位，不只是“网络慢”。
- 应用协议如果过度依赖本地假设，迁移后很容易暴露。

对应章节：

- 第17章
- 第19章
- 第20章

---

## 三、如何记录实验，避免只得到一次性直觉

每次实验至少记录：

- 工作负载：读写模式、文件大小、文件数量、并发度
- 缓存状态：冷启动、热 cache、是否清理或预热
- 持久化边界：是否包含 `fsync`、目录 `fsync`、close/open
- 环境信息：文件系统类型、挂载参数、设备类型、容器/虚拟化上下文
- 观测证据：命令输出、日志、`perf`、`mountinfo`、`lsof`、`findmnt`
- 结论：你验证了哪个假设，排除了哪个假设

如果没有这些记录，实验很容易退化成“那次机器上好像是这样”的模糊印象。

---

## 四、源码阅读入口图

Linux 文件系统源码很多，不建议从某个巨大的目录盲读。更好的方法是按问题切入口。

| 你想回答的问题 | 优先入口 |
|----------------|----------|
| 路径查找如何进行 | `fs/namei.c` |
| `open`/`close`/fd 生命周期怎么串起来 | `fs/open.c`、`include/linux/fs.h` |
| `openat2` 和路径约束怎么落地 | `fs/open.c`、`fs/namei.c` |
| page cache 与读路径 | `mm/filemap.c` |
| writeback 与脏页回写 | `mm/page-writeback.c` |
| ext4 的目录、inode、journal 更新路径 | `fs/ext4/` |
| 挂载、namespace、传播语义 | `fs/namespace.c` |
| overlayfs 的 copy-up / whiteout | `fs/overlayfs/` |
| NFS 客户端语义与缓存 | `fs/nfs/` |
| iomap 和现代 I/O 路径 | `fs/iomap/` |
| DAX / 持久内存路径 | `fs/dax.c`、相关文件系统实现中的 DAX 入口 |

### 一个更可行的源码阅读顺序

1. 先从第6章对应的 `fs/namei.c` 入手，建立路径查找直觉。
2. 再读 `include/linux/fs.h`，把核心对象关系串起来。
3. 再切 `mm/filemap.c` 和 `mm/page-writeback.c`，把 page cache 与 writeback 补上。
4. 然后选一个具体实现方向：`fs/ext4/`、`fs/overlayfs/` 或 `fs/nfs/`。
5. 最后再回到第18章和第20章，用“协议路径”视角理解代码。

### 读源码时最容易犯的三个错误

- 一上来就沿调用栈猛冲，不先建立对象图和层次图。
- 把某个具体文件系统实现细节误当成所有文件系统通用规律。
- 只读“快路径”，不问失败、撤销、回写、恢复和缓存失效路径。

---

## 五、如果你要继续深入，可以按专题推进

### 方向一：路径解析与安全边界

关注：

- `openat2`
- `RESOLVE_*` 约束
- symlink race
- `chroot`、namespace、bind mount 与安全绕过边界

### 方向二：缓存、脏页与持久化

关注：

- page cache
- writeback
- dirty throttling
- `fsync` 与设备 flush
- DAX / direct I/O / `mmap` 的不同代价

### 方向三：容器与联合视图

关注：

- mount namespace
- propagation
- overlayfs copy-up
- container image layer
- CSI / volume plugin 语义错位

### 方向四：远端与分布式语义

关注：

- close-to-open
- lease / delegation
- 元数据服务器
- 锁服务
- 网络分区、客户端崩溃与恢复

### 方向五：性能与基准方法

关注：

- 元数据风暴
- 小同步写
- flush/fua/queue depth
- cache 污染
- 基准对照组设计

---

## 六、最后的建议：把“现象 -> 层次 -> 证据 -> 协议”变成固定动作

真正的进阶，不是学到更多术语，而是每次遇到问题时都能稳定地做四件事：

1. 先定义现象在哪一层出现。
2. 再找能证明或推翻假设的证据。
3. 再判断真正的语义边界在哪里。
4. 最后把它落实为一个可重复执行的协议或排障清单。

如果你能做到这一步，文件系统知识才真正开始变成工程能力。
