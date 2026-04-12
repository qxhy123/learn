# 从零到高阶的 Linux 文件系统教程：路径、inode、VFS、ext4、缓存与一致性

## 项目简介

本教程试图回答一个看似朴素、实际上非常系统化的问题：

> 当你在 Linux 里输入一个路径并打开一个文件时，内核到底做了什么？

很多人会使用 `ls`、`cd`、`cp`、`mv`、`rm`，也知道“文件在磁盘上”“目录里有文件”“权限决定能不能访问”，但这些知识常常是割裂的：

- 路径和 inode 是什么关系？
- 文件名和文件内容为什么不是一回事？
- 为什么 `rm` 一个仍被进程打开的文件，空间不一定立刻释放？
- `rename` 为什么经常被说成“原子操作”，但为什么还会有人强调目录 `fsync`？
- page cache、writeback、mmap、journal、mount、VFS、overlayfs、NFS 又如何串成一条完整链路？

本教程会沿着“从用户看到的路径，到内核维护的对象，再到磁盘与缓存、容器和分布式场景”的路线，逐步建立一张完整的 Linux 文件系统地图。

这里的“高阶”不等于把某个内核源码文件逐行讲一遍，而是意味着你最终能够同时回答四类问题：

1. **对象问题**：路径、dentry、inode、file、superblock 各表示什么？
2. **语义问题**：`rename`、`fsync`、硬链接、软链接、page cache、close-to-open 各自保证什么，不保证什么？
3. **工程问题**：为什么 ext4、xfs、btrfs、zfs、overlayfs、NFS 在真实场景中的定位不同？
4. **排障问题**：为什么 `df` 和 `du` 会打架、为什么删文件后空间不回收、为什么容器里的目录树与宿主不同、为什么“写成功”却仍可能在断电后丢数据？

写作原则如下：

- **先从现象和命令出发，再进入内核抽象**
- **先建立对象关系，再进入性能和一致性**
- **面向初学者，但不牺牲系统准确性**
- **不只讲“是什么”，还讲“边界在哪里、代价是什么、为什么会误判”**

---

## 目标受众

- 刚接触 Linux，希望系统理解文件、目录、权限和挂载机制的学习者
- 会使用常见命令，但对 inode、VFS、page cache、日志机制仍然模糊的开发者
- 正在学习操作系统、存储系统、容器或基础设施的工程师
- 想把“日常命令体验”和“内核实现机制”连接起来的进阶读者
- 准备系统设计、内核、SRE、云原生相关面试的学习者

---

## 章节导航目录

### 开始之前

- [前言：如何阅读这门教程](./00-preface.md)

### 第一部分：文件系统入门地图

| 章节 | 标题 | 主要内容 |
|------|------|----------|
| 第1章 | [为什么需要文件系统](./part1-basics/01-why-filesystems.md) | 从裸设备到命名空间，理解文件系统解决的问题 |
| 第2章 | [路径、文件与目录](./part1-basics/02-paths-files-and-directories.md) | 绝对路径、相对路径、层级树、当前工作目录 |
| 第3章 | [权限、元数据与链接](./part1-basics/03-permissions-metadata-and-links.md) | `rwx`、所有者、时间戳、硬链接与软链接 |

### 第二部分：内核中的核心抽象

| 章节 | 标题 | 主要内容 |
|------|------|----------|
| 第4章 | [inode、dentry 与目录项](./part2-core-abstractions/04-inodes-dentries-and-directory-entries.md) | 名字、对象和元数据如何拆分 |
| 第5章 | [文件描述符与 open file](./part2-core-abstractions/05-file-descriptors-and-open-files.md) | 进程如何持有“打开中的文件” |
| 第6章 | [从路径到 inode：查找过程](./part2-core-abstractions/06-from-path-to-inode-how-lookup-works.md) | path walk、负 dentry、挂载穿越、符号链接与权限检查 |

### 第三部分：磁盘布局、挂载与一致性

| 章节 | 标题 | 主要内容 |
|------|------|----------|
| 第7章 | [块、超级块与空间分配](./part3-layout-and-mount/07-blocks-superblocks-and-allocation.md) | 文件系统在磁盘上的基本组织 |
| 第8章 | [ext4 布局与日志机制](./part3-layout-and-mount/08-ext4-layout-and-journaling.md) | ext4 的布局、extent、journal mode、提交与恢复直觉 |
| 第9章 | [VFS、挂载与文件系统家族](./part3-layout-and-mount/09-vfs-mount-and-filesystem-family.md) | VFS 对象图、挂载树、FUSE 与 ext4/xfs/btrfs/zfs 的工程定位 |
| 第10章 | [崩溃一致性与恢复](./part3-layout-and-mount/10-crash-consistency-and-recovery.md) | write ordering、rename/fsync、目录持久化、fsck 与一致性模型 |

### 第四部分：缓存、I/O 与可观测性

| 章节 | 标题 | 主要内容 |
|------|------|----------|
| 第11章 | [page cache 与读写路径](./part4-cache-and-io/11-page-cache-and-read-write-path.md) | readahead、脏页、writeback、内存压力与读写路径 |
| 第12章 | [mmap、direct I/O 与 fsync](./part4-cache-and-io/12-mmap-direct-io-and-fsync.md) | `mmap`、O_DIRECT、`fdatasync`、目录 `fsync` 与原子替换语义 |
| 第13章 | [空间管理、碎片与 TRIM](./part4-cache-and-io/13-space-management-fragmentation-and-trim.md) | inode 密度、配额、稀疏文件、预分配、碎片与 SSD 关注点 |
| 第14章 | [观测与排障](./part4-cache-and-io/14-observability-and-troubleshooting.md) | 从 `df`、`lsof`、`findmnt` 到 `perf`/`iostat` 的分层定位方法 |

### 第五部分：容器、隔离与分布式视角

| 章节 | 标题 | 主要内容 |
|------|------|----------|
| 第15章 | [bind mount、namespace 与隔离](./part5-containers-and-distributed/15-bind-mount-namespace-and-isolation.md) | mount namespace、传播语义、`pivot_root`/`chroot` 与容器视图隔离 |
| 第16章 | [overlayfs 与容器镜像分层](./part5-containers-and-distributed/16-overlayfs-and-container-images.md) | copy-up、whiteout、opaque dir、page cache 与镜像分层 |
| 第17章 | [网络文件系统与分布式存储](./part5-containers-and-distributed/17-network-filesystems-and-distributed-storage.md) | NFS close-to-open、属性缓存、锁、对象存储与分布式一致性边界 |

### 第六部分：高阶理解与综合实践

| 章节 | 标题 | 主要内容 |
|------|------|----------|
| 第18章 | [性能、基准与调优](./part6-advanced-and-synthesis/18-scaling-performance-and-benchmarks.md) | 工作负载建模、尾延迟、元数据瓶颈、基准陷阱与调优路径 |
| 第19章 | [设计权衡与常见误解](./part6-advanced-and-synthesis/19-design-tradeoffs-and-common-misconceptions.md) | 语义、可靠性、扩展性、FUSE/CoW/分布式等多维权衡 |
| 第20章 | [综合实践与下一步](./part6-advanced-and-synthesis/20-capstone-and-next-steps.md) | 用系统化排障框架把路径、缓存、一致性、容器与分布式问题串起来 |

### 附录

| 附录 | 标题 | 主要内容 |
|------|------|----------|
| A1 | [常用命令速查](./appendix/command-cheatsheet.md) | 文件系统观察、容量、挂载、inode、writeback 与排障命令 |
| A2 | [核心结构速查表](./appendix/structure-cheatsheet.md) | inode、dentry、superblock、page cache、writeback、journal、namespace 等速查 |
| A3 | [练习题答案索引](./appendix/answers.md) | 各章练习题方向与核对要点 |
| A4 | [高阶实验与源码阅读路线](./appendix/advanced-labs-and-source-roadmap.md) | 进阶实验、故障演练、内核源码入口与继续深入建议 |
| A5 | [分区、mkfs、挂载与动态扩容实操](./appendix/practical-mount-partition-and-resize.md) | 从裸盘初始化到持久挂载、LVM 与在线扩容的实践路线 |

---

## 学习路径建议

### 路径一：先建立整体地图（1-2 天）

适合第一次系统接触 Linux 文件系统的学习者：

1. 阅读[前言](./00-preface.md)
2. 学习第1章、第2章、第4章，先建立“名字、对象、路径”的基本框架
3. 学习第9章、第11章，知道 VFS、挂载和 page cache 在整张图里的位置
4. 选读第14章和附录 A1，建立命令观察能力
5. 最后阅读第20章，把完整链路串起来

### 路径二：系统打基础（1-2 周）

适合希望从命令使用一路走到内核抽象和存储实现的学习者：

1. 按顺序完成第1-6章，建立从路径到 inode 的完整理解
2. 按顺序完成第7-10章，理解磁盘布局、挂载与一致性
3. 按顺序完成第11-14章，掌握缓存、持久化与排障方法
4. 学习第15-17章，把文件系统知识带入容器和网络场景
5. 以第18-20章收尾，形成面向工程实践的综合视角

### 路径三：面向工程排障（3-5 天）

适合开发者、SRE 或平台工程师：

1. 重点阅读第3章、第5章、第6章，理解权限、文件描述符和路径解析
2. 阅读第9章、第10章、第11章、第12章，掌握挂载、缓存与一致性
3. 重点阅读第14章、第15章、第16章、第17章，连接容器、隔离和网络存储问题
4. 阅读第18章、第19章、第20章，形成性能与设计权衡视角
5. 将附录 A1 作为日常命令参考

### 路径四：深入语义与可靠性（4-7 天）

适合已经具备基础命令经验、希望理解“到底什么才算真正可靠”的读者：

1. 从第5章、第6章进入，弄清 `open`、文件描述符、路径查找与名字解析
2. 深读第8章、第10章、第12章，重点理解 journal mode、`rename`、`fsync`、目录 `fsync` 与崩溃恢复
3. 深读第11章、第18章，理解 page cache、writeback、基准与持久化之间的错位
4. 回看第16章、第17章，把本地语义和容器/远端语义区别开

### 路径五：高阶实验与源码阅读（1-2 周）

适合已经读完主要章节、希望把“知道概念”升级成“能做实验、能读源码、能复盘事故”的读者：

1. 先读第10章、第12章、第17章、第18章、第20章，建立一致性、远端语义和基准方法的高阶框架
2. 按照[附录 A4：高阶实验与源码阅读路线](./appendix/advanced-labs-and-source-roadmap.md)完成 4-6 个实验
3. 结合[附录 A1：常用命令速查](./appendix/command-cheatsheet.md)补上 `mountinfo`、writeback、NFS 观测、`perf`/`bpftrace` 的基本观察路径
4. 选择一个专题做源码阅读：`fs/namei.c`、`mm/filemap.c`、`fs/ext4/`、`fs/namespace.c`、`fs/overlayfs/` 或 `fs/nfs/`
5. 最后用第20章的案例分析框架，重写一次自己遇到过的文件系统或容器存储问题

### 路径六：实操初始化与扩容（2-4 天）

适合已经理解基本对象关系、希望把“会看图”和“会动手”连起来的读者：

1. 先读第9章，理解 mount、filesystem instance、`/etc/fstab` 和挂载视图是同一条链上的不同层次
2. 读第13章，理解空间、配额、预分配和动态扩容不是同一个问题
3. 按照[附录 A5：分区、mkfs、挂载与动态扩容实操](./appendix/practical-mount-partition-and-resize.md)完成一遍“新盘初始化 -> 持久挂载 -> 在线扩容”
4. 配合[附录 A1：常用命令速查](./appendix/command-cheatsheet.md)核对 `lsblk`、`blkid`、`findmnt`、`parted`、`pvresize`、`resize2fs`、`xfs_growfs`
5. 最后回到第10章，重新检查这些操作分别改变的是块层、分区层、卷管理层，还是文件系统层

---

## 前置要求

本教程不要求你先学过内核开发，也不要求你熟悉 ext4 源码，但具备以下经验会更轻松：

- 有基本 Linux 命令行使用经验
- 知道文件、目录、权限、进程这些基本概念
- 如果写过程序，理解“打开文件”“读写文件”“文件描述符”会更自然

如果这些还不熟，也可以直接开始。本教程会尽量从现象和命令出发，再进入内部机制。

---

## 如何使用本教程

1. **先看对象关系，不要急着背术语**：路径、目录项、inode、文件描述符各自解决的问题不同。
2. **每学一个概念，都问三个问题**：它表示什么对象？它位于哪一层？它和谁连接？
3. **带着真实命令现象去读**：建议边学边观察 `ls -li`、`stat`、`df`、`du`、`findmnt` 的输出。
4. **把“名字”和“内容”分开理解**：很多误区都来自把目录项、inode、数据块混为一谈。
5. **把“一致性”和“性能”分开思考**：写得快不等于落盘快，落盘快也不等于崩溃后一定一致。
6. **用综合问题检验自己**：例如“删除打开中文件为何空间不释放”“容器里看到的目录树为何与宿主不同”。
7. **把每个保证拆开问**：这个机制保证的是可见性、原子性、持久性、隔离性，还是只是性能？

---

## 教程特色

- **从用户态现象一路讲到内核抽象**：帮助你把命令体验与实现机制连接起来
- **强调对象关系图**：持续区分路径、dentry、inode、文件描述符、页缓存和数据块
- **兼顾经典本地文件系统与现代场景**：不仅讲 ext4，也讲 overlayfs、容器、网络存储与多种本地文件系统定位差异
- **显式区分“能工作”和“语义可靠”**：重点解释 `rename`、`fsync`、page cache、close-to-open 等常见误判点
- **重视一致性与恢复语义**：帮助你理解 `fsync`、journal、崩溃恢复为何如此关键
- **面向工程排障**：关注容量、inode 耗尽、挂载错位、缓存效应、性能瓶颈等真实问题
- **新增高阶实验与源码入口**：把概念、观测命令、实验设计和内核源码入口连成一条继续深入的路径
- **补入实操初始化与扩容路线**：把分区、`mkfs`、挂载、`/etc/fstab`、LVM 和在线扩容串成一条可操作路径
- **补足高阶机制细节**：补入 RCU path walk、`openat2()`、unwritten extent、writeback error、DAX、metadata server / split-brain 等更贴近生产系统的问题
- **中文编写**：术语统一，便于连续学习与复习

---

## 如果你觉得“前面还是太基础”，应该怎样读

可以直接跳过“按章顺读”的心态，改成按问题读：

- 想搞清路径和安全边界：第6章 + 附录 A1 + 附录 A4
- 想搞清 ext4 到底保证什么：第8章 + 第10章 + 第12章
- 想搞清为什么 benchmark 总是骗人：第11章 + 第18章
- 想真的从裸盘做到持久挂载和扩容：第9章 + 第13章 + 附录 A1 + 附录 A5
- 想搞清容器和远端共享存储为什么老出玄学问题：第15-17章 + 第20章

这条读法更适合已经熟悉 Linux 基本命令、但想快速建立高阶工程判断的人。

---

## 许可证

本项目采用 MIT 许可证开源。你可以自由使用、复制、修改和分发本教程内容。

---

*如有建议或发现错误，欢迎反馈。*
