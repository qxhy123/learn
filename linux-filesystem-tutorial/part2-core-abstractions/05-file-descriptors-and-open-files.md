# 第5章：文件描述符与 open file

> 文件描述符不是文件本身，甚至也不是“打开文件”的全部；真正的关键对象是 open file description，它把路径解析结果、运行时状态和后续 I/O 语义接了起来。

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

### 5.11 一个完整对象链模型

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

1. 为什么 fd 不能等同于文件对象？
2. `dup` 与再次 `open` 的根本区别是什么？
3. 为什么 `CLOEXEC` 是跨进程安全边界的重要工具？
4. deleted-but-open 现象具体说明了哪几层被拆开了？
5. 为什么说路径解析结束后，后续 I/O 主要围绕 open file description 展开？
