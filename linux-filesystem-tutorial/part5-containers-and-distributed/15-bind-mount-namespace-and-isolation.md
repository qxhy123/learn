# 第15章：bind mount、namespace 与隔离

> 容器里的文件系统隔离并不是“复制一整套根目录”，而是通过 mount namespace、bind mount、propagation、pivot_root 等机制，构造出一个进程可见但并不等于宿主视图的名字空间。

## 学习目标

完成本章后，你将能够：

1. 解释 bind mount、mount namespace、pivot_root、chroot 分别工作在哪一层
2. 理解 propagation（shared/private/slave）为什么决定隔离边界是否漏出
3. 说明容器卷挂载、根文件系统切换、宿主/容器路径错位的根因
4. 认识 idmapped mounts、user namespace 与 rootless 场景的关系
5. 把“同一路径在不同视图下不同对象”当成容器文件系统常态，而不是例外

---

## 正文内容

### 15.1 bind mount 改变的是挂载树，不是创建一个链接文件

bind mount 不是符号链接，也不是复制目录。它做的是：

- 选定一棵已有对象树
- 在挂载树里给它另开一个接入点

这意味着：

- 同一对象可通过多个路径入口被看见
- 新入口仍处在挂载语义层，而不是普通文件层
- 后续挂载遮蔽、只读策略、namespace 视图都可能继续作用在它上面

### 15.2 mount namespace 解决的是“谁看到哪棵挂载树”

mount namespace 不是重新发明文件系统，而是让不同进程组拥有不同的挂载视图。于是：

- 相同路径字符串不再保证指向同一对象
- 容器里的 `/proc`、`/sys`、`/app` 可能和宿主完全不是同一路径树
- 宿主上看到的“原目录内容”，在容器里可能已被 bind mount 或 overlay 覆盖

容器“有自己的根文件系统”这句话，更准确的理解是：“它有自己的挂载视图”。

### 15.3 `chroot` 不是完整隔离，`pivot_root` 更接近真正换根

- `chroot` 改变路径解析的根边界
- `pivot_root` 更接近把当前挂载树根切换到新根，并把旧根移到另一个位置等待处理

只靠 `chroot` 并不能自然提供完整容器隔离语义，因为：

- 挂载视图未必隔离
- 打开的 fd 仍可能指向旧根外部对象
- 许多内核对象和挂载点仍共享宿主视图

所以真正的容器根切换几乎总和 mount namespace 一起讨论。

### 15.4 propagation 决定隔离有没有“漏风”

共享传播语义（shared/private/slave/unbindable）决定挂载事件如何传播：

- shared：彼此传播
- private：完全不传播
- slave：接收上游传播但不向上游传播
- unbindable：不能被 bind mount

这会直接影响：

- 宿主新挂载是否进入容器
- 容器里的挂载是否反向影响宿主
- sidecar / CSI / Kubernetes volume 行为是否符合预期

很多“为什么容器里突然看到宿主新挂载”或“为什么 volume mount 没按预期更新”的问题，本质上是 propagation 没配对。

### 15.5 volume mount、镜像内容和覆盖关系

容器里一个路径常常同时受三层影响：

1. 镜像层提供的默认内容
2. overlayfs merged 视图提供运行时根文件系统
3. bind mount / volume mount 覆盖某个具体子路径

因此一个路径“看不到镜像里的原内容”，未必是文件没了，而是：

- 被 volume mount 遮住了
- 被 bind mount 替换成宿主目录了
- 在另一个 namespace 里看的是完全不同的挂载树

### 15.6 user namespace 和 idmapped mount 会进一步改变身份语义

在 rootless 容器或更现代的挂载设计里，除了“看哪棵树”，还要问“以什么 uid/gid 语义看这棵树”。

idmapped mounts 的直觉是：

- 同一底层对象
- 在某个挂载视图中
- 可以呈现不同的 uid/gid 映射解释

这对无特权容器和共享目录尤其重要，因为它把身份映射从“复制一份文件”变成了“挂载视图层翻译”。

### 15.7 rootless 场景为什么格外复杂

rootless 容器经常把这些问题叠在一起：

- user namespace 身份映射
- mount namespace 视图隔离
- FUSE 或用户态文件系统参与路径
- overlayfs / bind mount 的权限与能力边界

所以 rootless 不只是“没有 root”，而是多层对象与权限语义一起变化。

### 15.8 一个排障框架：容器里路径错了，到底错在哪层

可以先问：

1. 当前进程在哪个 mount namespace？
2. 这个路径是否被 bind mount/volume mount 覆盖？
3. propagation 是否导致宿主/容器视图相互影响？
4. 是对象权限问题，还是名字空间视图问题？
5. 是否存在 idmapped / user namespace 导致的 uid/gid 解释差异？

如果这些问题不先问清楚，容器里的路径问题看起来会特别“玄学”。

---

## 本章小结

| 主题 | 结论 |
|------|------|
| bind mount | 在挂载树层重新接入已有对象树 |
| mount namespace | 定义不同进程组看到的挂载视图 |
| `chroot` / `pivot_root` | 都影响根边界，但隔离强度和层次不同 |
| propagation | 决定挂载事件是否泄露或同步 |
| idmapped / user namespace | 让身份语义也变成视图的一部分 |
| 容器路径问题 | 往往首先是视图层问题，而不是文件消失 |

---

## 练习题

1. 为什么 bind mount 不应被理解成“目录快捷方式”？
2. mount namespace 和 `pivot_root` 在容器里分别解决什么问题？
3. propagation 为什么会影响 volume mount 的行为？
4. idmapped mount 的直觉是什么？
5. 容器里同一路径和宿主不一致时，第一批应该排查哪几层？
