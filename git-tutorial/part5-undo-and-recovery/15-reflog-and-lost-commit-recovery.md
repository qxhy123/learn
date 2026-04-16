# 第15章：reflog 与丢失提交恢复

## 学习目标

完成本章后，你应能：

1. 理解 `reflog` 记录的是什么
2. 使用 `reflog` 找回被 reset、切换分支或游离 HEAD 后“丢失”的提交
3. 知道为什么很多看似消失的提交其实还在
4. 形成“出事先看 reflog”的恢复习惯

---

## 15.1 `reflog` 看的是“你怎么移动过”，不是“项目怎么演化的”

`git log` 看的是提交历史；

`git reflog` 看的是：

> HEAD 或分支引用在本地如何移动过。

例如：

- 你切过哪些分支
- 你 reset 到过哪里
- 你 rebase 前在哪
- 你 amend 前 HEAD 指向哪

这就是它在恢复误操作时格外有用的原因。

---

## 15.2 为什么很多“丢掉的提交”其实没丢

很多时候，提交并不是被彻底删除了，而是：

- 当前分支不再指向它
- 你暂时失去了通往它的显式引用

例如执行了：

```bash
git reset --hard HEAD~1
```

你会感觉最近一次提交“消失了”，但它常常仍然存在于对象库中，只是当前分支不再指向它。

reflog 的价值，就是帮你找回“我曾经到过哪里”。

---

## 15.3 一个稳定的恢复流程

只要出现这些症状：

- 提交好像不见了
- reset 过头了
- 切分支后工作没了
- detached HEAD 上的提交找不到了

请先按这个顺序来：

1. 停止继续乱试命令
2. 运行 `git reflog`
3. 找到疑似正确位置
4. 先新建救援分支保存现场

```bash
git switch -c rescue-branch <target>
```

这个顺序比“直接 `reset --hard` 猜回去”安全得多。

---

## 15.4 使用 `reflog` 找回位置

查看记录：

```bash
git reflog
```

你可能看到：

```text
abc1234 HEAD@{0}: reset: moving to HEAD~1
def5678 HEAD@{1}: commit: add payment retry logic
```

这表示：

- `HEAD@{0}` 是当前状态
- `HEAD@{1}` 是上一步之前的状态

如果你只是想先保住现场：

```bash
git switch -c rescue-branch HEAD@{1}
```

如果你确认要回去，再考虑：

```bash
git reset --hard HEAD@{1}
```

但建议先建救援分支。

---

## 15.5 detached HEAD 提交为什么也能找回

如果你在 detached HEAD 上做了提交，后来切回分支，这些提交可能“看起来不见了”，但 reflog 往往仍然记得你曾到过那里。

找回方法通常是：

1. `git reflog`
2. 定位对应提交
3. 新建分支指向它

```bash
git switch -c recovered-work <commit>
```

这就是为什么 detached HEAD 不等于“仓库坏了”，只是“需要你给这段历史重新起名字”。

---

## 15.6 reflog 的边界

`reflog` 很强，但也有边界：

- 它主要是本地记录，不是团队共享历史
- 记录不会永久保留
- 如果对象最终被垃圾回收，过很久之后未必还能找回

所以它是强力恢复工具，但不是无限时光机。

如果 reflog 已经不够，有时还要借助更底层的对象排查工具；但对大多数常见误操作，reflog 已经是第一层最强入口。

---

## 15.7 一个必须动手练的恢复实验

建议按下面顺序做三组实验：

### 实验 1：错误 reset

1. 做一次提交
2. `git reset --hard HEAD~1`
3. 用 `git reflog` 找回

### 实验 2：detached HEAD 提交

1. 切到历史提交
2. 做一次新提交
3. 切回原分支
4. 用 `reflog` 与新分支救回

### 实验 3：先保现场再决定

1. 找到正确 reflog 位置
2. 先 `git switch -c rescue-branch ...`
3. 再决定是否真的 reset 回去

这三组练完，你对 Git 恢复的恐惧会明显下降。

---

## 常见误区

- **误区 1：当前分支上看不见的提交就彻底丢了。**
  很多时候只是引用丢了。

- **误区 2：reflog 和 log 是同一种历史。**
  一个看提交图，一个看引用移动。

- **误区 3：恢复时直接猜一个位置 `reset --hard`。**
  更稳妥的是先建救援分支。

- **误区 4：reflog 能永久兜底所有问题。**
  它很强，但不是无限保留。

---

## 本章练习

1. 做一次提交后执行 `git reset --hard HEAD~1`，再用 `git reflog` 找回它。
2. 进入 detached HEAD，创建一条提交，再切回主分支，然后尝试恢复。
3. 用 `git switch -c rescue-branch <target>` 练习“先保现场”的方式。
4. 思考题：为什么 reflog 更像“引用移动轨迹”，而不是“项目演化史”？

---

## 本章小结

`reflog` 是 Git 中最像“后悔药”的工具之一。下一章我们转向效率工具：什么时候该 stash，什么时候不该 stash，以及 `clean` 为什么经常比表面看起来更危险。
