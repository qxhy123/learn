# 05 — Binary Search Tree（融合版）

> **难度**：★★★☆☆
> **题数**：3
> **核心套路**：中序遍历有序性、区间验证 BST 合法性
> **本文件**：覆盖 binary_search_tree 3 题的算法套路总结 + 典型题精讲 + 自测

---

## 一例速记

> **中序遍历有序性**：BST 中序遍历（左→根→右）结果严格递增；$O(n)$ 时间 $O(h)$ 空间（$h$ 为树高），是 530 最小差、230 第 k 小的核心手段
> **迭代中序（栈模拟）**：用显式栈替代递归，可随时中断——找到第 k 个元素即停，避免全量遍历；空间 $O(h)$
> **区间验证**：递归传递 `(lo, hi)` 区间，每个节点值必须在区间内；根节点区间为 $(-\infty, +\infty)$，向左传 `hi = node.val`，向右传 `lo = node.val`（98 验证 BST）
> **BST 搜索**：从根出发，值小则走左子树，值大则走右子树，$O(h)$ 无需遍历全树；是普通二分查找的树形版本
> **AI 关联**：决策树（ID3/C4.5）的划分条件 = BST 分支逻辑；XGBoost 每棵树的节点判断 = BST 搜索路径；排序数据库索引（B-Tree）是 BST 的磁盘友好扩展

---

## 思维路径还原

> "看到 **530 Minimum Absolute Difference in BST**：BST 中任意两节点的最小绝对差 →
> BST 中序有序，所以最小差只可能出现在相邻节点之间。
> 维护 `prev`（中序前驱节点值），每次访问节点时计算 `node.val - prev`，更新全局最小值。
> 递归：先访问左子树，再处理当前节点（更新 prev 和 min_diff），再访问右子树。
> 初始 `prev = -inf`，第一个节点与 prev 的差必然很大，不影响结果。
>
> 看到 **230 Kth Smallest Element in a BST**：BST 中第 k 小的元素 →
> 中序遍历是升序，第 k 个访问的节点就是答案。
> 递推：先中序遍历左子树，每访问一个节点 k--，k 变为 0 时当前节点即为答案。
> 优化：迭代中序用显式栈，找到第 k 个即停，不继续遍历剩余节点——若频繁调用可进一步用"带 rank 的增强 BST"。
>
> 看到 **98 Validate Binary Search Tree**：验证是否是合法 BST →
> 关键陷阱：不能只比较节点与左右子节点，必须检查整棵子树的值域。
> 传递 `(lo, hi)` 区间：根节点为 `(-inf, +inf)`；向左递归时 `hi = node.val`；向右递归时 `lo = node.val`。
> 若 `node.val <= lo` 或 `node.val >= hi` 则非法（注意严格不等号，BST 通常要求严格大于）。
> 空节点返回 True（空树是合法 BST）。"

---

## 学习目标

- 掌握 BST 中序遍历（递归 + 迭代）的两种实现，理解中序结果严格递增的来源
- 能用"中序前驱 prev"技巧 $O(1)$ 额外空间解决最小差类问题
- 理解"区间验证"替代"局部比较"的必要性，避免 98 题最常见的错误思路
- 能写出迭代中序（显式栈）并在找到第 k 个时提前终止
- 认识 BST 与排序数据结构、决策树的工程联系

---

## 抽象成方法（标准模板代码）

### 套路 1：中序遍历（递归）+ 前驱追踪

适用题：530（最小绝对差）

```python
# 假设已有 TreeNode 定义
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val; self.left = left; self.right = right

def getMinimumDifference(root) -> int:
    """BST 中序遍历，相邻节点差的最小值。时间 O(n)，空间 O(h)。"""
    min_diff = float('inf')
    prev = float('-inf')        # 中序前驱的值

    def inorder(node) -> None:
        nonlocal min_diff, prev
        if node is None:
            return
        inorder(node.left)                          # 先遍历左子树
        min_diff = min(min_diff, node.val - prev)   # 处理当前节点
        prev = node.val                             # 更新前驱
        inorder(node.right)                         # 再遍历右子树

    inorder(root)
    return min_diff
```

> 关键：`prev` 初始化为 `float('-inf')` 使第一个节点的差足够大不影响结果；用 `nonlocal` 在嵌套函数中修改外层变量。

### 套路 2：迭代中序（显式栈）+ 计数中断

适用题：230（第 k 小元素）

```python
def kthSmallest(root, k: int) -> int:
    """迭代中序遍历，第 k 次访问即为答案。时间 O(h+k)，空间 O(h)。"""
    stack = []
    cur = root
    count = 0
    while cur is not None or stack:
        # 一路走到最左
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        # 弹出节点，访问
        cur = stack.pop()
        count += 1
        if count == k:
            return cur.val
        # 转向右子树
        cur = cur.right
    raise ValueError("k exceeds tree size")
```

> 迭代中序的"一路走左，弹出访问，转右"三步循环是标准模板，可随时中断而不必遍历整棵树。

### 套路 3：区间验证（递归传递上下界）

适用题：98（Validate Binary Search Tree）

```python
def isValidBST(root) -> bool:
    """时间 O(n)，空间 O(h)。传递 (lo, hi) 区间，每个节点值必须严格在区间内。"""
    def validate(node, lo: float, hi: float) -> bool:
        if node is None:
            return True
        if node.val <= lo or node.val >= hi:
            return False
        return (validate(node.left,  lo,       node.val) and
                validate(node.right, node.val, hi))

    return validate(root, float('-inf'), float('inf'))
```

> 严格不等号：标准 BST 定义为左子树所有值 **严格小于** 根，右子树所有值 **严格大于** 根；区间用开区间 `(lo, hi)` 配严格不等号。

### 套路 4：BST 搜索 / 插入（路径追踪）

适用题：搜索与插入基础操作（本 category 3 题的底层逻辑）

```python
def bst_search(root, target: int):
    """BST 搜索，时间 O(h)，空间 O(1)（迭代版）。"""
    cur = root
    while cur is not None:
        if target == cur.val:
            return cur
        elif target < cur.val:
            cur = cur.left
        else:
            cur = cur.right
    return None


def bst_insert(root, val: int):
    """BST 插入，返回根节点（递归版）。"""
    if root is None:
        return type(root)(val)  # 新建节点
    if val < root.val:
        root.left = bst_insert(root.left, val)
    elif val > root.val:
        root.right = bst_insert(root.right, val)
    return root
```

> BST 搜索是树形二分查找：每一步排除一半子树，平衡树时 $O(\log n)$，退化链时 $O(n)$。

### 套路 5：中序递归（通用框架）

适用题：530、230（复用此框架）

```python
def inorder_collect(root) -> list[int]:
    """收集 BST 中序遍历结果（升序列表）。仅用于理解，生产中可提前终止。"""
    result: list[int] = []

    def dfs(node) -> None:
        if node is None:
            return
        dfs(node.left)
        result.append(node.val)
        dfs(node.right)

    dfs(root)
    return result
    # 530: min(result[i+1] - result[i] for i in range(len(result)-1))
    # 230: result[k-1]
```

> 此版本便于理解但需 $O(n)$ 额外空间存储结果；套路 1/2 是其空间优化版本。

### 速查表

| 题型特征 | 套路 | 时间 | 空间 | 对应题目 |
|---|---|---|---|---|
| BST 中任意两节点最小差 | 中序递归 + prev 追踪 | $O(n)$ | $O(h)$ | 530 |
| BST 第 k 小的元素 | 迭代中序 + 计数中断 | $O(h+k)$ | $O(h)$ | 230 |
| 验证是否为合法 BST | 区间验证递归 `(lo, hi)` | $O(n)$ | $O(h)$ | 98 |
| BST 搜索目标值 | 比较走左/右，迭代 | $O(h)$ | $O(1)$ | 通用 |
| 收集 BST 所有值（升序） | 中序递归收集列表 | $O(n)$ | $O(n)$ | 通用 |

> $h$ 为树高；平衡 BST 时 $h = O(\log n)$，最坏退化链时 $h = O(n)$。

---

## 方法变形（4 类）

### 变形 1：中序前驱追踪系列

- **530**（最小绝对差）：中序前驱 `prev` 追踪，每次更新 `min_diff = min(min_diff, cur.val - prev)`。
- **783**（BST 节点最小距离，非本 category）：与 530 完全一致的模板，只是结果含义相同；LeetCode 两题实质相同。
- **中序转双向链表**（非本 category）：prev 不仅记值，还维护指针，遍历时修改节点 left/right 为前驱/后继。
- **平衡 BST 检查**：中序收集后验证是否严格递增，等价于 98 的区间验证但空间更大。

### 变形 2：迭代中序 / 中断系列

- **230**（第 k 小）：计数到 k 即 return，不继续遍历。
- **BST 迭代器**（173，非本 category）：将迭代中序的栈封装为 `next()` / `hasNext()` 接口，每次调用只前进一步；空间 $O(h)$。
- **中序逆序（右→根→左）**：把所有 `left/right` 互换，得到降序遍历；用于"第 k 大"问题。
- **Morris 中序遍历**：$O(1)$ 空间替代栈，通过临时修改树结构实现；本 category 不需要，但是空间极限优化的参考。

### 变形 3：区间验证扩展系列

- **98**（验证 BST）：`(lo, hi)` 开区间传递，严格不等号。
- **BST 插入**：搜索时路径上每个节点天然维护了"此节点值的合法范围"——插入位置就是搜索失败的叶子位置。
- **BST 删除**（450，非本 category）：找到节点后，若有两个子节点，用中序后继（右子树最左节点）替代；是 98 区间逻辑的逆操作。
- **进阶**：区间验证的 `(lo, hi)` 范围也可以用于剪枝——在搜索 / 计数问题中，若当前节点值已超出目标范围则整棵子树跳过。

### 变形 4：BST 与排序结构的工程联系

- **AI 类比 — 决策树**：每个内部节点是一个阈值判断（`feature_val <= threshold`），与 BST 分支完全同构；叶节点是分类/回归结果。
- **AI 类比 — XGBoost 推理**：对单棵 CART 树做推理 = 从根沿 BST 路径走到叶节点，$O(\log n)$ 次比较。
- **数据库索引 — B-Tree**：B-Tree 是 BST 的 m 叉推广，减少磁盘 I/O；B+Tree 所有数据在叶节点且相互链接，支持范围查询——对应 34 题的区间搜索。
- **平衡 BST（AVL / 红黑树）**：Python `sortedcontainers.SortedList` 底层是分块有序列表；C++ `std::map` / `std::set` 是红黑树；理解 BST 有助于分析其时间复杂度。

---

## 思考路标（条件反射）

1. 看到 **BST + "最小差 / 相邻节点"** → 中序遍历，`prev` 前驱追踪，相邻访问差即候选
2. 看到 **BST + "第 k 小 / 第 k 大"** → 中序（或逆中序）迭代，计数到 k 立即返回
3. 看到 **"验证是否为合法 BST"** → 区间验证 `(lo, hi)`，不要只比较父子节点
4. 看到 **BST + "原地操作 / 频繁查询"** → 迭代版优于递归（无栈溢出风险，可中断）
5. 看到 **"只比较 `node.val` 与 `node.left.val` / `node.right.val`"** → 这是 98 题最常见的错误，反例：`[5, 1, 4, null, null, 3, 6]` 根节点 4 < 5 不合法但局部比较通过
6. 看到 **BST 搜索** → 从根出发比较大小走左/右，平均 $O(\log n)$，不需要遍历全树
7. 看到 **"中序结果"** → 快速收集：`inorder_collect(root)` 拿到有序列表；再做任何有序数组操作
8. 看到 **树高 h** → 平衡 BST $h = O(\log n)$，链状 BST $h = O(n)$；空间/时间分析时需说明哪种情况
9. 看到 **BST + "范围查询 [lo, hi]"** → 递归时若 `node.val < lo` 只走右子树，若 `node.val > hi` 只走左子树，剪枝 $O(\log n + k)$
10. 看到 **"BST 转有序数组 / 链表"** → 中序遍历即可；逆中序得到降序；修改指针可原地转双向链表
11. 看到 **"决策树 / 随机森林推理"** → 联想 BST 的搜索路径；特征阈值比较 = 节点分支；叶节点 = 预测结果

---

## 易错点

1. **98 题"局部比较"陷阱**：只比较 `node.val > node.left.val` 和 `node.val < node.right.val` 会通过反例 `[5, 4, 6, null, null, 3, 7]`——根节点 5 的右子树中有 3 < 5，但局部比较看不出。必须用区间 `(lo, hi)` 验证每个节点的全局合法性。
2. **BST 严格不等号**：标准 BST 定义中左子树值严格小于根，右子树值严格大于根（无重复）；区间验证用 `node.val <= lo or node.val >= hi` 来拒绝等于边界的情况。若题目允许重复则需调整。
3. **530 / 230 的 `prev` 初始化**：`prev` 用于记录中序前驱值，初始化为 `float('-inf')`（不是 0 或 None）；若初始化为 0，当树只有一个节点时 `0 - 0 = 0` 可能错误地返回 0 差值。
4. **迭代中序的"一路走左"顺序**：外层 while 条件为 `cur is not None or stack`；内层先把 cur 一路压栈到最左，再弹出访问，再转右——顺序不能颠倒，否则访问顺序不是中序。
5. **230 迭代版的 `count` 计数时机**：`count += 1` 要在弹出节点后（访问当前节点时）而非压栈时；压栈时节点尚未"访问"，过早计数会导致结果偏移 k 位。
6. **平衡 BST vs 退化链**：时间复杂度通常标注为 $O(h)$，而非 $O(\log n)$；面试中应说明"若树平衡则 $O(\log n)$，退化链则 $O(n)$"，不要直接说 $O(\log n)$。
7. **98 题空节点处理**：`if node is None: return True`——空节点是合法 BST（边界条件），不要返回 False 或抛出异常；遗漏此行会在空树时崩溃。

---

## 典型应用例题

### 例 1：530. Minimum Absolute Difference in BST

**题目**：给定 BST，返回任意两节点之间最小绝对差。

**思路**：BST 中序有序，相邻元素差最小。维护前驱节点值 `prev`，遍历时计算 `node.val - prev` 更新答案。

**解**：

```python
# 参考：solutions/binary_search_tree/p530_minimum_absolute_difference_in_bst.py
def getMinimumDifference(root) -> int:
    min_diff = float('inf')
    prev = float('-inf')

    def inorder(node) -> None:
        nonlocal min_diff, prev
        if node is None:
            return
        inorder(node.left)
        min_diff = min(min_diff, node.val - prev)
        prev = node.val
        inorder(node.right)

    inorder(root)
    return min_diff
```

**分析**：$O(n)$ 时间，$O(h)$ 空间（递归调用栈）。利用 BST 中序严格递增的性质，将"任意两点最小差"转化为"相邻中序元素最小差"，一次遍历解决。

**为什么相邻差最小**：若数组 $a_1 < a_2 < \cdots < a_n$，则对任意 $i < j$，有 $a_j - a_i = \sum_{k=i}^{j-1}(a_{k+1}-a_k) \ge a_{i+1} - a_i$，故最小差必在相邻元素之间。

---

### 例 2：230. Kth Smallest Element in a BST

**题目**：给定 BST 和整数 k，返回 BST 中第 k 个最小元素（1-indexed）。

**思路**：中序遍历升序访问节点，第 k 次访问即为答案。用迭代版（显式栈）避免全量遍历——找到第 k 个立即返回。

**解**：

```python
# 参考：solutions/binary_search_tree/p230_kth_smallest_element_in_a_bst.py
def kthSmallest(root, k: int) -> int:
    stack = []
    cur = root
    count = 0
    while cur is not None or stack:
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        count += 1
        if count == k:
            return cur.val
        cur = cur.right
    raise ValueError("k exceeds tree node count")
```

**分析**：$O(h + k)$ 时间，$O(h)$ 空间。与递归版（$O(n)$ 时间，因为可能遍历全树）相比，迭代版在找到第 k 个后立即终止，对 k 较小或树较大的情况有显著优势。

**进阶**：若需要频繁查询不同的 k，可用"增强 BST"——每个节点额外存储左子树节点数，查询时根据左子树节点数与 k 的比较决定走哪侧，$O(\log n)$ 单次查询。

---

### 例 3：98. Validate Binary Search Tree

**题目**：给定二叉树的根节点，判断是否是有效的 BST（左子树所有节点严格小于根，右子树所有节点严格大于根，左右子树也均为合法 BST）。

**思路**：区间验证。递归时向下传递合法值域 `(lo, hi)`：访问左子节点时上界更新为 `node.val`，访问右子节点时下界更新为 `node.val`；任意节点值超出当前区间则非法。

**解**：

```python
# 参考：solutions/binary_search_tree/p098_validate_binary_search_tree.py
def isValidBST(root) -> bool:
    def validate(node, lo: float, hi: float) -> bool:
        if node is None:
            return True
        if node.val <= lo or node.val >= hi:
            return False
        return (validate(node.left,  lo,       node.val) and
                validate(node.right, node.val, hi))

    return validate(root, float('-inf'), float('inf'))
```

**分析**：$O(n)$ 时间，$O(h)$ 空间。每个节点恰好访问一次；区间验证确保了全局约束而非仅局部父子关系。

**反例演示**：树 `[5, 4, 6, null, null, 3, 7]`——根 5、左子 4、右子 6（根节点局部看合法），但 6 的左子 3 < 5，违反"右子树所有节点 > 5"；局部比较会漏判，区间验证在访问节点 3 时 `lo = 5`，`3 <= 5` 立即返回 False，正确判断非法。

---

## 自测题

**自测 1**（530 题 Minimum Absolute Difference）—— BST `[4,2,6,1,3]`，返回最小绝对差（答案为 1）。💡 提示：中序递归，`prev = -inf`，每次 `min_diff = min(min_diff, node.val - prev)`，再更新 `prev = node.val`；递归结束返回 `min_diff`。参考 `solutions/binary_search_tree/p530_minimum_absolute_difference_in_bst.py`。

**自测 2**（230 题 Kth Smallest Element）—— BST `[3,1,4,null,2]`，k=1 返回 1，k=3 返回 3。💡 提示：迭代中序三步循环："一路走左压栈 → 弹出访问计数 → 转向右子树"；count == k 时立即 return 当前节点值。参考 `solutions/binary_search_tree/p230_kth_smallest_element_in_a_bst.py`。

**自测 3**（98 题 Validate Binary Search Tree）—— 树 `[5,1,4,null,null,3,6]` 返回 False（右子树中有 3 < 5）；树 `[2,1,3]` 返回 True。💡 提示：区间验证，传 `(lo, hi)`；向左子树传 `(lo, node.val)`，向右子树传 `(node.val, hi)`；节点值必须满足 `lo < node.val < hi`。参考 `solutions/binary_search_tree/p098_validate_binary_search_tree.py`。

**自测 4**（综合）—— 手动构造一棵 BST（插入序列 `[5, 3, 7, 1, 4, 6, 8]`），验证其中序遍历是否严格递增，并找出第 3 小的元素。💡 提示：中序收集 `inorder_collect(root)` 得到 `[1, 3, 4, 5, 6, 7, 8]`，第 3 小为 `result[2] = 4`；或用迭代中序计数到 3。

**自测 5**（综合）—— 对树 `[10, 5, 15, null, null, 6, 20]` 调用 `isValidBST`：局部来看 `15 > 10` 合法，但 15 的左子 6 < 10，违反全局约束。手动追踪区间验证递归，说明哪一步 `lo`/`hi` 导致判断为 False。💡 提示：访问节点 6 时，递归路径是 root(10) → right(15) → left(6)，此时 `lo = 10`（从 10 右转传来），`hi = 15`（从 15 左转传来），`6 <= 10` 触发 `return False`。

---

## 题目全览（3 题）

| # | 题目 | 套路分类 | 难度 |
|---|---|---|---|
| 530 | Minimum Absolute Difference in BST | 中序递归 + prev 追踪 | Easy |
| 230 | Kth Smallest Element in a BST | 迭代中序 + 计数中断 | Medium |
| 98 | Validate Binary Search Tree | 区间验证递归 | Medium |

---

## 融合版说明

| 段 | 来源 | 价值 |
|---|---|---|
| 一例速记 | 本文件 | 3 题 2 类套路一览，含 AI/工程类比 |
| 思维路径还原 | 本文件 | 3 道题的解题内心独白，含核心陷阱提示 |
| 抽象成方法 | 本文件 | 5 个标准模板代码 + 速查表，可直接运行 |
| 方法变形 | 本文件 | 4 类变体扩展（前驱追踪 / 迭代中断 / 区间验证 / AI 类比） |
| 思考路标 | 本文件 | 11 条题型识别条件反射，含工程联系 |
| 易错点 | 本文件 | 7 条高频踩坑，覆盖局部比较陷阱、计数时机、空节点处理 |
| 典型应用例题 | solutions/ | 3 道精讲（530、230、98），代码 + 正确性分析 + 反例演示 |
| 自测题 | leetcode | 5 题带 💡 提示，含手动追踪练习 |
| 题目全览 | 本文件 | 3 题完整列表，套路分类一览 |

---

> **跨 category 导航**：
> - BST 搜索本质上是树形二分查找 → 二分查找理论基础见 `04-binary-search.md`
> - BST 的遍历（前序 / 中序 / 后序 / 层序）→ 见 `binary_tree_dfs` / `binary_tree_bfs` category
> - 若需要动态维护有序集合（插入 / 删除 / 查询第 k 小）→ 考虑平衡 BST（`sortedcontainers.SortedList`）或堆（`heap` category）
