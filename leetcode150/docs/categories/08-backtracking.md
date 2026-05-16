# 08 — Backtracking（融合版）

> **难度**：★★★★☆
> **题数**：7
> **核心套路**：选择 → 递归 → 撤销（三件套）、组合、排列、约束满足
> **本文件**：覆盖 backtracking 7 题的算法套路总结 + 典型题精讲 + 自测

---

## 一例速记

> **回溯三件套**：`for choice in choices: path.append(choice)  →  recurse(...)  →  path.pop()`；对应"选择→递归→撤销"三步，共享同一 `path` 列表
> **组合（77 / 39）**：从 `start` 开始枚举，避免重复；77 选 k 个不重复数，39 可重复选但剪 sum
> **排列（46）**：全排列用 `used[]` 标记已选元素，元素不固定起点
> **字母组合（17）**：电话号码每位独立选一个字母，DFS 深度 = 号码长度，宽度 = 按键字母数
> **N-Queens（52）**：用列 / 正对角线 / 反对角线三个集合做冲突检测，不需要检查每行（一行放一个）
> **生成括号（22）**：参数携带 `open` / `close` 计数，约束：`open <= n`，`close <= open`
> **网格搜索（79）**：在二维格子上 DFS，用"原地修改标记+回退"替代 visited 数组
> **AI 关联**：约束满足问题（CSP）/ 搜索剪枝 / 超参组合网格搜索 / 神经架构搜索（NAS）

---

## 思维路径还原

> "看到 **77 Combinations**：从 `[1..n]` 中取 k 个数的所有组合 →
> 回溯，`start` 指针保证不重复，`path` 长度到 k 时收集结果。
> 剪枝：若剩余数字数 `n - start + 1 < k - len(path)`，即使全选也凑不够，直接 `return`。
>
> 看到 **39 Combination Sum**：给定 candidates（无重复），找所有和为 target 的组合，可重复选 →
> 回溯时不推进 `start`（允许重复选同一元素），但用 `target - candidates[i]` 剪负值分支。
> 排序后若 `candidates[i] > target` 则后续全剪。
>
> 看到 **46 Permutations**：给定无重复整数，求所有全排列 →
> 每层从全集中选一个未用的元素，用 `used` 布尔数组标记，`path` 长度等于 n 时收集。
> 不需要 `start`——排列关心顺序，每次从 0 开始扫。
>
> 看到 **17 Letter Combinations**：电话号码 → 字母组合 →
> DFS 深度 = 号码长度，第 idx 层选 `digit_map[digits[idx]]` 中的一个字母；到底时收集。
> 注意空输入时直接返回 `[]`。
>
> 看到 **52 N-Queens II**：n×n 棋盘放 n 个不攻击的皇后，返回方案数 →
> 按行放置，每行恰好放一个；用集合 `cols`、`diag1`（行-列）、`diag2`（行+列）三个集合检测冲突；
> 到第 n 行时计数 +1。三个集合加/删对应选择/撤销。
>
> 看到 **22 Generate Parentheses**：n 对括号的所有合法组合 →
> 递归参数 `open` 和 `close`：`open < n` 时可加左括号，`close < open` 时可加右括号；
> `open == close == n` 时收集结果。约束代替回溯撤销——直接传参不修改全局状态更简洁。
>
> 看到 **79 Word Search**：在字符网格中找目标单词 →
> 对每个格子尝试作为起点，DFS 四方向匹配单词；原地标记已访问（`board[r][c] = '#'`），
> 递归结束后恢复（`board[r][c] = tmp`）；匹配完整单词返回 True 立即终止。"

---

## 学习目标

- 掌握回溯"选择 → 递归 → 撤销"三件套，能徒手写出标准框架
- 区分组合（有 `start` 去重）与排列（无 `start`、有 `used` 标记）的写法差异
- 理解剪枝的两类手段：约束剪（sum 超 target、凑不够 k 个）和状态剪（visited / 集合冲突）
- 掌握 N-Queens 的三集合冲突检测方案，避免逐行逐格扫描
- 理解网格回溯中"原地标记"的技巧，与 visited 数组的等价性
- 识别生成括号问题"参数约束"替代"撤销"的简洁写法

---

## 抽象成方法（标准模板代码）

### 回溯通用骨架

```python
from typing import List

def backtrack_skeleton(nums: List[int]) -> List[List[int]]:
    """回溯三件套骨架：选择 → 递归 → 撤销。"""
    result: List[List[int]] = []
    path: List[int] = []

    def backtrack(start: int) -> None:
        # 终止条件（视题目而定）
        if len(path) == len(nums):
            result.append(path[:])   # 收集当前路径的快照
            return
        for i in range(start, len(nums)):
            path.append(nums[i])     # 选择
            backtrack(i + 1)         # 递归（组合用 i+1，排列用 0）
            path.pop()               # 撤销

    backtrack(0)
    return result
```

---

### 套路 1：组合类（77 / 39）

适用题：77（Combinations）、39（Combination Sum）

```python
# 77: 从 [1..n] 中取 k 个数的所有组合
def combine(n: int, k: int) -> List[List[int]]:
    """时间 O(C(n,k)·k)，空间 O(k)（递归栈 + path）。"""
    result: List[List[int]] = []
    path: List[int] = []

    def backtrack(start: int) -> None:
        if len(path) == k:
            result.append(path[:])
            return
        # 剪枝：剩余候选数不足以凑成 k 个则跳出
        for i in range(start, n - (k - len(path)) + 2):
            path.append(i)
            backtrack(i + 1)
            path.pop()

    backtrack(1)
    return result


# 39: 候选数可重复选，找和为 target 的所有组合
def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
    """时间 O(n^(target/min))，空间 O(target/min)（最大递归深度）。"""
    candidates.sort()                # 排序后可提前剪枝
    result: List[List[int]] = []
    path: List[int] = []

    def backtrack(start: int, remaining: int) -> None:
        if remaining == 0:
            result.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:  # 剪枝：后续更大，全部跳过
                break
            path.append(candidates[i])
            backtrack(i, remaining - candidates[i])  # i 不 +1，允许重复选
            path.pop()

    backtrack(0, target)
    return result
```

---

### 套路 2：排列类（46）

适用题：46（Permutations）

```python
# 46: 无重复整数的全排列
def permute(nums: List[int]) -> List[List[int]]:
    """时间 O(n!·n)，空间 O(n)。"""
    result: List[List[int]] = []
    path: List[int] = []
    used = [False] * len(nums)

    def backtrack() -> None:
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            used[i] = False

    backtrack()
    return result
```

---

### 套路 3：字母组合 / 逐位选择（17）

适用题：17（Letter Combinations of a Phone Number）

```python
# 17: 电话号码字母组合
def letterCombinations(digits: str) -> List[str]:
    """时间 O(4^n·n)，空间 O(n)（n 为 digits 长度，最多 4 字母/键）。"""
    if not digits:
        return []

    digit_map = {
        '2': 'abc', '3': 'def',  '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz',
    }
    result: List[str] = []
    path: List[str] = []

    def backtrack(idx: int) -> None:
        if idx == len(digits):
            result.append(''.join(path))
            return
        for ch in digit_map[digits[idx]]:
            path.append(ch)
            backtrack(idx + 1)
            path.pop()

    backtrack(0)
    return result
```

---

### 套路 4：N-Queens 三集合约束（52）

适用题：52（N-Queens II）

```python
# 52: N 皇后方案数
def totalNQueens(n: int) -> int:
    """时间 O(n!)，空间 O(n)（三个集合各最多 n 个元素）。"""
    count = 0
    cols:  set[int] = set()
    diag1: set[int] = set()   # row - col（主对角线标识）
    diag2: set[int] = set()   # row + col（副对角线标识）

    def backtrack(row: int) -> None:
        nonlocal count
        if row == n:
            count += 1
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            backtrack(row + 1)
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return count
```

---

### 套路 5：生成括号（参数约束型回溯）（22）

适用题：22（Generate Parentheses）

```python
# 22: 生成所有合法括号组合
def generateParenthesis(n: int) -> List[str]:
    """时间 O(4^n / n^(3/2))（卡特兰数），空间 O(n)（递归深度 2n）。"""
    result: List[str] = []

    def backtrack(path: str, open_cnt: int, close_cnt: int) -> None:
        if len(path) == 2 * n:
            result.append(path)
            return
        if open_cnt < n:
            backtrack(path + '(', open_cnt + 1, close_cnt)
        if close_cnt < open_cnt:
            backtrack(path + ')', open_cnt, close_cnt + 1)

    backtrack('', 0, 0)
    return result
```

---

### 套路 6：网格 DFS + 原地标记（79）

适用题：79（Word Search）

```python
# 79: 单词搜索
def exist(board: List[List[str]], word: str) -> bool:
    """时间 O(m·n·4^L)（L 为 word 长度），空间 O(L)（递归栈）。"""
    rows, cols = len(board), len(board[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def dfs(r: int, c: int, idx: int) -> bool:
        if idx == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if board[r][c] != word[idx]:
            return False
        tmp = board[r][c]
        board[r][c] = '#'                # 原地标记：防止同一路径重复访问
        for dr, dc in directions:
            if dfs(r + dr, c + dc, idx + 1):
                board[r][c] = tmp        # 恢复（回溯）
                return True
        board[r][c] = tmp                # 恢复（回溯）
        return False

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False
```

---

### 速查表

| 题目 | 套路类型 | 去重方式 | 关键剪枝 | 时间复杂度 |
|---|---|---|---|---|
| 77 Combinations | 组合，有 `start` | `start` 推进 | 剩余元素不足 k 个 | $O(C(n,k) \cdot k)$ |
| 39 Combination Sum | 组合，可重选 | `start` 不推进 | `candidates[i] > remaining` | $O(n^{t/m})$ |
| 46 Permutations | 排列，有 `used[]` | `used[i]` 标记 | 无（全排列） | $O(n! \cdot n)$ |
| 17 Letter Combinations | 逐位 DFS | 按位推进 idx | 无 | $O(4^n \cdot n)$ |
| 52 N-Queens II | 按行放置 | 三集合冲突检测 | `col/diag1/diag2` 命中跳过 | $O(n!)$ |
| 22 Generate Parentheses | 参数约束型 | `open <= n`，`close <= open` | 约束即剪枝 | $O(4^n / n^{3/2})$ |
| 79 Word Search | 网格 DFS | 原地 `#` 标记 | 字符不匹配立即返回 | $O(mn \cdot 4^L)$ |

---

## 方法变形（4 类）

### 变形 1：组合系列

- **77**（C(n,k)）→ **39**（重复选，剪 sum）→ **40**（有重复元素，排序后跳过 `candidates[i] == candidates[i-1]`，非本 category）：三题均用 `start` 指针控制枚举起点，核心差别在于"是否允许重选"和"是否有重复元素"。
- **剪枝力度对比**：77 剪"剩余不足 k"，39 剪"超过 target"，重复元素题还要跳过同层重复项。
- **泛化**：所有"从集合中选若干元素、顺序无关"的组合类问题，首先考虑 `start` 指针 + 递归 + 剪枝框架。

### 变形 2：排列系列

- **46**（无重复元素）→ **47**（有重复元素，`used[i-1]` 控制同层跳过，非本 category）：46 用 `used[]` 数组，47 额外需要排序 + 同层去重。
- **交换法（swap）**：另一种实现排列的方式是 `nums[i], nums[start] = nums[start], nums[i]`，再递归 `backtrack(start+1)`，最后 swap 回来。代码更简洁但思路略难理解。
- **排列 vs 组合的区分**：排列关心顺序，`[1,2]` 和 `[2,1]` 是不同答案，不用 `start` 但要 `used`；组合不关心顺序，用 `start` 避免重复。

### 变形 3：约束满足（N-Queens / 生成括号）

- **52**（N-Queens II，计数）→ **51**（N-Queens，返回棋盘，非本 category）：51 额外在 `count += 1` 处构造并收集棋盘字符串。
- **三集合判断**：行不需要集合（每行恰好一个），列 `col in cols`，主对角 `row-col`，副对角 `row+col`——这三个标识在同一直线上的值相同，用 set 检测 O(1)。
- **22 的"无撤销"写法**：`generateParenthesis` 用字符串拼接（不可变）传参，天然无需撤销步骤，比维护 `path` 列表更简洁——适用于状态量小且用不可变对象表示的场景。

### 变形 4：网格搜索

- **79**（单词匹配）→ **200**（岛屿数量，不含本 category，属 graph）：79 每次 DFS 从当前格出发匹配单词，找到则立即返回；200 DFS 用来"淹没"整个连通分量，两者 DFS 框架相同，语义不同。
- **原地标记 vs visited 数组**：原地改 `board[r][c] = '#'` 节省空间，但必须在函数退出前还原；visited 数组更安全但需 `O(m·n)` 额外空间。
- **剪枝机会**：79 可提前检查 word 中各字符在 board 中的频率——若 board 里某字符出现次数少于 word 中的需求量，可直接返回 False（O(mn) 预处理，跳过大量无效搜索）。

---

## 思考路标（条件反射）

1. 看到 **"所有组合 / 子集"** → 回溯 + `start` 指针，去重靠"不往前看"
2. 看到 **"组合 + 可重复选"**（39）→ 递归时 `backtrack(i, ...)` 而非 `backtrack(i+1, ...)`
3. 看到 **"全排列"**（46）→ `used[]` 数组标记，每层从 0 扫到 n-1，跳过 `used[i]`
4. 看到 **"按键字母映射"**（17）→ DFS 深度 = digits 长度，第 `idx` 层选 `digit_map[digits[idx]]`
5. 看到 **"N 皇后 / 数独类约束"** → 三集合（列 / 两对角线）冲突检测，加前删后
6. 看到 **"合法括号"**（22）→ 参数携带 `open_cnt / close_cnt`，约束即剪枝，无需撤销
7. 看到 **"网格中找路径 / 单词"**（79）→ 对每格起点 DFS，原地 `#` 标记防回头，退出时还原
8. 看到 **"搜索树中有多余分支"** → 想剪枝：组合类剪"后续不够"，和类剪"超过目标"，排列类剪"已用"
9. 看到 **"result.append(path[:])"** → 必须是 `path[:]`（快照），否则结果全是同一引用
10. 看到 **"结果集中无重复"但候选有重复** → 排序后跳过同层 `candidates[i] == candidates[i-1]`
11. 看到 **"NAS / 超参组合搜索"**（AI 场景）→ 本质是 combinationSum / combinations 框架 + 约束剪枝
12. 看到 **"CSP 问题（约束满足）"** → 回溯框架 + 前向检验（forward checking）= 本节所有技巧的工程化版本

---

## 易错点

1. **path 快照**：`result.append(path)` 和 `result.append(path[:])` 的区别——前者只存引用，回溯撤销后 path 变化，最终结果全为空或相同；必须用 `path[:]` 或 `list(path)` 拷贝当前状态。
2. **排列中忘写 `used[i] = False`**：选择之后设 `used[i] = True`，递归后必须 `used[i] = False`；若忘记撤销，后续层看到元素被占用，全排列结果会严重缺失。
3. **39 题 start 不推进**：可重复选时递归传 `i`（不是 `i+1`）；若误传 `i+1` 则每个元素只能选一次，变成 77 题逻辑。
4. **77 题剪枝边界**：上界应为 `n - (k - len(path)) + 2`（Python range 右开），而非 `n + 1`；若上界过大，虽然结果正确但剪枝无效，时间复杂度退化。
5. **22 题字符串拼接**：`path + '('` 创建新字符串（不可变），不需要撤销；若改成 `path.append('(')` + `path.pop()` 的列表写法，逻辑等价但要记得撤销。
6. **52 题对角线标识**：主对角线用 `row - col`（同一主对角线上差值相同），副对角线用 `row + col`（同一副对角线上和相同）；两者容易混淆，写错则约束失效导致放置冲突皇后。
7. **79 题标记未还原**：`board[r][c] = '#'` 之后，无论 DFS 结果如何（True 或 False），退出前都必须 `board[r][c] = tmp`；若在返回 True 时忘记还原，下一个起点的搜索会受污染。
8. **17 题空输入**：`digits = ""` 时应返回 `[]` 而非 `[""]`；需在入口处显式判断 `if not digits: return []`，否则 backtrack(0) 在 `idx == len(digits)` 时会 append 一个空字符串。

---

## 典型应用例题

### 例 1：46. Permutations

**题目**：给定无重复整数数组 `nums`，返回所有全排列。

**思路**：每次从未使用的元素中选一个加入 path，`used` 数组标记已选。递归深度达到 `n` 时收集结果。没有 `start` 参数——排列关心顺序，每层都从头扫描全集。

**解**：

```python
# 参考：solutions/backtracking/p046_permutations.py
def permute(nums: List[int]) -> List[List[int]]:
    result: List[List[int]] = []
    path: List[int] = []
    used = [False] * len(nums)

    def backtrack() -> None:
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            used[i] = False

    backtrack()
    return result
```

**分析**：$O(n! \cdot n)$ 时间（$n!$ 个叶节点，每个收集 $O(n)$），$O(n)$ 空间（`path` + `used` + 递归栈均 $O(n)$）。与组合题的核心区别：无 `start` 参数，每层对所有元素重新扫描，靠 `used` 而非位置避免重复。

---

### 例 2：52. N-Queens II

**题目**：$n \times n$ 棋盘放 $n$ 个互不攻击的皇后，返回方案总数。

**思路**：按行放置（一行一个），对每列检查是否与已放皇后冲突。冲突检测：同列（`col in cols`）、主对角线（`row-col` 相同）、副对角线（`row+col` 相同）。三个集合的增删对应选择与撤销。

**解**：

```python
# 参考：solutions/backtracking/p052_n_queens_ii.py
def totalNQueens(n: int) -> int:
    count = 0
    cols: set[int] = set()
    diag1: set[int] = set()   # row - col
    diag2: set[int] = set()   # row + col

    def backtrack(row: int) -> None:
        nonlocal count
        if row == n:
            count += 1
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            cols.add(col);    diag1.add(row - col);    diag2.add(row + col)
            backtrack(row + 1)
            cols.remove(col); diag1.remove(row - col); diag2.remove(row + col)

    backtrack(0)
    return count
```

**分析**：$O(n!)$ 时间（实际因剪枝远小于 $n!$），$O(n)$ 空间。三集合检测 O(1)，比逐格扫描快。对角线标识 `row±col` 是经典技巧：同一对角线上所有格子的行列差（或和）相同，一个整数即可表示整条对角线。

---

### 例 3：79. Word Search

**题目**：给定字符矩阵 `board` 和字符串 `word`，判断 `word` 是否存在于矩阵中（每格只能用一次，可向上下左右移动）。

**思路**：枚举所有可能的起点，对每个起点 DFS 匹配单词。访问标记用原地修改（`board[r][c] = '#'`），避免额外 visited 数组；回退时还原。单词匹配完整（`idx == len(word)`）时返回 True，立即终止后续搜索。

**解**：

```python
# 参考：solutions/backtracking/p079_word_search.py
def exist(board: List[List[str]], word: str) -> bool:
    rows, cols = len(board), len(board[0])

    def dfs(r: int, c: int, idx: int) -> bool:
        if idx == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[idx]:
            return False
        tmp, board[r][c] = board[r][c], '#'
        found = (dfs(r+1, c, idx+1) or dfs(r-1, c, idx+1) or
                 dfs(r, c+1, idx+1) or dfs(r, c-1, idx+1))
        board[r][c] = tmp   # 回溯：恢复原始字符
        return found

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False
```

**分析**：$O(m \cdot n \cdot 4^L)$ 时间（$m \cdot n$ 个起点，每个起点 DFS 深度 $L$，每层最多 4 方向），$O(L)$ 空间（递归栈）。原地标记的回溯确保了正确性：每条 DFS 路径上已访问的格子不会被再次选取，且不影响其他起点的搜索。

---

## 自测题

**自测 1**（77 题 Combinations）—— `n=4, k=2` 输出 `[[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]`（顺序不限）。💡 提示：`backtrack(start)` 中循环 `range(start, n-(k-len(path))+2)`，`path` 长度到 k 时收集 `path[:]`。参考 `solutions/backtracking/p077_combinations.py`。

**自测 2**（39 题 Combination Sum）—— `candidates=[2,3,6,7], target=7` 输出 `[[2,2,3],[7]]`。💡 提示：排序后循环，`candidates[i] > remaining` 时 `break`；递归传 `i`（不是 `i+1`）允许重复选；`remaining == 0` 时收集。参考 `solutions/backtracking/p039_combination_sum.py`。

**自测 3**（17 题 Letter Combinations）—— `digits="23"` 输出 `["ad","ae","af","bd","be","bf","cd","ce","cf"]`（顺序不限）。💡 提示：DFS 第 `idx` 层遍历 `digit_map[digits[idx]]`，`idx == len(digits)` 时 `result.append(''.join(path))`；空输入直接返回 `[]`。参考 `solutions/backtracking/p017_letter_combinations_of_a_phone_number.py`。

**自测 4**（22 题 Generate Parentheses）—— `n=3` 输出 `["((()))","(()())","(())()","()(())","()()()"]`（顺序不限）。💡 提示：`open_cnt < n` 可加左括号，`close_cnt < open_cnt` 可加右括号，`len(path) == 2*n` 时收集；用字符串拼接不需撤销。参考 `solutions/backtracking/p022_generate_parentheses.py`。

**自测 5**（52 题 N-Queens II）—— `n=4` 输出 `2`，`n=1` 输出 `1`。💡 提示：三集合 `cols`、`diag1`（`row-col`）、`diag2`（`row+col`）；按行递归，第 n 行时 `count += 1`；选择后 add，递归后 remove。参考 `solutions/backtracking/p052_n_queens_ii.py`。

---

## 题目全览（7 题）

| # | 题目 | 套路分类 | 难度 |
|---|---|---|---|
| 17 | Letter Combinations of a Phone Number | 逐位 DFS，字母映射 | Medium |
| 77 | Combinations | 组合回溯，`start` 去重 | Medium |
| 46 | Permutations | 全排列，`used[]` 标记 | Medium |
| 39 | Combination Sum | 组合可重选，剪 sum | Medium |
| 52 | N-Queens II | 三集合约束，按行放置 | Hard |
| 22 | Generate Parentheses | 参数约束型回溯 | Medium |
| 79 | Word Search | 网格 DFS，原地标记 | Medium |

---

## 融合版说明

| 段 | 来源 | 价值 |
|---|---|---|
| 一例速记 | 本文件 | 7 题 6 类套路一览 + AI 关联，扫一眼知要用什么 |
| 思维路径还原 | 本文件 | 7 道题的解题内心独白，含关键决策点 |
| 抽象成方法 | 本文件 | 6 个标准模板代码（骨架 + 5 类子套路）+ 速查表 |
| 方法变形 | 本文件 | 4 类变体扩展（组合 / 排列 / 约束满足 / 网格） |
| 思考路标 | 本文件 | 12 条题型识别条件反射，覆盖全部 7 题 + AI 场景 |
| 易错点 | 本文件 | 8 条高频踩坑（path 快照 / 撤销遗漏 / 对角线标识等） |
| 典型应用例题 | solutions/ | 3 道精讲（46、52、79），代码 + 正确性分析 |
| 自测题 | leetcode | 5 题带 💡 提示，链接 solutions 文件 |
| 题目全览 | 本文件 | 7 题完整列表，套路分类一览 |

---

> **跨 category 导航**：
> - 图的 DFS（连通分量、拓扑排序）与网格搜索结构相同但语义不同 → 见 `graph_general` category
> - 动态规划可视为"回溯 + 记忆化"：若搜索树有大量重叠子问题，把 backtrack 的参数作为 key 存 memo → 见 `09-dynamic-programming` 系列
> - 二叉树 DFS 是回溯的特殊形式（树形搜索空间）→ 见 `07-binary-tree-dfs.md`
> - 约束满足问题（CSP）工程化：arc consistency + backtracking = 本节技巧在 AI Solver 中的应用
