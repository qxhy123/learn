# 23 — Matrix（融合版）

> **难度**：★★★☆☆
> **题数**：5
> **核心套路**：原地修改（标记法 / 状态压缩）、螺旋遍历（边界缩进）、矩阵旋转（转置 + 翻转）、数独验证（哈希 / 位集）、生命游戏（编码历史态）
> **本文件**：覆盖 matrix 5 题的算法套路总结 + 典型题精讲 + 自测

---

## 一例速记

> **Set Matrix Zeroes（73）**：用矩阵第一行 / 第一列作为标记空间，记录哪行哪列需要置零；需单独变量记录第一行 / 第一列本身是否含零，最后统一处理，避免标记与实际数据互相干扰，$O(1)$ 额外空间
> **Rotate Image（48）**：先沿主对角线转置（`matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]`），再水平翻转每行（`row.reverse()`）→ 顺时针旋转 90°；逆时针：先水平翻转再转置
> **Spiral Matrix（54）**：四个边界指针 top / bottom / left / right，按"右 → 下 → 左 → 上"顺序遍历，每遍历一条边后收缩对应边界
> **Valid Sudoku（36）**：同时维护行、列、3×3 宫格的已见数字集合，遍历一次完成验证；宫格编号 `(r//3, c//3)`
> **Game of Life（289）**：规则同时应用到所有格，用额外状态位编码"历史态"：当前为 1 将变 0 → 标记为 2；当前为 0 将变 1 → 标记为 3；第二次遍历用 `% 2` 还原最终态
> **AI 关联**：CNN 卷积核在 feature map 上的滑动窗口 = 矩阵分块遍历；转置 / 旋转对应数据增强（flip / rotate augmentation）；生命游戏是元胞自动机，与 GPU 上的并行 SIMD 更新模型高度相关

---

## 思维路径还原

> "看到 **'73 将含零元素的行列全置零'** → $O(1)$ 额外空间标记法：
> 第一步：扫描第一行是否有 0（标记 `row0_has_zero`），扫描第一列是否有 0（标记 `col0_has_zero`）。
> 第二步：对非第一行 / 列的元素，若 `matrix[i][j] == 0`，则 `matrix[i][0] = 0` 且 `matrix[0][j] = 0`。
> 第三步：根据第一行 / 列标记，将对应行 / 列置零（从内向外，避免覆盖标记）。
> 第四步：若 `row0_has_zero` 则第一行全置零，若 `col0_has_zero` 则第一列全置零。
>
> 看到 **'48 原地顺时针旋转 90°'** → 转置 + 翻转：
> ① 对每对 `(i, j)`（`i < j`）交换 `matrix[i][j]` 和 `matrix[j][i]`（转置）；
> ② 对每行调用 `row.reverse()`（水平翻转）。
> 不要想着四角环形替换（虽然也对），转置 + 翻转更容易记忆且不易出错。
>
> 看到 **'54 螺旋遍历'** → 四边界循环：
> 初始化 `top=0, bottom=m-1, left=0, right=n-1`。
> 循环：遍历顶行（left→right），top++；遍历右列（top→bottom），right--；
> 若 top <= bottom：遍历底行（right→left），bottom--；
> 若 left <= right：遍历左列（bottom→top），left++；
> 直到结果数组填满。
>
> 看到 **'36 数独验证'** → 三组集合同时维护：
> 遍历每个格子 `(r, c)`，维护 `rows[r]`、`cols[c]`、`boxes[r//3][c//3]` 三个集合，
> 若数字已在对应集合中则无效；否则加入。
>
> 看到 **'289 生命游戏'** → 编码多状态避免额外空间：
> `1→0`（活→死）用 2 标记，`0→1`（死→活）用 3 标记；
> 计算邻居时：`cell & 1` 取原始值（2 & 1 = 0，3 & 1 = 1）；
> 第二轮：2 → 0，3 → 1（或统一用 `cell % 2`）。"

---

## 学习目标

- 掌握矩阵原地标记法（73）：用第一行 / 列作标记空间，避免额外 $O(mn)$ 数组
- 熟练矩阵旋转的"转置 + 翻转"方法（48），以及四角环形替换的另一写法
- 掌握螺旋遍历的四边界指针模板（54），避免 off-by-one 错误
- 理解数独验证（36）的三维集合检查（行 / 列 / 宫格）
- 理解生命游戏（289）的多状态编码技巧，原地 $O(1)$ 额外空间实现
- 能识别 CNN feature map 操作与矩阵遍历的类比关系

---

## 抽象成方法（标准模板代码）

### 套路 1：原地标记法（Set Matrix Zeroes）

适用题：73

```python
from typing import List


def set_zeroes(matrix: List[List[int]]) -> None:
    """
    73: 将含零元素所在行列全部置零，原地修改，O(1) 额外空间。
    用第一行和第一列作为标记数组。
    """
    m, n = len(matrix), len(matrix[0])
    # 记录第一行 / 第一列本身是否含 0
    row0_zero = any(matrix[0][j] == 0 for j in range(n))
    col0_zero = any(matrix[i][0] == 0 for i in range(m))

    # 用第一行 / 列标记内部零元素
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0

    # 根据标记，将内部行列置零（先处理内部，再处理第一行/列）
    for i in range(1, m):
        if matrix[i][0] == 0:
            for j in range(1, n):
                matrix[i][j] = 0
    for j in range(1, n):
        if matrix[0][j] == 0:
            for i in range(1, m):
                matrix[i][j] = 0

    # 最后处理第一行 / 第一列
    if row0_zero:
        for j in range(n):
            matrix[0][j] = 0
    if col0_zero:
        for i in range(m):
            matrix[i][0] = 0
```

---

### 套路 2：矩阵旋转（转置 + 翻转）

适用题：48

```python
def rotate_image(matrix: List[List[int]]) -> None:
    """
    48: 原地顺时针旋转 90°，时间 O(n²)，空间 O(1)。
    步骤：① 沿主对角线转置；② 水平翻转每行。
    """
    n = len(matrix)
    # 步骤 1：转置（matrix[i][j] <-> matrix[j][i]，i < j 的部分）
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # 步骤 2：水平翻转每行
    for row in matrix:
        row.reverse()


# 逆时针旋转 90°（先水平翻转，再转置）
def rotate_ccw(matrix: List[List[int]]) -> None:
    n = len(matrix)
    for row in matrix:
        row.reverse()
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


# 四角环形替换（另一写法，直接模拟环形旋转）
def rotate_image_ring(matrix: List[List[int]]) -> None:
    """
    逐层从外到内，四角依次旋转。同样 O(n²) 时间，O(1) 空间。
    """
    n = len(matrix)
    for layer in range(n // 2):
        first, last = layer, n - 1 - layer
        for i in range(first, last):
            offset = i - first
            top = matrix[first][i]
            # 左 → 顶
            matrix[first][i] = matrix[last - offset][first]
            # 底 → 左
            matrix[last - offset][first] = matrix[last][last - offset]
            # 右 → 底
            matrix[last][last - offset] = matrix[i][last]
            # 顶 → 右
            matrix[i][last] = top
```

---

### 套路 3：螺旋遍历（四边界缩进）

适用题：54

```python
def spiral_order(matrix: List[List[int]]) -> List[int]:
    """
    54: 螺旋顺序遍历矩阵，时间 O(m·n)，空间 O(1)（不含输出）。
    四边界：top, bottom, left, right，按层收缩。
    """
    result: List[int] = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # 顶行：从左到右
        for c in range(left, right + 1):
            result.append(matrix[top][c])
        top += 1
        # 右列：从上到下
        for r in range(top, bottom + 1):
            result.append(matrix[r][right])
        right -= 1
        # 底行：从右到左（需判断仍有行）
        if top <= bottom:
            for c in range(right, left - 1, -1):
                result.append(matrix[bottom][c])
            bottom -= 1
        # 左列：从下到上（需判断仍有列）
        if left <= right:
            for r in range(bottom, top - 1, -1):
                result.append(matrix[r][left])
            left += 1
    return result
```

---

### 套路 4：数独验证（三组集合）

适用题：36

```python
def is_valid_sudoku(board: List[List[str]]) -> bool:
    """
    36: 验证数独是否有效（不需要能解出），时间 O(81)=O(1)，空间 O(81)=O(1)。
    三组集合：rows[9]、cols[9]、boxes[3][3]，各存储已出现的数字。
    """
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [[set() for _ in range(3)] for _ in range(3)]

    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == '.':
                continue
            # 宫格编号
            br, bc = r // 3, c // 3
            if val in rows[r] or val in cols[c] or val in boxes[br][bc]:
                return False
            rows[r].add(val)
            cols[c].add(val)
            boxes[br][bc].add(val)
    return True
```

---

### 套路 5：生命游戏（多状态编码）

适用题：289

```python
def game_of_life(board: List[List[int]]) -> None:
    """
    289: 按规则同时更新所有细胞，原地修改，O(1) 额外空间。
    状态编码：1→0 标记为 2（活死），0→1 标记为 3（死活）。
    读取原态：cell & 1（2 & 1 = 0，3 & 1 = 1，保持原义）。
    """
    m, n = len(board), len(board[0])
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def count_live(r: int, c: int) -> int:
        return sum(
            1 for dr, dc in dirs
            if 0 <= r+dr < m and 0 <= c+dc < n and board[r+dr][c+dc] & 1
        )

    # 第一轮：标记变化
    for r in range(m):
        for c in range(n):
            live = count_live(r, c)
            if board[r][c] == 1 and live not in (2, 3):
                board[r][c] = 2    # 活 → 死
            elif board[r][c] == 0 and live == 3:
                board[r][c] = 3    # 死 → 活

    # 第二轮：还原最终态
    for r in range(m):
        for c in range(n):
            board[r][c] %= 2       # 2→0（死），3→1（活），0/1 不变
```

---

### 速查表

| 题型特征 | 套路 | 时间 | 空间 |
|---|---|---|---|
| 含零行列全置零 | 第一行/列作标记 + 两轮遍历 | $O(mn)$ | $O(1)$ |
| 顺时针旋转 90° | 转置 + 水平翻转 | $O(n^2)$ | $O(1)$ |
| 逆时针旋转 90° | 水平翻转 + 转置 | $O(n^2)$ | $O(1)$ |
| 螺旋遍历 | 四边界缩进循环 | $O(mn)$ | $O(1)$ |
| 数独合法性验证 | 三组集合（行 / 列 / 宫格）| $O(1)$（固定 81 格）| $O(1)$ |
| 生命游戏原地更新 | 多状态编码（2/3）+ 两轮遍历 | $O(mn)$ | $O(1)$ |

---

## 方法变形（3 类）

### 变形 1：原地标记扩展

- **73**（Set Matrix Zeroes）→ 用第一行 / 列作标记，$O(1)$ 空间。
- **289**（Game of Life）→ 用额外状态位（`|= 2`）标记旧态，$O(1)$ 空间。
- 两者共同思路：**不能同时读旧值写新值**——必须先"标记"再"还原"，或保证读写顺序不互相影响。
- 若允许 $O(m+n)$ 额外空间：73 可用两个布尔数组 `zero_rows[m]` 和 `zero_cols[n]`，代码更简单。

### 变形 2：旋转 / 翻转系列

- **顺时针 90°**：转置 + 水平翻转（`row.reverse()`）。
- **逆时针 90°**：水平翻转 + 转置（两步顺序互换）。
- **旋转 180°**：垂直翻转（每行 reverse）+ 水平翻转（反转行顺序），或连续两次 90° 旋转。
- **数据增强（AI）**：PyTorch `torchvision.transforms.RandomHorizontalFlip` / `RandomRotation` 本质上就是矩阵翻转 / 旋转，与 48 题完全相同的操作。

### 变形 3：边界处理技巧

- **54 螺旋遍历**：底行遍历前需判断 `top <= bottom`（防止单行矩阵重复遍历），左列遍历前需判断 `left <= right`（防止单列矩阵重复遍历）。
- **59**（Spiral Matrix II，填充螺旋，非本 category）：框架完全相同，改为写入而非读取。
- **36 宫格编号**：`(r//3, c//3)` 将 9×9 网格划分为 9 个 3×3 宫格，编号范围 (0,0)~(2,2)，是固定技巧。

---

## 思考路标（条件反射）

1. 看到 **"含零行列全置零 / O(1) 空间"** → 第一行 / 第一列作标记，两阶段处理（先内部后边界）
2. 看到 **"顺时针旋转 90°"** → 转置 + 水平翻转；逆时针 → 水平翻转 + 转置
3. 看到 **"螺旋顺序 / 按层遍历"** → 四边界指针（top / bottom / left / right）收缩
4. 看到 **"数独 / 九宫格验证"** → 三组集合：行 set + 列 set + `[r//3][c//3]` 宫格 set
5. 看到 **"生命游戏 / 同时更新"** → 多状态编码（历史值嵌入当前格），`& 1` 读旧值，`% 2` 还原新值
6. 看到 **"矩阵搜索（240 Search 2D Matrix II）"** → 从右上角出发，比目标大则左移，比目标小则下移，$O(m+n)$（此题虽不在本 category 但常与矩阵题一起出现）
7. 看到 **"CNN feature map"** → 卷积 = 滑动矩阵窗口，与螺旋 / 边界遍历思路类比
8. 看到 **"数据增强 flip / rotate"** → 矩阵翻转 / 旋转，与 48 完全相同

---

## 易错点

1. **73 标记顺序**：必须先读内部区域再处理第一行 / 列；若先处理第一行 / 列，内部零元素写入的标记可能被错误置零，导致标记信息丢失。
2. **73 第一行第一列需单独标记**：`row0_has_zero` 和 `col0_has_zero` 需在第二步（用第一行 / 列作标记）之前记录，否则内部元素的标记会污染第一行 / 列的原始状态。
3. **48 转置范围**：转置只交换上三角部分（`j in range(i+1, n)`），否则 `(i,j)` 和 `(j,i)` 会被交换两次回到原始状态。
4. **54 底行和左列的条件判断**：处理底行前需判断 `top <= bottom`（否则单行矩阵会重复添加），处理左列前需判断 `left <= right`（否则单列矩阵会重复添加）；两个条件是必需的，不能省略。
5. **289 读旧值方式**：用 `board[r][c] & 1` 而非 `board[r][c]`（因为可能已被标记为 2 或 3）；若直接读 `board[r][c]` 会把 2 误判为活细胞（2 非 0）。
6. **289 还原新值**：第二轮 `board[r][c] %= 2`：0 → 0，1 → 1，2 → 0，3 → 1，正好对应最终状态。常见错误是用 `board[r][c] //= 2` 或忘记还原。
7. **36 宫格索引**：`boxes[r//3][c//3]` 而非 `boxes[r//3 * 3 + c//3]`（扁平化也可，但矩阵形式更直观）；注意 board 中数字是字符串 `'1'~'9'`，不是整数，set 中存字符串 key。

---

## 典型应用例题

### 例 1：73. Set Matrix Zeroes

**题目**：给定 `m×n` 整数矩阵，若某元素为 0，则将其所在行和列全部置零。原地修改，$O(1)$ 额外空间。

**思路**：用矩阵第一行标记哪列有零，第一列标记哪行有零；单独记录第一行和第一列自身是否含零。分四步：① 记录第一行/列是否有零；② 扫描内部标记；③ 根据标记置零内部；④ 最后处理第一行/列。

**解**：

```python
# 参考：solutions/matrix/p073_set_matrix_zeroes.py
def setZeroes(matrix: List[List[int]]) -> None:
    m, n = len(matrix), len(matrix[0])
    row0 = any(matrix[0][j] == 0 for j in range(n))
    col0 = any(matrix[i][0] == 0 for i in range(m))
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    for i in range(1, m):
        if matrix[i][0] == 0:
            for j in range(1, n):
                matrix[i][j] = 0
    for j in range(1, n):
        if matrix[0][j] == 0:
            for i in range(1, m):
                matrix[i][j] = 0
    if row0:
        for j in range(n): matrix[0][j] = 0
    if col0:
        for i in range(m): matrix[i][0] = 0
```

**分析**：时间 $O(mn)$，空间 $O(1)$（仅使用两个布尔变量和矩阵自身的第一行 / 列）。

---

### 例 2：48. Rotate Image

**题目**：给定 $n \times n$ 整数矩阵，将其顺时针旋转 90°，原地修改。

**思路**：顺时针旋转 90° = 先沿主对角线转置，再水平翻转每行。数学推导：旋转后 $(i,j) \to (j, n-1-i)$；转置后 $(i,j) \to (j,i)$，再水平翻转后 $(j,i) \to (j, n-1-i)$，恰好吻合。

**解**：

```python
# 参考：solutions/matrix/p048_rotate_image.py
def rotate(matrix: List[List[int]]) -> None:
    n = len(matrix)
    # 转置（上三角交换）
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # 水平翻转每行
    for row in matrix:
        row.reverse()
```

**分析**：转置 $O(n^2/2)$ 次交换，翻转 $O(n^2/2)$ 次交换，总体 $O(n^2)$ 时间，$O(1)$ 空间。

---

### 例 3：54. Spiral Matrix

**题目**：给定 `m×n` 矩阵，按螺旋顺序（顺时针）返回所有元素。

**思路**：四边界指针（top / bottom / left / right），按"右 → 下 → 左 → 上"顺序遍历每条边，每遍历一条边后收缩对应边界。注意底行和左列在遍历前需检查边界有效性（防止单行 / 单列重复）。

**解**：

```python
# 参考：solutions/matrix/p054_spiral_matrix.py
def spiralOrder(matrix: List[List[int]]) -> List[int]:
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1): result.append(matrix[top][c])
        top += 1
        for r in range(top, bottom + 1): result.append(matrix[r][right])
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1): result.append(matrix[bottom][c])
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1): result.append(matrix[r][left])
            left += 1
    return result
```

**分析**：时间 $O(mn)$（每个元素恰好访问一次），空间 $O(1)$（不含输出数组）。

---

## 自测题

**自测 1**（73 Set Matrix Zeroes）—— `matrix=[[1,1,1],[1,0,1],[1,1,1]]` 应得 `[[1,0,1],[0,0,0],[1,0,1]]`；`matrix=[[0,1,2,0],[3,4,5,2],[1,3,1,5]]` 应得 `[[0,0,0,0],[0,4,5,0],[0,3,1,0]]`。提示：先记录第一行/列有无零，再用第一行/列标记内部，最后处理内部行列，最后处理第一行/列。参考 `solutions/matrix/p073_set_matrix_zeroes.py`。

**自测 2**（48 Rotate Image）—— `matrix=[[1,2,3],[4,5,6],[7,8,9]]` 旋转后应为 `[[7,4,1],[8,5,2],[9,6,3]]`。提示：先转置（只遍历 `j > i` 的半三角），再 `row.reverse()`。参考 `solutions/matrix/p048_rotate_image.py`。

**自测 3**（54 Spiral Matrix）—— `matrix=[[1,2,3],[4,5,6],[7,8,9]]` 应返回 `[1,2,3,6,9,8,7,4,5]`；单行 `[[1,2,3,4]]` 应返回 `[1,2,3,4]`；单列 `[[1],[2],[3]]` 应返回 `[1,2,3]`。提示：四边界指针，底行左列前需判断 `top<=bottom` / `left<=right`。参考 `solutions/matrix/p054_spiral_matrix.py`。

**自测 4**（36 Valid Sudoku）—— 标准数独盘面返回 True；若某行有两个 `'5'` 返回 False；某 3×3 宫格有重复返回 False。提示：三组集合，宫格编号 `(r//3, c//3)`，跳过 `'.'`。参考 `solutions/matrix/p036_valid_sudoku.py`。

**自测 5**（289 Game of Life）—— `board=[[0,1,0],[0,0,1],[1,1,1],[0,0,0]]`，一步后应为 `[[0,0,0],[1,0,1],[0,1,1],[0,1,0]]`。提示：第一轮标记 `1→0` 为 2，`0→1` 为 3；`cell & 1` 读原值；第二轮 `% 2` 还原。参考 `solutions/matrix/p289_game_of_life.py`。

---

## 题目全览（5 题）

| # | 题目 | 套路分类 | 难度 |
|---|---|---|---|
| 36 | Valid Sudoku | 三组集合（行 / 列 / 宫格）验证 | Medium |
| 48 | Rotate Image | 转置 + 水平翻转 | Medium |
| 54 | Spiral Matrix | 四边界缩进循环 | Medium |
| 73 | Set Matrix Zeroes | 第一行 / 列作标记 + O(1) 空间 | Medium |
| 289 | Game of Life | 多状态编码原地更新 | Medium |

---

## 融合版说明

| 段 | 来源 | 价值 |
|---|---|---|
| 一例速记 | 本文件 | 5 题套路一览 + AI（CNN feature map / 数据增强）关联 |
| 思维路径还原 | 本文件 | 5 道题的解题独白，含关键技巧 |
| 抽象成方法 | 本文件 | 5 个标准模板（原地标记 / 旋转 / 螺旋 / 数独 / 生命游戏）+ 速查表 |
| 方法变形 | 本文件 | 3 类变体（原地标记扩展 / 旋转系列 / 边界处理） |
| 思考路标 | 本文件 | 8 条题型识别条件反射 |
| 易错点 | 本文件 | 7 条高频踩坑（标记顺序 / 转置范围 / 条件判断 / 读旧值） |
| 典型应用例题 | solutions/ | 3 道精讲（73、48、54），代码 + 分析 |
| 自测题 | leetcode | 5 题带提示，链接 solutions 文件 |
| 题目全览 | 本文件 | 5 题完整列表 |

---

> **跨 category 导航**：
> - 矩阵 BFS（岛屿 / 连通区域）→ `19-graph-general.md`（200 Number of Islands）
> - 矩阵二分搜索（74 / 240）→ `04-binary-search.md`（矩阵有序性利用）
> - CNN 卷积核在 feature map 上的滑动 = 矩阵分块遍历，padding 和 stride 决定边界处理方式
> - 生命游戏是冯·诺依曼元胞自动机的代表，GPU 上的并行细胞更新 = SIMD 向量化的矩阵原地操作
