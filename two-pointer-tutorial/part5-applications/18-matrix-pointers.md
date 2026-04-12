# 第18章：矩阵双指针

## 18.1 二维空间的双指针思维

矩阵双指针将一维双指针的思想扩展到二维：不再是在一个数组上移动两个指针，而是在矩阵的行列方向上利用有序性或单调性。

核心思路：**将二维问题降维到一维**。

---

## 18.2 行列均有序矩阵的搜索

**问题（LeetCode 74）**：每行升序，每行第一个元素大于上一行最后一个元素（完全有序矩阵），搜索目标值。

```python
def search_matrix_i(matrix, target):
    """
    方法一：当成一维有序数组做二分
    将矩阵下标 mid 映射为 (mid//cols, mid%cols)
    """
    if not matrix or not matrix[0]:
        return False

    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1

    while left <= right:
        mid = (left + right) // 2
        val = matrix[mid // cols][mid % cols]
        if val == target:
            return True
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1

    return False

# 测试
matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
print(search_matrix_i(matrix, 3))   # True
print(search_matrix_i(matrix, 13))  # False
```

**问题（LeetCode 240）**：每行升序，每列升序（行列均有序），搜索目标值（已在第17章介绍）。

```python
def search_matrix_ii(matrix, target):
    """从右上角出发的对撞指针"""
    row, col = 0, len(matrix[0]) - 1
    while row < len(matrix) and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    return False
```

---

## 18.3 矩阵中的双指针：岛屿面积

**问题（LeetCode 695）**：找最大岛屿面积（1 代表陆地，0 代表海洋）。

```python
def max_area_of_island(grid):
    """DFS 探索每个岛屿，双指针思想体现在边界扩展"""
    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == 0:
            return 0
        grid[r][c] = 0  # 标记已访问（原地修改避免额外空间）
        return 1 + dfs(r+1,c) + dfs(r-1,c) + dfs(r,c+1) + dfs(r,c-1)

    return max(dfs(r, c) for r in range(rows) for c in range(cols))
```

---

## 18.4 矩阵对角线遍历

**问题（LeetCode 498）**：按对角线 zigzag 顺序遍历矩阵。

```python
def find_diagonal_order(mat):
    """
    对角线规律：第 d 条对角线上，所有元素满足 i + j = d
    偶数对角线从下往上，奇数对角线从上往下
    """
    if not mat or not mat[0]:
        return []

    rows, cols = len(mat), len(mat[0])
    result = []

    for d in range(rows + cols - 1):
        if d % 2 == 0:  # 从下往上（行减小，列增大）
            r = min(d, rows - 1)
            c = d - r
            while r >= 0 and c < cols:
                result.append(mat[r][c])
                r -= 1
                c += 1
        else:            # 从上往下（行增大，列减小）
            c = min(d, cols - 1)
            r = d - c
            while c >= 0 and r < rows:
                result.append(mat[r][c])
                r += 1
                c -= 1

    return result

# 测试
print(find_diagonal_order([[1,2,3],[4,5,6],[7,8,9]]))
# [1, 2, 4, 7, 5, 3, 6, 8, 9]
```

---

## 18.5 矩阵螺旋遍历（四指针）

```python
def spiral_order(matrix):
    """
    四指针维护上下左右边界
    top, bottom, left, right
    """
    if not matrix:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # 向右：top 行
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1

        # 向下：right 列
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1

        # 向左：bottom 行（如果还有行）
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1

        # 向上：left 列（如果还有列）
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

    return result

# 测试
print(spiral_order([[1,2,3],[4,5,6],[7,8,9]]))
# [1, 2, 3, 6, 9, 8, 7, 4, 5]

print(spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))
# [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
```

**四指针不变量**：`[top, bottom] × [left, right]` 是未处理的矩形区域，每处理一圈，四个边界各收缩一次。

---

## 18.6 矩阵旋转（原地）

```python
def rotate_matrix(matrix):
    """
    顺时针旋转90度，原地操作
    步骤：先转置，再水平翻转（每行翻转）
    """
    n = len(matrix)

    # 步骤1：转置（沿主对角线翻转）
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # 步骤2：每行用对撞指针翻转
    for row in matrix:
        left, right = 0, n - 1
        while left < right:
            row[left], row[right] = row[right], row[left]
            left += 1
            right -= 1

# 测试
matrix = [[1,2,3],[4,5,6],[7,8,9]]
rotate_matrix(matrix)
print(matrix)  # [[7,4,1],[8,5,2],[9,6,3]]
```

**数学推导**：
```
原位置 (i, j) → 旋转后 (j, n-1-i)

分解为两步：
  转置：(i, j) → (j, i)
  水平翻转：(j, i) → (j, n-1-i) ✓
```

---

## 18.7 矩阵中的双指针路径

**问题（LeetCode 329）**：矩阵中最长递增路径。

```python
def longest_increasing_path(matrix):
    """记忆化DFS：每个位置只计算一次"""
    rows, cols = len(matrix), len(matrix[0])
    memo = {}

    def dfs(r, c):
        if (r, c) in memo:
            return memo[(r, c)]

        best = 1
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and matrix[nr][nc] > matrix[r][c]):
                best = max(best, 1 + dfs(nr, nc))

        memo[(r, c)] = best
        return best

    return max(dfs(r, c) for r in range(rows) for c in range(cols))

# 测试
print(longest_increasing_path([[9,9,4],[6,6,8],[2,1,1]]))  # 4 (1→2→6→9)
print(longest_increasing_path([[3,4,5],[3,2,6],[2,2,1]]))  # 4 (1→2→3→4)
```

---

## 18.8 本章小结

矩阵双指针的核心模式：

| 模式 | 典型题目 | 关键技术 |
|------|----------|----------|
| 行列缩减（对撞） | 搜索矩阵II | 从角落出发，排除行列 |
| 边界收缩（四指针） | 螺旋矩阵 | 四边界依次处理 |
| 对角线扫描 | 对角线遍历 | d = i + j 决定对角线 |
| 转置+翻转 | 旋转图像 | 两步分解旋转 |

**下一章：最小覆盖子串与复杂字符串——need/have 模型的极限应用**

---

## LeetCode 推荐题目

- [74. 搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/) ⭐⭐
- [240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/) ⭐⭐
- [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/) ⭐⭐
- [48. 旋转图像](https://leetcode.cn/problems/rotate-image/) ⭐⭐
- [498. 对角线遍历](https://leetcode.cn/problems/diagonal-traverse/) ⭐⭐
- [329. 矩阵中的最长递增路径](https://leetcode.cn/problems/longest-increasing-path-in-a-matrix/) ⭐⭐⭐
