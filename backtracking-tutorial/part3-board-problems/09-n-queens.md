# 第9章：N 皇后问题

## 9.1 问题描述

**N 皇后**：在 n×n 棋盘上放置 n 个皇后，使得任意两个皇后不在同一行、同一列或同一对角线上。（LeetCode 51/52）

```
4×4 棋盘的一个解：
. Q . .
. . . Q
Q . . .
. . Q .
```

## 9.2 关键约束分析

皇后不能互相攻击的条件：

```
同列：col[i] == col[j]
同主对角线：row[i] - col[i] == row[j] - col[j]
同副对角线：row[i] + col[i] == row[j] + col[j]
（同行由逐行放置保证不冲突）
```

用集合记录已占用的列和对角线，O(1) 检查冲突：

```python
cols = set()       # 已使用的列
diag1 = set()      # 主对角线：row - col 相同
diag2 = set()      # 副对角线：row + col 相同
```

## 9.3 标准回溯解法

```python
def solve_n_queens(n):
    """
    逐行放置皇后（每行恰好一个）
    时间 O(n!)，空间 O(n)
    """
    result = []
    queens = [-1] * n  # queens[row] = col
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col
    
    def backtrack(row):
        if row == n:
            # 构造棋盘
            board = []
            for r in range(n):
                board.append('.' * queens[r] + 'Q' + '.' * (n - queens[r] - 1))
            result.append(board)
            return
        
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue  # 剪枝
            
            # 放置皇后
            queens[row] = col
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            
            backtrack(row + 1)
            
            # 撤销
            queens[row] = -1
            cols.discard(col)
            diag1.discard(row - col)
            diag2.discard(row + col)
    
    backtrack(0)
    return result

# 测试
solutions = solve_n_queens(4)
for board in solutions:
    for row in board:
        print(row)
    print()
# 输出2个解
```

## 9.4 只求解的数量（LC 52）

```python
def total_n_queens(n):
    """只统计解的数量，不构造棋盘"""
    count = [0]
    cols = set()
    diag1 = set()
    diag2 = set()
    
    def backtrack(row):
        if row == n:
            count[0] += 1
            return
        
        for col in range(n):
            if col in cols or (row-col) in diag1 or (row+col) in diag2:
                continue
            cols.add(col); diag1.add(row-col); diag2.add(row+col)
            backtrack(row + 1)
            cols.discard(col); diag1.discard(row-col); diag2.discard(row+col)
    
    backtrack(0)
    return count[0]

# 各 n 的解数
for n in range(1, 10):
    print(f"N={n}: {total_n_queens(n)} 种解")
# N=1: 1, N=4: 2, N=5: 10, N=8: 92
```

## 9.5 位运算优化（极速版）

使用位运算代替集合，大幅加速：

```python
def total_n_queens_bit(n):
    """
    位运算 N 皇后，比集合版快 5-10 倍
    
    cols, diag1, diag2 分别用一个整数的位表示
    第 i 位为 1 表示该列/对角线已被占用
    """
    count = [0]
    limit = (1 << n) - 1  # n 位全 1
    
    def backtrack(cols, diag1, diag2):
        if cols == limit:
            count[0] += 1
            return
        
        # 可用位置 = 当前行中未被占用的列
        available = limit & (~(cols | diag1 | diag2))
        
        while available:
            pos = available & (-available)  # 取最低位（选一列）
            available &= available - 1      # 清除最低位
            
            backtrack(
                cols | pos,
                (diag1 | pos) << 1,   # 主对角线向下移位
                (diag2 | pos) >> 1    # 副对角线向下移位
            )
    
    backtrack(0, 0, 0)
    return count[0]

# 性能对比
import time
for n in [10, 12, 14]:
    t = time.time()
    r = total_n_queens_bit(n)
    print(f"N={n}: {r} 解, 用时 {time.time()-t:.3f}s")
```

## 9.6 N 皇后的对称性优化

利用棋盘的对称性，只搜索第一行的前 n//2 列，结果乘以 2（若 n 为奇数，中间列单独处理）：

```python
def total_n_queens_symmetric(n):
    """利用对称性减少一半搜索量"""
    cols = set(); diag1 = set(); diag2 = set()
    
    def backtrack(row):
        if row == n:
            return 1
        total = 0
        for col in range(n):
            if col in cols or (row-col) in diag1 or (row+col) in diag2:
                continue
            cols.add(col); diag1.add(row-col); diag2.add(row+col)
            total += backtrack(row + 1)
            cols.discard(col); diag1.discard(row-col); diag2.discard(row+col)
        return total
    
    # 第一行只搜索左半边
    total = 0
    cols = set(); diag1 = set(); diag2 = set()
    for col in range(n // 2):
        cols.add(col); diag1.add(-col); diag2.add(col)
        total += backtrack(1)
        cols.discard(col); diag1.discard(-col); diag2.discard(col)
    
    total *= 2  # 左右对称
    
    # 若 n 为奇数，中间列单独处理
    if n % 2 == 1:
        mid = n // 2
        cols = {mid}; diag1 = {-mid}; diag2 = {mid}
        total += backtrack(1)
    
    return total
```

## 9.7 解的可视化

```python
def visualize_queens(board):
    """可视化皇后棋盘"""
    n = len(board)
    print("+" + "-+" * n)
    for row in board:
        print("|" + "|".join("♛" if c == "Q" else " " for c in row) + "|")
        print("+" + "-+" * n)

solutions = solve_n_queens(5)
print(f"5皇后共 {len(solutions)} 种解，显示第一种：")
visualize_queens(solutions[0])
```

## 小结

| 技术 | 效果 |
|------|------|
| 集合记录约束 | O(1) 冲突检查 |
| 位运算 | 5-10 倍加速 |
| 对称性 | 减少约一半搜索量 |
| 逐行放置 | 自动保证不同行 |

## 练习

1. 实现"骑士巡游"：马在 n×n 棋盘上走哈密顿路径，经过每个格子恰好一次
2. 将 N 皇后解中的对称解（水平/垂直/对角翻转）去重
3. 分析为什么 N=8 恰好有 92 种解（而非 8! = 40320 种）

---

**上一章（Part 2）：** [组合总和](../part2-combinations/08-combination-sum.md) | **下一章：** [数独求解](10-sudoku.md)
