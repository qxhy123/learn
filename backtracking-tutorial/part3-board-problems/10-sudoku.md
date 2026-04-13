# 第10章：数独求解

## 10.1 问题描述

**解数独**：填入 1-9，使得每行、每列、每个 3×3 宫格都包含 1-9 恰好一次。（LeetCode 37）

```
初始棋盘（0 表示空格）：
5 3 0 | 0 7 0 | 0 0 0
6 0 0 | 1 9 5 | 0 0 0
0 9 8 | 0 0 0 | 0 6 0
------+-------+------
8 0 0 | 0 6 0 | 0 0 3
4 0 0 | 8 0 3 | 0 0 1
7 0 0 | 0 2 0 | 0 0 6
------+-------+------
0 6 0 | 0 0 0 | 2 8 0
0 0 0 | 4 1 9 | 0 0 5
0 0 0 | 0 8 0 | 0 7 9
```

## 10.2 约束表示

```python
class SudokuSolver:
    def __init__(self):
        # 三种约束：行、列、3x3宫格
        self.rows = [set() for _ in range(9)]   # rows[i] 已使用的数字
        self.cols = [set() for _ in range(9)]   # cols[j] 已使用的数字
        self.boxes = [set() for _ in range(9)]  # boxes[k] 已使用的数字
    
    def box_idx(self, r, c):
        """3x3宫格的索引 (0-8)"""
        return (r // 3) * 3 + (c // 3)
    
    def can_place(self, r, c, num):
        k = self.box_idx(r, c)
        return (num not in self.rows[r] and 
                num not in self.cols[c] and 
                num not in self.boxes[k])
    
    def place(self, r, c, num):
        k = self.box_idx(r, c)
        self.rows[r].add(num)
        self.cols[c].add(num)
        self.boxes[k].add(num)
    
    def remove(self, r, c, num):
        k = self.box_idx(r, c)
        self.rows[r].discard(num)
        self.cols[c].discard(num)
        self.boxes[k].discard(num)
```

## 10.3 基础回溯解法

```python
def solve_sudoku(board):
    """
    逐格填写，遇到空格尝试 1-9
    时间 O(9^m)，m = 空格数
    """
    solver = SudokuSolver()
    
    # 初始化约束（已有数字）
    for r in range(9):
        for c in range(9):
            if board[r][c] != '.':
                num = int(board[r][c])
                solver.place(r, c, num)
    
    def backtrack():
        # 找第一个空格
        for r in range(9):
            for c in range(9):
                if board[r][c] == '.':
                    for num in range(1, 10):
                        if solver.can_place(r, c, num):
                            board[r][c] = str(num)
                            solver.place(r, c, num)
                            
                            if backtrack():
                                return True
                            
                            board[r][c] = '.'
                            solver.remove(r, c, num)
                    
                    return False  # 没有合法数字，回溯
        
        return True  # 所有格子已填，找到解
    
    backtrack()
```

## 10.4 优化一：最少可能值优先（MRV 启发式）

**关键优化**：不按顺序填格，而是优先填**可选数字最少**的格子（约束最强的格子）。

```python
def solve_sudoku_mrv(board):
    """
    MRV (Minimum Remaining Values) 启发式
    优先处理约束最多的格子，大幅减少搜索空间
    """
    solver = SudokuSolver()
    empties = []
    
    for r in range(9):
        for c in range(9):
            if board[r][c] != '.':
                solver.place(r, c, int(board[r][c]))
            else:
                empties.append((r, c))
    
    def get_candidates(r, c):
        """获取 (r,c) 处的所有合法候选数字"""
        return [n for n in range(1, 10) if solver.can_place(r, c, n)]
    
    def backtrack(filled):
        if filled == len(empties):
            return True
        
        # MRV：找候选数最少的空格
        best_r, best_c = -1, -1
        best_candidates = None
        min_count = 10
        
        for r, c in empties:
            if board[r][c] != '.':
                continue
            candidates = get_candidates(r, c)
            if len(candidates) < min_count:
                min_count = len(candidates)
                best_r, best_c = r, c
                best_candidates = candidates
                if min_count == 0:
                    return False  # 死局，立即回溯
                if min_count == 1:
                    break  # 只有一个选择，不必再找
        
        for num in best_candidates:
            board[best_r][best_c] = str(num)
            solver.place(best_r, best_c, num)
            
            if backtrack(filled + 1):
                return True
            
            board[best_r][best_c] = '.'
            solver.remove(best_r, best_c, num)
        
        return False
    
    backtrack(0)
```

## 10.5 优化二：约束传播（前向检验）

当放置一个数字后，更新受影响格子的候选集：

```python
class ConstraintPropagation:
    """
    约束传播：维护每个空格的候选集
    当某格只剩一个候选时，强制填入（Arc Consistency）
    """
    
    def __init__(self, board):
        self.board = [row[:] for row in board]
        self.candidates = {}
        
        # 初始化候选集
        for r in range(9):
            for c in range(9):
                if self.board[r][c] == '.':
                    self.candidates[(r,c)] = set(range(1, 10))
        
        # 根据已有数字消除候选
        for r in range(9):
            for c in range(9):
                if self.board[r][c] != '.':
                    self._eliminate(r, c, int(self.board[r][c]))
    
    def _peers(self, r, c):
        """与 (r,c) 同行、同列、同宫格的所有格子"""
        peers = set()
        for i in range(9):
            peers.add((r, i))
            peers.add((i, c))
        box_r, box_c = (r//3)*3, (c//3)*3
        for dr in range(3):
            for dc in range(3):
                peers.add((box_r+dr, box_c+dc))
        peers.discard((r, c))
        return peers
    
    def _eliminate(self, r, c, num):
        """从 (r,c) 的相邻格子中消除 num"""
        for pr, pc in self._peers(r, c):
            if (pr, pc) in self.candidates:
                self.candidates[(pr, pc)].discard(num)
```

## 10.6 性能对比

```python
import time

def benchmark_sudoku():
    # 困难数独（少量已知数字）
    hard_board = [
        ['.','.','.','.','.','.','.','.','.'],
        ['.','.','.','.','.','.','.','.','1'],
        ['.','.','.','.','.','.','.','2','.'],
        # ... 省略其他行
    ]
    
    # 基础回溯
    import copy
    b1 = copy.deepcopy(hard_board)
    t = time.time()
    solve_sudoku(b1)
    print(f"基础回溯: {time.time()-t:.3f}s")
    
    # MRV 优化
    b2 = copy.deepcopy(hard_board)
    t = time.time()
    solve_sudoku_mrv(b2)
    print(f"MRV优化: {time.time()-t:.3f}s")
    # MRV 通常快 10-100 倍
```

## 10.7 数独的变体

```python
def is_valid_sudoku(board):
    """
    验证数独是否有效（不需要是完整解）
    LeetCode 36
    """
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    
    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == '.':
                continue
            
            box_idx = (r // 3) * 3 + (c // 3)
            
            if val in rows[r] or val in cols[c] or val in boxes[box_idx]:
                return False
            
            rows[r].add(val)
            cols[c].add(val)
            boxes[box_idx].add(val)
    
    return True
```

## 小结

| 优化方法 | 加速效果 | 实现复杂度 |
|---------|---------|----------|
| 基础回溯 | 基准 | 简单 |
| MRV 启发式 | 10-100x | 中等 |
| 约束传播 | 100-1000x | 较复杂 |
| AC-3 算法 | 最优 | 复杂 |

## 练习

1. 实现"生成数独谜题"（先生成完整解，再随机移除数字）
2. 统计不同难度数独（已知数字数量不同）的求解时间
3. 扩展为 16×16 数独（使用 hex 数字 1-G）

---

**上一章：** [N 皇后](09-n-queens.md) | **下一章：** [单词搜索](11-word-search.md)
