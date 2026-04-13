# 第12章：泛洪填充与岛屿问题

## 12.1 问题描述

**泛洪填充**：将图像中从指定像素出发的所有连通同色像素替换为新颜色。（LeetCode 733）

```
初始图像：          执行 floodFill(image, 1, 1, 2)：
1 1 1              2 2 2
1 1 0     →        2 2 0
1 0 0              2 0 0
起始点 (1,1)，颜色 1 → 2
```

## 12.2 基础实现

```python
def floodFill(image, sr, sc, color):
    """
    从 (sr,sc) 出发，将所有连通同色区域替换为 color
    时间 O(m×n)，空间 O(m×n)（递归栈）
    """
    old_color = image[sr][sc]
    if old_color == color:
        return image  # 颜色相同，无需处理
    
    m, n = len(image), len(image[0])
    
    def dfs(r, c):
        if r < 0 or r >= m or c < 0 or c >= n:
            return
        if image[r][c] != old_color:
            return
        
        image[r][c] = color  # 填充
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
    
    dfs(sr, sc)
    return image
```

## 12.3 岛屿数量（经典 DFS）

**问题**：计算由 '1'（陆地）和 '0'（水）组成的网格中岛屿的数量。（LeetCode 200）

```python
def numIslands(grid):
    """
    遍历每格，遇到 '1' 就 DFS 淹没整个岛屿，计数
    时间 O(m×n)，空间 O(m×n)
    """
    if not grid:
        return 0
    
    m, n = len(grid), len(grid[0])
    count = 0
    
    def dfs(r, c):
        if r < 0 or r >= m or c < 0 or c >= n:
            return
        if grid[r][c] != '1':
            return
        grid[r][c] = '0'  # 标记已访问
        dfs(r+1, c); dfs(r-1, c)
        dfs(r, c+1); dfs(r, c-1)
    
    for r in range(m):
        for c in range(n):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1
    
    return count
```

## 12.4 岛屿问题系列

### 岛屿的最大面积（LeetCode 695）

```python
def maxAreaOfIsland(grid):
    m, n = len(grid), len(grid[0])
    
    def dfs(r, c):
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] == 0:
            return 0
        grid[r][c] = 0
        return 1 + dfs(r+1, c) + dfs(r-1, c) + dfs(r, c+1) + dfs(r, c-1)
    
    return max(dfs(r, c) for r in range(m) for c in range(n))
```

### 岛屿的周长（LeetCode 463）

```python
def islandPerimeter(grid):
    """
    每个陆地格子贡献 4 条边，减去与相邻陆地共享的边
    """
    m, n = len(grid), len(grid[0])
    perimeter = 0
    
    for r in range(m):
        for c in range(n):
            if grid[r][c] == 1:
                perimeter += 4
                # 减去和相邻陆地的共享边
                if r > 0 and grid[r-1][c] == 1:
                    perimeter -= 2
                if c > 0 and grid[r][c-1] == 1:
                    perimeter -= 2
    
    return perimeter
```

### 统计封闭岛屿（LeetCode 1254）

```python
def closedIsland(grid):
    """
    封闭岛屿：不接触边界的岛屿
    先淹没边界连通的陆地，再统计剩余岛屿
    """
    m, n = len(grid), len(grid[0])
    
    def dfs(r, c):
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != 0:
            return
        grid[r][c] = 1  # 淹没
        dfs(r+1, c); dfs(r-1, c)
        dfs(r, c+1); dfs(r, c-1)
    
    # 淹没边界上的陆地
    for r in range(m):
        dfs(r, 0); dfs(r, n-1)
    for c in range(n):
        dfs(0, c); dfs(m-1, c)
    
    # 统计封闭岛屿
    count = 0
    for r in range(m):
        for c in range(n):
            if grid[r][c] == 0:
                dfs(r, c)
                count += 1
    return count
```

### 飞地的数量（LeetCode 1020）

```python
def numEnclaves(grid):
    """飞地：不能到达边界的陆地格子数量"""
    m, n = len(grid), len(grid[0])
    
    def dfs(r, c):
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != 1:
            return
        grid[r][c] = 0
        dfs(r+1, c); dfs(r-1, c)
        dfs(r, c+1); dfs(r, c-1)
    
    # 淹没边界连通的陆地
    for r in range(m):
        dfs(r, 0); dfs(r, n-1)
    for c in range(n):
        dfs(0, c); dfs(m-1, c)
    
    return sum(grid[r][c] for r in range(m) for c in range(n))
```

## 12.5 并查集方案（岛屿数量）

对于大规模数据，并查集比递归 DFS 更能避免栈溢出：

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = 0
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.count -= 1

def numIslands_uf(grid):
    m, n = len(grid), len(grid[0])
    uf = UnionFind(m * n)
    
    for r in range(m):
        for c in range(n):
            if grid[r][c] == '1':
                uf.count += 1
                for dr, dc in [(1,0),(0,1)]:  # 只向右和下
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == '1':
                        uf.union(r*n+c, nr*n+nc)
    
    return uf.count
```

## 12.6 迭代 DFS（避免栈溢出）

对于超大网格，递归可能栈溢出，改用显式栈：

```python
def floodFill_iterative(image, sr, sc, color):
    old_color = image[sr][sc]
    if old_color == color:
        return image
    
    m, n = len(image), len(image[0])
    stack = [(sr, sc)]
    image[sr][sc] = color
    
    while stack:
        r, c = stack.pop()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < m and 0 <= nc < n and image[nr][nc] == old_color:
                image[nr][nc] = color
                stack.append((nr, nc))
    
    return image
```

## 12.7 最短路径变体：腐烂的橘子（BFS）

**注意**：多源最短路径用 BFS，不是 DFS/回溯！

```python
from collections import deque

def orangesRotting(grid):
    """
    所有腐烂橘子同时扩散（BFS 多源最短路径）
    """
    m, n = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    
    for r in range(m):
        for c in range(n):
            if grid[r][c] == 2:
                queue.append((r, c, 0))  # (行, 列, 时间)
            elif grid[r][c] == 1:
                fresh += 1
    
    minutes = 0
    while queue:
        r, c, t = queue.popleft()
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                minutes = t + 1
                queue.append((nr, nc, t+1))
    
    return minutes if fresh == 0 else -1
```

## 小结

| 问题类型 | 推荐方法 | 核心技巧 |
|---------|---------|---------|
| 泛洪填充/DFS | 递归 DFS | 原地修改标记访问 |
| 大规模网格 | 迭代 DFS | 显式栈避免溢出 |
| 动态合并 | 并查集 | 路径压缩 + 按秩合并 |
| 最短扩散时间 | BFS | 多源 BFS |

## 练习

1. 实现"太平洋大西洋水流问题"（LeetCode 417）：找出既能流向太平洋又能流向大西洋的坐标
2. 解决"被围绕的区域"（LeetCode 130）：将被 'X' 包围的 'O' 替换为 'X'
3. 分析为什么"腐烂橘子"用 BFS 而不是 DFS？

---

**上一章：** [单词搜索](11-word-search.md) | **下一章（Part 4）：** [回文分割](../part4-string-partition/13-palindrome-partition.md)
