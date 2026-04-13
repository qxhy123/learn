# 第11章：单词搜索

## 11.1 问题描述

**单词搜索**：在 m×n 的字符网格中，判断是否存在单词 `word`，单词必须按顺序连接相邻格子（上下左右），每个格子只能使用一次。（LeetCode 79）

```
网格：
A B C E
S F C S
A D E E

word = "ABCCED" → True
word = "SEE"    → True
word = "ABCB"   → False（不能重复使用 B）
```

## 11.2 标准回溯解法

```python
def exist(board, word):
    """
    从每个格子出发，DFS 搜索单词
    时间 O(m×n×4^L)，L = 单词长度
    """
    m, n = len(board), len(board[0])
    
    def backtrack(r, c, idx):
        if idx == len(word):
            return True  # 找到完整单词
        if r < 0 or r >= m or c < 0 or c >= n:
            return False
        if board[r][c] != word[idx]:
            return False
        
        # 标记已访问（避免重复）
        temp = board[r][c]
        board[r][c] = '#'
        
        # 向四个方向搜索
        found = (backtrack(r+1, c, idx+1) or
                 backtrack(r-1, c, idx+1) or
                 backtrack(r, c+1, idx+1) or
                 backtrack(r, c-1, idx+1))
        
        # 恢复现场
        board[r][c] = temp
        return found
    
    for r in range(m):
        for c in range(n):
            if backtrack(r, c, 0):
                return True
    return False
```

## 11.3 剪枝优化

### 优化一：字符频率剪枝

如果单词中某个字符在网格中出现次数不足，直接返回 False：

```python
def exist_optimized(board, word):
    from collections import Counter
    
    # 频率剪枝
    board_count = Counter(c for row in board for c in row)
    word_count = Counter(word)
    for c, cnt in word_count.items():
        if board_count[c] < cnt:
            return False
    
    m, n = len(board), len(board[0])
    
    def backtrack(r, c, idx):
        if idx == len(word):
            return True
        if r < 0 or r >= m or c < 0 or c >= n:
            return False
        if board[r][c] != word[idx]:
            return False
        
        temp = board[r][c]
        board[r][c] = '#'
        found = (backtrack(r+1, c, idx+1) or backtrack(r-1, c, idx+1) or
                 backtrack(r, c+1, idx+1) or backtrack(r, c-1, idx+1))
        board[r][c] = temp
        return found
    
    for r in range(m):
        for c in range(n):
            if backtrack(r, c, 0):
                return True
    return False
```

### 优化二：反向搜索

如果单词首尾的字符频率，末尾更稀少，从末尾开始搜索：

```python
def exist_smart(board, word):
    from collections import Counter
    
    board_count = Counter(c for row in board for c in row)
    
    # 决定搜索方向（从出现少的那端开始）
    if board_count[word[0]] > board_count[word[-1]]:
        word = word[::-1]
    
    m, n = len(board), len(board[0])
    
    def backtrack(r, c, idx):
        if idx == len(word):
            return True
        if r < 0 or r >= m or c < 0 or c >= n or board[r][c] != word[idx]:
            return False
        
        board[r][c] = '#'
        res = any(backtrack(r+dr, c+dc, idx+1) 
                  for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)])
        board[r][c] = word[idx]
        return res
    
    for r in range(m):
        for c in range(n):
            if backtrack(r, c, 0):
                return True
    return False
```

## 11.4 单词搜索 II（找出所有存在的单词）

**问题**：给定网格和单词列表，找出网格中所有存在的单词。（LeetCode 212）

**关键优化**：用 **Trie（前缀树）** 避免重复搜索前缀。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None  # 非 None 表示路径构成完整单词

class Solution:
    def findWords(self, board, words):
        # 构建 Trie
        root = TrieNode()
        for word in words:
            node = root
            for c in word:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
            node.word = word
        
        m, n = len(board), len(board[0])
        result = []
        
        def backtrack(r, c, node):
            if r < 0 or r >= m or c < 0 or c >= n:
                return
            ch = board[r][c]
            if ch == '#' or ch not in node.children:
                return
            
            next_node = node.children[ch]
            
            if next_node.word:
                result.append(next_node.word)
                next_node.word = None  # 避免重复添加
            
            board[r][c] = '#'
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                backtrack(r+dr, c+dc, next_node)
            board[r][c] = ch
            
            # 剪枝：删除已无子节点的 Trie 节点（已被匹配）
            if not next_node.children and not next_node.word:
                del node.children[ch]
        
        for r in range(m):
            for c in range(n):
                backtrack(r, c, root)
        
        return result
```

**Trie 优化的效果**：
- 所有单词共享前缀，避免重复搜索
- 动态修剪 Trie（删除已匹配的叶节点），加速后续搜索
- 时间复杂度：O(m×n×4×3^(L-1))，L = 最长单词长度

## 11.5 搜索路径记录

```python
def exist_with_path(board, word):
    """返回单词在网格中的路径坐标"""
    m, n = len(board), len(board[0])
    
    def backtrack(r, c, idx, path):
        if idx == len(word):
            return path[:]
        if r < 0 or r >= m or c < 0 or c >= n:
            return None
        if board[r][c] != word[idx]:
            return None
        
        board[r][c] = '#'
        path.append((r, c))
        
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            result = backtrack(r+dr, c+dc, idx+1, path)
            if result:
                board[r][c] = word[idx]
                return result
        
        path.pop()
        board[r][c] = word[idx]
        return None
    
    for r in range(m):
        for c in range(n):
            path = backtrack(r, c, 0, [])
            if path:
                return path
    return None

# 测试
board = [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']]
print(exist_with_path(board, "ABCCED"))
# [(0,0),(0,1),(0,2),(1,2),(2,2),(2,1)]
```

## 11.6 统计单词出现次数

```python
def count_word_occurrences(board, word):
    """统计单词在网格中的出现次数（允许重叠路径）"""
    m, n = len(board), len(board[0])
    count = [0]
    
    def backtrack(r, c, idx):
        if idx == len(word):
            count[0] += 1
            return
        if r < 0 or r >= m or c < 0 or c >= n:
            return
        if board[r][c] != word[idx]:
            return
        
        temp = board[r][c]
        board[r][c] = '#'
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            backtrack(r+dr, c+dc, idx+1)
        board[r][c] = temp
    
    for r in range(m):
        for c in range(n):
            backtrack(r, c, 0)
    
    return count[0]
```

## 小结

| 技术 | 效果 |
|------|------|
| 原地标记 '#' | O(1) 访问控制，无需额外 visited 数组 |
| 频率剪枝 | 提前过滤不可能的情况 |
| 反向搜索 | 从稀有字符端出发减少分支 |
| Trie 优化 | 多单词搜索时共享前缀 |
| 动态修剪 Trie | 删除已匹配节点减少后续开销 |

## 练习

1. 实现"单词接龙"（LeetCode 127）：找从 beginWord 到 endWord 的最短转换序列
2. 扩展 findWords，返回每个单词的所有路径（不只是是否存在）
3. 分析为什么 Trie 的动态修剪能显著提升性能（给出具体复杂度对比）

---

**上一章：** [数独求解](10-sudoku.md) | **下一章：** [泛洪填充](12-flood-fill.md)
