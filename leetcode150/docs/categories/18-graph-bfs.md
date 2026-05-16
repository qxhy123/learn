# 18 — Graph BFS（融合版）

> **难度**：★★★☆☆
> **题数**：3
> **核心套路**：最短路径 BFS（无权图）、字符串变换 BFS、蛇梯棋盘 BFS
> **本文件**：覆盖 graph_bfs 3 题的算法套路总结 + 典型题精讲 + 自测

---

## 一例速记

> **最短路径 BFS（无权图）**：队列 + 已访问集合，按"层"扩展节点，每推进一层距离 +1；BFS 天然保证第一次到达终点时路径最短（127 / 433）
> **字符串变换 BFS**：将每个单词/基因序列视为图中节点，两节点有边当且仅当它们恰好相差一个字符；BFS 找最短变换链（127 Word Ladder / 433 Minimum Genetic Mutation）
> **双向 BFS 优化**：从起点和终点同时扩展，当两侧相遇时路径即为最短；将搜索空间从 $O(b^d)$ 压缩到 $O(b^{d/2})$（127 进阶优化）
> **棋盘 BFS / 状态压缩**：将二维棋盘格子编号，记录已访问格子，处理蛇和梯子的跳转（909 Snakes and Ladders）
> **AI 关联**：图神经网络（GNN）消息传递——每一轮聚合邻居信息，与 BFS 按层传播结构完全对应；知识图谱推理中的多跳问答（multi-hop QA）也依赖图的层次遍历

---

## 思维路径还原

> "看到 **'最短变换路径'（127）** → 立刻想 BFS：
> 把 `wordList` 转成 `set`（O(1) 查找），队列初始放入 `(beginWord, 1)`（单词, 当前步数）。
> 对当前单词的每个字符位置，逐一枚举 26 个字母，若变换后在 `word_set` 中且未访问，
> 则入队，并从 `word_set` 中删除（标记已访问）。
> 找到 `endWord` 时返回当前步数；队列为空则返回 0。
> 时间 $O(26 \times L \times N)$，其中 L 为单词长度，N 为单词表大小。
>
> 看到 **'433 最小基因变化'** → 与 127 完全相同框架：
> 8 个字符，字符集为 `{'A','C','G','T'}`（只有 4 种），bank 作为合法节点集合。
> 代码几乎可以直接复用，只需把字符集从 26 个字母换成 4 个碱基。
>
> 看到 **'909 蛇和梯子'** → 棋盘 BFS：
> 先将 $n \times n$ 棋盘"蛇形展开"为一维数组（注意奇偶行方向交替），
> 然后从格子 1 开始 BFS，每次从当前格子出发掷骰子（1~6），
> 到达的格子若有蛇/梯子则跳转（取 `board` 数组的值 -1），
> 若已访问则跳过，否则入队。到达格子 $n^2 - 1$ 时返回步数。
> 关键：坐标变换——将格子编号 s 转为 `(row, col)`，注意行从底部向上、奇偶行列方向不同。"

---

## 学习目标

- 掌握无权图 BFS 最短路径模板：队列 + 访问集合 + 按层扩展
- 熟练字符串变换 BFS：逐字符枚举替换，从 word_set 中删除标记已访问
- 理解双向 BFS 的剪枝原理及适用场景（起终点已知、分支因子较大时）
- 掌握棋盘编号 ↔ 坐标变换（蛇形展开）的技巧
- 能识别"最短步数 / 最少操作"题型并直接套 BFS 模板

---

## 抽象成方法（标准模板代码）

### 套路 1：BFS 最短路径（无权图通用模板）

适用题：127、433、909

```python
from collections import deque
from typing import List


def bfs_shortest_path(start, end, neighbors_fn) -> int:
    """
    无权图 BFS 最短路径通用模板。
    neighbors_fn(node) -> List[node]：返回邻居节点列表。
    返回从 start 到 end 的最短距离，不可达返回 -1。
    """
    if start == end:
        return 0
    visited = {start}
    queue = deque([start])
    dist = 0
    while queue:
        dist += 1
        for _ in range(len(queue)):          # 按层扩展
            node = queue.popleft()
            for nxt in neighbors_fn(node):
                if nxt == end:
                    return dist
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
    return -1
```

> 关键不变式：每轮循环处理同一"层"的所有节点，`dist` 代表当前层到起点的距离。
> 先检查终点再入队（Early Exit）可减少一层多余扩展。

---

### 套路 2：字符串变换 BFS（Word Ladder / Genetic Mutation）

适用题：127、433

```python
def word_ladder_bfs(begin: str, end: str, word_set: set[str]) -> int:
    """
    127 / 433 通用框架：每次变换一个字符，求最短变换步数。
    时间 O(26 * L * N)，空间 O(N)，L=单词长度，N=词典大小。
    """
    if end not in word_set:
        return 0
    queue = deque([(begin, 1)])
    visited = {begin}
    alphabet = 'abcdefghijklmnopqrstuvwxyz'  # 127 用 26 字母

    while queue:
        word, steps = queue.popleft()
        for i in range(len(word)):
            for c in alphabet:
                if c == word[i]:
                    continue
                new_word = word[:i] + c + word[i+1:]
                if new_word == end:
                    return steps + 1
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, steps + 1))
    return 0


def min_mutation_bfs(start: str, end: str, bank: List[str]) -> int:
    """
    433: 基因变化，字符集为 {'A','C','G','T'}，bank 为合法节点集合。
    时间 O(4 * 8 * N)，空间 O(N)。
    """
    bank_set = set(bank)
    if end not in bank_set:
        return -1
    queue = deque([(start, 0)])
    visited = {start}
    gene_chars = 'ACGT'

    while queue:
        gene, steps = queue.popleft()
        for i in range(len(gene)):
            for c in gene_chars:
                if c == gene[i]:
                    continue
                new_gene = gene[:i] + c + gene[i+1:]
                if new_gene == end:
                    return steps + 1
                if new_gene in bank_set and new_gene not in visited:
                    visited.add(new_gene)
                    queue.append((new_gene, steps + 1))
    return -1
```

---

### 套路 3：双向 BFS 优化（Word Ladder 进阶）

适用题：127（大规模输入时）

```python
def word_ladder_bidirectional(begin: str, end: str, word_list: List[str]) -> int:
    """
    双向 BFS：从 begin 和 end 同时向中间扩展，相遇即终止。
    将时间复杂度从 O(b^d) 降到 O(b^(d/2))，b=分支因子，d=最短路径长度。
    """
    word_set = set(word_list)
    if end not in word_set:
        return 0

    front, back = {begin}, {end}   # 两端的当前层节点集合
    visited = {begin, end}
    steps = 1

    while front and back:
        # 总是扩展较小的那一侧（BFS 平衡优化）
        if len(front) > len(back):
            front, back = back, front

        next_front = set()
        for word in front:
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = word[:i] + c + word[i+1:]
                    if new_word in back:         # 两侧相遇
                        return steps + 1
                    if new_word in word_set and new_word not in visited:
                        visited.add(new_word)
                        next_front.add(new_word)
        front = next_front
        steps += 1
    return 0
```

---

### 套路 4：棋盘 BFS + 坐标变换（Snakes and Ladders）

适用题：909

```python
def snakes_and_ladders(board: List[List[int]]) -> int:
    """
    909: n×n 蛇形棋盘，格子 1..n² 编号，蛇/梯子用 board 矩阵记录。
    BFS 求从格子 1 到格子 n² 的最少步数。时间 O(n²)，空间 O(n²)。
    """
    n = len(board)
    total = n * n

    def label_to_coord(s: int):
        """将格子编号 s（1-indexed）转为 board[row][col]。"""
        s -= 1                      # 转为 0-indexed
        row_from_bottom = s // n    # 从底部算第几行
        col_in_row = s % n
        # 偶数行（从底部）从左到右，奇数行从右到左
        if row_from_bottom % 2 == 0:
            col = col_in_row
        else:
            col = n - 1 - col_in_row
        row = n - 1 - row_from_bottom
        return row, col

    visited = {1}
    queue = deque([(1, 0)])         # (当前格子编号, 步数)

    while queue:
        pos, steps = queue.popleft()
        for dice in range(1, 7):    # 掷骰子 1~6
            nxt = pos + dice
            if nxt > total:
                break
            r, c = label_to_coord(nxt)
            dest = board[r][c]
            if dest != -1:          # 有蛇 / 梯子，跳转
                nxt = dest
            if nxt == total:
                return steps + 1
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, steps + 1))
    return -1
```

---

### 速查表

| 题型特征 | 套路 | 时间 | 空间 |
|---|---|---|---|
| 无权图最短路径 | BFS 按层扩展 | $O(V+E)$ | $O(V)$ |
| 字符串单字符变换最短链 | BFS + 字符枚举 | $O(26 \times L \times N)$ | $O(N)$ |
| 字符集小（4 种）基因变化 | BFS + 字符集枚举 | $O(4 \times L \times N)$ | $O(N)$ |
| 两端已知、分支因子大 | 双向 BFS | $O(b^{d/2})$ | $O(b^{d/2})$ |
| 棋盘格子最短步数 | BFS + 蛇形坐标变换 | $O(n^2)$ | $O(n^2)$ |

---

## 方法变形（3 类）

### 变形 1：Word Ladder 系列

- **127**（Word Ladder）：字符集 26 字母，返回步数（含起始词），不可达返回 0。
- **433**（Minimum Genetic Mutation）：字符集 4 碱基，返回步数（不含起始），bank 作为合法集合，不可达返回 -1。
- 框架完全相同，差异仅在字符集、合法节点集合、以及返回值定义（含不含起始节点）。

### 变形 2：BFS 与 Dijkstra 的选择

- 无权图（每条边权重相同）→ BFS 即可，$O(V+E)$。
- 有权图（边权不同）→ Dijkstra（最小堆），$O((V+E) \log V)$。
- 本 category 3 题均为无权图（每次变换 = 步数 +1），直接用 BFS。

### 变形 3：909 坐标变换的常见错误

- 棋盘蛇形展开：行号从底部开始（`row_from_bottom = s // n`），偶数行从左到右，奇数行从右到左。
- 访问判断要在跳转（蛇/梯子）**之后**记录，避免同一目标格子被重复入队。
- 掷骰子上限：`nxt = pos + dice` 可能超过 `total`，需要 `if nxt > total: break`（6 面骰，超过则后续也不用试）。

---

## 思考路标（条件反射）

1. 看到 **"最短步数 / 最少操作次数 / 无权图"** → BFS
2. 看到 **"两个字符串相差一个字符"** → 字符串变换 BFS，枚举每个位置的替换
3. 看到 **"word list / bank 作为合法节点"** → 转成 set，BFS 时从 set 删除已访问（等价于 visited 集合）
4. 看到 **"双向 BFS"** → 起终点都已知 + 分支因子较大时，分别从两端扩展，取较小的一侧推进
5. 看到 **"棋盘 / 蛇和梯子"** → BFS + 坐标变换，注意蛇形展开奇偶行方向
6. 看到 **"有权图最短路"** → 跳到 Dijkstra（最小堆），不要用 BFS
7. 看到 **"GNN 消息传递 / 图卷积"** → 类比 BFS 按层聚合邻居特征

---

## 易错点

1. **127 返回值含起始节点**：题目要求返回"变换序列的长度"（包含 beginWord），所以初始 steps=1，到达 endWord 时返回 steps+1；不少人初始化为 0 导致差 1。
2. **433 返回值不含起始节点**：与 127 相反，返回的是变换次数（不含 startGene），注意两题定义的差异。
3. **word_set 删除 vs visited 集合**：两者等价，但直接从 word_set 中 `discard` 已访问节点可以省去 visited 集合；若不删除则需单独维护 visited，否则同一节点会被重复入队导致超时。
4. **909 坐标变换**：`row_from_bottom` 决定行的奇偶性（从底部数），易与"从顶部数"混淆；建议单独写 `label_to_coord` 函数并用简单用例验证（n=2 时格子 1 对应 `board[1][0]`）。
5. **909 掷骰子边界**：`pos + dice` 可能超过 `n²`，若超过则不合法，需 `break`（因为 dice 从 1~6 递增，一旦超过后续都超过）。
6. **双向 BFS 步数计算**：`steps` 变量代表"已扩展了多少层"，相遇时答案是 `steps + 1`（前向已走 steps 步，后向再走 1 步相遇）；若两侧都已经扩展了 steps/2 层则相遇时答案为 steps，根据实现方式仔细校验。

---

## 典型应用例题

### 例 1：127. Word Ladder

**题目**：给定 `beginWord`、`endWord` 和单词表 `wordList`，每次只能改变一个字母，且变换后的词必须在 `wordList` 中。求从 `beginWord` 变换到 `endWord` 的最短序列长度（含首尾），若不可达返回 0。

**思路**：将每个单词视为图节点，两词相差一个字母则有边（无权）。BFS 从 `beginWord` 出发，逐层扩展，第一次到达 `endWord` 时的层数即为答案。枚举邻居时，逐字符位置替换 26 个字母，若在 word_set 中则为合法邻居。

**解**：

```python
# 参考：solutions/graph_bfs/p127_word_ladder.py
def ladderLength(beginWord: str, endWord: str, wordList: List[str]) -> int:
    word_set = set(wordList)
    if endWord not in word_set:
        return 0
    queue = deque([(beginWord, 1)])
    visited = {beginWord}
    while queue:
        word, steps = queue.popleft()
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                if new_word == endWord:
                    return steps + 1
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, steps + 1))
    return 0
```

**分析**：单词长度 L，词典大小 N。每个单词生成 $26L$ 个候选邻居，每次哈希查找 $O(L)$，总体 $O(26L^2 N)$；因 L 通常很小（5~8），实际运行快。BFS 保证最短路径。

---

### 例 2：433. Minimum Genetic Mutation

**题目**：基因序列长度固定为 8，字符集为 `{'A','C','G','T'}`。给定 `startGene`、`endGene` 和合法基因库 `bank`，每次变换一个字符且结果必须在 `bank` 中。求最少变换次数，不可达返回 -1。

**思路**：与 127 框架相同，字符集缩小到 4 个字符，合法节点从 `wordList` 换成 `bank`。注意返回值定义与 127 不同（不含起始节点）。

**解**：

```python
# 参考：solutions/graph_bfs/p433_minimum_genetic_mutation.py
def minMutation(startGene: str, endGene: str, bank: List[str]) -> int:
    bank_set = set(bank)
    if endGene not in bank_set:
        return -1
    queue = deque([(startGene, 0)])
    visited = {startGene}
    while queue:
        gene, steps = queue.popleft()
        for i in range(len(gene)):
            for c in 'ACGT':
                new_gene = gene[:i] + c + gene[i+1:]
                if new_gene == endGene:
                    return steps + 1
                if new_gene in bank_set and new_gene not in visited:
                    visited.add(new_gene)
                    queue.append((new_gene, steps + 1))
    return -1
```

**分析**：基因长度固定为 8，字符集 4 个，`bank` 大小 N。时间 $O(4 \times 8 \times N) = O(32N)$，空间 $O(N)$。

---

### 例 3：909. Snakes and Ladders

**题目**：$n \times n$ 棋盘，格子从底部按蛇形编号 $1 \sim n^2$。`board[r][c]` 为 -1（正常格子）或目标格子编号（蛇/梯子）。从格子 1 出发，每次掷骰子（1~6），若落点有蛇/梯子则自动跳转。求到达格子 $n^2$ 的最少步数，不可达返回 -1。

**思路**：BFS。将格子编号展开为状态，队列存储 `(当前格子, 步数)`。关键是将格子编号转为棋盘坐标（蛇形展开，偶数行从左到右，奇数行从右到左，行从底部向上）。

**解**：

```python
# 参考：solutions/graph_bfs/p909_snakes_and_ladders.py
def snakesAndLadders(board: List[List[int]]) -> int:
    n = len(board)
    total = n * n

    def label_to_coord(s: int):
        s -= 1
        q, r = divmod(s, n)
        col = r if q % 2 == 0 else n - 1 - r
        row = n - 1 - q
        return row, col

    visited = {1}
    queue = deque([(1, 0)])
    while queue:
        pos, steps = queue.popleft()
        for dice in range(1, 7):
            nxt = pos + dice
            if nxt > total:
                break
            r, c = label_to_coord(nxt)
            if board[r][c] != -1:
                nxt = board[r][c]
            if nxt == total:
                return steps + 1
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, steps + 1))
    return -1
```

**分析**：格子数 $n^2$，每个格子最多入队一次，每次最多扩展 6 个邻居。时间 $O(n^2)$，空间 $O(n^2)$。

---

## 自测题

**自测 1**（127 Word Ladder）—— `beginWord='hit', endWord='cog', wordList=['hot','dot','dog','lot','log','cog']` 应返回 5（hit→hot→dot→dog→cog）；`wordList` 不含 `'cog'` 时返回 0。提示：BFS，word_set 记录合法节点，逐字符枚举 26 字母替换，first reach endWord 时返回步数 +1。参考 `solutions/graph_bfs/p127_word_ladder.py`。

**自测 2**（433 Minimum Genetic Mutation）—— `startGene='AACCGGTT', endGene='AACCGGTA', bank=['AACCGGTA']` 返回 1；`bank=[]` 返回 -1。提示：框架同 127，字符集换成 `'ACGT'`，不可达返回 -1 而非 0。参考 `solutions/graph_bfs/p433_minimum_genetic_mutation.py`。

**自测 3**（909 Snakes and Ladders）—— $n=2$ 的棋盘格子编号为 `[3,4],[1,2]`（底行左起），`board[1][0]=3` 表示格子 1 有梯子跳到格子 3。手动验证坐标变换：格子 1 → `(1,0)`，格子 2 → `(1,1)`，格子 3 → `(0,1)`，格子 4 → `(0,0)`。提示：BFS，`label_to_coord` 函数须处理奇偶行方向，跳转后记录 `nxt` 再判断是否已访问。参考 `solutions/graph_bfs/p909_snakes_and_ladders.py`。

---

## 题目全览（3 题）

| # | 题目 | 套路分类 | 难度 |
|---|---|---|---|
| 127 | Word Ladder | 字符串变换 BFS + 26 字母枚举 | Hard |
| 433 | Minimum Genetic Mutation | 字符串变换 BFS + 4 碱基枚举 | Medium |
| 909 | Snakes and Ladders | 棋盘 BFS + 蛇形坐标变换 | Medium |

---

## 融合版说明

| 段 | 来源 | 价值 |
|---|---|---|
| 一例速记 | 本文件 | 3 题套路一览 + AI（GNN 消息传递）关联 |
| 思维路径还原 | 本文件 | 从题目条件到 BFS 实现的解题独白 |
| 抽象成方法 | 本文件 | 4 个标准模板（通用 BFS / 字符串变换 BFS / 双向 BFS / 棋盘 BFS）+ 速查表 |
| 方法变形 | 本文件 | 3 类变体（Word Ladder 系列 / BFS vs Dijkstra / 坐标变换易错） |
| 思考路标 | 本文件 | 7 条题型识别条件反射 |
| 易错点 | 本文件 | 6 条高频踩坑（返回值定义 / 坐标变换 / 边界处理） |
| 典型应用例题 | solutions/ | 3 道精讲（127、433、909），代码 + 复杂度分析 |
| 自测题 | leetcode | 3 题带提示，链接 solutions 文件 |
| 题目全览 | 本文件 | 3 题完整列表，套路分类一览 |

---

> **跨 category 导航**：
> - 有权图最短路 → Dijkstra（优先队列），见 heap category
> - BFS 按层思路同样适用于树的层序遍历 → 见 `06-binary-tree-bfs.md`
> - 图的 DFS 遍历 / 连通分量 → 见 `19-graph-general.md`
> - GNN 中每层聚合邻居信息 = BFS 每层扩展，是图深度学习的核心操作
