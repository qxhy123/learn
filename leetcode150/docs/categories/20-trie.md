# 20 — Trie（融合版）

> **难度**：★★★☆☆
> **题数**：3
> **核心套路**：Trie 标准实现（前缀树）、通配符 DFS 搜索、Trie + 回溯单词搜索
> **本文件**：覆盖 trie 3 题的算法套路总结 + 典型题精讲 + 自测

---

## 一例速记

> **Trie 标准实现**：每个节点包含 `children[26]`（或 `dict`）和 `is_end` 标志；`insert` 逐字符插入，`search` 逐字符查找并检查末节点 `is_end`，`startsWith` 只查前缀是否存在（208）
> **通配符 DFS 搜索**：211 题中 `'.'` 匹配任意字符，遇到 `'.'` 时对当前节点的所有非空子节点递归搜索，普通字符则直接沿确定路径走
> **Trie + 回溯（Word Search II）**：先将所有词插入 Trie，再对棋盘每个格子做 DFS + 回溯；在 Trie 上同步走，若当前 Trie 节点为 `is_end` 则收集该词，回溯时撤销棋盘格子标记（212）
> **剪枝优化**：212 中找到一个单词后将对应 Trie 节点的 `word` 置为 None（防止重复收集）；若 Trie 节点已无子节点可剪去，减少后续无效搜索
> **AI 关联**：NLP 词典 / 输入法候选词（前缀查询）/ 搜索引擎自动补全 / 拼写检查；BERT tokenizer 的词表也基于前缀树管理子词（subword）词典

---

## 思维路径还原

> "看到 **'208 实现前缀树'** → 直接套 Trie 标准模板：
> 每个 `TrieNode` 有 `children = {}` 和 `is_end = False`。
> `insert(word)`：从根逐字符走，不存在则新建节点，最后标记 `is_end = True`。
> `search(word)`：逐字符走，若中途找不到字符返回 False，末尾返回 `node.is_end`。
> `startsWith(prefix)`：逐字符走，成功走完 prefix 则返回 True，中途断则返回 False。
> 时间：三个操作均 $O(L)$（L 为字符串长度），空间 $O(\sum L_i)$（插入词的总长度）。
>
> 看到 **'211 通配符搜索'** → Trie + DFS：
> `insert` 正常插入；`search` 遇到普通字符走确定路径，遇到 `'.'` 则对所有非空子节点递归搜索，
> 任一子路径搜索成功则返回 True。因为有通配符，无法走确定路径，必须用 DFS/回溯。
>
> 看到 **'212 单词搜索 II'** → Trie + 棋盘回溯：
> 先把所有单词 insert 进 Trie，再对棋盘每个格子启动 DFS（四方向），
> DFS 过程中同时在 Trie 上向下走：若当前字符不在 Trie 子节点中则剪枝；
> 若 Trie 节点 `is_end` 则收集当前词，将 `is_end = False`（防重复）；
> 回溯时恢复棋盘格子（`board[r][c] = c_saved`）。"

---

## 学习目标

- 掌握 Trie 节点的两种实现（`dict` vs 定长数组 `children[26]`）及其权衡
- 熟练实现 `insert` / `search` / `startsWith` 三个核心操作
- 理解通配符 `'.'` 的 DFS 处理方式（枚举所有子节点）
- 掌握 Trie + 棋盘回溯的组合技（212），包括剪枝策略
- 能识别"前缀匹配 / 自动补全 / 词典搜索"题型并直接套模板

---

## 抽象成方法（标准模板代码）

### 套路 1：Trie 标准实现（字典版）

适用题：208

```python
class TrieNode:
    def __init__(self):
        self.children: dict[str, 'TrieNode'] = {}
        self.is_end: bool = False


class Trie:
    """
    208: 前缀树标准实现。insert/search/startsWith 均 O(L)，L 为字符串长度。
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._walk(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        return self._walk(prefix) is not None

    def _walk(self, s: str) -> 'TrieNode | None':
        """沿 s 逐字符走，返回末节点；中途断则返回 None。"""
        node = self.root
        for c in s:
            if c not in node.children:
                return None
            node = node.children[c]
        return node
```

---

### 套路 2：Trie 定长数组版（26 字符，性能更优）

适用题：208、212

```python
class TrieNodeArray:
    """定长数组版，空间固定但访问更快（无哈希冲突）。"""
    __slots__ = ['children', 'is_end']

    def __init__(self):
        self.children: list['TrieNodeArray | None'] = [None] * 26
        self.is_end: bool = False


class TrieArray:
    def __init__(self):
        self.root = TrieNodeArray()

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            idx = ord(c) - ord('a')
            if node.children[idx] is None:
                node.children[idx] = TrieNodeArray()
            node = node.children[idx]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            idx = ord(c) - ord('a')
            if node.children[idx] is None:
                return False
            node = node.children[idx]
        return node.is_end

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            idx = ord(c) - ord('a')
            if node.children[idx] is None:
                return False
            node = node.children[idx]
        return True
```

> 两种实现对比：字典版节省空间（字符集稀疏时），数组版访问速度快（下标直接寻址，无哈希）。字符集为全小写 26 字母时首选数组版。

---

### 套路 3：Trie + 通配符 DFS（Word Search）

适用题：211

```python
class WordDictionary:
    """
    211: 支持 '.' 通配符的单词搜索。insert O(L)，search O(26^L) 最坏情况。
    """

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word: str) -> bool:
        return self._dfs(self.root, word, 0)

    def _dfs(self, node: TrieNode, word: str, i: int) -> bool:
        """从 node 开始，匹配 word[i:]；'.' 匹配任意字符。"""
        if i == len(word):
            return node.is_end
        c = word[i]
        if c == '.':
            # 枚举所有子节点
            for child in node.children.values():
                if self._dfs(child, word, i + 1):
                    return True
            return False
        else:
            if c not in node.children:
                return False
            return self._dfs(node.children[c], word, i + 1)
```

---

### 套路 4：Trie + 棋盘回溯（Word Search II）

适用题：212

```python
from typing import List


class TrieNodeWord:
    """扩展 TrieNode，存储到达该节点所拼成的单词（仅在 is_end 节点）。"""
    def __init__(self):
        self.children: dict[str, 'TrieNodeWord'] = {}
        self.word: str | None = None   # 非 None 时表示此处是一个完整单词


def find_words(board: List[List[str]], words: List[str]) -> List[str]:
    """
    212: Trie + 棋盘 DFS 回溯。时间 O(M·N·4^L)，L=最长单词长度。
    """
    # 构建 Trie
    root = TrieNodeWord()
    for w in words:
        node = root
        for c in w:
            if c not in node.children:
                node.children[c] = TrieNodeWord()
            node = node.children[c]
        node.word = w

    m, n = len(board), len(board[0])
    result: List[str] = []

    def dfs(r: int, c: int, trie_node: TrieNodeWord) -> None:
        ch = board[r][c]
        if ch not in trie_node.children:
            return
        nxt = trie_node.children[ch]
        if nxt.word is not None:
            result.append(nxt.word)
            nxt.word = None         # 防止同一单词被重复收集

        board[r][c] = '#'          # 标记已访问
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and board[nr][nc] != '#':
                dfs(nr, nc, nxt)
        board[r][c] = ch           # 回溯：恢复格子

    for r in range(m):
        for c in range(n):
            dfs(r, c, root)
    return result
```

---

### 速查表

| 题型特征 | 套路 | 时间（每次操作）| 空间 |
|---|---|---|---|
| 插入 / 精确搜索 / 前缀搜索 | Trie 标准实现 | $O(L)$ | $O(\sum L_i)$ |
| 支持 `'.'` 通配符搜索 | Trie + DFS 枚举子节点 | $O(26^L)$ 最坏 | $O(\sum L_i)$ |
| 棋盘中搜索多个单词 | Trie + 棋盘回溯 + 剪枝 | $O(MN \cdot 4^L)$ | $O(\sum L_i + MN)$ |

---

## 方法变形（3 类）

### 变形 1：Trie 节点设计扩展

- **208**：最基础，仅需 `children` + `is_end`。
- **211**：同 208，但 `search` 改为 DFS 以支持 `'.'`。
- **212**：`TrieNode` 增加 `word` 字段（或 `is_end` + 另存单词到列表），收集整词更方便。
- 进阶：自动补全场景中，每个节点可额外存 top-k 热词列表（LRU 缓存思路）。

### 变形 2：棋盘 DFS 剪枝策略

- **212 基础剪枝**：发现单词后置 `nxt.word = None` 防重复；Trie 节点若 `children` 为空且 `word` 为 None 则可删除，减少后续无效搜索（但实现稍复杂，面试中通常只做第一层剪枝即可）。
- **79**（Word Search I，非本 category）：仅需在棋盘找一个单词，不用 Trie，直接 DFS + 回溯。
- **212 vs 79**：单词 → DFS 回溯；多单词 → Trie + DFS 回溯（避免对每个词各做一次完整 DFS）。

### 变形 3：前缀树应用场景

- **输入法候选词**：`startsWith(prefix)` 找所有匹配前缀的单词（BFS / DFS Trie 子树）。
- **最长公共前缀**（14，非本 category）：也可用 Trie 解决（走到分叉点为止），但排序 + 比首尾更简洁。
- **XOR 最大化**（421 非本 category）：二进制 Trie，每位贪心选择与当前位相反的路径，最大化 XOR。

---

## 思考路标（条件反射）

1. 看到 **"前缀树 / 插入 / 精确搜索 / startsWith"** → Trie 标准实现，三个操作各 $O(L)$
2. 看到 **"通配符 `'.'` / 模式匹配"** → Trie + DFS，`'.'` 时枚举所有子节点
3. 看到 **"棋盘中找多个单词"** → Trie + 棋盘 DFS 回溯（而非对每个词单独 DFS）
4. 看到 **"棋盘中找单个单词"** → 直接 DFS + 回溯（79 Word Search），不需要 Trie
5. 看到 **"字符集为小写字母"** → 考虑定长数组版 `children[26]`，性能优于 `dict`
6. 看到 **"防止重复输出"** → 找到单词后将 Trie 节点的 `word` 置为 None（而非用结果集 set）
7. 看到 **"自动补全 / 输入法 / 词典前缀"** → Trie，AI 场景中 subword tokenizer 也用前缀树管理词表

---

## 易错点

1. **208 search vs startsWith 的区别**：`search` 要求完整匹配（必须检查末节点 `is_end`），`startsWith` 只要前缀存在即可（不检查 `is_end`）；两者代码相差仅最后一行，容易混淆。
2. **211 递归终止条件**：`i == len(word)` 时返回 `node.is_end`，而不是 `True`；若此时节点不是词尾则返回 False。
3. **212 回溯标记**：进入 DFS 前必须将 `board[r][c]` 标记为 `'#'`，DFS 结束后必须恢复为原字符 `ch`；常见错误是忘记恢复导致后续路径无法使用该格子。
4. **212 防重复收集**：同一单词可能在棋盘多处找到，用 `nxt.word = None`（而非 result 用 set）来防重复，因为 set 需要额外的包含检查。
5. **Trie 根节点**：根节点本身不代表任何字符，是所有字符的入口；`insert` 和 `search` 从 `root.children` 开始，不要把根节点本身当作字符节点使用。
6. **定长数组越界**：若题目包含非小写字母字符（如大写字母、数字），定长数组 `[None]*26` 会越界（`ord(c) - ord('a')` 可能为负或超过 25），此时改用 `dict` 版。
7. **211 最坏时间复杂度**：全为 `'.'` 时 DFS 遍历整个 Trie，时间 $O(26^L)$；实际上 Trie 中插入的词数量有限，不会退化到理论最坏。

---

## 典型应用例题

### 例 1：208. Implement Trie (Prefix Tree)

**题目**：实现 Trie 数据结构，支持 `insert(word)`、`search(word)`（精确匹配）、`startsWith(prefix)`（前缀匹配）三个操作。

**思路**：每个 `TrieNode` 维护 `children`（字典或定长数组）和 `is_end` 标志。`insert` 逐字符创建路径，`search` 逐字符走并检查末尾 `is_end`，`startsWith` 只验证路径存在性。

**解**：

```python
# 参考：solutions/trie/p208_implement_trie_prefix_tree.py
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            if c not in node.children:
                return False
            node = node.children[c]
        return node.is_end

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children[c]
        return True
```

**分析**：三个操作时间均 $O(L)$，$L$ 为字符串长度。空间 $O(\sum L_i)$，即所有插入单词长度之和（最坏情况下无公共前缀）。

---

### 例 2：211. Design Add and Search Words Data Structure

**题目**：实现数据结构支持 `addWord(word)` 和 `search(word)`，`search` 中 `'.'` 可匹配任意单个字母。

**思路**：`addWord` 同标准 Trie 的 `insert`；`search` 在 Trie 上 DFS，遇到普通字符走确定路径，遇到 `'.'` 则枚举所有非空子节点递归搜索。

**解**：

```python
# 参考：solutions/trie/p211_design_add_and_search_words_data_structure.py
class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word: str) -> bool:
        def dfs(node: TrieNode, i: int) -> bool:
            if i == len(word):
                return node.is_end
            c = word[i]
            if c == '.':
                return any(dfs(child, i + 1) for child in node.children.values())
            if c not in node.children:
                return False
            return dfs(node.children[c], i + 1)
        return dfs(self.root, 0)
```

**分析**：`addWord` $O(L)$；`search` 最坏 $O(26^L)$（全 `'.'`），实际取决于 Trie 中实际存储的词，远小于理论上界。

---

### 例 3：212. Word Search II

**题目**：给定 $m \times n$ 棋盘和单词列表 `words`，找出所有出现在棋盘中的单词（字符需 4 连通且不重复使用）。

**思路**：将所有单词插入 Trie。对棋盘每个格子做 DFS，同时在 Trie 上行走；若 Trie 节点有 `word`（词尾）则收集该词并清除（防重复）；DFS 时回溯恢复格子。

**解**：

```python
# 参考：solutions/trie/p212_word_search_ii.py
def findWords(board: List[List[str]], words: List[str]) -> List[str]:
    root = TrieNodeWord()
    for w in words:
        node = root
        for c in w:
            if c not in node.children:
                node.children[c] = TrieNodeWord()
            node = node.children[c]
        node.word = w

    m, n = len(board), len(board[0])
    result = []

    def dfs(r, c, trie):
        ch = board[r][c]
        if ch not in trie.children:
            return
        nxt = trie.children[ch]
        if nxt.word:
            result.append(nxt.word)
            nxt.word = None
        board[r][c] = '#'
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<m and 0<=nc<n and board[nr][nc] != '#':
                dfs(nr, nc, nxt)
        board[r][c] = ch

    for r in range(m):
        for c in range(n):
            dfs(r, c, root)
    return result
```

**分析**：棋盘格子 $MN$ 个，每个格子启动 DFS，最坏深度为最长单词长度 $L$，每层最多 4 个方向。时间 $O(MN \cdot 4^L)$，Trie 剪枝大幅减少实际搜索量。

---

## 自测题

**自测 1**（208 Implement Trie）—— 依次调用 `insert('apple')`、`search('apple')`（True）、`search('app')`（False）、`startsWith('app')`（True）、`insert('app')`、`search('app')`（True）。提示：`search` 需检查末节点 `is_end`，`startsWith` 不检查。参考 `solutions/trie/p208_implement_trie_prefix_tree.py`。

**自测 2**（211 Add and Search）—— `addWord('bad')`、`addWord('dad')`、`addWord('mad')`，`search('pad')`→ False，`search('bad')`→ True，`search('.ad')`→ True，`search('b..')`→ True。提示：DFS，遇 `'.'` 枚举所有子节点，普通字符走确定路径。参考 `solutions/trie/p211_design_add_and_search_words_data_structure.py`。

**自测 3**（212 Word Search II）—— `board=[['o','a','a','n'],['e','t','a','e'],['i','h','k','r'],['i','f','l','v']], words=['oath','pea','eat','rain']`，应返回 `['oath','eat']`（顺序不限）。提示：先建 Trie，棋盘每格启动 DFS + 回溯，找到词后清除 Trie 节点 `word` 防重复。参考 `solutions/trie/p212_word_search_ii.py`。

---

## 题目全览（3 题）

| # | 题目 | 套路分类 | 难度 |
|---|---|---|---|
| 208 | Implement Trie (Prefix Tree) | Trie 标准实现（insert / search / startsWith） | Medium |
| 211 | Design Add and Search Words | Trie + 通配符 DFS | Medium |
| 212 | Word Search II | Trie + 棋盘回溯 + 剪枝 | Hard |

---

## 融合版说明

| 段 | 来源 | 价值 |
|---|---|---|
| 一例速记 | 本文件 | 3 题套路一览 + AI（NLP 词典 / 输入法）关联 |
| 思维路径还原 | 本文件 | 3 道题的解题独白，含关键判断点 |
| 抽象成方法 | 本文件 | 4 个标准模板（字典版 Trie / 数组版 Trie / 通配符 DFS / 棋盘回溯）+ 速查表 |
| 方法变形 | 本文件 | 3 类变体（节点设计扩展 / 棋盘剪枝 / 前缀树应用场景） |
| 思考路标 | 本文件 | 7 条题型识别条件反射 |
| 易错点 | 本文件 | 7 条高频踩坑（is_end / 回溯恢复 / 防重复） |
| 典型应用例题 | solutions/ | 3 道精讲（208、211、212），代码 + 分析 |
| 自测题 | leetcode | 3 题带提示，链接 solutions 文件 |
| 题目全览 | 本文件 | 3 题完整列表 |

---

> **跨 category 导航**：
> - 棋盘 DFS 回溯（79 Word Search 单词）→ `08-backtracking.md`
> - 字符串前缀最长公共前缀（14）→ `01-array-string.md`
> - 图的 DFS（连通分量）→ `19-graph-general.md`
> - BERT / GPT tokenizer 的 BPE（Byte Pair Encoding）也需要维护前缀词表，与 Trie 高度相关
