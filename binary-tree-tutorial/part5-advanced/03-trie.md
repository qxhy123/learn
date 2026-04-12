# 第17章：字典树（Trie）

## 17.1 什么是 Trie

Trie（发音 "try"，来自 re**trie**val）是一种多叉树，专为字符串的**前缀匹配**设计：

```
插入 "cat", "car", "card", "care", "dog" 后的 Trie：

        root
       /    \
      c      d
      |      |
      a      o
     / \     |
    t   r    g(*)
   (*)  |
       / \
      d   e
     (*) (*)

(*) 表示该节点是某个单词的结尾
```

**Trie vs 哈希表**：

| | Trie | 哈希表 |
|-|------|--------|
| 精确查找 | O(m) | O(m) |
| 前缀查找 | O(m) | O(n) |
| 按字典序遍历 | O(n) | O(n log n) |
| 空间 | 可能大 | 紧凑 |

m = 字符串长度，n = 字符串数量

## 17.2 基本实现

```python
class TrieNode:
    def __init__(self):
        self.children = {}   # char → TrieNode
        self.is_end = False  # 是否为单词结尾
        self.count = 0       # 经过该节点的单词数（前缀计数）

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """插入单词，O(m)"""
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.count += 1
        node.is_end = True
    
    def search(self, word):
        """精确查找单词，O(m)"""
        node = self._find_prefix(word)
        return node is not None and node.is_end
    
    def starts_with(self, prefix):
        """是否存在以 prefix 开头的单词，O(m)"""
        return self._find_prefix(prefix) is not None
    
    def _find_prefix(self, prefix):
        """找到 prefix 对应的节点，O(m)"""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node
    
    def count_starts_with(self, prefix):
        """统计以 prefix 开头的单词数，O(m)"""
        node = self._find_prefix(prefix)
        return node.count if node else 0
    
    def delete(self, word):
        """删除单词，O(m)"""
        def _delete(node, word, depth):
            if depth == len(word):
                if not node.is_end:
                    return False  # 单词不存在
                node.is_end = False
                return len(node.children) == 0  # 是否可以删除此节点
            
            ch = word[depth]
            if ch not in node.children:
                return False
            
            should_delete = _delete(node.children[ch], word, depth + 1)
            if should_delete:
                del node.children[ch]
                return not node.is_end and len(node.children) == 0
            
            return False
        
        _delete(self.root, word, 0)

# 测试
trie = Trie()
for word in ["cat", "car", "card", "care", "dog"]:
    trie.insert(word)

print(trie.search("car"))         # True
print(trie.search("ca"))          # False（不是完整单词）
print(trie.starts_with("ca"))     # True
print(trie.count_starts_with("car"))  # 3（car, card, care）
```

## 17.3 固定字符集优化（26个字母）

```python
class TrieNodeArray:
    """用数组代替字典，更快但更耗内存"""
    def __init__(self):
        self.children = [None] * 26
        self.is_end = False
    
    def idx(self, ch):
        return ord(ch) - ord('a')

class TrieFast:
    def __init__(self):
        self.root = TrieNodeArray()
    
    def insert(self, word):
        node = self.root
        for ch in word:
            i = node.idx(ch)
            if node.children[i] is None:
                node.children[i] = TrieNodeArray()
            node = node.children[i]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for ch in word:
            i = node.idx(ch)
            if node.children[i] is None:
                return False
            node = node.children[i]
        return node.is_end
```

## 17.4 经典应用

### 单词搜索（带通配符）

```python
class WordDictionary:
    """支持 '.' 通配符的单词查找（LeetCode 211）"""
    
    def __init__(self):
        self.root = TrieNode()
    
    def add_word(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True
    
    def search(self, word):
        def dfs(node, i):
            if i == len(word):
                return node.is_end
            
            ch = word[i]
            if ch == '.':
                # 通配符：尝试所有子节点
                return any(dfs(child, i + 1) 
                          for child in node.children.values())
            elif ch in node.children:
                return dfs(node.children[ch], i + 1)
            return False
        
        return dfs(self.root, 0)
```

### 最长公共前缀

```python
def longest_common_prefix(words):
    """找到所有单词的最长公共前缀"""
    if not words:
        return ""
    
    trie = Trie()
    for word in words:
        trie.insert(word)
    
    # 从根出发，沿唯一路径走
    node = trie.root
    prefix = []
    
    while True:
        # 只有一个子节点且不是单词结尾
        if len(node.children) == 1 and not node.is_end:
            ch = next(iter(node.children))
            prefix.append(ch)
            node = node.children[ch]
        else:
            break
    
    return "".join(prefix)

print(longest_common_prefix(["flower", "flow", "flight"]))  # "fl"
```

### 单词替换（词根）

```python
def replace_words(dictionary, sentence):
    """
    用词典中的词根替换句子中的单词（LeetCode 648）
    例：词根 ["cat", "bat", "rat"]
    "the cattle was rattled" → "the cat was rat"
    """
    trie = Trie()
    for root in dictionary:
        trie.insert(root)
    
    def find_root(word):
        node = trie.root
        for i, ch in enumerate(word):
            if node.is_end:
                return word[:i]  # 找到词根
            if ch not in node.children:
                return word
            node = node.children[ch]
        return word
    
    return ' '.join(find_root(word) for word in sentence.split())
```

## 17.5 二进制 Trie（异或问题）

将整数按二进制位存入 Trie，高效解决异或相关问题：

```python
class BinaryTrie:
    """
    二进制 Trie，存储 32 位整数
    用于最大/最小异或问题
    """
    def __init__(self):
        self.root = {}
    
    def insert(self, num):
        node = self.root
        for i in range(31, -1, -1):  # 从高位到低位
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]
    
    def max_xor(self, num):
        """找出与 num 异或结果最大的已插入数"""
        node = self.root
        result = 0
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            # 贪心：尽量选相反的位（使异或为1）
            want = 1 - bit
            if want in node:
                result |= (1 << i)
                node = node[want]
            else:
                node = node[bit]
        return result

def find_max_xor(nums):
    """
    数组中两个数的最大异或值（LeetCode 421）
    O(n) 时间（Trie 方法）
    """
    trie = BinaryTrie()
    for num in nums:
        trie.insert(num)
    
    return max(trie.max_xor(num) for num in nums)

print(find_max_xor([3, 10, 5, 25, 2, 8]))  # 28
```

## 17.6 压缩 Trie（Patricia Tree）

当大量节点只有一个子节点时，可以压缩路径：

```
普通 Trie（插入 "abc", "abd"）:
root → a → b → c(*)
                \→ d(*)

压缩 Trie：
root → "ab" → c(*)
                \→ d(*)
```

用于路由表（IP 地址匹配）、文件系统路径等场景。

## 小结

- Trie 是前缀匹配的首选数据结构
- 插入/查找 O(m)，m 为字符串长度
- 二进制 Trie 解决整数异或问题
- 压缩 Trie 节省空间，用于实际路由器

## 练习

1. 实现"单词搜索 II"：在矩阵中找出所有出现在字典中的单词
2. 用 Trie 实现自动补全功能（返回最多 3 个热门推荐）
3. 实现 Trie 的序列化和反序列化

---

**上一章：** [树状数组](02-fenwick-tree.md) | **下一章：** [树形动态规划](04-tree-dp.md)
