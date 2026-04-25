# 146. LRU Cache

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/lru-cache/
- Official Group: Linked List
- Pattern Group: Linked List
- Patterns: linked-list

## First-Principles Explanation

### What The Problem Is Asking
An `LRUCache` stores key-value pairs with a fixed positive `capacity`. It must support two operations:

- `get(key)`: return the value for `key` if it exists, otherwise return `-1`.
- `put(key, value)`: insert or update `key` with `value`.

The special rule is the eviction rule. If a `put` would make the cache hold more than `capacity` keys, it must remove the **least recently used** key. A key becomes recently used whenever it is successfully read by `get` or written by `put`.

So the cache must answer two questions at all times:

1. Given a key, where is its value?
2. If the cache is full, which key has gone the longest without being touched?

The challenge is that both operations must run in `O(1)` average time. That rules out approaches that scan all keys to find the oldest one.

### Brute-Force Baseline
A direct implementation could store entries in a list ordered from least recently used to most recently used:

```text
[(oldest key, value), ..., (newest key, value)]
```

For `get(key)`, scan the list to find the key. If found, remove that pair from its current position and append it to the end because it is now most recently used. For `put(key, value)`, scan for the key. If it exists, update it and move it to the end. If it does not exist, append it, and if the list is too large, remove the first pair.

This is conceptually correct, but it is too slow:

- Finding a key in the list costs `O(capacity)`.
- Removing an entry from the middle of a Python list costs `O(capacity)` because later elements shift.
- The oldest entry is easy to find, but fast eviction alone is not enough.

A second brute-force version could use a dictionary from key to value plus a timestamp or counter recording the last use. Then `get` and updates are fast, but eviction requires scanning all timestamps to find the minimum, again `O(capacity)`.

The baseline teaches the real requirement: we need fast lookup by key **and** fast maintenance of recency order.

### Key Observation
The cache is two data structures pretending to be one:

1. A hash map gives `O(1)` average access from `key` to the entry.
2. A doubly linked list stores the entries in recency order and lets us remove or move any known node in `O(1)` time.

A singly linked list is not enough once a hash map points directly to a node. If we know the node for `key`, we still need to unlink it from its current position. Unlinking a singly linked node requires knowing its previous node, which we do not have in `O(1)` unless we store extra predecessor information. A doubly linked node carries both `prev` and `next`, so it can remove itself locally:

```text
node.prev.next = node.next
node.next.prev = node.prev
```

This is why the official pattern group is linked list even though the cache also depends on hashing. The hash map finds the node; the linked list makes recency changes constant-time.

### Hash Map + Doubly Linked List Invariant
Maintain a dictionary and a doubly linked list with two dummy sentinel nodes:

```text
left/sentinel head  <->  least recent  <->  ...  <->  most recent  <->  right/sentinel tail
```

The invariant is:

- Every real cache entry has exactly one linked-list node.
- `cache[key]` points to the node whose `key` is `key`.
- Nodes appear from least recently used near the left sentinel to most recently used near the right sentinel.
- The left and right sentinels are never real entries; they exist only to make insertion and deletion uniform.

With this invariant, each operation becomes simple:

- To mark a node as recently used, remove it from its current position and insert it immediately before the right sentinel.
- To evict the least recently used entry, remove the real node immediately after the left sentinel.
- To look up an entry by key, ask the dictionary.

The sentinels remove edge cases. The cache may be empty, have one item, or have many items, but insertion before `right` and removal of a known node use the same pointer rewiring every time.

### Detailed Algorithm
Define a node with `key`, `value`, `prev`, and `next`.

Initialize the cache:

1. Store `capacity`.
2. Create an empty dictionary `cache` mapping keys to nodes.
3. Create dummy nodes `left` and `right`.
4. Link them as `left <-> right`.

Use two helper operations.

`remove(node)` unlinks a real node from wherever it is:

1. Let `before = node.prev` and `after = node.next`.
2. Set `before.next = after`.
3. Set `after.prev = before`.

`insert_at_most_recent(node)` inserts a node immediately before `right`:

1. Let `before = right.prev`.
2. Set `before.next = node`.
3. Set `node.prev = before`.
4. Set `node.next = right`.
5. Set `right.prev = node`.

`get(key)`:

1. If `key` is not in the dictionary, return `-1`.
2. Otherwise, retrieve the node.
3. Remove it from its current list position.
4. Insert it at the most-recent position.
5. Return its value.

`put(key, value)`:

1. If `key` already exists, remove the old node from the list. This discards its old recency position.
2. Create a node containing `key` and `value`.
3. Store it in the dictionary under `key`.
4. Insert it at the most-recent position.
5. If the dictionary now has more than `capacity` entries:
   - The least recently used node is `left.next`.
   - Remove that node from the list.
   - Delete its key from the dictionary.

Updating an existing key counts as a use, so the updated key becomes most recent. A successful `get` also counts as a use. An unsuccessful `get` does not insert anything and does not change recency.

### Detailed Example Walkthrough
Use the official sequence with capacity `2`:

```text
operations: LRUCache(2), put(1,1), put(2,2), get(1), put(3,3), get(2), put(4,4), get(1), get(3), get(4)
outputs:    null,        null,     null,     1,      null,     -1,     null,     -1,     3,      4
```

Represent the linked list from least recent to most recent.

Start:

```text
cache = {}
list  = []
```

After `put(1, 1)`, key `1` is inserted and is the most recent item:

```text
cache = {1 -> node(1,1)}
list  = [1]
```

After `put(2, 2)`, key `2` is newer than key `1`:

```text
cache = {1 -> node(1,1), 2 -> node(2,2)}
list  = [1, 2]
```

`get(1)` finds value `1`. Reading key `1` makes it most recent, so move node `1` after node `2`:

```text
return 1
list = [2, 1]
```

`put(3, 3)` inserts key `3` as most recent:

```text
temporary list = [2, 1, 3]
```

The capacity is `2`, so evict the least recent node, which is the leftmost real node: key `2`.

```text
cache = {1 -> node(1,1), 3 -> node(3,3)}
list  = [1, 3]
```

`get(2)` misses because key `2` was evicted. A miss does not change the list:

```text
return -1
list = [1, 3]
```

`put(4, 4)` inserts key `4` as most recent:

```text
temporary list = [1, 3, 4]
```

Evict the least recent key `1`:

```text
cache = {3 -> node(3,3), 4 -> node(4,4)}
list  = [3, 4]
```

`get(1)` misses because key `1` was evicted:

```text
return -1
list = [3, 4]
```

`get(3)` returns `3` and moves key `3` to most recent:

```text
return 3
list = [4, 3]
```

`get(4)` returns `4` and moves key `4` to most recent:

```text
return 4
list = [3, 4]
```

The entire behavior comes from one invariant: the leftmost real node is always the next eviction candidate, and the rightmost real node is always the most recently touched key.

### Code / Pseudocode
One common Python shape is:

```python
class Node:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.left = Node()   # least-recent side sentinel
        self.right = Node()  # most-recent side sentinel
        self.left.next = self.right
        self.right.prev = self.left

    def remove(self, node):
        before = node.prev
        after = node.next
        before.next = after
        after.prev = before

    def insert_most_recent(self, node):
        before = self.right.prev
        before.next = node
        node.prev = before
        node.next = self.right
        self.right.prev = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self.remove(node)
        self.insert_most_recent(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])

        node = Node(key, value)
        self.cache[key] = node
        self.insert_most_recent(node)

        if len(self.cache) > self.capacity:
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]
```

Some implementations reuse the existing node when updating a key instead of creating a new node. That is also fine: update `node.value`, remove the node, and insert it at the most-recent position. The important requirement is that the dictionary and linked list still agree about the single live node for each key.

### Correctness
We prove that the algorithm returns the required values and evicts exactly the least recently used key.

First, the dictionary/list invariant holds after initialization. The dictionary is empty, and the linked list contains only the two sentinels, so there are no real entries that could violate the invariant.

Consider `get(key)`. If `key` is absent from the dictionary, the cache contains no entry for that key, so returning `-1` is correct and no recency order should change. If `key` is present, the dictionary points to exactly the node containing that key's value. The algorithm returns that value. Because a successful `get` makes the key most recently used, the algorithm removes the node from its old position and reinserts it immediately before the right sentinel. Removing a node preserves the relative order of all other nodes, and inserting it before the right sentinel places it after every other real node. Therefore the list order again matches least-to-most recent usage.

Consider `put(key, value)` when `key` already exists. The old node is the cache entry for `key`; removing it eliminates the old value and old recency position. The new node with the new value is stored in the dictionary and inserted as most recent, which is correct because writing the key counts as using it. The cache size does not increase after replacing an existing key, so no unrelated key is evicted.

Consider `put(key, value)` when `key` is new. The new node is added to both the dictionary and the most-recent end of the list, which correctly records the write as the latest use. If the size is still at most `capacity`, all entries are valid and ordered correctly. If the size exceeds `capacity`, exactly one key must be removed. Since the list is ordered from least recent to most recent, the real node after the left sentinel is exactly the least recently used key. Removing that node from the list and deleting its key from the dictionary restores the capacity limit while preserving the invariant for every remaining key.

By induction over the operation sequence, after every operation the dictionary contains exactly the keys currently in the cache, the linked list stores those same keys in least-to-most recent order, `get` returns the correct value or `-1`, and eviction removes the correct least recently used key.

### Complexity
Let `n` be the cache capacity.

- `get`: `O(1)` average time, because dictionary lookup, node removal, and node insertion are constant-time operations.
- `put`: `O(1)` average time, because dictionary insert/update/delete, node removal, node insertion, and eviction of `left.next` are constant-time operations.
- Space: `O(n)`, because at most `capacity` real nodes and dictionary entries are stored.

The `O(1)` time is average-case because it depends on hash map operations being average constant time.

### Common Pitfalls
- Treating `get` as read-only. A successful `get` must move the key to the most-recent position.
- Evicting after an update to an existing key. Replacing a key should not increase the number of live keys.
- Forgetting to delete the evicted key from the dictionary. The linked list and dictionary must describe the same live entries.
- Using a singly linked list without predecessor access. A hash map pointing to a singly linked node does not by itself allow `O(1)` removal.
- Rewiring pointers in an unsafe order. Save `prev` and `next` before changing links.
- Accidentally evicting a sentinel node. Only `left.next` is evicted after confirming the cache is over capacity; with positive capacity, it is a real node at that moment.
- Creating a new node for an existing key but leaving the old node in the list. That produces duplicate list nodes for one dictionary key and breaks eviction.
- Returning the node instead of the value from `get`.

### First-Principles Summary
An LRU cache is hard only if value lookup and recency tracking are mixed together. Split them. The hash map answers, "Where is the entry for this key?" The doubly linked list answers, "Which entry is oldest, and how do I move a touched entry to newest in constant time?"

The invariant is the whole solution: dictionary keys point to exactly the real linked-list nodes, and the list is ordered from least recently used to most recently used. Every successful access removes one known node and appends it near the most-recent sentinel. Every overflow removes the node near the least-recent sentinel. Because these are local pointer operations, both required cache operations run in constant average time.

## Implementation
See `solutions/linked_list/p146_lru_cache.py`.

## Tests
See `tests/linked_list/test_p146_lru_cache.py`.

## Examples

### Example 1
- Input: `{'raw': '["LRUCache","put","put","get","put","get","put","get","get","get"]\n[[2],[1,1],[2,2],[1],[3,3],[2],[4,4],[1],[3],[4]]'}`
- Output: `'See official examples'`

## Follow-up Practice
- Implement the same cache while reusing nodes on update instead of replacing them.
- Trace a capacity-1 cache through repeated `put` and `get` operations.
- Draw the sentinel nodes and verify that every helper operation preserves both directions of each link.
