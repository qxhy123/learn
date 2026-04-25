# 199. Binary Tree Right Side View

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/binary-tree-right-side-view/
- Official Group: Binary Tree BFS
- Pattern Group: Binary Tree BFS
- Patterns: binary-tree-bfs, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given the root of a binary tree.

Imagine standing on the right side of the tree and looking toward it. At each vertical depth, some nodes may be hidden behind other nodes. The problem asks for the values of the nodes you can see, from top to bottom.

The important detail is that visibility is decided separately for each depth.

For a tree like this:

```text
        1
       / \
      2   3
       \   \
        5   4
```

The levels are:

```text
depth 0: [1]
depth 1: [2, 3]
depth 2: [5, 4]
```

From the right side:

- at depth `0`, you see `1`
- at depth `1`, `3` hides `2`
- at depth `2`, `4` hides `5`

So the answer is:

```text
[1, 3, 4]
```

The problem is not asking for all right children. It is asking for the rightmost node at every depth.

That distinction matters. A node can be visible from the right even if it is a left child, as long as it is the rightmost existing node on its level.

For example:

```text
        1
       /
      2
     /
    3
```

The right side view is:

```text
[1, 2, 3]
```

Even though there are no right children, every level still has one visible node.

So the real problem is:

> For each depth of the tree, return the value of the node that appears farthest to the right on that depth.

---

### 2. Start From the Baseline Idea

A direct way to think about the problem is:

1. Group all nodes by depth.
2. For each depth, keep the nodes from left to right.
3. Take the last value from each group.

For the first example:

```text
depth 0 -> [1]       -> take 1
depth 1 -> [2, 3]    -> take 3
depth 2 -> [5, 4]    -> take 4
```

This gives:

```text
[1, 3, 4]
```

This baseline is correct because the visible node at a depth is exactly the rightmost node in that depth's left-to-right order.

One possible implementation would be to build a list of lists:

```python
levels = []

# after traversal:
# levels[0] = [1]
# levels[1] = [2, 3]
# levels[2] = [5, 4]

answer = [level[-1] for level in levels]
```

But we do not actually need to store every node in every level. Once we know the last node of a level, the earlier nodes from that same level no longer matter.

So the useful target is smaller:

> Visit nodes level by level, and record only the last node of each level.

---

### 3. The Key Observation

The phrase "right side view" sounds geometric, but the algorithmic property is simple:

```text
right side visible node at a depth = last node in that depth when scanned left to right
```

If we can isolate one depth at a time, then the problem becomes easy.

For one level:

```text
[2, 3, 7, 9]
```

The right side visible node is:

```text
9
```

For another level:

```text
[8]
```

The right side visible node is:

```text
8
```

So the whole problem can be solved by repeatedly answering this local question:

> Among the nodes currently on this level, which one is processed last?

Breadth-first search is a natural fit because it processes a tree by depth: root first, then all depth-1 nodes, then all depth-2 nodes, and so on.

---

### 4. Why a Queue Represents Levels

Breadth-first search uses a queue.

A queue removes items in the same order they were inserted:

```text
first in -> first out
```

For a binary tree, if we process each node and enqueue its left child before its right child, then nodes of the next level enter the queue in left-to-right order.

For example:

```text
        1
       / \
      2   3
     / \   \
    4   5   6
```

Start:

```text
queue = [1]
```

Process level 0:

```text
pop 1
enqueue 2, then 3
queue = [2, 3]
```

Now the queue contains exactly level 1, from left to right.

Process level 1:

```text
pop 2 -> enqueue 4, then 5
pop 3 -> enqueue 6
queue = [4, 5, 6]
```

Now the queue contains exactly level 2, from left to right.

This is the reason BFS gives us the right structure for the problem.

---

### 5. The Queue / Level Invariant

The central invariant is:

```text
At the start of each outer loop iteration, the queue contains exactly the nodes of one tree level, ordered from left to right.
```

This invariant gives us two powerful facts.

First, the number of nodes currently in the queue is the size of the current level:

```python
level_size = len(queue)
```

Second, if we process exactly `level_size` nodes, then the final node processed in that batch is the rightmost node of that level.

We must store `level_size` before the inner loop starts.

Why?

Because while processing this level, we enqueue children for the next level. If we keep checking `len(queue)` dynamically, the current level and next level will get mixed together.

The frozen size separates the two concepts:

```text
nodes to pop now     = current level
nodes being enqueued = next level
```

That separation is the heart of the algorithm.

---

### 6. Detailed Algorithm

If the tree is empty, there is no side view:

```text
root = null -> []
```

Otherwise:

1. Create an empty answer list.
2. Put the root in a queue.
3. While the queue is not empty:
   1. Freeze the current level size with `level_size = len(queue)`.
   2. Process exactly `level_size` nodes.
   3. For each node:
      - remove it from the front of the queue
      - enqueue its left child if it exists
      - enqueue its right child if it exists
      - if this node is the last node of the current level, append its value to the answer
4. Return the answer.

The last-node test is usually written as:

```python
if i == level_size - 1:
    answer.append(node.val)
```

where `i` is the index inside the current level.

This works because the level is processed from left to right. Therefore, index `level_size - 1` is the rightmost node in that level.

---

### 7. Pseudocode

```python
from collections import deque


def rightSideView(root):
    if root is None:
        return []

    answer = []
    queue = deque([root])

    while queue:
        level_size = len(queue)

        for i in range(level_size):
            node = queue.popleft()

            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

            if i == level_size - 1:
                answer.append(node.val)

    return answer
```

There is a small variation that records `node.val` after the loop instead of checking the index, but the idea is the same: each level contributes exactly its final left-to-right node.

---

### 8. Detailed Example Walkthrough

Use the official-style example:

```text
root = [1,2,3,null,5,null,4]
```

This represents:

```text
        1
       / \
      2   3
       \   \
        5   4
```

Initialize:

```text
answer = []
queue  = [1]
```

#### Level 0

At the start of the level:

```text
queue = [1]
level_size = 1
```

Process exactly one node.

`i = 0`:

```text
pop 1
enqueue left child 2
enqueue right child 3
```

Since `i == level_size - 1`, node `1` is the last node of this level.

Append it:

```text
answer = [1]
queue  = [2, 3]
```

#### Level 1

At the start of the level:

```text
queue = [2, 3]
level_size = 2
```

Process exactly two nodes.

`i = 0`:

```text
pop 2
enqueue left child: none
enqueue right child: 5
```

Node `2` is not the last node of the level because `i = 0` and `level_size - 1 = 1`.

Current state:

```text
answer = [1]
queue  = [3, 5]
```

Notice that `5` is already in the queue, but it belongs to the next level. This is why we froze `level_size = 2` before the loop.

`i = 1`:

```text
pop 3
enqueue left child: none
enqueue right child: 4
```

Now `i == level_size - 1`, so node `3` is the last node of level 1.

Append it:

```text
answer = [1, 3]
queue  = [5, 4]
```

#### Level 2

At the start of the level:

```text
queue = [5, 4]
level_size = 2
```

Process exactly two nodes.

`i = 0`:

```text
pop 5
no children
```

Node `5` is not the rightmost node of this level.

```text
answer = [1, 3]
queue  = [4]
```

`i = 1`:

```text
pop 4
no children
```

Node `4` is the last node of level 2.

Append it:

```text
answer = [1, 3, 4]
queue  = []
```

The queue is empty, so traversal is complete.

Return:

```text
[1, 3, 4]
```

---

### 9. Why This Is Correct

We prove that the algorithm returns exactly the right side view.

#### Lemma 1: At the start of each outer loop iteration, the queue contains exactly one level in left-to-right order.

Initially, the queue contains only the root. The root is the only node on level `0`, so the invariant is true.

Assume the invariant is true for some level. The algorithm freezes that level's size and pops exactly those nodes. For each popped node, it enqueues the left child before the right child. Because the current level is processed left to right, its children are enqueued in the left-to-right order of the next level.

After exactly the current level's nodes are popped, the queue contains all and only the children that form the next level, in left-to-right order.

So the invariant holds for the next iteration.

#### Lemma 2: The value appended during each level is the rightmost value of that level.

By Lemma 1, the queue contains the current level in left-to-right order at the start of the iteration.

The algorithm processes exactly `level_size` nodes from that queue. Therefore, the node processed when `i == level_size - 1` is the last node in the current level's left-to-right order.

The last node in left-to-right order is exactly the node visible from the right side at that depth.

So the appended value is correct for that level.

#### Lemma 3: Every depth contributes exactly one value.

The outer loop runs once per non-empty level. Inside that iteration, the algorithm appends a value exactly once: when it processes the final node of that level.

Therefore, every level contributes one visible value, and no level contributes more than one.

#### Theorem: The returned list is the binary tree's right side view from top to bottom.

By Lemma 2, each appended value is the correct visible node for its level. By Lemma 3, the algorithm appends exactly one value for every depth that exists in the tree. Because BFS processes levels from top to bottom, those values appear in the required order.

Therefore, the returned list is exactly the right side view.

---

### 10. Complexity

Let `n` be the number of nodes in the tree.

Every node is enqueued once and dequeued once.

So the time complexity is:

```text
O(n)
```

The queue stores at most one level's worth of nodes plus some children from the next level while a level is being processed. This is bounded by the maximum width of the tree.

Let `w` be the maximum number of nodes on any level.

The auxiliary space complexity is:

```text
O(w)
```

In the worst case, a complete binary tree can have width proportional to `n`, so worst-case space can be:

```text
O(n)
```

The answer list stores one value per level. If the tree height is `h`, the output size is `O(h)`. Usually, output space is not counted as auxiliary space, but it still exists as part of the returned result.

---

### 11. Common Pitfalls

#### Mistake 1: Thinking only right children matter

This is wrong:

```text
        1
       /
      2
       \
        5
```

The right side view is:

```text
[1, 2, 5]
```

Node `2` is a left child, but it is still visible because it is the only node at its depth.

The rule is not "follow right pointers." The rule is "take the rightmost node on each level."

#### Mistake 2: Not freezing the level size

If the code uses `while queue:` inside a level without saving `level_size`, it can accidentally process children from the next level as part of the current level.

Correct separation:

```python
level_size = len(queue)
for i in range(level_size):
    ...
```

The fixed size is what prevents levels from blending together.

#### Mistake 3: Appending every node

The answer should contain one value per depth, not all node values.

For this tree:

```text
        1
       / \
      2   3
```

The answer is:

```text
[1, 3]
```

not:

```text
[1, 2, 3]
```

#### Mistake 4: Returning the last node visited overall

The right side view is not a single node. It is one node per level.

For:

```text
        1
       / \
      2   3
     /
    4
```

The right side view is:

```text
[1, 3, 4]
```

Even though `4` is on the left side structurally, it is the only node at depth `2`, so it is visible.

#### Mistake 5: Using a Python list as a queue with `pop(0)`

A list can work logically, but `pop(0)` is `O(n)` because all remaining elements shift left.

Use `collections.deque` instead:

```python
from collections import deque
```

Then removing from the front is efficient:

```python
node = queue.popleft()
```

#### Mistake 6: Forgetting the empty tree

If `root` is `None`, there are no levels and no visible nodes.

Return:

```text
[]
```

---

### 12. Alternative First-Principles View: DFS

Although BFS is the most direct level-based solution, there is another useful way to reason about the problem.

If we do a depth-first traversal that visits the right child before the left child, then the first node we encounter at each depth is the right side visible node.

The DFS invariant would be:

```text
If this is the first time we have reached depth d, record this node.
```

Pseudocode:

```python
def dfs(node, depth):
    if node is None:
        return

    if depth == len(answer):
        answer.append(node.val)

    dfs(node.right, depth + 1)
    dfs(node.left, depth + 1)
```

This also works. But for this tutorial's BFS approach, the level invariant is more explicit: isolate one level, take its last node, move to the next level.

---

### 13. First-Principles Summary

The right side view is determined depth by depth.

At each depth, the visible node is not necessarily a right child. It is simply the node farthest to the right among all nodes at that depth.

Breadth-first search naturally groups nodes by depth. If children are enqueued left child first and right child second, each level is processed from left to right.

The only subtle implementation point is to freeze the queue size at the beginning of each level. That frozen size says:

```text
These nodes belong to the current level.
Everything enqueued while processing them belongs to the next level.
```

Once a level is isolated, the answer for that level is the final node processed in that fixed batch.

So the algorithm is:

```text
BFS by levels -> take the last node of each level -> return those values top to bottom
```

That is exactly the right side view.

## Implementation
See `solutions/binary_tree_bfs/p199_binary_tree_right_side_view.py`.

## Tests
See `tests/binary_tree_bfs/test_p199_binary_tree_right_side_view.py`.

## Examples

### Example 1
- Input: `{'raw': '[1,2,3,null,5,null,4]\n[1,2,3,4,null,null,null,5]\n[1,null,3]\n[]'}`
- Output: `'See official examples'`

## Follow-up Practice
- Trace the queue level by level and write down the frozen `level_size` before each level starts.
- Implement the BFS solution with `collections.deque`.
- Implement the right-first DFS variation and compare its invariant with the BFS invariant.
- Test a tree where the visible node on a level is a left child.
