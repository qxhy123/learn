# 103. Binary Tree Zigzag Level Order Traversal

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
- Official Group: Binary Tree BFS
- Pattern Group: Binary Tree BFS
- Patterns: binary-tree-bfs, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given the root of a binary tree, return the node values level by level, but alternate the reading direction on each level.

A normal level-order traversal reads every level from left to right:

```text
level 0: left -> right
level 1: left -> right
level 2: left -> right
...
```

This problem asks for a zigzag order:

```text
level 0: left -> right
level 1: right -> left
level 2: left -> right
level 3: right -> left
...
```

So the problem has two separate requirements:

1. Group nodes by depth.
2. Reverse the value order on every other depth.

The important detail is that the tree structure itself does not change. We are not rewiring children, swapping nodes, or changing how descendants are discovered. We are only deciding how to output the values for each completed level.

For this tree:

```text
        3
       / \
      9  20
        /  \
       15   7
```

The levels are:

```text
level 0: [3]
level 1: [9, 20]
level 2: [15, 7]
```

After applying zigzag direction:

```text
level 0 left-to-right: [3]
level 1 right-to-left: [20, 9]
level 2 left-to-right: [15, 7]
```

Final answer:

```text
[[3], [20, 9], [15, 7]]
```

### 2. Start From the Brute Force Idea

A very direct way to think about the task is:

1. Visit every node.
2. Record its depth.
3. Store values in a list for that depth.
4. After traversal finishes, reverse the lists for odd-numbered depths.

For example, a DFS baseline could do this:

```python
def dfs(node, depth):
    if node is None:
        return

    if depth == len(levels):
        levels.append([])

    levels[depth].append(node.val)
    dfs(node.left, depth + 1)
    dfs(node.right, depth + 1)

levels = []
dfs(root, 0)

for depth in range(len(levels)):
    if depth % 2 == 1:
        levels[depth].reverse()
```

This is a valid baseline: each node is visited once, and values are grouped by depth.

But it has a small mismatch with the shape of the output. The answer is naturally level-by-level, and breadth-first search already processes a tree one level at a time. If we use BFS, the grouping by depth falls out directly from the queue instead of being reconstructed from recursive depth numbers.

The first-principles question becomes:

> How can we make the queue contain exactly one level at a time?

Once that is true, the zigzag rule is just a direction flag attached to each completed level.

### 3. The Key Observation

In a binary tree, all nodes at depth `d + 1` are children of nodes at depth `d`.

That means if a queue currently holds all nodes at one level, then processing exactly those nodes and enqueueing their children creates the next level.

For example, suppose the queue contains:

```text
[9, 20]
```

These are exactly the nodes at level `1`. While processing them, we enqueue their children:

```text
9 has no children
20 has children 15 and 7
```

After processing exactly two nodes, the queue becomes:

```text
[15, 7]
```

That is exactly level `2`.

This is the main invariant:

```text
At the start of each outer loop iteration, the queue contains exactly the nodes of the next level, ordered from left to right.
```

Notice the phrase “at the start.” During the loop, the queue temporarily contains a mix of remaining current-level nodes and newly added next-level nodes. That is why we must freeze the current level size before processing the level.

### 4. Why the Queue Must Process a Fixed Level Size

Suppose the queue starts a level like this:

```text
queue = [9, 20]
```

The current level has size `2`.

If we process until the queue is empty, we will not stop after `9` and `20`, because processing `20` adds `15` and `7` to the same queue:

```text
before level: [9, 20]
after 9:      [20]
after 20:     [15, 7]
```

If the loop condition is simply “while queue is not empty,” the traversal would continue into `15` and `7`, mixing level `1` and level `2` into one output row.

So before processing a level, store:

```python
level_size = len(queue)
```

Then process exactly `level_size` nodes. Newly enqueued children belong to the next level and must wait for the next outer loop iteration.

This gives us a precise level boundary.

### 5. Where the Zigzag Direction Belongs

There are two common ways to build the zigzag row.

#### Option A: Collect Left-To-Right, Then Reverse When Needed

For each level:

1. Pop nodes from the queue in normal left-to-right BFS order.
2. Append their values to `level`.
3. If this level should be right-to-left, reverse `level` before appending it to the answer.

Conceptually:

```python
if left_to_right:
    result.append(level)
else:
    result.append(level[::-1])
```

This is simple and easy to reason about. The queue always stays in natural left-to-right order, and the output direction is handled only after the level is complete.

#### Option B: Insert Values at the Front for Right-To-Left Levels

For right-to-left levels, insert each value at the front of the current row.

Conceptually:

```python
if left_to_right:
    level.append(node.val)
else:
    level.appendleft(node.val)
```

This avoids a separate reverse operation if `level` is a deque, but it introduces another data structure and slightly more bookkeeping.

Both approaches are correct. The simplest explanation is Option A: keep traversal order stable, then reverse only the level values when the zigzag rule says to.

### 6. Detailed Algorithm

Handle the empty tree first:

```text
If root is None, there are no levels, so return [].
```

Otherwise:

1. Create a queue containing `root`.
2. Create an empty `result` list.
3. Set `left_to_right = True` for level `0`.
4. While the queue is not empty:
   1. Store `level_size = len(queue)`.
   2. Create an empty list `level`.
   3. Repeat `level_size` times:
      1. Remove the next node from the front of the queue.
      2. Append `node.val` to `level`.
      3. If `node.left` exists, append it to the queue.
      4. If `node.right` exists, append it to the queue.
   4. If `left_to_right` is `False`, reverse `level`.
   5. Append `level` to `result`.
   6. Flip `left_to_right`.
5. Return `result`.

The children are always enqueued left child before right child. This preserves the queue invariant that the next level is stored left-to-right. The zigzag effect is applied only to the values placed in the answer.

### 7. Pseudocode

```python
from collections import deque


def zigzagLevelOrder(root):
    if root is None:
        return []

    result = []
    queue = deque([root])
    left_to_right = True

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

        if not left_to_right:
            level.reverse()

        result.append(level)
        left_to_right = not left_to_right

    return result
```

The same idea can be implemented with `level[::-1]` instead of `level.reverse()`. If using `level.reverse()`, reverse before appending, because it mutates the list in place.

### 8. Detailed Example Walkthrough

Use the first example:

```text
root = [3, 9, 20, None, None, 15, 7]
```

Tree shape:

```text
        3
       / \
      9  20
        /  \
       15   7
```

Initialize:

```text
queue = [3]
result = []
left_to_right = True
```

#### Level 0

At the start of the level:

```text
queue = [3]
level_size = 1
```

Process exactly one node.

Pop `3`:

```text
level = [3]
```

Enqueue its children, left child first, then right child:

```text
queue = [9, 20]
```

The direction is left-to-right, so keep the level as is:

```text
result = [[3]]
```

Flip direction:

```text
left_to_right = False
```

#### Level 1

At the start of the level:

```text
queue = [9, 20]
level_size = 2
```

Process exactly two nodes.

Pop `9`:

```text
level = [9]
queue = [20]
```

`9` has no children, so nothing is enqueued.

Pop `20`:

```text
level = [9, 20]
```

Enqueue `20`'s children:

```text
queue = [15, 7]
```

The direction is right-to-left, so reverse this completed level:

```text
level = [20, 9]
result = [[3], [20, 9]]
```

Flip direction:

```text
left_to_right = True
```

#### Level 2

At the start of the level:

```text
queue = [15, 7]
level_size = 2
```

Process exactly two nodes.

Pop `15`:

```text
level = [15]
queue = [7]
```

Pop `7`:

```text
level = [15, 7]
queue = []
```

Both are leaves, so no children are enqueued.

The direction is left-to-right, so keep the level as is:

```text
result = [[3], [20, 9], [15, 7]]
```

The queue is empty, so traversal stops.

Final answer:

```text
[[3], [20, 9], [15, 7]]
```

### 9. Correctness

We prove that the algorithm returns exactly the binary tree zigzag level order traversal.

#### Lemma 1: At the start of each outer loop iteration, the queue contains exactly one tree level in left-to-right order.

Initially, the queue contains only `root`, which is exactly level `0` in left-to-right order.

Assume the queue contains exactly level `d` in left-to-right order at the start of an iteration. The algorithm stores the queue length as `level_size` and processes exactly those `level_size` nodes. For each processed node, it enqueues the left child before the right child. Since the parents are processed from left to right, their children are enqueued in the natural left-to-right order for level `d + 1`. No other nodes are enqueued. Therefore, after the iteration finishes, the queue contains exactly level `d + 1` in left-to-right order.

By induction, the invariant holds for every level.

#### Lemma 2: Each list appended to `result` contains exactly the values from one level.

During each outer loop iteration, the algorithm processes exactly the number of nodes that were in the queue at the start of that iteration. By Lemma 1, those nodes are exactly one tree level. The algorithm appends each processed node's value to `level` once. Therefore, before any optional reversal, `level` contains exactly the values from that level.

#### Lemma 3: Each appended level has the required zigzag direction.

The variable `left_to_right` starts as `True`, matching level `0`. After each level is appended, the algorithm flips the flag, so the flag alternates on every subsequent level.

If `left_to_right` is `True`, the algorithm appends the level values in their natural left-to-right order. If `left_to_right` is `False`, the algorithm reverses that completed level before appending it, giving right-to-left order. Therefore, each level is appended in the required direction.

#### Theorem: The algorithm returns the correct zigzag level order traversal.

By Lemma 2, every appended row contains exactly one tree level. By Lemma 3, every row has the correct alternating direction. By Lemma 1, levels are processed from top to bottom until all reachable nodes have been processed. Therefore, `result` is exactly the required zigzag level order traversal.

### 10. Complexity

Let `n` be the number of nodes in the tree, and let `w` be the maximum number of nodes on any level.

- Time: `O(n)`. Every node is removed from the queue once, every child pointer is considered once, and the total number of values reversed across all levels is `n`.
- Space: `O(w)` auxiliary space for the queue, excluding the output. The widest level determines the largest queue size. If counting the returned answer, the output itself stores `n` values, so total space including output is `O(n)`.

### 11. Common Pitfalls

- Not handling `root is None`. The empty tree should return `[]`, not `[[]]`.
- Forgetting to freeze `level_size` before processing a level. Without this, children can be mixed into the current row.
- Enqueueing children in different orders depending on the zigzag direction. The queue should usually stay left-to-right; only the output row changes direction.
- Reversing the queue instead of the completed level values. Reversing the queue can disturb the parent order needed to build the next level correctly.
- Flipping `left_to_right` inside the inner loop. The direction changes once per level, not once per node.
- Using `list.pop(0)` for the queue in Python. It works logically but costs `O(n)` per pop; `collections.deque.popleft()` is the appropriate queue operation.
- Calling `level.reverse()` inside `result.append(...)`. In Python, `reverse()` mutates the list and returns `None`, so `result.append(level.reverse())` appends `None`.

### 12. First-Principles Summary

The problem asks for top-to-bottom levels with alternating horizontal direction. BFS is natural because a queue can represent the current frontier of the tree. The essential invariant is that, at the start of each level, the queue contains exactly that level's nodes in left-to-right order. Freezing the queue length gives a clean boundary between the current level and the next one. Once a level's values are collected, the zigzag rule is only a question of whether to keep that row as is or reverse it before adding it to the answer.

## Implementation
See `solutions/binary_tree_bfs/p103_binary_tree_zigzag_level_order_traversal.py`.

## Tests
See `tests/binary_tree_bfs/test_p103_binary_tree_zigzag_level_order_traversal.py`.

## Examples

### Example 1
- Input: `{'root': [3, 9, 20, None, None, 15, 7]}`
- Output: `[[3], [20, 9], [15, 7]]`

### Example 2
- Input: `{'root': [1]}`
- Output: `[[1]]`

### Example 3
- Input: `{'root': []}`
- Output: `[]`

## Follow-up Practice
- Trace the queue level by level and write down `level_size` before each row.
- Implement the row-building step with a `deque` and `appendleft` instead of reversing odd levels.
- Modify the traversal to return normal level order, right-side view, or average value per level.
