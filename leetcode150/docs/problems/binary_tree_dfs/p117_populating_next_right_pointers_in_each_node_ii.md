# 117. Populating Next Right Pointers in Each Node II

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Each tree node has four fields:

```text
val
left
right
next
```

The `left` and `right` pointers already describe the vertical tree structure. The problem asks us to fill the extra horizontal pointer:

```text
node.next = the node immediately to its right on the same level
```

If there is no node to the right on that level, `node.next` should be `None`.

For this tree:

```text
        1
      /   \
     2     3
    / \     \
   4   5     7
```

The desired `next` chains are:

```text
Level 0: 1 -> None
Level 1: 2 -> 3 -> None
Level 2: 4 -> 5 -> 7 -> None
```

The output format used by LeetCode serializes each level by following `next` pointers and writing `#` after the end of a level:

```text
[1,#,2,3,#,4,5,7,#]
```

The important detail is that this is **Populating Next Right Pointers in Each Node II**, not the perfect-binary-tree version. Nodes may be missing anywhere:

```text
        1
       / \
      2   3
       \   \
        5   7
```

So we cannot assume:

```text
node.left.next = node.right
node.right.next = node.next.left
```

Those formulas only work when every internal node has exactly two children. Here, the next node for a child might be under a far-away parent on the same level.

### 2. Start From the Brute-Force / Baseline Idea

The easiest way to think about the problem is level order traversal.

If we put nodes of one level into a queue, then we can connect adjacent nodes in that queue:

```python
queue = [root]

while queue:
    level_size = len(queue)
    previous = None

    for _ in range(level_size):
        node = queue.pop_left()

        if previous is not None:
            previous.next = node
        previous = node

        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)

    previous.next = None
```

This is correct because a queue visits nodes level by level and left to right. When we process one level, the next node removed from the queue is exactly the node that should be connected by `next`.

The cost is:

- Time: `O(n)`, because every node is visited once.
- Extra space: `O(w)`, where `w` is the maximum width of the tree.

That baseline is useful because it exposes the real structure of the problem:

> We need to scan each level from left to right and build the next level's left-to-right chain.

The follow-up challenge is to do that without an explicit queue.

### 3. Key Observation: Existing `next` Pointers Can Replace the Queue

When one level's `next` pointers are already connected, that level becomes a linked list.

For example, after connecting level 1:

```text
2 -> 3 -> None
```

we can traverse that entire level without a queue:

```python
current = first_node_on_level
while current is not None:
    visit current
    current = current.next
```

While scanning this current level, we see all children that belong to the next level:

```text
current.left
current.right
```

If we append those children in the order we encounter them, they appear in exactly the required left-to-right order for the next level.

So the problem can be solved one level at a time:

```text
Use the current level's next chain to walk across parents.
While walking, build the next level's next chain from their children.
Move down to the first node of that next chain.
```

This is the central first-principles shift:

> A queue stores the next nodes to visit. But once a level is connected, its `next` pointers already store the horizontal order we need.

### 4. The Pointer / Level Invariant

At the start of each outer loop iteration, maintain this invariant:

```text
level_start points to the leftmost node of the current level,
and every node on the current level can be reached by repeatedly following next.
```

For the root level, this is true immediately:

```text
level_start = root
root.next = None
```

Inside that level, we build the next level using two helper pointers:

```text
next_head = first child found on the next level
next_tail = last child appended to the next level chain
```

They maintain a second invariant while scanning the current level:

```text
The children already discovered for the next level are connected from
next_head through next_tail, in left-to-right order.
```

Whenever we find a child:

1. If no child has been found yet, that child becomes both `next_head` and `next_tail`.
2. Otherwise, attach it after `next_tail` and move `next_tail` forward.

In pointer form:

```python
if next_head is None:
    next_head = child
    next_tail = child
else:
    next_tail.next = child
    next_tail = child
```

After scanning all parents on the current level, `next_head` is the leftmost node of the next level. That means it is the correct `level_start` for the next outer iteration.

### 5. Why Children Must Be Appended Left Before Right

The horizontal order of the next level is determined by two rules:

1. Parents are visited from left to right.
2. For each parent, its left child comes before its right child.

So when visiting a parent node, process:

```text
left child first
right child second
```

For this tree:

```text
        1
      /   \
     2     3
    / \     \
   4   5     7
```

Scanning level 1 gives parents in this order:

```text
2, then 3
```

Their children should be appended as:

```text
2.left  = 4
2.right = 5
3.left  = None
3.right = 7
```

So the next level chain becomes:

```text
4 -> 5 -> 7 -> None
```

If we processed right before left, or scanned parents in the wrong order, the chain would be wrong.

### 6. Detailed Algorithm

Handle the empty tree first:

```text
If root is None, return None.
```

Then repeat level by level:

1. Let `level_start` be the first node of the current level.
2. Create an empty chain for the next level:
   - `next_head = None`
   - `next_tail = None`
3. Walk across the current level using `current = current.next`.
4. For every `current`, append `current.left` if it exists.
5. Append `current.right` if it exists.
6. After the current level is exhausted, move down:
   - `level_start = next_head`
7. Continue until there is no next level.
8. Return `root`.

A small helper function often makes the code cleaner:

```python
def append(child):
    nonlocal next_head, next_tail

    if child is None:
        return

    if next_head is None:
        next_head = child
        next_tail = child
    else:
        next_tail.next = child
        next_tail = child
```

Then each parent contributes at most two children:

```python
append(current.left)
append(current.right)
```

### 7. Pseudocode

```python
def connect(root):
    if root is None:
        return None

    level_start = root

    while level_start is not None:
        next_head = None
        next_tail = None
        current = level_start

        while current is not None:
            for child in (current.left, current.right):
                if child is None:
                    continue

                if next_head is None:
                    next_head = child
                    next_tail = child
                else:
                    next_tail.next = child
                    next_tail = child

            current = current.next

        level_start = next_head

    return root
```

This version uses constant extra working space. It uses the tree's own `next` fields as the horizontal links instead of allocating a queue for entire levels.

### 8. Example Walkthrough

Use the official example:

```text
root = [1, 2, 3, 4, 5, None, 7]
```

The tree is:

```text
        1
      /   \
     2     3
    / \     \
   4   5     7
```

#### Initial State

```text
level_start = 1
```

The current level is just:

```text
1 -> None
```

The next level chain is empty:

```text
next_head = None
next_tail = None
```

#### Build Level 1

Visit `1`.

Its left child is `2`:

```text
next_head = 2
next_tail = 2
```

Its right child is `3`, so attach it after `2`:

```text
2.next = 3
next_tail = 3
```

Now the next level chain is:

```text
2 -> 3 -> None
```

Move down:

```text
level_start = 2
```

#### Build Level 2

The current level is available through `next` pointers:

```text
2 -> 3 -> None
```

Reset the next-level builder:

```text
next_head = None
next_tail = None
```

Visit `2`.

Append `2.left = 4`:

```text
next_head = 4
next_tail = 4
```

Append `2.right = 5`:

```text
4.next = 5
next_tail = 5
```

Now move horizontally:

```text
current = 2.next = 3
```

Visit `3`.

`3.left` is missing, so skip it.

Append `3.right = 7`:

```text
5.next = 7
next_tail = 7
```

The next level chain is now:

```text
4 -> 5 -> 7 -> None
```

Move down:

```text
level_start = 4
```

#### Build Level 3

The current level is:

```text
4 -> 5 -> 7 -> None
```

None of these nodes has children, so no `next_head` is created.

After scanning the level:

```text
next_head = None
level_start = None
```

The loop stops.

The final connected tree is:

```text
1 -> None
2 -> 3 -> None
4 -> 5 -> 7 -> None
```

Serialized by levels, this is:

```text
[1,#,2,3,#,4,5,7,#]
```

### 9. Correctness

We prove that the algorithm correctly assigns every `next` pointer.

#### Lemma 1: At the start of each outer loop, the current level can be traversed from left to right using `next` pointers.

For the first iteration, the current level contains only the root. Following `root.next` reaches `None`, so the invariant holds.

Assume the invariant holds for some current level. The algorithm scans that level by repeatedly following `current.next`, so it visits the parents from left to right. While doing so, it appends each existing left child before each existing right child. Therefore, it builds the next level's children in exactly left-to-right order and connects them using `next` pointers. The first discovered child becomes `next_head`, which is the leftmost node of the next level. Thus, when the algorithm moves to `level_start = next_head`, the invariant holds for the next level.

By induction, the invariant holds for every level the algorithm processes.

#### Lemma 2: While scanning one level, the chain from `next_head` to `next_tail` contains exactly the already discovered nodes of the next level in left-to-right order.

Before any child is discovered, the chain is empty, which is correct.

When the algorithm sees a missing child, it does nothing, so the chain remains correct.

When it sees an existing child, that child is the next node in left-to-right order because parents are scanned left to right and each parent's left child is considered before its right child. Appending the child after `next_tail` preserves the chain order and extends it by exactly one correct node.

Therefore, after scanning the whole current level, the chain contains all nodes of the next level in the correct order.

#### Lemma 3: Every node's `next` pointer is assigned to the node immediately to its right on the same level, or remains `None` if no such node exists.

By Lemma 2, each next-level chain is built in exact left-to-right order. Whenever a new child is appended after an existing `next_tail`, the old tail's `next` pointer is set to that new child, which is precisely the next node to its right. The last node on the chain has no later node appended, so its `next` remains `None`, which is the required value.

#### Theorem: The algorithm returns the tree with all `next` pointers populated correctly.

Every non-root node appears as a child of exactly one parent, so it is appended exactly once while building its level. By Lemma 3, every appended node receives the correct relationship to its right neighbor. The root level is also correct because the root has no right neighbor. Therefore all levels are connected correctly, and the returned `root` is the required tree.

### 10. Complexity

Let `n` be the number of nodes.

Each node is scanned once as `current` on its own level. Each node is also considered at most once as a child of its parent. Therefore:

```text
Time: O(n)
```

The algorithm stores only a few pointers:

```text
level_start
current
next_head
next_tail
```

No queue, stack, or recursion proportional to the tree size is required. Therefore:

```text
Extra space: O(1)
```

This does not count the output `next` pointers because they are part of the given nodes.

### 11. Common Pitfalls

#### Assuming the tree is perfect

The simpler problem, LeetCode 116, allows logic like:

```python
node.left.next = node.right
node.right.next = node.next.left
```

That fails here because `node.right`, `node.next`, or `node.next.left` may not exist. In problem 117, the next child might be several parents away.

#### Forgetting to skip missing children

Only real nodes should be appended to the next-level chain. A missing left child should not create a gap or stop the scan.

#### Losing the first node of the next level

`next_tail` is enough to append nodes, but it is not enough to move down after finishing the level. You also need `next_head`, the first discovered child.

#### Moving down before finishing the current level

Do not switch to the next level after processing one parent. You must scan every node connected by the current level's `next` chain first, because later parents may have children that belong on the same next level.

#### Processing children in the wrong order

Always append `left` before `right`. The next level's order depends on that local ordering.

#### Accidentally reusing stale builder pointers

`next_head` and `next_tail` must be reset to `None` for each level. Otherwise children from different levels may be incorrectly connected.

### 12. First-Principles Summary

The problem is about horizontal order within each tree level.

The baseline queue solution works because a queue gives us nodes level by level, left to right. The constant-space solution keeps the same idea but notices that once a level has been connected, its `next` pointers already form the queue-like order needed to scan that level.

So the algorithm is built around one invariant:

```text
At the start of a level, the current level is already a linked list through next pointers.
```

Using that linked list, we walk across the current level and construct the linked list for the next level. Then we move down and repeat.

The whole solution follows from this local rule:

```text
scan parents left to right;
append each existing left child, then right child;
the appended children form the next level's next chain.
```

## Implementation
See `solutions/binary_tree_dfs/p117_populating_next_right_pointers_in_each_node_ii.py`.

## Tests
See `tests/binary_tree_dfs/test_p117_populating_next_right_pointers_in_each_node_ii.py`.

## Examples

### Example 1
- Input: `{'root': [1, 2, 3, 4, 5, None, 7]}`
- Output: `'[1,#,2,3,#,4,5,7,#]'

### Example 2
- Input: `{'root': []}`
- Output: `[]`

## Follow-up Practice
- Solve the problem with a queue first, then remove the queue by using existing `next` pointers.
- Trace a sparse tree where a node's next neighbor is under a different parent.
- Explain why `next_head` and `next_tail` are enough to build one level at a time.
