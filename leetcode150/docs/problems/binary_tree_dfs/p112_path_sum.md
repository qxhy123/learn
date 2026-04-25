# 112. Path Sum

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/path-sum/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs, sum

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given the root of a binary tree and an integer `targetSum`, decide whether the tree contains at least one root-to-leaf path whose node values add up to exactly `targetSum`.

A root-to-leaf path is not just any downward path. It must:

1. Start at the root.
2. Move only from parent to child.
3. End at a leaf.

A leaf is a node with no left child and no right child.

For example, in this tree:

```text
        5
       / \
      4   8
     /   / \
   11   13  4
   / \       \
  7   2       1
```

with:

```text
targetSum = 22
```

one valid path is:

```text
5 -> 4 -> 11 -> 2
```

because:

```text
5 + 4 + 11 + 2 = 22
```

So the answer is `True`.

The problem is not asking for the path itself, and it is not asking for how many valid paths exist. It only asks a yes/no question:

> Does at least one root-to-leaf path have sum exactly equal to `targetSum`?

---

### 2. Start From the Brute-Force Idea

The most direct way to think about the problem is:

1. List every root-to-leaf path.
2. Compute the sum of each path.
3. Return `True` if any path sum equals `targetSum`.
4. If all paths fail, return `False`.

Conceptually:

```python
paths = all_root_to_leaf_paths(root)

for path in paths:
    if sum(path) == targetSum:
        return True

return False
```

This is correct, but it stores more information than we need.

For a path such as:

```text
5 -> 4 -> 11 -> 2
```

we do not actually need the full list `[5, 4, 11, 2]` if our only goal is to test its sum. We only need to know how much target remains after consuming the nodes seen so far.

After visiting `5`, the remaining amount is:

```text
22 - 5 = 17
```

After visiting `4`, the remaining amount is:

```text
17 - 4 = 13
```

After visiting `11`, the remaining amount is:

```text
13 - 11 = 2
```

At the leaf `2`, the remaining amount becomes:

```text
2 - 2 = 0
```

That means the path is valid.

So instead of building every path and summing later, we can check each path while walking down the tree.

---

### 3. The Key Observation

At any node, the only history that matters is:

```text
How much sum is still needed from this node down to a leaf?
```

Suppose we call a function on a node with a value called `remaining`.

That call should answer:

```text
Is there a root-to-leaf continuation starting at this node whose values add up to remaining?
```

When we stand on the current node, we must include `node.val` in the path. After including it, the children need to supply:

```text
remaining - node.val
```

This turns the original problem:

```text
Does root have a root-to-leaf path summing to targetSum?
```

into smaller problems:

```text
Does root.left have a root-to-leaf path summing to targetSum - root.val?
Does root.right have a root-to-leaf path summing to targetSum - root.val?
```

If either side succeeds, the whole tree succeeds.

This is why depth-first search is natural here: a candidate path is formed by repeatedly choosing one child until reaching a leaf.

---

### 4. Recursive Contract

Define the recursive helper like this:

```text
has_path(node, remaining)
```

Contract:

```text
Return True if and only if there exists a path that starts at node, ends at a leaf in node's subtree, and has total sum exactly remaining.
```

This contract is the center of the solution.

It says that `remaining` is not the original target anymore. It is the amount still required from the current node down to some leaf.

There are three important cases.

#### Case 1: Empty Node

If `node` is `None`, there is no path starting here.

```python
if node is None:
    return False
```

Even if `remaining == 0`, an empty child is not a root-to-leaf path. This detail matters for cases like an empty tree with `targetSum = 0`; the answer is still `False` because there is no root-to-leaf path at all.

#### Case 2: Leaf Node

If the node is a leaf, this is the only place where we are allowed to decide that a full path is complete.

```python
if node.left is None and node.right is None:
    return node.val == remaining
```

Why compare `node.val` to `remaining` instead of checking children?

Because the path must end at a leaf. Once we are at a leaf, there are no more nodes available. The path is valid exactly when this final node supplies exactly the remaining amount.

Equivalently, we could write:

```python
return remaining - node.val == 0
```

Both mean the same thing.

#### Case 3: Internal Node

If the node is not a leaf, include the current node and ask the children to finish the path:

```python
next_remaining = remaining - node.val
return (
    has_path(node.left, next_remaining)
    or has_path(node.right, next_remaining)
)
```

The `or` is important. We only need one valid path.

---

### 5. Detailed Algorithm

1. Start at the root with the full target:

```text
has_path(root, targetSum)
```

2. If the current node is missing, return `False`.

3. If the current node is a leaf, return whether its value exactly equals the remaining target.

4. Otherwise, subtract the current node's value from the remaining target.

5. Recursively search the left child and right child with the reduced remaining target.

6. Return `True` if either child can complete a valid root-to-leaf path.

This algorithm does not store paths. It lets the call stack represent the current path, and it carries only one number: the remaining sum needed.

---

### 6. Pseudocode

```python
def hasPathSum(root, targetSum):
    def has_path(node, remaining):
        if node is None:
            return False

        if node.left is None and node.right is None:
            return node.val == remaining

        next_remaining = remaining - node.val

        return (
            has_path(node.left, next_remaining)
            or has_path(node.right, next_remaining)
        )

    return has_path(root, targetSum)
```

The same logic can also be written without a nested helper by making the public function recursive:

```python
def hasPathSum(root, targetSum):
    if root is None:
        return False

    if root.left is None and root.right is None:
        return root.val == targetSum

    next_target = targetSum - root.val

    return (
        hasPathSum(root.left, next_target)
        or hasPathSum(root.right, next_target)
    )
```

Both versions use the same invariant.

---

### 7. Detailed Walkthrough of Example 1

Input:

```text
root = [5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1]
targetSum = 22
```

Tree:

```text
        5
       / \
      4   8
     /   / \
   11   13  4
   / \       \
  7   2       1
```

Start at the root:

```text
has_path(5, 22)
```

Node `5` is not a leaf, so include it:

```text
next remaining = 22 - 5 = 17
```

Search left subtree:

```text
has_path(4, 17)
```

Node `4` is not a leaf:

```text
next remaining = 17 - 4 = 13
```

Search left subtree again:

```text
has_path(11, 13)
```

Node `11` is not a leaf:

```text
next remaining = 13 - 11 = 2
```

Search its left child:

```text
has_path(7, 2)
```

Node `7` is a leaf. A leaf succeeds only if its value equals the remaining amount:

```text
7 == 2  -> False
```

So the path:

```text
5 -> 4 -> 11 -> 7
```

has sum:

```text
5 + 4 + 11 + 7 = 27
```

It is not valid.

Now search `11`'s right child:

```text
has_path(2, 2)
```

Node `2` is a leaf, and:

```text
2 == 2  -> True
```

So the path:

```text
5 -> 4 -> 11 -> 2
```

is valid.

Because we only need one valid path, the recursion can return `True` all the way back to the original call.

Final answer:

```text
True
```

---

### 8. Walkthrough of the Empty Tree Example

Input:

```text
root = []
targetSum = 0
```

The call is:

```text
has_path(None, 0)
```

There is no node, so there is no root-to-leaf path.

Return:

```text
False
```

This is an important distinction:

```text
A missing path with sum 0 is not a valid path.
```

The path must contain actual tree nodes.

---

### 9. Correctness

We prove that the algorithm returns `True` if and only if the tree contains a root-to-leaf path whose sum is `targetSum`.

Use the recursive contract:

```text
has_path(node, remaining) returns True if and only if there exists a path from node to a leaf whose sum is remaining.
```

#### Base Case: `node is None`

If `node` is `None`, no path can start there. Returning `False` is correct.

#### Base Case: `node` Is a Leaf

If `node` is a leaf, the only path from this node to a leaf is the one-node path containing `node` itself.

That path has sum:

```text
node.val
```

So it is valid exactly when:

```text
node.val == remaining
```

The algorithm returns exactly that condition, so the leaf case is correct.

#### Recursive Step: `node` Is an Internal Node

If `node` is not a leaf, every path from `node` to a leaf must:

1. Include `node.val`, and then
2. Continue either into the left subtree or into the right subtree.

After using `node.val`, the rest of the path must sum to:

```text
remaining - node.val
```

By the recursive contract, `has_path(node.left, remaining - node.val)` correctly tells whether the left subtree can supply such a continuation, and `has_path(node.right, remaining - node.val)` correctly tells whether the right subtree can supply such a continuation.

A valid path exists from `node` if and only if at least one of those child calls returns `True`. The algorithm returns their logical `or`, so the internal-node case is correct.

#### Whole Tree

The initial call is:

```text
has_path(root, targetSum)
```

By the contract, this returns `True` exactly when there is a path from the root to a leaf whose sum is `targetSum`. That is exactly what the problem asks.

---

### 10. Complexity

Let `n` be the number of nodes in the tree, and let `h` be the height of the tree.

#### Time Complexity

Each node is visited at most once.

At each node, the work is constant: check whether the node exists, check whether it is a leaf, subtract one value, and combine two boolean results.

So the time complexity is:

```text
O(n)
```

In the best case, the algorithm may stop early after finding a valid path, but the worst case still visits every node.

#### Space Complexity

The recursion stack stores one call for each node on the current root-to-node path.

So the auxiliary space is:

```text
O(h)
```

For a balanced tree:

```text
h = O(log n)
```

For a completely skewed tree:

```text
h = O(n)
```

No extra list of paths is required.

---

### 11. Common Pitfalls

#### Pitfall 1: Treating Any Node as a Valid Stopping Point

It is not enough for the running sum to equal `targetSum` at an internal node.

For example:

```text
    1
   /
  2
 /
3
```

with:

```text
targetSum = 3
```

The values `1 + 2` equal `3`, but node `2` is not a leaf. The path must continue to node `3`, so `1 -> 2` is not a valid answer.

Only leaf nodes can complete a path.

#### Pitfall 2: Returning `True` for an Empty Tree When `targetSum == 0`

An empty tree has no root-to-leaf path.

So:

```text
root = []
targetSum = 0
```

must return:

```text
False
```

This is why the `None` case returns `False` immediately.

#### Pitfall 3: Forgetting Negative Values Are Possible

The problem can include negative node values. Do not prune just because the remaining target becomes negative.

For example, a later negative value could bring the path sum back to the target.

The safe DFS rule is to continue until a leaf, not to stop based on whether the remaining amount is positive or negative.

#### Pitfall 4: Checking `remaining == 0` Before Consuming the Leaf

At a leaf, the correct check is:

```python
node.val == remaining
```

or:

```python
remaining - node.val == 0
```

Checking `remaining == 0` when entering the leaf is one step too early.

#### Pitfall 5: Confusing Root-to-Leaf Paths With Any Downward Path

This problem always starts at the root. A path that starts in the middle of the tree does not count.

It also must end at a leaf. A path ending at an internal node does not count.

---

### 12. First-Principles Summary

A path sum is an accumulated quantity along a single root-to-leaf route.

Instead of storing the route, carry the amount still needed:

```text
remaining = targetSum - sum(values already chosen above this node)
```

At each node:

1. The current node must be part of the path.
2. If it is a leaf, it succeeds exactly when its value equals `remaining`.
3. If it is not a leaf, subtract its value and ask either child to complete the path.

The core invariant is:

```text
has_path(node, remaining) answers whether node can start a downward path to a leaf whose sum is exactly remaining.
```

Once that contract is clear, the implementation is a direct translation of the definition of a root-to-leaf path.

## Implementation
See `solutions/binary_tree_dfs/p112_path_sum.py`.

## Tests
See `tests/binary_tree_dfs/test_p112_path_sum.py`.

## Examples

### Example 1
- Input: `{'root': [5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1], 'targetSum': 22}`
- Output: `True`

### Example 2
- Input: `{'root': [1, 2, 3], 'targetSum': 5}`
- Output: `False`

### Example 3
- Input: `{'root': [], 'targetSum': 0}`
- Output: `False`

## Follow-up Practice
- Write the recursive contract in one sentence before writing code.
- Trace the empty tree, a one-node tree, and a two-level tree by hand.
- Implement the same logic with an explicit stack of `(node, remaining)` pairs.
