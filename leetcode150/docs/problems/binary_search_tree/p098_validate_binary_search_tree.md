# 98. Validate Binary Search Tree

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/validate-binary-search-tree/
- Official Group: Binary Search Tree
- Pattern Group: Binary Search Tree
- Patterns: binary-search-tree, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given the root of a binary tree, decide whether that tree is a valid binary search tree.

A binary search tree is not just a tree where each node looks correct compared with its immediate children. The ordering rule applies to entire subtrees:

```text
Every value in the left subtree must be strictly less than the node's value.
Every value in the right subtree must be strictly greater than the node's value.
Both subtrees must also be valid binary search trees.
```

The word **strictly** matters. Duplicates are not allowed by this problem's definition.

So for a node with value `x`:

```text
left subtree values  < x <  right subtree values
```

That rule must hold at every node, all the way down the tree.

The hidden difficulty is that a node can satisfy its parent but still violate an older ancestor.

For example:

```text
        5
       / \
      1   7
         /
        4
```

The node `4` is less than its parent `7`, so the local parent-child comparison looks fine. But `4` is inside the right subtree of `5`, so it must be greater than `5`. It is not. Therefore the whole tree is invalid.

The problem is really asking:

> Can every node be placed inside the value range forced by all of its ancestors?

### 2. Start From the Baseline Idea

A direct way to validate the definition is to check every subtree from scratch.

For each node:

1. Find the maximum value in its left subtree.
2. Find the minimum value in its right subtree.
3. Check that:

```text
max(left subtree) < node.val < min(right subtree)
```

4. Recursively validate the left and right subtrees.

In pseudocode:

```python
def is_valid(root):
    if root is None:
        return True

    if root.left is not None and max_value(root.left) >= root.val:
        return False

    if root.right is not None and min_value(root.right) <= root.val:
        return False

    return is_valid(root.left) and is_valid(root.right)
```

This matches the definition, but it repeats work.

The same subtree can be scanned many times while computing `max_value` or `min_value` for different ancestors. On a skewed tree, this can degrade to `O(n^2)` time.

The baseline teaches the important idea, though:

> A node is valid only relative to a range, not only relative to its parent.

### 3. Why Checking Only Children Is Wrong

A tempting shortcut is:

```text
If left child exists, require left.val < node.val.
If right child exists, require right.val > node.val.
Then recurse.
```

That is not enough.

Consider the official invalid example:

```text
        5
       / \
      1   4
         / \
        3   6
```

At node `4`, both children look locally valid:

```text
3 < 4 < 6
```

At root `5`, the right child `4` is already invalid because `4` is not greater than `5`. But even if the direct right child were greater, a deeper descendant could still violate `5`.

The true rule is inherited:

```text
When we go left, we create a new upper bound.
When we go right, we create a new lower bound.
```

Every descendant must obey every bound created by its ancestors.

### 4. The Key Observation: Carry the Allowed Range

Instead of recomputing subtree minimums and maximums, carry the valid range as we descend.

At the root, there is no restriction yet:

```text
allowed range = (-infinity, +infinity)
```

If the root has value `5`, then:

- every node in the left subtree must be in `(-infinity, 5)`
- every node in the right subtree must be in `(5, +infinity)`

If we then move to the right child, say `7`, its left subtree must obey both facts:

```text
greater than 5, because it is in root's right subtree
less than 7, because it is in 7's left subtree
```

So the allowed range becomes:

```text
(5, 7)
```

This is the central invariant:

```text
When validating a node, low < node.val < high.
```

Here `low` and `high` are not arbitrary helper variables. They summarize all ordering promises made by the path from the root to the current node.

### 5. Recursive Invariant and Bounds

Define a helper:

```python
valid(node, low, high)
```

Meaning:

```text
The subtree rooted at node is a valid BST,
assuming every value in this subtree must be strictly between low and high.
```

The invariant at entry is:

```text
All ancestors of node have already determined the open interval (low, high).
```

For the current node to be valid:

```text
low < node.val < high
```

If that fails, the tree cannot be a valid BST.

If it succeeds, the children receive tighter bounds:

```text
Left child:  values must be in (low, node.val)
Right child: values must be in (node.val, high)
```

Why open intervals?

Because duplicates are invalid. A value equal to `low`, `high`, or the parent value must fail.

### 6. Detailed Algorithm

1. Start DFS from the root with no finite bounds.
2. If the current node is `None`, return `True` because an empty subtree cannot violate the BST rule.
3. Check whether the current value is strictly inside its allowed range.
4. If it is outside the range, return `False` immediately.
5. Recursively validate the left subtree with the current value as the new upper bound.
6. Recursively validate the right subtree with the current value as the new lower bound.
7. The current subtree is valid only if both recursive calls are valid.

Pseudocode:

```python
def isValidBST(root):
    def valid(node, low, high):
        if node is None:
            return True

        if not (low < node.val < high):
            return False

        return (
            valid(node.left, low, node.val)
            and valid(node.right, node.val, high)
        )

    return valid(root, -infinity, +infinity)
```

The important part is not the syntax. The important part is that each recursive call receives exactly the range that the BST definition requires for that subtree.

### 7. Walkthrough: Valid Tree

Example 1:

```text
root = [2, 1, 3]

      2
     / \
    1   3
```

Start at `2`:

```text
allowed range: (-infinity, +infinity)
check: -infinity < 2 < +infinity  yes
```

Move left to `1`:

```text
allowed range: (-infinity, 2)
check: -infinity < 1 < 2  yes
```

Both children of `1` are empty, so they return `True`.

Move right to `3`:

```text
allowed range: (2, +infinity)
check: 2 < 3 < +infinity  yes
```

Both children of `3` are empty, so they return `True`.

Every visited node satisfies the range created by its ancestors, so the answer is:

```text
True
```

### 8. Walkthrough: Invalid Tree

Example 2:

```text
root = [5, 1, 4, None, None, 3, 6]

        5
       / \
      1   4
         / \
        3   6
```

Start at `5`:

```text
allowed range: (-infinity, +infinity)
check: -infinity < 5 < +infinity  yes
```

Move left to `1`:

```text
allowed range: (-infinity, 5)
check: -infinity < 1 < 5  yes
```

The left subtree is fine.

Move right to `4`:

```text
allowed range: (5, +infinity)
check: 5 < 4 < +infinity  no
```

The node `4` is in the right subtree of `5`, so it must be greater than `5`. It is not. The algorithm can stop and return:

```text
False
```

Notice that the invalidity is not about `4`'s children. The node `4` itself violates the lower bound created by ancestor `5`.

### 9. Alternative View: In-Order Traversal

There is another valid first-principles approach.

In a valid BST, an in-order traversal visits values in strictly increasing order:

```text
left subtree, node, right subtree
```

So we can traverse in-order and remember the previous visited value. If the current value is ever less than or equal to the previous value, the tree is invalid.

Pseudocode:

```python
previous = None

def inorder(node):
    nonlocal previous
    if node is None:
        return True

    if not inorder(node.left):
        return False

    if previous is not None and node.val <= previous:
        return False
    previous = node.val

    return inorder(node.right)
```

This works because the in-order sequence of a BST must be sorted. The bounds method is often easier to reason about for this problem because it directly encodes the ancestor constraints that make child-only checks fail.

### 10. Correctness

We prove the bound-based DFS is correct.

The helper `valid(node, low, high)` is called with the invariant that every value in `node`'s subtree must be strictly between `low` and `high` because of the ancestors above `node`.

If `node` is `None`, the subtree is empty. An empty subtree contains no value that can violate the BST property, so returning `True` is correct.

If `node.val` is not strictly between `low` and `high`, then `node` violates a constraint imposed by one of its ancestors or by its parent. Therefore no tree containing this node at this position can be a valid BST, so returning `False` is correct.

If `node.val` is inside the range, then any value in the left subtree must also be less than `node.val`, while still satisfying the old lower bound. That is exactly the range `(low, node.val)`. Similarly, any value in the right subtree must be greater than `node.val`, while still satisfying the old upper bound. That is exactly the range `(node.val, high)`.

The algorithm recursively checks both subtrees with those exact required ranges. If both return `True`, then the current value is valid, every value in the left subtree is valid and less than the current value, and every value in the right subtree is valid and greater than the current value. Therefore the subtree rooted at `node` is a valid BST.

By induction over the tree structure, `valid(root, -infinity, +infinity)` returns `True` exactly when the whole input tree is a valid binary search tree.

### 11. Complexity

Let `n` be the number of nodes and `h` be the height of the tree.

- Time: `O(n)`, because each node is visited at most once.
- Space: `O(h)`, because the recursion stack stores one call per level of the tree.

For a balanced tree, `h = O(log n)`. For a completely skewed tree, `h = O(n)`.

### 12. Common Pitfalls

- Checking only `node.left.val < node.val < node.right.val`. This misses violations against grandparents and older ancestors.
- Using non-strict comparisons. This problem requires `low < node.val < high`, not `low <= node.val <= high`.
- Forgetting that duplicates are invalid even if they appear on a consistent side.
- Initializing bounds to fixed 32-bit sentinels when node values may equal or exceed those sentinels. Use `None`, language infinities, or wider numeric bounds.
- Updating only one side of the range incorrectly. The left child gets a new upper bound; the right child gets a new lower bound.
- Treating an empty subtree as invalid. `None` should return `True` because it has no violating node.
- Using global state for in-order traversal without resetting it between calls.

### 13. First-Principles Summary

A BST is a recursive ordering promise.

Each node does two things:

```text
It must fit inside the range created by its ancestors.
It creates tighter ranges for its children.
```

So the natural algorithm is to carry that range downward.

The root begins with no restrictions. Moving left tightens the upper bound. Moving right tightens the lower bound. A node is valid exactly when its value lies strictly inside the current open interval.

This solves the real problem directly: not whether each parent-child edge looks sorted, but whether every node respects the complete chain of ancestor constraints that defines a binary search tree.

## Implementation
See `solutions/binary_search_tree/p098_validate_binary_search_tree.py`.

## Tests
See `tests/binary_search_tree/test_p098_validate_binary_search_tree.py`.

## Examples

### Example 1
- Input: `{'root': [2, 1, 3]}`
- Output: `True`

### Example 2
- Input: `{'root': [5, 1, 4, None, None, 3, 6]}`
- Output: `False`

## Follow-up Practice
- Solve the same task recursively and iteratively.
- Trace a case where a violation is hidden below a grandparent.
- Compare bound-based DFS with in-order traversal.
