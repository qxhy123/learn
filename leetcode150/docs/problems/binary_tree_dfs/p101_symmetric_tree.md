# 101. Symmetric Tree

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/symmetric-tree/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given the root of a binary tree, decide whether the tree is symmetric around its center.

A symmetric tree looks the same if reflected in a vertical mirror through the root.

For example:

```text
        1
      /   \
     2     2
    / \   / \
   3   4 4   3
```

This tree is symmetric because every node on the left side has a matching node on the right side in the mirrored position:

```text
left 2 matches right 2
left-left 3 matches right-right 3
left-right 4 matches right-left 4
```

But this tree is not symmetric:

```text
        1
      /   \
     2     2
      \     \
       3     3
```

The values `3` appear on both sides, but they are not in mirrored positions. The left subtree has its `3` as a right child, while the right subtree also has its `3` as a right child. A mirror would require the right subtree's matching `3` to be a left child.

So the real question is not just:

```text
Do the left and right subtrees contain the same values?
```

It is:

```text
Are the left and right subtrees mirror images of each other in both structure and value?
```

---

### 2. Start From a Baseline Idea

A first attempt might be to traverse the tree level by level and check whether every level reads the same forward and backward.

For the symmetric tree:

```text
        1
      /   \
     2     2
    / \   / \
   3   4 4   3
```

The levels are:

```text
[1]
[2, 2]
[3, 4, 4, 3]
```

Each level is a palindrome, so this tree is symmetric.

But there is a subtle problem: missing children matter.

Consider:

```text
        1
      /   \
     2     2
      \     \
       3     3
```

If we ignore missing children, level order values look like:

```text
[1]
[2, 2]
[3, 3]
```

Those levels look palindromic, but the tree is not symmetric. To make level-order checking correct, we would need to include `None` placeholders:

```text
[1]
[2, 2]
[None, 3, None, 3]
```

Now the last level is not symmetric.

This baseline can work, but it forces us to carefully preserve missing-child positions. It also hides the real structure of the problem.

The cleaner idea is to compare the two sides directly as mirrors.

---

### 3. The Key Observation

A binary tree is symmetric if and only if:

```text
the root's left subtree is a mirror of the root's right subtree
```

Now define what it means for two trees `a` and `b` to be mirrors.

They are mirrors if:

1. Both are empty.
2. Or both are non-empty, their root values are equal, and:
   - `a.left` is a mirror of `b.right`
   - `a.right` is a mirror of `b.left`

The crossed comparison is the whole problem.

For normal equality between two trees, we would compare:

```text
a.left  with b.left
a.right with b.right
```

For mirror equality, we compare:

```text
a.left  with b.right
a.right with b.left
```

That is the first-principles reason DFS fits naturally: after checking one pair of nodes, the remaining work is the same mirror question on smaller pairs of subtrees.

---

### 4. Recursive Contract

Define a helper function:

```text
is_mirror(left_node, right_node)
```

Its contract is:

```text
Return True if the subtree rooted at left_node is the mirror image of the subtree rooted at right_node.
```

This contract is about two nodes, not one node.

That distinction is important. A single subtree cannot tell whether it is in the correct mirrored position by looking only at itself. Symmetry is a relationship between the left side and the right side.

The helper must answer exactly this question for the current pair:

```text
Can these two nodes occupy opposite mirrored positions in a symmetric tree?
```

The invariant during recursion is:

```text
Every call compares two nodes that should be mirror counterparts.
```

If that invariant is true for the current pair, then the next pairs must be crossed:

```text
left_node.left   vs right_node.right
left_node.right  vs right_node.left
```

---

### 5. Base Cases

For a pair of nodes `(left_node, right_node)`, there are three structural possibilities.

#### Case 1: Both Are Missing

```text
left_node is None
right_node is None
```

Two empty subtrees are mirrors of each other.

Return:

```text
True
```

#### Case 2: Exactly One Is Missing

```text
left_node is None, right_node is not None
```

or:

```text
left_node is not None, right_node is None
```

The structure differs. One side has a node where the mirror side has nothing.

Return:

```text
False
```

#### Case 3: Both Exist

Now structure at this position is compatible, so compare values:

```text
left_node.val == right_node.val
```

If the values differ, the tree cannot be symmetric.

If the values match, recursively verify the crossed children.

---

### 6. Algorithm

1. If the root is missing, the tree is symmetric.
   - An empty tree has no asymmetric part.

2. Otherwise call:

```text
is_mirror(root.left, root.right)
```

3. Inside `is_mirror(left_node, right_node)`:

```text
if both nodes are None:
    return True

if exactly one node is None:
    return False

if left_node.val != right_node.val:
    return False

return (
    is_mirror(left_node.left, right_node.right)
    and
    is_mirror(left_node.right, right_node.left)
)
```

The `and` is useful because one failed mirror pair is enough to prove the whole tree is not symmetric.

---

### 7. Python-Style Implementation

The repository solution file is scaffolded, but the intended implementation follows this shape:

```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def is_mirror(left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
            if left is None and right is None:
                return True

            if left is None or right is None:
                return False

            if left.val != right.val:
                return False

            return (
                is_mirror(left.left, right.right)
                and is_mirror(left.right, right.left)
            )

        if root is None:
            return True

        return is_mirror(root.left, root.right)
```

Some implementations skip the explicit `root is None` check and call `is_mirror(root.left, root.right)` directly. That is fine on LeetCode because the input root can be `None` only if handled first. Keeping the check makes the empty-tree behavior explicit.

---

### 8. Detailed Walkthrough: Symmetric Example

Input:

```text
root = [1, 2, 2, 3, 4, 4, 3]
```

Tree:

```text
        1
      /   \
     2     2
    / \   / \
   3   4 4   3
```

Start from the root:

```text
is_mirror(root.left, root.right)
```

So we compare the two `2` nodes:

```text
is_mirror(left 2, right 2)
```

Both exist, and their values match:

```text
2 == 2
```

Now compare crossed children.

First crossed pair:

```text
is_mirror(left 2's left child, right 2's right child)
is_mirror(3, 3)
```

Both exist, and their values match:

```text
3 == 3
```

Their crossed children are all missing:

```text
is_mirror(None, None) -> True
is_mirror(None, None) -> True
```

So the pair `(3, 3)` is mirrored.

Second crossed pair:

```text
is_mirror(left 2's right child, right 2's left child)
is_mirror(4, 4)
```

Both exist, and their values match:

```text
4 == 4
```

Their crossed children are also all missing:

```text
is_mirror(None, None) -> True
is_mirror(None, None) -> True
```

So the pair `(4, 4)` is mirrored.

Both crossed pairs under the `2` nodes returned `True`, so:

```text
is_mirror(left 2, right 2) -> True
```

Therefore the whole tree is symmetric.

Output:

```text
True
```

---

### 9. Detailed Walkthrough: Non-Symmetric Example

Input:

```text
root = [1, 2, 2, None, 3, None, 3]
```

Tree:

```text
        1
      /   \
     2     2
      \     \
       3     3
```

Start:

```text
is_mirror(root.left, root.right)
is_mirror(left 2, right 2)
```

Both `2` nodes exist and their values match.

Now compare crossed children.

First crossed pair:

```text
is_mirror(left 2's left child, right 2's right child)
is_mirror(None, 3)
```

Exactly one side is missing.

That means the structure is not mirrored:

```text
None does not match 3
```

So this call returns:

```text
False
```

Once one required mirror pair fails, the whole tree fails.

Output:

```text
False
```

The important lesson from this example is that equal values at the same depth are not enough. The missing-child positions must also mirror each other.

---

### 10. Correctness Argument

We prove that the algorithm returns `True` exactly when the tree is symmetric.

#### Lemma: `is_mirror(a, b)` returns `True` exactly when the subtree rooted at `a` is a mirror of the subtree rooted at `b`.

Consider any pair of nodes `a` and `b`.

If both are `None`, both subtrees are empty. Empty subtrees are mirrors, and the function returns `True`.

If exactly one is `None`, one subtree has a node where the other has no node. Their structures cannot be mirrors, and the function returns `False`.

If both exist but their values differ, the two roots cannot occupy mirrored positions in a symmetric tree. The function returns `False`.

If both exist and their values match, then the only remaining requirement is that their children mirror in crossed order:

```text
a.left  mirrors b.right
a.right mirrors b.left
```

The function checks exactly those two smaller mirror relationships recursively and returns `True` only if both are true.

Therefore, by structural induction on the pair of subtrees, `is_mirror(a, b)` is correct.

#### Theorem: `isSymmetric(root)` returns `True` exactly when the whole tree is symmetric.

If `root` is `None`, the tree is empty and symmetric, so returning `True` is correct.

Otherwise, the whole tree is symmetric exactly when its left subtree is a mirror of its right subtree. The algorithm returns:

```text
is_mirror(root.left, root.right)
```

By the lemma, this is `True` exactly when those two subtrees are mirrors.

Therefore the algorithm is correct.

---

### 11. Complexity

Let `n` be the number of nodes in the tree, and let `h` be the height of the tree.

#### Time Complexity

Each real node is compared at most once as part of one mirror pair.

The helper also visits `None` child positions, but there are only `O(n)` such positions in a binary tree.

So the time complexity is:

```text
O(n)
```

#### Space Complexity

The recursive call stack follows a path down the tree.

The maximum depth of recursion is the height of the tree:

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

So the worst-case space complexity is:

```text
O(n)
```

---

### 12. Iterative Alternative

The same mirror contract can be implemented with an explicit stack or queue.

Instead of recursive calls, store pairs of nodes that should mirror each other:

```python
stack = [(root.left, root.right)]

while stack:
    left, right = stack.pop()

    if left is None and right is None:
        continue

    if left is None or right is None:
        return False

    if left.val != right.val:
        return False

    stack.append((left.left, right.right))
    stack.append((left.right, right.left))

return True
```

This is the same algorithm expressed without the language call stack.

The key is still the same pair invariant:

```text
Every stored pair contains two nodes that must be mirror counterparts.
```

---

### 13. Common Pitfalls

#### Comparing Same-Side Children

A frequent mistake is to write:

```text
is_mirror(left.left, right.left)
is_mirror(left.right, right.right)
```

That checks whether the two subtrees are identical, not whether they are mirrors.

For symmetry, the comparisons must cross:

```text
is_mirror(left.left, right.right)
is_mirror(left.right, right.left)
```

#### Ignoring Missing Children

The tree:

```text
        1
      /   \
     2     2
      \     \
       3     3
```

has matching level values if `None` positions are ignored, but it is not symmetric.

Always treat `None` as structural information.

#### Thinking Values Alone Are Enough

Two sides can contain the same values but still fail symmetry because the shape differs. Symmetry requires both:

```text
same value at mirrored positions
same structure at mirrored positions
```

#### Defining the Recursive Function Around One Node

A helper like:

```text
check(node)
```

is usually not enough, because symmetry is not a property of one isolated subtree. It is a relationship between two subtrees.

The natural helper is:

```text
check(left_node, right_node)
```

#### Forgetting the Empty Tree

An empty tree is symmetric. If the platform allows `root = None`, return `True` for that case.

---

### 14. First-Principles Summary

The problem is about reflection.

Reflection pairs positions across the center of the tree. Therefore, the unit of work is not one node but a pair of nodes that should be mirror counterparts.

For each pair:

```text
both missing       -> valid mirror pair
one missing        -> invalid mirror pair
values differ      -> invalid mirror pair
values match       -> check crossed children
```

The crossed children are the essential insight:

```text
left.left  must mirror right.right
left.right must mirror right.left
```

Once that contract is clear, the recursive DFS is just a direct translation of the definition of mirror symmetry.

## Implementation
See `solutions/binary_tree_dfs/p101_symmetric_tree.py`.

## Tests
See `tests/binary_tree_dfs/test_p101_symmetric_tree.py`.

## Examples

### Example 1
- Input: `{'root': [1, 2, 2, 3, 4, 4, 3]}`
- Output: `True`

### Example 2
- Input: `{'root': [1, 2, 2, None, 3, None, 3]}`
- Output: `False`

## Follow-up Practice
- Write the `is_mirror(left_node, right_node)` contract in one sentence before coding.
- Trace a tree with only one child under the left subtree and confirm why the matching right-side child must be on the opposite side.
- Implement the iterative stack version and verify that it stores pairs, not individual nodes.
