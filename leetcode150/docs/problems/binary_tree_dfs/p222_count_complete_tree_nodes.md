# 222. Count Complete Tree Nodes

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/count-complete-tree-nodes/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given the root of a **complete binary tree**, return the number of nodes in the tree.

A complete binary tree is not just any binary tree. It has a very specific shape:

```text
Every level except possibly the last is completely full.
The last level is filled from left to right with no gaps.
```

For example:

```text
        1
      /   \
     2     3
    / \   /
   4   5 6
```

This tree has `6` nodes.

It is complete because:

```text
level 0: 1                    full
level 1: 2, 3                 full
level 2: 4, 5, 6              filled from left to right
```

The problem could be solved by simply visiting every node and counting it. But the input guarantee gives us more structure than an ordinary tree. The real question is:

> How can we count the nodes while exploiting the fact that the tree is complete?

The complete-tree condition lets us recognize large perfect subtrees and count them instantly instead of walking through every node one by one.

---

### 2. Start From the Baseline: Count Every Node

The most direct solution is normal tree traversal.

For each node:

1. Count the current node.
2. Count all nodes in the left subtree.
3. Count all nodes in the right subtree.
4. Add the three values.

Conceptually:

```python
def count(node):
    if node is None:
        return 0

    return 1 + count(node.left) + count(node.right)
```

This is correct for every binary tree, not only complete trees.

For the example:

```text
        1
      /   \
     2     3
    / \   /
   4   5 6
```

The traversal eventually visits:

```text
1, 2, 4, 5, 3, 6
```

and returns `6`.

The cost is:

```text
Time:  O(n)
Space: O(h)
```

where `n` is the number of nodes and `h` is the height of the tree.

That is already fine for many tree problems. But this problem specifically says the tree is complete, so we should ask:

> Can a complete tree reveal the size of some subtrees without visiting every node inside them?

Yes.

---

### 3. The Key Observation: Perfect Subtrees Are Cheap to Count

A **perfect binary tree** is a tree where every level is completely full.

For example:

```text
        1
      /   \
     2     3
    / \   / \
   4   5 6   7
```

Its height in nodes is `3`:

```text
1 -> 2 -> 4
```

and its node count is:

```text
1 + 2 + 4 = 7
```

In general, a perfect binary tree with height `h` has:

```text
2^h - 1
```

nodes.

So if we can prove a subtree is perfect, we do not need to recursively count its children. We can compute its size immediately.

The important question becomes:

> How do we quickly detect that a subtree inside a complete tree is perfect?

For a complete tree, compare two boundary paths:

```text
leftmost height  = keep going left until None
rightmost height = keep going right until None
```

If these two heights are equal, the subtree is perfect.

Why?

In a complete tree, missing nodes can only appear on the last level, and they can only be missing from the right side. If the far-left path and far-right path have the same height, then the far-right node at the deepest level exists. Because the last level is filled from left to right, every position before that far-right node must also exist. Therefore every level is full, and the subtree is perfect.

---

### 4. The Complete-Tree Invariant

The algorithm relies on this invariant:

```text
Every recursive call receives the root of a complete binary tree.
```

This matters because the height comparison is only enough under the complete-tree guarantee.

In an arbitrary binary tree, equal leftmost and rightmost heights do not necessarily imply the tree is perfect. For example:

```text
        1
      /   \
     2     3
      \   /
       4 5
```

The leftmost and rightmost boundary paths both have height `2`:

```text
1 -> 2
1 -> 3
```

but the tree is not perfect.

The LeetCode input guarantee rules out shapes like that. Inside a complete tree:

```text
root.left is complete when it exists
root.right is complete when it exists
missing nodes, if any, are pushed as far right as possible on the last level
```

So for each subtree encountered by recursion, the same reasoning remains valid.

The invariant lets us make this decision for a subtree rooted at `node`:

```text
If leftmost_height(node) == rightmost_height(node):
    the subtree is perfect, so count it as 2^height - 1
else:
    it is not perfect, so count the root and recurse into both children
```

---

### 5. Detailed Algorithm

For each subtree root `node`:

1. If `node` is `None`, return `0`.
2. Compute the height of the left boundary:
   - start at `node`
   - repeatedly move to `.left`
   - count how many nodes are seen
3. Compute the height of the right boundary:
   - start at `node`
   - repeatedly move to `.right`
   - count how many nodes are seen
4. If the two heights are equal:
   - this subtree is perfect
   - return `(1 << height) - 1`
5. Otherwise:
   - the subtree is complete but not perfect
   - return `1 + countNodes(node.left) + countNodes(node.right)`

The expression:

```python
(1 << height) - 1
```

is the same as:

```python
2 ** height - 1
```

It uses a bit shift because powers of two are natural for perfect binary trees.

---

### 6. Pseudocode

```python
def countNodes(root):
    if root is None:
        return 0

    left_height = height_by_following_left(root)
    right_height = height_by_following_right(root)

    if left_height == right_height:
        return 2 ** left_height - 1

    return 1 + countNodes(root.left) + countNodes(root.right)


def height_by_following_left(node):
    height = 0
    while node is not None:
        height += 1
        node = node.left
    return height


def height_by_following_right(node):
    height = 0
    while node is not None:
        height += 1
        node = node.right
    return height
```

A direct Python implementation would look like:

```python
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        def left_height(node: Optional[TreeNode]) -> int:
            height = 0
            while node:
                height += 1
                node = node.left
            return height

        def right_height(node: Optional[TreeNode]) -> int:
            height = 0
            while node:
                height += 1
                node = node.right
            return height

        if root is None:
            return 0

        left = left_height(root)
        right = right_height(root)

        if left == right:
            return (1 << left) - 1

        return 1 + self.countNodes(root.left) + self.countNodes(root.right)
```

---

### 7. Detailed Example Walkthrough

Consider the official example:

```text
root = [1, 2, 3, 4, 5, 6]
```

The tree is:

```text
        1
      /   \
     2     3
    / \   /
   4   5 6
```

Start at node `1`.

The leftmost path is:

```text
1 -> 2 -> 4
```

So:

```text
left_height = 3
```

The rightmost path is:

```text
1 -> 3
```

So:

```text
right_height = 2
```

The heights are not equal, so the whole tree is complete but not perfect. We cannot count it with `2^h - 1` yet.

Now recurse:

```text
count(1) = 1 + count(2) + count(3)
```

For subtree rooted at `2`:

```text
     2
    / \
   4   5
```

Leftmost path:

```text
2 -> 4
```

Rightmost path:

```text
2 -> 5
```

Both heights are `2`, so this subtree is perfect.

Count it immediately:

```text
2^2 - 1 = 3
```

So:

```text
count(2) = 3
```

For subtree rooted at `3`:

```text
   3
  /
 6
```

Leftmost path:

```text
3 -> 6
```

Rightmost path:

```text
3
```

The heights are different, so recurse:

```text
count(3) = 1 + count(6) + count(None)
```

For subtree rooted at `6`:

```text
6
```

Leftmost height and rightmost height are both `1`, so it is perfect:

```text
count(6) = 2^1 - 1 = 1
```

For the missing right child:

```text
count(None) = 0
```

Therefore:

```text
count(3) = 1 + 1 + 0 = 2
```

Finally:

```text
count(1) = 1 + count(2) + count(3)
         = 1 + 3 + 2
         = 6
```

So the answer is:

```text
6
```

Notice what was saved: the algorithm did not need to visit both children of node `2` individually after proving that subtree was perfect.

---

### 8. Why the Algorithm Is Correct

We prove that `countNodes(node)` returns the exact number of nodes in the complete subtree rooted at `node`.

#### Base Case

If `node` is `None`, the subtree is empty.

The algorithm returns `0`, which is exactly the number of nodes in an empty tree.

#### Perfect-Subtree Case

If `left_height(node) == right_height(node)`, then the leftmost and rightmost boundary paths reach the same depth.

Because the subtree is complete, the last level is filled from left to right. The existence of the rightmost deepest node means every node position before it on that level also exists. All earlier levels are already full by the definition of completeness.

Therefore the subtree is perfect.

A perfect binary tree of height `h` has exactly:

```text
2^h - 1
```

nodes, so the algorithm returns the correct count for this case.

#### Non-Perfect Complete-Subtree Case

If the two heights are different, the subtree is not perfect. But it is still a binary tree rooted at `node`, so its node count is exactly:

```text
1 + number of nodes in the left subtree + number of nodes in the right subtree
```

The recursive calls count the left and right complete subtrees correctly by the same argument. Adding `1` for the current root gives the exact count for the whole subtree.

#### Conclusion

Every call either:

```text
returns the exact formula for a proven perfect subtree
```

or decomposes the tree into its root, left subtree, and right subtree.

Thus the returned value is the exact number of nodes in the original complete tree.

---

### 9. Complexity

Let `h` be the height of the tree.

Each height computation walks down one boundary path, so it costs:

```text
O(h)
```

At a non-perfect complete subtree, the algorithm recurses into children. In the common analysis for this approach, each level needs boundary-height work, and the recursion quickly skips perfect subtrees.

For a complete tree with `n` nodes:

```text
h = O(log n)
```

The optimized complete-tree counting approach runs in:

```text
Time:  O(log^2 n)
Space: O(log n)
```

The space is the recursion stack height.

The baseline traversal would be:

```text
Time:  O(n)
Space: O(log n) for a complete tree
```

So the improvement comes from replacing many full subtree traversals with the perfect-tree formula.

---

### 10. Common Pitfalls

#### Pitfall 1: Forgetting the Complete-Tree Guarantee

The height comparison is safe because the input tree is complete.

Do not apply this exact shortcut blindly to arbitrary binary trees.

#### Pitfall 2: Mixing Edge Height and Node Height

This tutorial counts height as the number of nodes on a boundary path.

So a single-node tree has:

```text
height = 1
count  = 2^1 - 1 = 1
```

If you instead count edges, the formula changes. Pick one definition and use it consistently.

#### Pitfall 3: Returning `2 ** height` Instead of `2 ** height - 1`

A perfect tree of height `h` has levels:

```text
1, 2, 4, ..., 2^(h - 1)
```

The sum is:

```text
2^h - 1
```

The `-1` is required.

#### Pitfall 4: Only Checking the Root Once

The whole tree may not be perfect, but one of its subtrees may be.

The shortcut should be applied at every recursive subtree root, not just at the original root.

#### Pitfall 5: Assuming the Last Level Can Have Gaps

In a complete tree, the last level is filled from left to right.

This is why a rightmost deepest node proves all earlier last-level nodes exist. Without that invariant, the proof fails.

---

### 11. First-Principles Summary

The baseline idea is simple:

```text
To count a tree, count the root plus both subtrees.
```

The complete-tree insight is stronger:

```text
Some complete subtrees are perfect, and perfect subtrees have a closed-form node count.
```

So the algorithm repeatedly asks one local structural question:

```text
Do the leftmost and rightmost boundary paths have the same height?
```

If yes:

```text
count the entire subtree as 2^height - 1
```

If no:

```text
split into left and right subtrees and continue
```

The solution is not about a generic DFS template. It is about using the shape guarantee of a complete binary tree to replace unnecessary traversal with math.

## Implementation
See `solutions/binary_tree_dfs/p222_count_complete_tree_nodes.py`.

## Tests
See `tests/binary_tree_dfs/test_p222_count_complete_tree_nodes.py`.

## Examples

### Example 1
- Input: `{'root': [1, 2, 3, 4, 5, 6]}`
- Output: `6`

### Example 2
- Input: `{'root': []}`
- Output: `0`

### Example 3
- Input: `{'root': [1]}`
- Output: `1`

## Follow-up Practice
- Prove why equal boundary heights imply a perfect subtree only under the complete-tree invariant.
- Rewrite the solution using a helper function instead of calling `self.countNodes` recursively.
- Compare the optimized algorithm against the baseline traversal on a tree with the last level almost full.
