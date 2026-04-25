# 106. Construct Binary Tree from Inorder and Postorder Traversal

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given two lists containing the same distinct node values from one binary tree:

```text
inorder
postorder
```

The task is to reconstruct the original binary tree.

This is not asking whether the traversals are valid, and it is not asking for just one possible tree among many. Under the usual LeetCode constraint that all node values are unique and the traversals came from the same tree, the answer is uniquely determined.

The two traversals describe the same tree from two different viewpoints:

```text
inorder:    left subtree, root, right subtree
postorder:  left subtree, right subtree, root
```

So the entire problem reduces to this question:

> Given a region of the inorder list and the matching region of the postorder list, how do we identify that region's root and split the remaining values into its left and right subtrees?

If we can answer that question for one subtree, then we can answer it recursively for every smaller subtree.

### 2. Start From the Baseline Idea

A very direct recursive solution is:

1. The last value in the current postorder slice is the root.
2. Search for that root value inside the current inorder slice.
3. Everything left of that root in inorder belongs to the left subtree.
4. Everything right of that root in inorder belongs to the right subtree.
5. Use those subtree sizes to cut the postorder slice into left and right parts.
6. Recurse.

For example, with:

```text
inorder    = [9, 3, 15, 20, 7]
postorder  = [9, 15, 7, 20, 3]
```

The last postorder value is `3`, so `3` is the root.

In inorder, `3` splits the list:

```text
[9]  3  [15, 20, 7]
left root right
```

So the root's left subtree contains one node and the root's right subtree contains three nodes.

That is enough information to split postorder:

```text
[9]  [15, 7, 20]  3
left right        root
```

This baseline is correct, but there are two common inefficiencies:

- If each recursive call scans the inorder slice to find the root, the worst-case time can become `O(n^2)`.
- If each recursive call creates new list slices, the code copies many values and uses extra memory.

The optimized version keeps the same idea, but avoids both costs.

### 3. The Key Observation

The two traversals each answer a different structural question.

Postorder answers:

```text
What is the root of this subtree?
```

Because postorder visits:

```text
left, right, root
```

the final value of any subtree's postorder region is that subtree's root.

Inorder answers:

```text
Which values belong to the left subtree, and which values belong to the right subtree?
```

Because inorder visits:

```text
left, root, right
```

the root's position divides the inorder region into exactly two contiguous regions.

So the reconstruction rule is:

```text
postorder root value -> locate in inorder -> split into left and right subproblems
```

The only operation that would be expensive is repeatedly locating a value in inorder. We remove that cost by building a dictionary once:

```python
index = {value: position_in_inorder}
```

Then every root lookup is `O(1)`.

### 4. The Recursive Contract

A clean recursive function should mean one precise thing.

Use index ranges instead of copying slices:

```text
build(in_left, in_right, post_left, post_right)
```

Contract:

> Build and return the tree whose inorder traversal is `inorder[in_left:in_right + 1]` and whose postorder traversal is `postorder[post_left:post_right + 1]`.

This contract contains everything the function needs:

- The inorder range tells us which values belong to this subtree.
- The postorder range tells us the order in which those same values appear, with the root at the end.
- The two ranges always describe the same set of values.

The base case is when the range is empty:

```text
in_left > in_right
```

or equivalently:

```text
post_left > post_right
```

An empty traversal region represents an empty subtree, so return `None`.

### 5. Deriving the Subproblem Ranges

Suppose we are inside:

```text
build(in_left, in_right, post_left, post_right)
```

The root value is:

```text
root_val = postorder[post_right]
```

Find its inorder position:

```text
root_index = index[root_val]
```

Then the left subtree in inorder is:

```text
inorder[in_left : root_index]
```

and the right subtree in inorder is:

```text
inorder[root_index + 1 : in_right + 1]
```

The number of nodes in the left subtree is:

```text
left_size = root_index - in_left
```

This size is the bridge between inorder and postorder.

Postorder for the current subtree has this layout:

```text
left subtree values, right subtree values, root
```

So the left subtree's postorder range has `left_size` values starting at `post_left`:

```text
postorder[post_left : post_left + left_size]
```

The right subtree's postorder range comes after that and stops before the root:

```text
postorder[post_left + left_size : post_right]
```

In inclusive-index form:

```text
left:
  inorder   [in_left, root_index - 1]
  postorder [post_left, post_left + left_size - 1]

right:
  inorder   [root_index + 1, in_right]
  postorder [post_left + left_size, post_right - 1]
```

That is the whole algorithm.

### 6. Detailed Algorithm

1. If `inorder` is empty, return `None`.
2. Build a dictionary mapping each node value to its index in `inorder`.
3. Define `build(in_left, in_right, post_left, post_right)`.
4. If the inorder range is empty, return `None`.
5. Read `postorder[post_right]`; this is the current root value.
6. Create a `TreeNode` for that root value.
7. Use the dictionary to find the root's index in `inorder`.
8. Compute `left_size = root_index - in_left`.
9. Recursively build the left subtree from the left inorder and left postorder ranges.
10. Recursively build the right subtree from the right inorder and right postorder ranges.
11. Attach both children to the root.
12. Return the root.

The recursion naturally mirrors the definition of the traversals: each call consumes exactly one root and delegates the remaining values to the two child subtrees.

### 7. Example Walkthrough

Use the first official example:

```text
inorder   = [9, 3, 15, 20, 7]
postorder = [9, 15, 7, 20, 3]
```

Build the index map:

```text
9  -> 0
3  -> 1
15 -> 2
20 -> 3
7  -> 4
```

Start with the full ranges:

```text
build(0, 4, 0, 4)
```

The root is the last value in the postorder range:

```text
postorder[4] = 3
```

In inorder, `3` is at index `1`:

```text
[9] 3 [15, 20, 7]
```

So:

```text
left_size = 1 - 0 = 1
```

The left subtree ranges are:

```text
inorder   [0, 0] -> [9]
postorder [0, 0] -> [9]
```

The right subtree ranges are:

```text
inorder   [2, 4] -> [15, 20, 7]
postorder [1, 3] -> [15, 7, 20]
```

Now build the left subtree:

```text
build(0, 0, 0, 0)
```

The root is:

```text
postorder[0] = 9
```

`9` has no values to its left or right in the inorder range, so it becomes a leaf node.

Now build the right subtree:

```text
build(2, 4, 1, 3)
```

The root is:

```text
postorder[3] = 20
```

In the relevant inorder region:

```text
[15] 20 [7]
```

So `20` has left child `15` and right child `7`.

The resulting tree is:

```text
      3
     / \
    9   20
       /  \
      15   7
```

In level-order list form, that is:

```text
[3, 9, 20, None, None, 15, 7]
```

which matches the expected output.

### 8. Code

Python-style implementation:

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder:
            return None

        inorder_index = {value: index for index, value in enumerate(inorder)}

        def build(in_left: int, in_right: int, post_left: int, post_right: int) -> Optional[TreeNode]:
            if in_left > in_right:
                return None

            root_value = postorder[post_right]
            root = TreeNode(root_value)

            root_index = inorder_index[root_value]
            left_size = root_index - in_left

            root.left = build(
                in_left,
                root_index - 1,
                post_left,
                post_left + left_size - 1,
            )
            root.right = build(
                root_index + 1,
                in_right,
                post_left + left_size,
                post_right - 1,
            )

            return root

        return build(0, len(inorder) - 1, 0, len(postorder) - 1)
```

There is also a compact variant that walks `postorder` backward with a shared pointer and builds the right subtree before the left subtree. That version works because when reading postorder from the end, the order becomes:

```text
root, right, left
```

The range-based version above is usually easier to reason about because every recursive call explicitly states which traversal regions belong to that subtree.

### 9. Correctness

We prove that `build(in_left, in_right, post_left, post_right)` returns exactly the subtree described by those traversal ranges.

Base case:

If `in_left > in_right`, the inorder range is empty. Since the inorder and postorder ranges describe the same subtree, the subtree has no nodes. The algorithm returns `None`, which is the correct empty tree.

Recursive step:

Assume the range is non-empty. In postorder traversal, the final value of a subtree is its root, so `postorder[post_right]` is the correct root value. The algorithm creates a node with that value.

In inorder traversal, all values before the root position belong to the left subtree, and all values after the root position belong to the right subtree. Because node values are unique, the dictionary gives the root's exact inorder position. Therefore `left_size = root_index - in_left` is exactly the number of nodes in the left subtree.

Postorder visits all left-subtree nodes first, then all right-subtree nodes, then the root. Since the left subtree contains `left_size` nodes, the algorithm's computed postorder ranges give exactly the left and right subtree traversals.

By the recursive assumption, the left recursive call returns the correct left subtree and the right recursive call returns the correct right subtree. Attaching those subtrees to the created root produces exactly the tree represented by the current traversal ranges.

The initial call uses the full traversal ranges, so the algorithm returns the original tree.

### 10. Complexity

Let `n` be the number of nodes.

- Time: `O(n)`. Building the dictionary costs `O(n)`, and each node becomes the root of exactly one recursive call.
- Space: `O(n)` for the dictionary plus `O(h)` recursion stack space, where `h` is the tree height. In the worst case, `h = n` for a completely skewed tree; in a balanced tree, `h = log n`.

If the dictionary is not used and each call scans inorder to find the root, the worst-case time can degrade to `O(n^2)`.

### 11. Common Pitfalls

- Confusing postorder with preorder. In postorder, the root is at the end of the current range, not the beginning.
- Splitting postorder at the root's inorder index directly. The inorder index is a position in `inorder`, not in `postorder`; use `left_size` to translate between the arrays.
- Building the right subtree from the wrong postorder range. The current root at `post_right` must be excluded from both child ranges.
- Forgetting that the left subtree's postorder segment comes before the right subtree's segment.
- Creating list slices in every recursive call. Slices are simpler to write, but they copy data and obscure the exact range invariant.
- Assuming duplicate values work automatically. This reconstruction relies on each value identifying one unique inorder position.
- Using only `post_left > post_right` as the base case while accidentally passing inconsistent ranges. A well-defined range contract prevents those off-by-one errors.

### 12. First-Principles Summary

The problem becomes simple once each traversal is given one job:

```text
postorder tells us the root
inorder tells us the left/right split
```

Every recursive call represents one subtree. The last value of that call's postorder range creates the subtree root. The root's position in inorder tells how many nodes belong to the left side and how many belong to the right side. Those counts determine the next postorder ranges.

So the algorithm is not guessing a tree. It is repeatedly applying the same forced fact:

```text
root at end of postorder + root position in inorder = exact subtree boundaries
```

Once the boundaries are exact, recursion rebuilds the tree one root at a time.

## Implementation
See `solutions/binary_tree_dfs/p106_construct_binary_tree_from_inorder_and_postorder_traversal.py`.

## Tests
See `tests/binary_tree_dfs/test_p106_construct_binary_tree_from_inorder_and_postorder_traversal.py`.

## Examples

### Example 1
- Input: `{'inorder': [9, 3, 15, 20, 7], 'postorder': [9, 15, 7, 20, 3]}`
- Output: `[3, 9, 20, None, None, 15, 7]`

### Example 2
- Input: `{'inorder': [-1], 'postorder': [-1]}`
- Output: `[-1]`

## Follow-up Practice
- Write the recursive contract in one sentence before writing code.
- Trace the index ranges for a one-node tree, a root with only a left child, and a root with only a right child.
- Implement the range-based version first, then compare it with the backward-postorder-pointer version.
