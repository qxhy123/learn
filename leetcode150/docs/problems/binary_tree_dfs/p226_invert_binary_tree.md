# 226. Invert Binary Tree

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/invert-binary-tree/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given the root of a binary tree, return the root of the same tree after it has been inverted.

Inverting a binary tree means mirroring it across its vertical center line.

For a single node, the mirror operation is simple:

```text
the old left child becomes the new right child
the old right child becomes the new left child
```

But the operation must happen at every node, not only at the root.

For example:

```text
        4
      /   \
     2     7
    / \   / \
   1   3 6   9
```

After inversion:

```text
        4
      /   \
     7     2
    / \   / \
   9   6 3   1
```

The root value `4` stays where it is. What changes is the direction of every edge below every node.

So the real task is:

> For every node in the tree, swap its left and right subtrees, and return the original root pointer after all swaps are complete.

The output is still a binary tree. It is not asking for a list traversal, a sorted order, or a new collection of values.

---

### 2. Start From the Baseline Idea

A direct way to think about the problem is:

1. Visit every node in the tree.
2. At each node, exchange its two child pointers.
3. Continue until no nodes remain unvisited.

Conceptually:

```python
for each node in the tree:
    node.left, node.right = node.right, node.left
```

This is already close to optimal, because every node must be considered at least once.

If a node is never visited, then its children would never be swapped, and the tree might not be fully inverted.

The only remaining design question is:

> What is the cleanest way to visit every node and swap each node exactly once?

For a binary tree, recursion is the natural fit because each subtree has the same shape as the whole problem.

---

### 3. The Key Observation

A binary tree is recursively defined:

```text
a tree is either empty
or a root node with a left subtree and a right subtree
```

The desired output has the same recursive structure.

To invert a tree rooted at `node`:

```text
invert the left subtree
invert the right subtree
swap the two subtrees under node
return node
```

Equivalently, you can swap first and then recursively invert the children. Both orders work, as long as every original subtree is eventually inverted.

The important observation is that the operation is local plus recursive:

```text
one local pointer swap at the current node
plus the same operation on each child subtree
equals inverted tree
```

There is no need to compare values, track depths, build a traversal array, or know anything about nodes outside the current subtree.

---

### 4. The Recursive Contract

Define the recursive function like this:

```text
invertTree(node) returns the root of the inverted version of the subtree rooted at node.
```

This one sentence is the contract.

It means:

- If `node` is `None`, the inverted empty tree is still `None`.
- If `node` is a real node, then after the call finishes, every node inside that subtree has had its left and right children swapped.
- The function returns the same `node` pointer, now serving as the root of the inverted subtree.

The contract is stronger than merely saying “visit nodes.” It says exactly what the caller can rely on after the call returns.

For a parent node, this is useful because the parent does not need to know how a child subtree was inverted. It only needs to trust:

```text
invertTree(child) gives me that child subtree after inversion
```

Then the parent can place the inverted child subtrees on opposite sides.

---

### 5. Base Case

The smallest possible tree is empty:

```text
root = None
```

There are no child pointers to swap.

The inverted result is also empty:

```python
if root is None:
    return None
```

This base case matters because every leaf eventually calls the function on its missing children.

For a leaf:

```text
    1
   / \
None None
```

Both recursive calls hit the empty-tree case. Swapping two `None` children changes nothing, which is correct.

---

### 6. Detailed Algorithm

For a node `root`:

1. If `root` is `None`, return `None`.
2. Recursively invert the left subtree.
3. Recursively invert the right subtree.
4. Assign the inverted right subtree to `root.left`.
5. Assign the inverted left subtree to `root.right`.
6. Return `root`.

In code-like form:

```python
def invertTree(root):
    if root is None:
        return None

    inverted_left = invertTree(root.left)
    inverted_right = invertTree(root.right)

    root.left = inverted_right
    root.right = inverted_left

    return root
```

This version stores the recursive results before assigning them back. That makes the data movement explicit:

```text
old left subtree  -> inverted_left  -> new right side
old right subtree -> inverted_right -> new left side
```

A shorter equivalent version is:

```python
def invertTree(root):
    if root is None:
        return None

    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root
```

Both are expressing the same recursive contract.

---

### 7. Example Walkthrough

Use the first example:

```text
root = [4, 2, 7, 1, 3, 6, 9]
```

The tree is:

```text
        4
      /   \
     2     7
    / \   / \
   1   3 6   9
```

Call:

```text
invertTree(4)
```

According to the contract, this call must return the root of the fully inverted tree rooted at `4`.

#### Invert the left subtree rooted at `2`

Before inversion:

```text
    2
   / \
  1   3
```

Call `invertTree(1)`:

```text
1 has no children, so it remains 1
```

Call `invertTree(3)`:

```text
3 has no children, so it remains 3
```

Now swap the two child subtrees under `2`:

```text
    2
   / \
  3   1
```

So `invertTree(2)` returns the root `2`, but its subtree has been mirrored.

#### Invert the right subtree rooted at `7`

Before inversion:

```text
    7
   / \
  6   9
```

Call `invertTree(6)`:

```text
6 remains 6
```

Call `invertTree(9)`:

```text
9 remains 9
```

Swap the two child subtrees under `7`:

```text
    7
   / \
  9   6
```

So `invertTree(7)` returns the root `7`, with its subtree mirrored.

#### Finish the root `4`

At this point, the two recursively inverted subtrees are:

```text
left result:
    2
   / \
  3   1

right result:
    7
   / \
  9   6
```

Now swap them under `4`:

```text
        4
      /   \
     7     2
    / \   / \
   9   6 3   1
```

Level-order form:

```text
[4, 7, 2, 9, 6, 3, 1]
```

That matches the expected output.

---

### 8. Why Swapping at Every Node Is Enough

Mirroring a tree does not change the identity of nodes or their values. It changes only whether each child edge points left or right.

Every path from the root can be described as a sequence of directions:

```text
left, right, left, ...
```

After inversion, each direction on that path is flipped:

```text
right, left, right, ...
```

Swapping children at the root flips the first direction. Swapping children inside each subtree flips the next direction. Continuing recursively flips every direction in every root-to-node path.

That is exactly what a mirror image requires.

---

### 9. Correctness Argument

We prove that `invertTree(node)` returns the root of the inverted version of the subtree rooted at `node`.

#### Base case

If `node` is `None`, the subtree is empty.

The algorithm returns `None`.

The inverted version of an empty tree is also empty, so the result is correct.

#### Recursive step

Assume the function works correctly for the left and right subtrees of `node`.

That means:

```text
invertTree(node.left) returns the inverted old left subtree
invertTree(node.right) returns the inverted old right subtree
```

To invert the subtree rooted at `node`, the old right subtree must become the new left subtree, and the old left subtree must become the new right subtree.

The algorithm assigns:

```text
node.left  = inverted old right subtree
node.right = inverted old left subtree
```

Therefore the subtree rooted at `node` is correctly inverted.

By structural induction over the tree, the algorithm is correct for the entire input tree.

---

### 10. Complexity

Let `n` be the number of nodes in the tree.

#### Time Complexity

```text
O(n)
```

Each node is visited once, and each visit does constant work: two recursive calls and one child-pointer swap.

No node needs to be revisited after its subtree has been inverted.

#### Space Complexity

```text
O(h)
```

where `h` is the height of the tree.

This space comes from the recursion call stack.

- For a balanced tree, `h = O(log n)`.
- For a completely skewed tree, `h = O(n)`.

The algorithm does not allocate a new tree, so aside from recursion stack space it uses constant extra memory.

---

### 11. Common Pitfalls

#### Forgetting the empty tree

Input may be:

```text
root = []
```

That corresponds to `root is None`. Return `None` immediately.

#### Swapping only the root

This is not enough:

```python
root.left, root.right = root.right, root.left
return root
```

It mirrors only one level. The children inside each subtree remain in their original orientation.

#### Losing one subtree during assignment

Be careful with sequential assignments like:

```python
root.left = invertTree(root.right)
root.right = invertTree(root.left)  # wrong: root.left no longer means the old left subtree
```

After the first assignment, `root.left` has changed. The second line no longer refers to the original left subtree.

Use temporary variables or tuple assignment:

```python
left = invertTree(root.left)
right = invertTree(root.right)
root.left = right
root.right = left
```

#### Returning the wrong value

The function should return the root of the inverted subtree.

For a non-empty subtree, that root is still `root`.

Do not return one of the children after swapping.

#### Confusing values with pointers

Inversion does not swap node values.

It swaps child references.

For example, you should not transform:

```text
node.val
```

The values stay attached to their original nodes; only the left/right structure changes.

---

### 12. Iterative Alternative

The same idea can be implemented with an explicit stack or queue.

The invariant becomes:

```text
every node removed from the stack has its children swapped exactly once
```

Pseudocode:

```python
def invertTree(root):
    if root is None:
        return None

    stack = [root]

    while stack:
        node = stack.pop()
        node.left, node.right = node.right, node.left

        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

    return root
```

Notice that after swapping, `node.left` and `node.right` refer to the swapped children. That is fine: both child subtrees still need to be processed, and their relative order does not matter.

The recursive version is usually shorter and matches the tree definition more directly, but the iterative version avoids recursion depth concerns.

---

### 13. First-Principles Summary

The tree is made of repeated smaller trees.

To mirror the whole tree, mirror each smaller tree and exchange the two sides at every root.

The essential contract is:

```text
invertTree(node) returns the same subtree root after all left/right directions inside that subtree have been flipped.
```

Once that contract is clear, the implementation follows naturally:

```text
empty tree -> empty tree
non-empty tree -> invert children, swap children, return root
```

The solution is optimal because every node must be touched at least once, and the algorithm touches each node exactly once.

## Implementation
See `solutions/binary_tree_dfs/p226_invert_binary_tree.py`.

## Tests
See `tests/binary_tree_dfs/test_p226_invert_binary_tree.py`.

## Examples

### Example 1
- Input: `{'root': [4, 2, 7, 1, 3, 6, 9]}`
- Output: `[4, 7, 2, 9, 6, 3, 1]`

### Example 2
- Input: `{'root': [2, 1, 3]}`
- Output: `[2, 3, 1]`

### Example 3
- Input: `{'root': []}`
- Output: `[]`

## Follow-up Practice
- Write the recursive contract in one sentence before writing code.
- Trace the empty tree, a one-node tree, and a root with two leaf children.
- Implement both the recursive and iterative versions, then compare their stack-space behavior.
