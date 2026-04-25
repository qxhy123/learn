# 104. Maximum Depth of Binary Tree

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/maximum-depth-of-binary-tree/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given the root of a binary tree, return its maximum depth.

A binary tree is made of nodes. Each node may have:

```text
left child
right child
```

A path from the root down to a leaf follows child pointers:

```text
root -> child -> grandchild -> ... -> leaf
```

The **maximum depth** is the number of nodes on the longest such root-to-leaf path.

For example, in the tree represented by:

```text
[3, 9, 20, None, None, 15, 7]
```

The shape is:

```text
      3
     / \
    9   20
       /  \
      15   7
```

The longest root-to-leaf paths are:

```text
3 -> 20 -> 15
3 -> 20 -> 7
```

Each path contains `3` nodes, so the answer is:

```text
3
```

The problem is not asking for the number of nodes in the tree. It is not asking for the number of edges either. It asks:

> Starting at the root, how many node levels can we go down along the deepest branch?

---

### 2. Start From the Brute Force Idea

A direct way to think about the problem is:

1. List every path from the root to a leaf.
2. Count how many nodes are in each path.
3. Return the largest count.

For the example tree:

```text
      3
     / \
    9   20
       /  \
      15   7
```

The root-to-leaf paths are:

```text
3 -> 9        length 2
3 -> 20 -> 15 length 3
3 -> 20 -> 7  length 3
```

The maximum length is `3`.

This idea is correct, but explicitly building every path is unnecessary. A tree with many branches can have many root-to-leaf paths, and storing complete paths repeats the same prefix nodes again and again.

The deeper question is:

> Do we need the actual deepest path, or only its length?

We only need the length. That means each subtree can summarize itself with one integer: its own maximum depth.

---

### 3. The Key Observation

Look at one node in isolation:

```text
        node
       /    \
 left subtree right subtree
```

If we already know:

```text
left_depth  = maximum depth of the left subtree
right_depth = maximum depth of the right subtree
```

then the maximum depth starting at `node` is:

```text
1 + max(left_depth, right_depth)
```

The `1` counts the current node.

The `max(...)` chooses the deeper side.

This is the whole problem.

A binary tree is recursive: every child is itself the root of a smaller binary tree. So the answer for a tree can be defined using the answers for its two subtrees.

That gives the recurrence:

```text
maxDepth(None) = 0
maxDepth(node) = 1 + max(maxDepth(node.left), maxDepth(node.right))
```

The empty tree has depth `0` because it contains no nodes.

A leaf has depth `1` because both children are empty:

```text
maxDepth(leaf) = 1 + max(0, 0) = 1
```

---

### 4. Recursive Contract / Invariant

The most important part of this problem is defining exactly what the recursive function means.

Use this contract:

> `maxDepth(node)` returns the number of nodes on the longest downward path starting at `node` and ending at a leaf inside `node`'s subtree.

This contract is local and self-contained.

It does not say:

```text
Return the depth from the original root.
```

It says:

```text
If this node were treated as the root of its own subtree, how deep is that subtree?
```

That distinction matters. Once the function has this meaning, every recursive call asks the exact same question on a smaller tree:

```text
maxDepth(node.left)
maxDepth(node.right)
```

The invariant is:

```text
Whenever a recursive call returns, its returned integer is the correct maximum depth of that subtree.
```

The parent can safely rely on those two returned integers and combine them with:

```text
1 + max(left_depth, right_depth)
```

---

### 5. Detailed Algorithm

Use depth-first search in postorder form: solve children first, then combine their answers at the parent.

For a given `root`:

1. If `root` is `None`, return `0`.
   - There is no node here.
   - An empty subtree contributes no depth.
2. Recursively compute the maximum depth of the left subtree.
3. Recursively compute the maximum depth of the right subtree.
4. Choose the larger of those two depths.
5. Add `1` for the current node.
6. Return that value.

In Python-like pseudocode:

```python
def maxDepth(root):
    if root is None:
        return 0

    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)

    return 1 + max(left_depth, right_depth)
```

The algorithm does not need a global variable. The depth naturally flows upward as return values.

---

### 6. Detailed Example Walkthrough

Use the first example:

```text
root = [3, 9, 20, None, None, 15, 7]
```

Tree shape:

```text
      3
     / \
    9   20
       /  \
      15   7
```

Call:

```text
maxDepth(3)
```

To answer that, we need:

```text
maxDepth(9)
maxDepth(20)
```

First consider node `9`:

```text
  9
 / \
None None
```

Both children are empty:

```text
maxDepth(None) = 0
maxDepth(None) = 0
```

So:

```text
maxDepth(9) = 1 + max(0, 0) = 1
```

Now consider node `20`:

```text
   20
  /  \
 15   7
```

For node `15`:

```text
maxDepth(15) = 1 + max(0, 0) = 1
```

For node `7`:

```text
maxDepth(7) = 1 + max(0, 0) = 1
```

So:

```text
maxDepth(20) = 1 + max(1, 1) = 2
```

Now return to the original root `3`:

```text
left_depth  = maxDepth(9)  = 1
right_depth = maxDepth(20) = 2
```

Therefore:

```text
maxDepth(3) = 1 + max(1, 2) = 3
```

The answer is:

```text
3
```

Notice how the answer is assembled bottom-up. Leaf nodes return `1`; their parents return one more than their deepest child; the original root returns the depth of the whole tree.

---

### 7. Code

The implementation follows the recurrence directly:

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0

        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)

        return 1 + max(left_depth, right_depth)
```

This is a bottom-up recursive DFS.

The call stack temporarily holds the path from the original root to the node currently being processed. Once a subtree has returned its depth, the algorithm does not need to remember all nodes inside that subtree anymore.

---

### 8. Why This Is DFS

Depth-first search means we follow a branch downward before finishing other branches.

For this problem, recursion naturally performs DFS:

```text
start at node
go into left subtree
return its depth
go into right subtree
return its depth
combine both results
```

The exact order of visiting left before right is not important for correctness. We only need both subtree depths before returning the current node's depth.

This is why the solution is often described as postorder DFS:

```text
left -> right -> node
```

The parent result is computed after child results are known.

---

### 9. Correctness

We prove that the algorithm returns the maximum depth of the input tree.

#### Base Case

If `root` is `None`, the tree is empty.

An empty tree has no nodes and therefore has maximum depth `0`.

The algorithm returns `0`, so it is correct for the empty tree.

#### Recursive Step

Assume the algorithm correctly returns the maximum depth for smaller subtrees.

Now consider a non-empty node `root`.

Any downward path starting at `root` must do exactly one of these things:

```text
root -> path in the left subtree
root -> path in the right subtree
root alone, if both children are empty
```

By the recursive assumption:

```text
left_depth
```

is the maximum depth of the left subtree, and:

```text
right_depth
```

is the maximum depth of the right subtree.

The deepest path starting at `root` must choose the deeper of those two subtrees. It also includes `root` itself, so its length is:

```text
1 + max(left_depth, right_depth)
```

That is exactly what the algorithm returns.

Therefore the algorithm is correct for `root` if it is correct for its children.

#### Conclusion

By structural induction over the tree, the algorithm returns the correct maximum depth for every binary tree.

---

### 10. Complexity

Let `n` be the number of nodes in the tree, and let `h` be the height of the tree.

#### Time Complexity

```text
O(n)
```

Each node is visited once.

At each node, the algorithm does only constant extra work:

```text
compare two integers
add 1
return the result
```

So the total time is linear in the number of nodes.

#### Space Complexity

```text
O(h)
```

The recursive call stack stores one frame for each node on the current root-to-current-node path.

In a balanced tree:

```text
h = O(log n)
```

In a completely skewed tree, such as:

```text
1
 \
  2
   \
    3
     \
      4
```

we have:

```text
h = O(n)
```

So the worst-case recursion stack space is `O(n)`.

---

### 11. Common Pitfalls

#### Confusing Nodes With Edges

This problem counts nodes, not edges.

A single-node tree has maximum depth:

```text
1
```

not `0`.

That is why the recurrence adds `1` for the current node.

#### Returning `1` for `None`

The empty subtree should return `0`, because it contributes no nodes.

If `None` returned `1`, then a leaf would incorrectly have depth:

```text
1 + max(1, 1) = 2
```

#### Adding Both Subtree Depths

Do not write:

```python
1 + left_depth + right_depth
```

That would count nodes from both sides, which describes a different idea. Maximum depth follows one downward path, so it can choose only one child at each node.

The correct combination is:

```python
1 + max(left_depth, right_depth)
```

#### Overcomplicating With Stored Paths

You do not need to store every root-to-leaf path.

The problem asks only for the length of the deepest path. One integer per subtree is enough.

#### Using Global State Without Needing It

A top-down DFS with a global maximum can work, but it is not necessary here.

The bottom-up version is simpler because the recursive return value already represents the depth of the current subtree.

---

### 12. First-Principles Summary

A binary tree's maximum depth is determined by the maximum depths of its two child subtrees.

For any node:

```text
depth at this node = 1 + deeper child depth
```

For an empty child:

```text
depth = 0
```

So the entire problem reduces to a small recursive contract:

> Return the maximum number of nodes on a downward path starting from this node.

Once that contract is clear, the code is just the recurrence:

```text
maxDepth(None) = 0
maxDepth(node) = 1 + max(maxDepth(node.left), maxDepth(node.right))
```

This is why the solution is short: the tree's structure already contains the recursion needed to solve the problem.

## Implementation
See `solutions/binary_tree_dfs/p104_maximum_depth_of_binary_tree.py`.

## Tests
See `tests/binary_tree_dfs/test_p104_maximum_depth_of_binary_tree.py`.

## Examples

### Example 1
- Input: `{'root': [3, 9, 20, None, None, 15, 7]}`
- Output: `3`

### Example 2
- Input: `{'root': [1, None, 2]}`
- Output: `2`

## Follow-up Practice

- Explain why an empty subtree has depth `0`.
- Trace the recursive calls for a single-node tree.
- Trace a completely skewed tree and compare its call stack depth with a balanced tree.
- Rewrite the solution as an iterative DFS that stores `(node, depth)` pairs on a stack.
