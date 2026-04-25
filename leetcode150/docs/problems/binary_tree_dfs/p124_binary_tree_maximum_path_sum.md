# 124. Binary Tree Maximum Path Sum

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/binary-tree-maximum-path-sum/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs, tree-traversal, sum

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given the root of a binary tree where every node stores an integer value.

A **path** is any sequence of nodes connected by parent-child edges, with no node repeated.

The path does **not** have to start at the root.
The path does **not** have to end at a leaf.
The path may contain only one node.

For example, in this tree:

```text
    1
   / \
  2   3
```

The best path is:

```text
2 -> 1 -> 3
```

Its sum is:

```text
2 + 1 + 3 = 6
```

So the answer is `6`.

The real problem is:

> Among all connected, non-repeating paths in the tree, find the largest possible sum of node values.

The hard part is not adding numbers. The hard part is understanding what shapes a valid path can have.

---

### 2. What Shapes Can a Path Have?

A path in a tree is simple because there is only one route between any two nodes.

If a path passes through some node `x`, then relative to `x` it can use:

```text
only x
x plus one downward branch
x plus left branch plus right branch
```

For example:

```text
      x
     / \
 left   right
```

A path whose highest turning point is `x` can look like:

```text
left branch -> x -> right branch
```

That is a complete path. It cannot continue upward to `x`'s parent after using both children, because then `x` would have three path directions:

```text
left child
right child
parent
```

That would no longer be a single line path. It would branch.

This distinction is the central idea of the problem:

> A path may use both children at the node where it turns, but a path returned to a parent may use only one child branch.

---

### 3. Start From the Brute Force Idea

A direct brute force approach is:

1. Consider every node as a possible start.
2. Explore every connected path from that start.
3. Compute each path sum.
4. Keep the maximum.

Conceptually:

```python
best = -infinity

for start in all_nodes:
    for path starting at start without repeating nodes:
        best = max(best, sum(path))
```

This is correct, but it repeats the same subtree work many times.

For example, if many candidate paths pass through the same left subtree, brute force recomputes the best downward contribution from that subtree again and again.

So we ask a better first-principles question:

> What information does a parent actually need from each child?

A parent does not need every possible path inside the child subtree. To build a path that continues upward through the parent, it only needs the best one-sided path that starts at the child and goes downward.

---

### 4. The Key Observation

At each node, there are two different quantities:

```text
1. The best complete path whose highest turning point may be this node.
2. The best extendable path that this node can offer to its parent.
```

They are not the same.

Suppose we are at node `x`:

```text
      x
     / \
    L   R
```

A complete path through `x` may use both sides:

```text
best downward path from L + x + best downward path from R
```

But the value returned to `x`'s parent must be a path that can still be extended upward. It can use only one side:

```text
x + max(best downward path from L, best downward path from R)
```

Why only one side?

Because if `x` returns a path that already uses both children, then the parent would attach above it and create a forked shape:

```text
left branch -> x -> right branch
                 |
               parent
```

That is not a valid path.

So the algorithm needs both behaviors:

```text
update the global answer with a complete path
return a one-sided extendable path to the parent
```

---

### 5. Recursive Invariant and Return Contract

Define a DFS function:

```text
dfs(node)
```

Return contract:

```text
dfs(node) returns the maximum sum of a non-empty path that:
- starts at node,
- goes downward only,
- chooses at most one child direction at every step,
- and can be extended by node's parent.
```

This returned value is sometimes called the node's **gain**.

For a missing child:

```text
dfs(None) = 0
```

This does not mean an empty path is the final answer. It means:

```text
A missing child contributes nothing to a parent.
```

For a real node, first compute child gains:

```text
left_gain  = dfs(node.left)
right_gain = dfs(node.right)
```

If a child gain is negative, using that child would make the path worse. Since paths are allowed to stop at the current node, negative child contributions should be discarded:

```text
left_gain  = max(left_gain, 0)
right_gain = max(right_gain, 0)
```

Now the best complete path that turns at `node` is:

```text
left_gain + node.val + right_gain
```

Use this to update the global answer.

The value returned upward is:

```text
node.val + max(left_gain, right_gain)
```

That is the best one-sided path starting at `node`.

---

### 6. Detailed Algorithm

1. Initialize `best` to negative infinity.
   - This matters because all node values may be negative.
2. Run DFS from the root.
3. For each node:
   - Recursively compute the best extendable gain from the left child.
   - Recursively compute the best extendable gain from the right child.
   - Replace negative gains with `0`, because a harmful branch should not be included.
   - Treat `left_gain + node.val + right_gain` as a complete path passing through this node.
   - Update `best` with that complete path.
   - Return `node.val + max(left_gain, right_gain)` to the parent.
4. After DFS finishes, return `best`.

The traversal is postorder in spirit:

```text
solve left subtree
solve right subtree
combine at current node
```

We need children first because the current node's best path depends on the best downward contribution from each child.

---

### 7. Pseudocode

```python
def maxPathSum(root):
    best = -infinity

    def dfs(node):
        nonlocal best

        if node is None:
            return 0

        left_gain = max(dfs(node.left), 0)
        right_gain = max(dfs(node.right), 0)

        path_through_node = left_gain + node.val + right_gain
        best = max(best, path_through_node)

        return node.val + max(left_gain, right_gain)

    dfs(root)
    return best
```

The important line is not just the formula. It is the meaning behind the formula:

```text
path_through_node may be final, so it can use both sides.
return value may be extended upward, so it can use only one side.
```

---

### 8. Walkthrough: `[1, 2, 3]`

Tree:

```text
    1
   / \
  2   3
```

Start with:

```text
best = -infinity
```

Visit node `2`:

```text
left_gain = 0
right_gain = 0
path_through_2 = 0 + 2 + 0 = 2
best = 2
return 2
```

Visit node `3`:

```text
left_gain = 0
right_gain = 0
path_through_3 = 0 + 3 + 0 = 3
best = 3
return 3
```

Visit node `1`:

```text
left_gain = 2
right_gain = 3
path_through_1 = 2 + 1 + 3 = 6
best = 6
return 1 + max(2, 3) = 4
```

The returned value `4` represents the extendable path:

```text
1 -> 3
```

But the global answer is `6`, representing the complete path:

```text
2 -> 1 -> 3
```

This shows why the algorithm needs a global answer separate from the returned value.

---

### 9. Walkthrough: `[-10, 9, 20, None, None, 15, 7]`

Tree:

```text
      -10
      /  \
     9    20
         /  \
        15   7
```

Visit node `9`:

```text
left_gain = 0
right_gain = 0
path_through_9 = 9
best = 9
return 9
```

Visit node `15`:

```text
path_through_15 = 15
best = 15
return 15
```

Visit node `7`:

```text
path_through_7 = 7
best = 15
return 7
```

Visit node `20`:

```text
left_gain = 15
right_gain = 7
path_through_20 = 15 + 20 + 7 = 42
best = 42
return 20 + max(15, 7) = 35
```

The path `15 -> 20 -> 7` is complete. It uses both children of `20`, so it cannot be returned upward.

Visit node `-10`:

```text
left_gain = 9
right_gain = 35
path_through_-10 = 9 + (-10) + 35 = 34
best = 42
return -10 + max(9, 35) = 25
```

The best remains `42`, from:

```text
15 -> 20 -> 7
```

This example shows another important point:

> The maximum path does not have to include the root.

---

### 10. Why Negative Branches Are Ignored

Consider this tree:

```text
   5
  /
-3
```

A path through `5` could include `-3`:

```text
-3 + 5 = 2
```

But stopping at `5` gives:

```text
5
```

So the negative branch should not be used.

That is why we write:

```python
left_gain = max(dfs(node.left), 0)
right_gain = max(dfs(node.right), 0)
```

This does not incorrectly allow an empty final path, because `best` is only updated at real nodes with:

```text
left_gain + node.val + right_gain
```

Every candidate global answer includes at least the current node.

For an all-negative tree like:

```text
  -3
  / \
-2  -5
```

The algorithm still works:

```text
best starts at -infinity
node -2 updates best to -2
node -5 does not improve it
node -3 does not improve it
answer = -2
```

If `best` started at `0`, this case would incorrectly return `0`, which is not a valid path sum because the path must contain at least one node.

---

### 11. Correctness Argument

We prove the algorithm returns the maximum path sum.

First, consider the DFS return value.

For `None`, the returned gain is `0`, which is correct because a missing child contributes no extendable path to its parent.

For a real node, assume the recursive calls correctly return the best extendable downward gains from the left and right children.

Any extendable path starting at the current node has only three possible forms:

```text
node alone
node + an extendable path from the left child
node + an extendable path from the right child
```

It cannot use both children and still be extendable to the parent. Therefore returning:

```text
node.val + max(max(left_gain, 0), max(right_gain, 0))
```

is exactly the best extendable path starting at this node.

Now consider the global answer update.

Every valid path in a tree has a highest node relative to the root: the node where the path's two endpoints, if both exist below it, meet. At that highest node, the path consists of:

```text
some downward path from the left side
that node
some downward path from the right side
```

Either side may be absent.

When DFS processes that highest node, the recursive return values provide the best possible downward gains from each side, and negative sides are correctly omitted. Therefore the algorithm considers a path at least as good as any valid path whose highest node is that node.

Since every valid path has exactly one such highest node, and every node is processed once, the global `best` is updated with the optimal path sum somewhere during the traversal.

Therefore, after DFS finishes, `best` equals the maximum path sum in the tree.

---

### 12. Complexity

Let `n` be the number of nodes and `h` be the height of the tree.

Each node is visited once, and each visit does constant work:

```text
Time: O(n)
```

The recursion stack contains at most one root-to-leaf chain at a time:

```text
Space: O(h)
```

For a balanced tree, `h = O(log n)`.
For a completely skewed tree, `h = O(n)`.

---

### 13. Common Pitfalls

- Returning `left_gain + node.val + right_gain` to the parent. That value may use both children, so it is a complete path, not an extendable path.
- Initializing the answer to `0`. This fails when every node value is negative.
- Forgetting to discard negative child gains. A negative branch should never be forced into a path that can stop earlier.
- Thinking the path must include the root. The optimal path may be entirely inside one subtree.
- Thinking the path must end at leaves. A single internal node can be the best path if all surrounding branches are harmful.
- Treating `dfs(None) = 0` as the final answer for an empty path. The global answer is updated only at real nodes.
- Losing the distinction between a candidate global path and the value returned upward.

---

### 14. First-Principles Summary

A binary tree path is a line, not a branching structure.

At each node, there are two different questions:

```text
What is the best complete path that turns here?
What is the best one-sided path I can pass to my parent?
```

The complete path may use both children:

```text
left_gain + node.val + right_gain
```

The returned path may use only one child:

```text
node.val + max(left_gain, right_gain)
```

Negative child gains are ignored because adding them only makes a path worse.

The algorithm is efficient because each subtree reports exactly the one piece of information its parent needs: the best extendable downward gain. A global answer records complete paths that cannot be returned upward.

## Implementation
See `solutions/binary_tree_dfs/p124_binary_tree_maximum_path_sum.py`.

## Tests
See `tests/binary_tree_dfs/test_p124_binary_tree_maximum_path_sum.py`.

## Examples

### Example 1
- Input: `{'root': [1, 2, 3]}`
- Output: `6`

### Example 2
- Input: `{'root': [-10, 9, 20, None, None, 15, 7]}`
- Output: `42`

## Follow-up Practice
- Write the DFS return contract before writing code.
- Trace an all-negative tree and verify that the answer is the least negative node.
- Explain why the path through a node can use two children, but the value returned to a parent cannot.
- Compare the global candidate `left_gain + node.val + right_gain` with the returned gain `node.val + max(left_gain, right_gain)` on several small trees.
