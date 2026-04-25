# 236. Lowest Common Ancestor of a Binary Tree

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given:

```text
root = the root of a binary tree
p    = one node in that tree
q    = another node in that tree
```

Return the **lowest common ancestor** of `p` and `q`.

An ancestor of a node is any node on the path from the root down to that node. A node is also considered an ancestor of itself.

So if the tree is:

```text
        3
      /   \
     5     1
    / \   / \
   6   2 0   8
      / \
     7   4
```

and:

```text
p = 5
q = 1
```

then the path from the root to `5` is:

```text
3 -> 5
```

The path from the root to `1` is:

```text
3 -> 1
```

Their common ancestors are:

```text
3
```

So the answer is `3`.

If instead:

```text
p = 5
q = 4
```

then the path to `5` is:

```text
3 -> 5
```

and the path to `4` is:

```text
3 -> 5 -> 2 -> 4
```

The common ancestors are:

```text
3, 5
```

The **lowest** one means the common ancestor deepest in the tree, closest to the targets. That is `5`.

The real problem is:

> Find the deepest node whose subtree contains both target nodes, allowing that the node itself can be one of the targets.

---

### 2. Start From the Baseline Idea

A direct way to think about the problem is path-based.

1. Find the path from `root` to `p`.
2. Find the path from `root` to `q`.
3. Walk both paths from the beginning while the nodes are equal.
4. The last equal node is the lowest common ancestor.

For example, for `p = 5` and `q = 4`:

```text
path_to_p = [3, 5]
path_to_q = [3, 5, 2, 4]
```

Compare them from left to right:

```text
3 == 3   common
5 == 5   common
```

Then `path_to_p` ends, so the last common node is `5`.

Conceptually:

```python
path_p = path_from_root_to(root, p)
path_q = path_from_root_to(root, q)

answer = None

for a, b in zip(path_p, path_q):
    if a is b:
        answer = a
    else:
        break

return answer
```

This is correct, and it is a useful mental baseline.

But it does extra work:

- It searches for `p`.
- It searches for `q`.
- It stores two full root-to-node paths.
- It solves the problem from the top down, even though the key information is naturally discovered from the bottom up.

The deeper question is:

> Can each subtree tell its parent whether it contains `p`, `q`, or the answer?

Yes. That leads to the standard recursive DFS solution.

---

### 3. The Key Observation

For any node `x`, there are only a few possibilities:

1. Neither `p` nor `q` appears in `x`'s subtree.
2. Exactly one of `p` or `q` appears in `x`'s subtree.
3. Both `p` and `q` appear in `x`'s subtree.

If both targets appear in `x`'s subtree, there are two subcases:

- The lowest common ancestor is already deeper inside one child subtree.
- Or `x` itself is the first place where the two targets come together.

The second case happens when:

```text
one target is found in the left subtree
and the other target is found in the right subtree
```

or when:

```text
x is one target
and the other target is found below x
```

This is why the problem is a natural postorder DFS problem:

```text
first ask the left subtree what it found
first ask the right subtree what it found
then decide what the current node means
```

The tree structure itself tells us how to combine the answers.

---

### 4. The Recursive Return Contract

The most important part of this problem is defining exactly what the recursive function returns.

Let:

```python
dfs(node)
```

mean:

> Search the subtree rooted at `node`. Return the lowest common ancestor if that ancestor has already been determined inside this subtree. Otherwise, return whichever one of `p` or `q` was found in this subtree. If neither target was found, return `None`.

That sounds like one return value doing two jobs, so it helps to spell it out:

```text
dfs(node) returns None
```

means:

```text
this subtree contains neither p nor q
```

```text
dfs(node) returns p or q
```

means:

```text
this subtree contains that target, but the LCA has not been formed below this point
```

```text
dfs(node) returns some other node
```

means:

```text
the LCA has already been found in this subtree
```

In LeetCode 236, both `p` and `q` are guaranteed to exist in the tree. Because of that guarantee, the final return value from `dfs(root)` is the answer.

---

### 5. Why Returning the Target Node Works

When the recursion reaches `p` or `q`, it can immediately return that node.

For example:

```python
if node is p or node is q:
    return node
```

This is not ignoring descendants. It is using the ancestor definition.

If `node` is `p`, there are two possibilities:

1. `q` is somewhere below `p`.
2. `q` is not below `p`.

If `q` is below `p`, then `p` is the LCA because a node can be an ancestor of itself.

If `q` is not below `p`, then `p` is still the useful signal that must travel upward until some ancestor also receives a signal from the other side.

So returning `node` when `node` is one of the targets is exactly the right information to give the parent.

---

### 6. The Local Decision at Each Node

For a current node `node`, recursively compute:

```python
left = dfs(node.left)
right = dfs(node.right)
```

Now interpret the two returned signals.

#### Case 1: Both Sides Return Something

```text
left is not None
right is not None
```

This means one target was found somewhere in the left subtree and the other target was found somewhere in the right subtree.

The current node is the first node where those two discoveries meet.

So:

```python
return node
```

#### Case 2: Only the Left Side Returns Something

```text
left is not None
right is None
```

Everything important found so far is in the left subtree.

The current node does not combine two sides, so pass the left result upward:

```python
return left
```

That result may be:

- `p`,
- `q`, or
- an already-discovered LCA from deeper in the left subtree.

In all cases, the parent should receive it.

#### Case 3: Only the Right Side Returns Something

This is symmetric:

```python
return right
```

#### Case 4: Neither Side Returns Anything

```text
left is None
right is None
```

No target was found in this subtree, so return:

```python
return None
```

---

### 7. Algorithm

1. Define a recursive helper `dfs(node)`.
2. If `node` is `None`, return `None`.
3. If `node` is `p` or `node` is `q`, return `node`.
4. Recursively search the left subtree.
5. Recursively search the right subtree.
6. If both recursive calls return non-`None`, return the current node.
7. Otherwise, return whichever side is non-`None`.
8. The answer is `dfs(root)`.

In code-like form:

```python
def lowestCommonAncestor(root, p, q):
    def dfs(node):
        if node is None:
            return None

        if node is p or node is q:
            return node

        left = dfs(node.left)
        right = dfs(node.right)

        if left is not None and right is not None:
            return node

        if left is not None:
            return left

        return right

    return dfs(root)
```

The same final return can be written compactly as:

```python
return left or right
```

after the two-sided case is handled.

---

### 8. Detailed Walkthrough: Example 1

Use the tree:

```text
        3
      /   \
     5     1
    / \   / \
   6   2 0   8
      / \
     7   4
```

with:

```text
p = 5
q = 1
```

Start at `3`.

The DFS asks the left subtree rooted at `5` what it contains.

Since `5` is `p`, `dfs(5)` returns `5`.

Then the DFS asks the right subtree rooted at `1` what it contains.

Since `1` is `q`, `dfs(1)` returns `1`.

Now at node `3`:

```text
left  = 5
right = 1
```

Both sides returned a target signal.

That means:

```text
one target is in the left subtree of 3
one target is in the right subtree of 3
```

So `3` is the first node where the two target paths meet.

Return `3`.

The answer is:

```text
3
```

---

### 9. Detailed Walkthrough: Example 2

Use the same tree, but now:

```text
p = 5
q = 4
```

The important part of the tree is:

```text
     5
    / \
   6   2
      / \
     7   4
```

Node `5` is one of the targets.

Because a node can be an ancestor of itself, if `4` is below `5`, then `5` should be the answer.

With the concise LeetCode solution, `dfs(5)` immediately returns `5` because `5 is p`.

Then at the root `3`, the right subtree rooted at `1` returns `None` because it contains neither `5` nor `4`.

So at `3`:

```text
left  = 5
right = None
```

The root passes `5` upward:

```text
return 5
```

The final answer is `5`.

This works because the problem guarantees both target nodes exist. If `p` is found and `q` is not in the other side of the tree, then `q` must be somewhere inside `p`'s subtree or below a node already represented by that returned signal.

---

### 10. A More Explicit Variant

Some people find the concise version surprising because it returns immediately when it sees `p` or `q`.

An alternative is to let every subtree return two pieces of information:

```text
whether p was found
whether q was found
```

and record the first node where both become true.

Conceptually:

```python
answer = None

def dfs(node):
    nonlocal answer

    if node is None:
        return False, False

    left_has_p, left_has_q = dfs(node.left)
    right_has_p, right_has_q = dfs(node.right)

    has_p = left_has_p or right_has_p or node is p
    has_q = left_has_q or right_has_q or node is q

    if has_p and has_q and answer is None:
        answer = node

    return has_p, has_q
```

This version makes the logic very explicit: a node is an ancestor of both targets if its subtree contains both.

The standard solution compresses this information into one returned node signal. That is why it is shorter, but the underlying idea is the same.

---

### 11. Correctness

We prove that `dfs(node)` follows its return contract for every subtree rooted at `node`.

#### Base Case

If `node` is `None`, the subtree is empty.

An empty subtree contains neither `p` nor `q`, so returning `None` is correct.

#### Target Case

If `node` is `p` or `node` is `q`, returning `node` is correct.

The current subtree contains that target, and the current node itself is the highest useful signal to pass upward. If the other target is below this node, then this node is the LCA. If the other target is elsewhere, an ancestor will combine this signal with a signal from another subtree.

#### Recursive Step

Assume the recursive calls correctly follow the return contract for `node.left` and `node.right`.

Let:

```text
left  = dfs(node.left)
right = dfs(node.right)
```

If both `left` and `right` are non-`None`, then the left subtree contains one relevant target signal and the right subtree contains the other. Since the two targets are in different child subtrees, no descendant of `node` can be an ancestor of both. Therefore `node` is exactly the lowest common ancestor, and returning `node` is correct.

If only one side is non-`None`, then all discovered target information in this subtree lies on that side. The current node does not combine two target-containing branches, so the correct result for this subtree is the result already found by that child. Returning the non-`None` side preserves the contract.

If both sides are `None`, neither subtree contains either target, and the current node is not a target because that case was handled earlier. Therefore this subtree contains neither target, so returning `None` is correct.

By structural induction over the tree, `dfs(root)` follows the return contract for the whole tree. Since the problem guarantees that both `p` and `q` are in the tree, the value returned by `dfs(root)` is the lowest common ancestor.

---

### 12. Complexity

Let `n` be the number of nodes in the tree and `h` be its height.

Each node is visited at most once.

So the time complexity is:

```text
O(n)
```

The recursion stack contains at most one root-to-leaf path at a time.

So the auxiliary space complexity is:

```text
O(h)
```

In a balanced tree:

```text
h = O(log n)
```

In a completely skewed tree:

```text
h = O(n)
```

The algorithm does not allocate path arrays or a parent map.

---

### 13. Common Pitfalls

#### Comparing Values Instead of Nodes

The LeetCode function receives actual `TreeNode` objects for `p` and `q`.

The safest comparison is object identity:

```python
node is p or node is q
```

Using values can be wrong if duplicate values exist. Even if the problem examples have distinct values, the concept of LCA is about node identity, not value equality.

#### Forgetting That a Node Is Its Own Ancestor

For `p = 5` and `q = 4`, the answer is `5`, not `3`.

That is because `5` is allowed to be an ancestor of itself.

#### Returning Too Early at the Parent

The recursive function should only return the current node when both sides report non-`None`, or when the current node itself is a target.

If only one child returns a signal, pass that signal upward.

#### Losing an Already-Found LCA

When a child returns a node, that node might be a target or it might already be the LCA found deeper in the tree.

Do not discard it just because the other side is `None`.

#### Forgetting the Existence Guarantee

The concise solution relies on the standard LeetCode 236 guarantee that both `p` and `q` exist in the tree.

If the problem variant does not guarantee that, use an explicit found-count or boolean-return version so you can distinguish:

```text
found only p
found only q
found both
```

---

### 14. First-Principles Summary

The lowest common ancestor is the deepest node whose subtree contains both target nodes.

Instead of building root-to-target paths, ask each subtree to return a compact signal:

```text
None          -> found neither target
p or q        -> found one target
some ancestor -> already found the LCA
```

At each node, the decisive moment is when both children return non-`None` signals. That means the targets lie in different child subtrees, so the current node is exactly where the paths meet.

If only one side returns a signal, the current node has not combined the two targets, so it simply passes that signal upward.

This is the core first-principles idea:

> A subtree can summarize everything its parent needs to know with one returned node: no target, one target, or the already-determined ancestor.

## Implementation
See `solutions/binary_tree_dfs/p236_lowest_common_ancestor_of_a_binary_tree.py`.

## Tests
See `tests/binary_tree_dfs/test_p236_lowest_common_ancestor_of_a_binary_tree.py`.

## Examples

### Example 1
- Input: `{'root': [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4], 'p': 5, 'q': 1}`
- Output: `3`

### Example 2
- Input: `{'root': [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4], 'p': 5, 'q': 4}`
- Output: `5`

### Example 3
- Input: `{'root': [1, 2], 'p': 1, 'q': 2}`
- Output: `1`

## Follow-up Practice
- Write the recursive contract in one sentence before writing code.
- Trace the case where one target is an ancestor of the other.
- Rewrite the solution using explicit `found_p` and `found_q` booleans for a variant where targets may be missing.
