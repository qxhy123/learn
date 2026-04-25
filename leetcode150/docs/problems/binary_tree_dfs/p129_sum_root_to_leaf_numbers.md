# 129. Sum Root to Leaf Numbers

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/sum-root-to-leaf-numbers/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs, sum

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a binary tree where every node contains one digit from `0` to `9`.

A number is formed by starting at the root and walking down to a leaf. Each node on that path contributes one digit, in order.

For example, in this tree:

```text
    1
   / \
  2   3
```

There are two root-to-leaf paths:

```text
1 -> 2  forms the number 12
1 -> 3  forms the number 13
```

The answer is the sum of all such numbers:

```text
12 + 13 = 25
```

The key phrase is **root-to-leaf**. A path only counts when it starts at the root and ends at a leaf. Stopping at an internal node does not produce a complete number.

So the real task is:

> Visit every root-to-leaf path, interpret each path as a base-10 number, and add those numbers together.

---

### 2. Start From the Baseline Idea

The most direct way to think about the problem is to store every path explicitly.

For each root-to-leaf path:

1. Collect the digits along the path.
2. Convert those digits into a number.
3. Add that number to the answer.

For the tree:

```text
      4
     / \
    9   0
   / \
  5   1
```

The paths are:

```text
4 -> 9 -> 5  gives digits [4, 9, 5]  gives number 495
4 -> 9 -> 1  gives digits [4, 9, 1]  gives number 491
4 -> 0       gives digits [4, 0]     gives number 40
```

The sum is:

```text
495 + 491 + 40 = 1026
```

A baseline recursive version might carry a list of digits:

```python
def dfs(node, digits):
    if node is None:
        return 0

    digits.append(node.val)

    if node.left is None and node.right is None:
        number = convert_digits_to_integer(digits)
        digits.pop()
        return number

    total = dfs(node.left, digits) + dfs(node.right, digits)
    digits.pop()
    return total
```

This is correct, but it does extra work. At every leaf, it converts the whole path into an integer. If a path has length `h`, that conversion costs `O(h)`. Across many leaves, the repeated conversions are unnecessary.

We can do better by building the number as we descend.

---

### 3. The Key Observation: A Path Number Can Be Updated One Digit at a Time

Suppose the digits seen so far form the number:

```text
prefix
```

When we move to a child with digit:

```text
d
```

The new number is not `prefix + d`. The new digit goes at the end of the decimal representation.

In base 10, appending a digit means:

```text
new_prefix = prefix * 10 + d
```

Examples:

```text
prefix = 4,   d = 9  -> 4 * 10 + 9 = 49
prefix = 49,  d = 5  -> 49 * 10 + 5 = 495
prefix = 49,  d = 1  -> 49 * 10 + 1 = 491
prefix = 4,   d = 0  -> 4 * 10 + 0 = 40
```

This means we do not need to store the whole path. We only need to carry the numeric value of the path prefix built so far.

That is the central simplification:

> The entire ancestor path can be summarized by one integer: the number formed before entering the current node.

---

### 4. Why DFS Fits the Shape of the Problem

A root-to-leaf number is defined by a path from parent to child. Once we are standing at a node, the choices are simply:

```text
go left
go right
```

Both choices inherit the same kind of information from the parent: the number formed by the path so far.

That is exactly what depth-first search is good at:

1. Carry context from ancestors down to descendants.
2. Detect when a complete path has been reached.
3. Return a contribution from each completed path.
4. Add the contributions from the left and right subtrees.

The important point is that this is not a generic tree traversal where visiting nodes is enough. A node's value only has meaning together with its ancestors. The DFS state must therefore include the prefix number built before that node.

---

### 5. Recursive Invariant and Accumulator Contract

Define the recursive helper like this:

```python
def dfs(node, prefix):
```

The contract is:

> `prefix` is the number formed by the path from the root to the parent of `node`. `dfs(node, prefix)` returns the sum of all complete root-to-leaf numbers that pass through `node`.

Inside the call, the first thing we do is include `node.val`:

```python
current = prefix * 10 + node.val
```

After this line, the invariant becomes:

> `current` is the number formed by the path from the root to `node`.

Now there are two cases.

#### Case 1: `node` is a leaf

A leaf has no children:

```text
node.left is None and node.right is None
```

At a leaf, the current path is complete. There is exactly one root-to-leaf number ending here, and it is `current`.

So return:

```python
return current
```

#### Case 2: `node` is not a leaf

The path is not complete yet. We must continue into the children.

The total contribution from this node's subtree is:

```text
sum from left subtree + sum from right subtree
```

Each child receives `current` as its prefix, because `current` is now the number built up to this node.

So return:

```python
return dfs(node.left, current) + dfs(node.right, current)
```

The `None` case contributes `0`, because an absent child contains no root-to-leaf paths.

---

### 6. Detailed Algorithm

1. If the root is `None`, return `0`.
2. Define a DFS helper that accepts:
   - `node`: the current tree node.
   - `prefix`: the number formed before this node is appended.
3. If `node` is `None`, return `0`.
4. Compute the number represented by the path through this node:

   ```python
   current = prefix * 10 + node.val
   ```

5. If the node is a leaf, return `current`.
6. Otherwise, recursively compute:
   - the sum of all root-to-leaf numbers in the left subtree,
   - the sum of all root-to-leaf numbers in the right subtree.
7. Return the sum of those two recursive results.
8. Start the recursion with `prefix = 0` at the root.

---

### 7. Pseudocode

```python
def sumNumbers(root):
    def dfs(node, prefix):
        if node is None:
            return 0

        current = prefix * 10 + node.val

        if node.left is None and node.right is None:
            return current

        return dfs(node.left, current) + dfs(node.right, current)

    return dfs(root, 0)
```

This version returns values instead of using a global variable. That keeps the meaning of each recursive call clear: every call returns the total contribution of its subtree under the prefix it was given.

---

### 8. Walk Through Example 1

Input:

```text
root = [1, 2, 3]
```

Tree:

```text
    1
   / \
  2   3
```

Start:

```text
dfs(1, 0)
```

At node `1`:

```text
current = 0 * 10 + 1 = 1
```

Node `1` is not a leaf, so recurse into both children.

Left child:

```text
dfs(2, 1)
current = 1 * 10 + 2 = 12
```

Node `2` is a leaf, so it returns `12`.

Right child:

```text
dfs(3, 1)
current = 1 * 10 + 3 = 13
```

Node `3` is a leaf, so it returns `13`.

Back at node `1`:

```text
left contribution  = 12
right contribution = 13
total              = 25
```

So the answer is:

```text
25
```

---

### 9. Walk Through Example 2

Input:

```text
root = [4, 9, 0, 5, 1]
```

Tree:

```text
      4
     / \
    9   0
   / \
  5   1
```

Start at the root:

```text
dfs(4, 0)
current = 0 * 10 + 4 = 4
```

Node `4` is not a leaf.

Go left to `9`:

```text
dfs(9, 4)
current = 4 * 10 + 9 = 49
```

Node `9` is not a leaf.

Go left to `5`:

```text
dfs(5, 49)
current = 49 * 10 + 5 = 495
```

Node `5` is a leaf, so it contributes:

```text
495
```

Go right from `9` to `1`:

```text
dfs(1, 49)
current = 49 * 10 + 1 = 491
```

Node `1` is a leaf, so it contributes:

```text
491
```

Therefore the subtree rooted at `9` contributes:

```text
495 + 491 = 986
```

Now go right from `4` to `0`:

```text
dfs(0, 4)
current = 4 * 10 + 0 = 40
```

Node `0` is a leaf, so it contributes:

```text
40
```

Finally, the root combines both sides:

```text
986 + 40 = 1026
```

So the answer is:

```text
1026
```

---

### 10. Correctness

We prove that the algorithm returns the sum of all root-to-leaf numbers.

The recursive contract is:

> `dfs(node, prefix)` returns the sum of all root-to-leaf numbers that pass through `node`, assuming `prefix` is the number formed by the path from the root to `node`'s parent.

#### Base Case: Empty Node

If `node` is `None`, there is no subtree and no root-to-leaf path through that child. Returning `0` is correct.

#### Leaf Case

When `node` is a leaf, the path from the root to this node is complete.

The algorithm computes:

```text
current = prefix * 10 + node.val
```

By the meaning of decimal notation, this is exactly the number represented by the root-to-leaf path ending at this leaf. Since there are no child paths below a leaf, the total contribution of this subtree is exactly `current`. Returning `current` is correct.

#### Recursive Case

When `node` is not a leaf, every complete root-to-leaf path passing through `node` must continue either into the left subtree or into the right subtree.

After computing:

```text
current = prefix * 10 + node.val
```

`current` is exactly the number formed by the path from the root to `node`. Therefore it is the correct prefix to pass to each child.

By the recursive contract:

```text
dfs(node.left, current)
```

returns the sum of all complete root-to-leaf numbers in the left subtree, and:

```text
dfs(node.right, current)
```

returns the sum of all complete root-to-leaf numbers in the right subtree.

These two sets of paths are disjoint and together include every complete path through `node`. Adding the two returned sums is therefore correct.

#### Whole Tree

The initial call is:

```text
dfs(root, 0)
```

Before the root, no digits have been read, so the correct prefix is `0`. By the recursive contract, the result is the sum of all root-to-leaf numbers in the entire tree.

---

### 11. Complexity

Let `n` be the number of nodes in the tree and `h` be the height of the tree.

#### Time Complexity

Each node is visited once. The work done at each node is constant: one multiplication, one addition, a leaf check, and at most two recursive calls.

So the time complexity is:

```text
O(n)
```

#### Space Complexity

The algorithm does not store all paths. It only stores the recursion stack.

The maximum recursion depth is the height of the tree:

```text
O(h)
```

For a balanced tree, `h = O(log n)`. For a completely skewed tree, `h = O(n)`.

---

### 12. Common Pitfalls

#### Returning `0` at a Leaf

A leaf is the moment when a complete number has been formed. The leaf should return `current`, not `0`.

#### Adding Node Values Instead of Appending Digits

The path `4 -> 9 -> 5` represents `495`, not:

```text
4 + 9 + 5 = 18
```

The update must be:

```python
current = prefix * 10 + node.val
```

not:

```python
current = prefix + node.val
```

#### Counting Internal Nodes as Complete Numbers

In this tree:

```text
    1
   /
  2
```

The answer is `12`, not `1 + 12`. The root is not a complete path unless it is also a leaf.

#### Mishandling Digit `0`

A zero is still a digit that must be appended.

For path:

```text
4 -> 0
```

The number is:

```text
40
```

The formula handles this naturally:

```text
4 * 10 + 0 = 40
```

#### Using Shared Mutable Path State Incorrectly

A list-based path solution must carefully backtrack with `pop()`. The accumulator approach avoids that entire class of bugs by passing an integer value down the recursion instead of mutating shared path state.

---

### 13. First-Principles Summary

A root-to-leaf path is a sequence of digits. A sequence of digits can be built incrementally because appending a digit `d` to an existing decimal number `prefix` gives:

```text
prefix * 10 + d
```

So the only context a child needs from its parent is the number formed so far.

At every node:

1. Append the node's digit to the inherited prefix.
2. If the node is a leaf, return that completed number.
3. Otherwise, ask the left and right children for their completed-path sums.
4. Return the sum of those child contributions.

The algorithm is efficient because it never materializes every path as a list or string. It turns the tree traversal itself into the number-building process.

## Implementation
See `solutions/binary_tree_dfs/p129_sum_root_to_leaf_numbers.py`.

## Tests
See `tests/binary_tree_dfs/test_p129_sum_root_to_leaf_numbers.py`.

## Examples

### Example 1
- Input: `{'root': [1, 2, 3]}`
- Output: `25`

### Example 2
- Input: `{'root': [4, 9, 0, 5, 1]}`
- Output: `1026`

## Follow-up Practice
- Write the recursive contract in one sentence before writing code.
- Trace the accumulator value on the smallest non-empty tree: a single node.
- Trace a path containing `0` to confirm that digits are appended, not summed.
- Rewrite the recursive DFS using an explicit stack of `(node, prefix)` pairs.
