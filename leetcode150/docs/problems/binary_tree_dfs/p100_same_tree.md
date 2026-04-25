# 100. Same Tree

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/same-tree/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given two binary trees:

```text
p = root of the first tree
q = root of the second tree
```

Return `True` if the two trees are exactly the same, and return `False` otherwise.

"Exactly the same" means two things must be true at every corresponding position:

1. The same node positions must exist in both trees.
2. The node values at those positions must be equal.

So this is not enough:

```text
Tree p:     1
           /
          2

Tree q:     1
             \
              2
```

Both trees contain the values `1` and `2`, but they are not the same tree. In `p`, the `2` is the left child. In `q`, the `2` is the right child.

The shape matters just as much as the values.

A binary tree is recursive by nature:

```text
a tree = empty
      or a node with a value, a left subtree, and a right subtree
```

Therefore, two binary trees are the same if:

```text
both are empty
```

or:

```text
both roots exist
and root values are equal
and left subtrees are the same
and right subtrees are the same
```

That sentence is the whole problem. The algorithm is just a direct translation of it.

---

### 2. Start From the Brute-Force/Baseline Idea

A natural baseline is to serialize both trees, then compare the serialized results.

For example, we could traverse each tree and write down every node plus every missing child marker:

```text
Tree:       1
           / \
          2   3

Serialization with null markers:
1, 2, null, null, 3, null, null
```

Then the algorithm would be:

1. Convert `p` into a sequence.
2. Convert `q` into a sequence.
3. Return whether the two sequences are equal.

This works only if the serialization includes empty-child markers. Without null markers, different shapes can look identical.

For example:

```text
Tree p:     1
           /
          2

Tree q:     1
             \
              2
```

A preorder traversal that records only values gives:

```text
1, 2
```

for both trees, even though the shapes differ.

With null markers, they become different:

```text
p: 1, 2, null, null, null
q: 1, null, 2, null, null
```

So serialization can solve the problem, but it does extra work:

- It builds two intermediate sequences.
- It keeps comparing even after many traversals have already revealed a mismatch.
- It separates the real idea from the tree structure.

The better baseline is to compare the two trees directly, one corresponding pair of nodes at a time.

---

### 3. Key Observation: Compare Positions, Not Sets of Values

The problem is not asking whether the two trees contain the same multiset of values.

It is asking whether every position matches.

A position means a path from the root, such as:

```text
root
root.left
root.right
root.left.right
root.right.left.left
```

For every such position, the two trees must agree:

```text
both have no node there
```

or:

```text
both have a node there, and the values are equal
```

This gives a very local test for each pair of nodes:

1. If both nodes are missing, this position matches.
2. If exactly one node is missing, the trees differ.
3. If both nodes exist but values differ, the trees differ.
4. If both nodes exist and values match, compare their left children and right children.

The important first-principles move is that once the current pair of nodes matches, the rest of the question splits into two independent smaller questions:

```text
Are p.left and q.left the same?
Are p.right and q.right the same?
```

Both must be true.

---

### 4. Recursive Invariant / Contract

Define a helper function conceptually:

```text
same(a, b)
```

Contract:

```text
same(a, b) returns True exactly when the subtree rooted at a
and the subtree rooted at b are identical in both shape and values.
```

This contract is precise enough to write the solution.

For a call `same(a, b)`, there are four cases.

Case 1: both are empty.

```text
a is None
b is None
```

There is no value and no child structure on either side. Two empty trees are identical, so return `True`.

Case 2: exactly one is empty.

```text
a is None, b is not None
```

or:

```text
a is not None, b is None
```

One tree has a node at this position and the other does not. The shapes differ, so return `False`.

Case 3: both exist, but values differ.

```text
a.val != b.val
```

The shapes may still look similar below this point, but this position already fails. Return `False`.

Case 4: both exist and values match.

Now the current position is fine. The subtrees are identical only if both child pairs are identical:

```text
same(a.left, b.left) and same(a.right, b.right)
```

That is the recursive invariant: every recursive call answers the same question for a smaller pair of subtrees.

---

### 5. Detailed Algorithm

Use depth-first search on both trees at the same time.

At each step, compare one node from `p` with the corresponding node from `q`.

Algorithm:

1. Start with the pair `(p, q)`.
2. If both nodes are `None`, return `True` for this pair.
3. If only one node is `None`, return `False`.
4. If both nodes exist but their values differ, return `False`.
5. Recursively compare the left children.
6. Recursively compare the right children.
7. Return `True` only if both recursive comparisons return `True`.

The algorithm can stop as soon as it finds a mismatch.

For example, if the root values differ, there is no reason to inspect any descendants. If one left child is missing while the other exists, there is no reason to compare the corresponding right subtrees for the final answer, because the answer is already `False`.

This early stopping is one advantage over building complete serialized representations first.

---

### 6. Code / Pseudocode

Python-style pseudocode:

```python
def isSameTree(p, q):
    if p is None and q is None:
        return True

    if p is None or q is None:
        return False

    if p.val != q.val:
        return False

    return (
        isSameTree(p.left, q.left)
        and isSameTree(p.right, q.right)
    )
```

The order of checks matters.

You must handle `None` before reading `.val`, `.left`, or `.right`. A missing node has no fields.

A slightly compressed version is possible:

```python
def isSameTree(p, q):
    if not p or not q:
        return p is q

    return (
        p.val == q.val
        and isSameTree(p.left, q.left)
        and isSameTree(p.right, q.right)
    )
```

The expanded version is often easier to reason about because each structural case is visible.

---

### 7. Detailed Example Walkthrough

Consider Example 1:

```text
p = [1, 2, 3]
q = [1, 2, 3]
```

The trees are:

```text
p:        1
         / \
        2   3

q:        1
         / \
        2   3
```

Start at the roots:

```text
same(p root 1, q root 1)
```

Both nodes exist and both values are `1`, so the root position matches.

Now compare left subtrees:

```text
same(p.left 2, q.left 2)
```

Both nodes exist and both values are `2`.

Compare their left children:

```text
same(None, None) -> True
```

Compare their right children:

```text
same(None, None) -> True
```

So the two leaf nodes with value `2` match.

Now compare right subtrees of the root:

```text
same(p.right 3, q.right 3)
```

Both nodes exist and both values are `3`.

Again, both left children are missing and both right children are missing:

```text
same(None, None) -> True
same(None, None) -> True
```

So the right subtrees match.

The root returns:

```text
root values match
and left subtrees match
and right subtrees match
```

Therefore the final answer is:

```text
True
```

---

### 8. Walkthrough of a Shape Mismatch

Consider Example 2:

```text
p = [1, 2]
q = [1, None, 2]
```

The trees are:

```text
p:        1
         /
        2

q:        1
           \
            2
```

Start at the roots:

```text
same(1, 1)
```

The roots both exist and values match.

Now compare left children:

```text
same(p.left 2, q.left None)
```

One side has a node and the other side does not. That means the shapes differ at the path:

```text
root.left
```

So this call returns:

```text
False
```

Because both left and right subtrees must match, the whole answer is already `False`.

Notice why value-only thinking fails here. Both trees contain the values `1` and `2`, but the position of `2` is different.

---

### 9. Walkthrough of a Value Mismatch

Consider Example 3:

```text
p = [1, 2, 1]
q = [1, 1, 2]
```

The trees are:

```text
p:        1
         / \
        2   1

q:        1
         / \
        1   2
```

Start at the roots:

```text
same(1, 1)
```

The roots match.

Compare the left children:

```text
same(2, 1)
```

Both nodes exist, but their values differ:

```text
2 != 1
```

So the left subtree comparison returns `False`, and the final answer is `False`.

Even though both trees contain two `1` values and one `2` value overall, they are not the same because corresponding positions do not match.

---

### 10. Correctness

We prove that the algorithm returns `True` if and only if the two input trees are identical in shape and values.

Use structural induction on the pair of subtrees passed to `isSameTree`.

Base case 1: both subtrees are empty.

The algorithm returns `True`. This is correct because two empty trees have the same shape and no values that could differ.

Base case 2: exactly one subtree is empty.

The algorithm returns `False`. This is correct because one tree has a node at the current position while the other does not, so their shapes are different.

Recursive case: both roots exist.

If the root values differ, the algorithm returns `False`. This is correct because identical trees must have equal values at every corresponding position, including the current root position.

If the root values are equal, the algorithm recursively checks:

```text
isSameTree(p.left, q.left)
isSameTree(p.right, q.right)
```

By the induction hypothesis, the first recursive call returns `True` exactly when the left subtrees are identical, and the second recursive call returns `True` exactly when the right subtrees are identical.

The algorithm returns the logical `and` of these two results together with the current value match. Therefore it returns `True` exactly when:

```text
current root values match
and left subtrees are identical
and right subtrees are identical
```

That is exactly the definition of two non-empty binary trees being the same.

Therefore, by structural induction, the algorithm is correct for the original input trees.

---

### 11. Complexity

Let `n` be the number of nodes in `p`, and let `m` be the number of nodes in `q`.

Time complexity:

```text
O(min(n, m)) in early-mismatch cases
O(n) when both trees have the same size and all corresponding positions must be checked
```

A simple worst-case bound is:

```text
O(n + m)
```

because the algorithm never visits a corresponding position more than once, and it stops as soon as a mismatch proves the answer is `False`.

If the two trees are identical and contain `n` nodes, every node pair is checked once, so the time is:

```text
O(n)
```

Space complexity:

```text
O(h)
```

where `h` is the height of the recursion stack.

For balanced trees:

```text
h = O(log n)
```

For completely skewed trees:

```text
h = O(n)
```

No extra data structure proportional to all nodes is needed. The recursion stack is the main additional memory cost.

---

### 12. Common Pitfalls

#### Pitfall 1: Comparing Traversal Values Without Null Markers

A traversal that records only values can miss shape differences.

These two trees both have preorder values `1, 2`:

```text
1        1
/          \
2            2
```

But they are not the same tree.

If using serialization, include null markers. If using direct recursion, the `None` checks naturally handle this.

#### Pitfall 2: Reading `.val` Before Checking for `None`

This is unsafe:

```python
if p.val != q.val:
    return False
```

If either `p` or `q` is `None`, this crashes.

Handle missing nodes first.

#### Pitfall 3: Using `or` Instead of `and` for Child Comparisons

The two trees are the same only if both sides match:

```python
left_same and right_same
```

Using `or` would incorrectly accept trees where only one subtree matches.

#### Pitfall 4: Checking Values But Not Structure

This is incomplete:

```python
if p and q and p.val == q.val:
    return True
```

Matching root values are not enough. The entire left and right subtree structure must also match.

#### Pitfall 5: Thinking the Problem Requires Tree Balancing or Ordering

This is not a binary search tree problem. There is no ordering rule to use. The left child must match the left child, and the right child must match the right child.

---

### 13. First-Principles Summary

A binary tree is defined recursively, so equality of binary trees is also recursive.

At any position, there are only three meaningful outcomes:

```text
both missing        -> match at this position
one missing         -> shape mismatch
both present        -> values must match, then children must match
```

The recursive contract is:

```text
isSameTree(p, q) tells whether the subtree rooted at p
is identical to the subtree rooted at q.
```

The algorithm works because every call checks exactly one corresponding position and then delegates the same question to the two smaller corresponding child positions.

In short:

```text
same tree = same root + same left subtree + same right subtree
```

That is why DFS recursion is the natural solution.

## Implementation
See `solutions/binary_tree_dfs/p100_same_tree.py`.

## Tests
See `tests/binary_tree_dfs/test_p100_same_tree.py`.

## Examples

### Example 1
- Input: `{'p': [1, 2, 3], 'q': [1, 2, 3]}`
- Output: `True`

### Example 2
- Input: `{'p': [1, 2], 'q': [1, None, 2]}`
- Output: `False`

### Example 3
- Input: `{'p': [1, 2, 1], 'q': [1, 1, 2]}`
- Output: `False`

## Follow-up Practice
- Write the recursive contract in one sentence.
- Trace the smallest tree: empty, one node, then two children.
- Convert the recursion to an explicit stack when useful.
