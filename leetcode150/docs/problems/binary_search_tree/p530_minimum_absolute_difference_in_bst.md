# 530. Minimum Absolute Difference in BST

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/minimum-absolute-difference-in-bst/
- Official Group: Binary Search Tree
- Pattern Group: Binary Search Tree
- Patterns: binary-search-tree

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given the root of a binary search tree, return the minimum absolute difference between the values of any two different nodes.

For example, in this tree:

```text
        4
       / \
      2   6
     / \
    1   3
```

The node values are:

```text
4, 2, 6, 1, 3
```

All pairwise absolute differences include:

```text
|4 - 2| = 2
|4 - 6| = 2
|4 - 1| = 3
|4 - 3| = 1
|2 - 6| = 4
|2 - 1| = 1
|2 - 3| = 1
|6 - 1| = 5
|6 - 3| = 3
|1 - 3| = 2
```

The smallest difference is:

```text
1
```

So the real problem is:

> Among all pairs of node values in the BST, find the pair whose values are closest together.

The phrase **absolute difference** means the order of the pair does not matter:

```text
|a - b| == |b - a|
```

So we only care about distance on the number line.

---

### 2. Start From the Brute Force Idea

The most direct approach ignores the BST property completely:

1. Traverse the whole tree and collect every value into a list.
2. Compare every pair of values.
3. Keep the smallest absolute difference.

Conceptually:

```python
values = all node values
best = infinity

for i in range(len(values)):
    for j in range(i + 1, len(values)):
        best = min(best, abs(values[i] - values[j]))
```

This is correct because it checks every possible pair.

But it is inefficient.

If there are `n` nodes, there are about:

```text
n * (n - 1) / 2
```

pairs, so the pair comparison costs `O(n^2)` time.

The question is:

> Can the BST ordering tell us which pairs are even worth comparing?

Yes.

---

### 3. The Key Number-Line Observation

Forget trees for a moment.

Suppose the values are just numbers:

```text
[4, 2, 6, 1, 3]
```

If the numbers are unsorted, it is not obvious which pair is closest.

Now sort them:

```text
[1, 2, 3, 4, 6]
```

The closest pair must be next to each other in this sorted order.

Why?

Take any two non-adjacent sorted values:

```text
a ... b ... c
```

where `a < b < c`.

Then:

```text
c - a = (b - a) + (c - b)
```

Both pieces are non-negative, so `c - a` cannot be smaller than both neighboring gaps. At least one adjacent gap inside that interval is no larger than the non-adjacent gap.

So a non-adjacent pair cannot be the uniquely necessary pair to check. The minimum difference in a sorted list is found by scanning adjacent values only:

```text
min(values[i] - values[i - 1]) for i from 1 to n - 1
```

This changes the problem from:

```text
compare every pair
```

to:

```text
visit values in sorted order and compare each value with the previous value
```

---

### 4. Why a BST Gives Sorted Order

A binary search tree has this rule at every node:

```text
all values in the left subtree  < node.val
all values in the right subtree > node.val
```

So if we traverse the tree in this order:

```text
left subtree -> node -> right subtree
```

we see the values from smallest to largest.

This is called **in-order traversal**.

For the tree:

```text
        4
       / \
      2   6
     / \
    1   3
```

In-order traversal visits:

```text
1, 2, 3, 4, 6
```

That is exactly the sorted order of the BST values.

So we do not need to collect all values and sort them. The tree already stores enough ordering information. We only need to read it in the right order.

---

### 5. The Traversal Invariant

During in-order traversal, maintain two pieces of state:

```text
previous = the value visited immediately before the current node in sorted order
best     = the smallest adjacent sorted-order gap seen so far
```

The invariant is:

```text
Before processing the current node, previous is the greatest value already visited,
and best is the minimum difference among all adjacent visited values so far.
```

When the current node is visited, its value is the next value in sorted order.

Therefore, the only new adjacent gap introduced by this node is:

```text
current.val - previous
```

if `previous` exists.

Then update:

```text
best = min(best, current.val - previous)
previous = current.val
```

That is the whole algorithm.

The important point is that we are not comparing the current node with its parent, its children, or every earlier node. We compare it only with the immediately previous in-order value, because only adjacent sorted values can produce the minimum difference.

---

### 6. Detailed Algorithm

Use a depth-first in-order traversal.

1. Initialize:

```text
previous = none
best = infinity
```

2. Traverse the left subtree.

3. Process the current node:

```text
if previous exists:
    best = min(best, node.val - previous)
previous = node.val
```

4. Traverse the right subtree.

5. Return `best` after the traversal finishes.

Because LeetCode guarantees the tree has at least two nodes for this problem, `best` will be updated at least once.

---

### 7. Example Walkthrough: `[4, 2, 6, 1, 3]`

Tree:

```text
        4
       / \
      2   6
     / \
    1   3
```

Start:

```text
previous = none
best = infinity
```

In-order traversal first goes as far left as possible.

#### Visit `1`

There is no previous value yet.

```text
previous = 1
best = infinity
```

No difference can be computed until we have two values.

#### Visit `2`

The previous sorted value is `1`.

```text
gap = 2 - 1 = 1
best = min(infinity, 1) = 1
previous = 2
```

#### Visit `3`

The previous sorted value is `2`.

```text
gap = 3 - 2 = 1
best = min(1, 1) = 1
previous = 3
```

#### Visit `4`

The previous sorted value is `3`.

```text
gap = 4 - 3 = 1
best = min(1, 1) = 1
previous = 4
```

#### Visit `6`

The previous sorted value is `4`.

```text
gap = 6 - 4 = 2
best = min(1, 2) = 1
previous = 6
```

Traversal is finished.

Final answer:

```text
1
```

Notice that the closest values are discovered naturally by adjacent in-order comparisons. We never explicitly compare `4` with `3` because they happen to be parent/child. We compare them because they are consecutive values in sorted order.

---

### 8. Example Walkthrough: `[1, 0, 48, None, None, 12, 49]`

Tree:

```text
        1
       / \
      0   48
         /  \
        12  49
```

In-order traversal visits:

```text
0, 1, 12, 48, 49
```

Track adjacent gaps:

```text
1 - 0   = 1
12 - 1  = 11
48 - 12 = 36
49 - 48 = 1
```

The minimum is:

```text
1
```

So the output is `1`.

---

### 9. Code

Recursive version:

```python
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        previous = None
        best = float("inf")

        def inorder(node: Optional[TreeNode]) -> None:
            nonlocal previous, best

            if node is None:
                return

            inorder(node.left)

            if previous is not None:
                best = min(best, node.val - previous)
            previous = node.val

            inorder(node.right)

        inorder(root)
        return best
```

Iterative version:

```python
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        stack = []
        node = root
        previous = None
        best = float("inf")

        while node is not None or stack:
            while node is not None:
                stack.append(node)
                node = node.left

            node = stack.pop()

            if previous is not None:
                best = min(best, node.val - previous)
            previous = node.val

            node = node.right

        return best
```

Both versions implement the same idea. The recursive version lets the call stack remember where to return after finishing a left subtree. The iterative version stores that path explicitly in `stack`.

---

### 10. Why This Code Is Correct

We prove the recursive algorithm correct; the iterative algorithm visits nodes in the same in-order sequence, so the same reasoning applies.

First, in-order traversal of a BST visits node values in strictly increasing sorted order. This follows directly from the BST rule: every value in the left subtree is smaller than the node, and every value in the right subtree is larger than the node.

Second, the algorithm maintains this invariant:

```text
After processing each visited node, previous is that node's value,
and best is the smallest difference between adjacent visited values so far.
```

At the first visited node, there is no previous value, so no pair has been formed yet. Setting `previous` to that value makes the invariant true.

For every later visited node, the traversal order guarantees that `previous` is the immediately preceding value in sorted order. Therefore, the gap:

```text
node.val - previous
```

is exactly the new adjacent sorted-order difference introduced by visiting this node.

The algorithm compares that gap with `best`, so after the update, `best` is still the smallest adjacent gap among all values visited so far. Then it sets `previous` to the current value, preparing the invariant for the next node.

Finally, after all nodes are visited, every adjacent pair in sorted order has been considered exactly once.

The minimum absolute difference among any two values in a sorted list must occur between adjacent values. Since the algorithm computes the minimum over all adjacent sorted-order gaps, the returned `best` is the minimum absolute difference between any two nodes in the BST.

---

### 11. Complexity

Let `n` be the number of nodes and `h` be the height of the tree.

Each node is visited once, and each visit does constant work:

```text
Time: O(n)
```

The extra space depends on the traversal stack.

For recursion, the call stack can contain one path from the root to a leaf:

```text
Space: O(h)
```

For the iterative version, the explicit stack stores the same kind of path:

```text
Space: O(h)
```

If the tree is balanced, `h = O(log n)`. If the tree is completely skewed, `h = O(n)`.

---

### 12. Common Pitfalls

#### Comparing Only Parent and Child Nodes

The closest values are not necessarily connected by an edge.

For example:

```text
      10
     /
    5
     \
      9
```

The minimum difference is:

```text
10 - 9 = 1
```

Nodes `9` and `10` are not parent and direct child in this drawing. They are adjacent in sorted order, which is what matters.

#### Forgetting to Use In-Order Traversal

Pre-order traversal visits:

```text
node -> left -> right
```

Post-order traversal visits:

```text
left -> right -> node
```

Neither one guarantees sorted values in a BST.

The adjacent-gap idea only works when the values are visited in sorted order:

```text
left -> node -> right
```

#### Taking `abs(node.val - previous)` Unnecessarily

In a valid BST in-order traversal, values appear in increasing order, so:

```text
node.val - previous >= 0
```

Using `abs` is harmless, but it can hide a traversal-order mistake. If the traversal is correct, subtraction in this direction is enough.

#### Resetting State Incorrectly

`previous` and `best` belong to one complete traversal.

If they are stored as object attributes, they must be reset every time `getMinimumDifference` is called. Otherwise, a second call on the same `Solution` object may accidentally reuse state from a previous tree.

Using local variables with `nonlocal`, as shown above, avoids that problem.

#### Assuming Balanced Height

The algorithm is `O(n)` time for every BST shape, but the stack space is `O(h)`, not always `O(log n)`.

A skewed tree like:

```text
1
 \
  2
   \
    3
     \
      4
```

has height `n`, so recursive depth can also be `n`.

---

### 13. First-Principles Summary

The problem asks for the two closest node values, not for a tree-specific relationship such as closest parent-child pair.

Closest values are easiest to reason about on a sorted number line. In sorted order, the minimum difference must occur between adjacent values, because any non-adjacent pair has at least one value between them and therefore contains a no-larger adjacent gap inside the interval.

A BST gives sorted order for free through in-order traversal.

So the full solution is:

```text
Use in-order traversal to stream values from smallest to largest.
Remember the previous streamed value.
For each current value, update the answer with current - previous.
Return the smallest gap seen.
```

The traversal invariant is the bridge between the tree and the number-line argument: at every node, `previous` is the immediately preceding sorted value, so comparing only with `previous` is both sufficient and complete.

## Implementation
See `solutions/binary_search_tree/p530_minimum_absolute_difference_in_bst.py`.

## Tests
See `tests/binary_search_tree/test_p530_minimum_absolute_difference_in_bst.py`.

## Examples

### Example 1
- Input: `{'root': [4, 2, 6, 1, 3]}`
- Output: `1`

### Example 2
- Input: `{'root': [1, 0, 48, None, None, 12, 49]}`
- Output: `1`

## Follow-up Practice
- Solve the same task recursively and iteratively.
- Trace a case where the closest pair is not a parent-child pair.
- Explain why adjacent values in sorted order are sufficient.
