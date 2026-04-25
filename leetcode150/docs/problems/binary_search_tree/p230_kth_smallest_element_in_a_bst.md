# 230. Kth Smallest Element in a BST

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/kth-smallest-element-in-a-bst/
- Official Group: Binary Search Tree
- Pattern Group: Binary Search Tree
- Patterns: binary-search-tree

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given:

```text
root = the root of a binary search tree
k    = a positive integer
```

Return the `k`th smallest value in the tree.

The tree is not just any binary tree. It is a binary search tree, which means every node divides the values around it:

```text
all values in node.left  < node.val
all values in node.right > node.val
```

So the problem is really asking:

> If all values in this BST were written in increasing order, what value would appear at position `k`?

For example, this tree:

```text
    3
   / \
  1   4
   \
    2
```

contains the values:

```text
1, 2, 3, 4
```

If `k = 1`, the answer is the first value in that sorted order:

```text
1
```

The key is that we do not need to sort the values from scratch if we use the BST property directly.

---

### 2. Start From the Baseline Idea

The most direct approach is to ignore the tree ordering during traversal:

1. Visit every node.
2. Store every value in an array.
3. Sort the array.
4. Return `values[k - 1]`.

Conceptually:

```python
values = []

def collect(node):
    if node is None:
        return
    values.append(node.val)
    collect(node.left)
    collect(node.right)

collect(root)
values.sort()
return values[k - 1]
```

This is correct because sorting all values puts the `k`th smallest value at index `k - 1`.

But it wastes the structure the input already gives us.

Sorting costs extra time, and storing every value costs extra space. A BST already stores the relative ordering of values in its shape, so the better question is:

> Can we read the values from the BST in sorted order without explicitly sorting them?

Yes: use in-order traversal.

---

### 3. The Key Observation: In-Order Traversal Is Sorted

In-order traversal visits a binary tree in this order:

```text
left subtree -> current node -> right subtree
```

For a normal binary tree, that order has no special meaning.

For a BST, it is exactly increasing order.

Why?

At any node:

```text
all left-subtree values are smaller than node.val
all right-subtree values are larger than node.val
```

So if we first list the left subtree, then the node, then the right subtree, we get:

```text
smaller values -> this value -> larger values
```

The same argument applies recursively inside every subtree. Therefore, an in-order traversal of the whole BST visits nodes from smallest to largest.

That changes the problem from sorting to counting:

> Walk through the tree in sorted order and stop when you have visited `k` nodes.

---

### 4. The Traversal Invariant

The algorithm maintains one central invariant:

```text
After visiting some number of nodes in in-order traversal,
those visited nodes are exactly the smallest values seen so far,
in increasing order.
```

So when the traversal visits its first node, that node is the smallest value in the tree.

When it visits its second node, that node is the second smallest value.

In general:

```text
the node visited when count == k is the kth smallest value
```

The traversal order is what makes the counter meaningful. If we used pre-order or post-order traversal, `count == k` would only mean “the kth node visited by that traversal,” not “the kth smallest value.”

---

### 5. Detailed Algorithm

There are two common ways to implement the same idea:

1. Recursive in-order DFS.
2. Iterative in-order traversal with an explicit stack.

The iterative version is often preferred here because it can stop as soon as the `k`th node is reached without relying on nonlocal state.

The stack represents the path of nodes whose left side has been explored or is about to be explored.

Algorithm:

1. Start at `root`.
2. Push nodes while moving left, because the smallest remaining value is always as far left as possible.
3. When there is no more left child, pop the top node.
4. That popped node is the next value in sorted order.
5. Decrease `k` because one more sorted value has been consumed.
6. If `k == 0`, return this node's value.
7. Move to the popped node's right child, because values larger than the popped node may live there.
8. Repeat until the answer is found.

The loop shape is:

```text
while there is a current node or pending stack nodes:
    go left as far as possible
    visit the next smallest node
    go right
```

---

### 6. Why the Stack Works

Consider what happens when we repeatedly move left from a node.

Every step left moves to a smaller value. The leftmost reachable node is therefore the smallest node in that part of the tree.

But while moving left, we cannot forget the ancestors. After finishing an ancestor's left subtree, the ancestor itself must be visited before its right subtree.

The stack stores exactly those postponed ancestors.

For example, if we push:

```text
5, then 3, then 2, then 1
```

the next node to visit is `1`, then maybe `2`, then `3`, then `5`, depending on right subtrees encountered along the way.

That last-in-first-out behavior matches the order needed after walking down the left edge.

---

### 7. Example Walkthrough: `root = [3, 1, 4, None, 2]`, `k = 1`

The tree is:

```text
    3
   / \
  1   4
   \
    2
```

The sorted order should be:

```text
1, 2, 3, 4
```

Start:

```text
current = 3
stack = []
k = 1
```

Move left as far as possible:

```text
push 3, current = 1
push 1, current = None
stack = [3, 1]
```

Now pop:

```text
node = 1
```

This is the first node visited in in-order traversal, so it is the smallest value.

Decrease `k`:

```text
k = 0
```

Because `k == 0`, return:

```text
1
```

The algorithm does not need to visit `2`, `3`, or `4`, because the answer has already been identified.

---

### 8. Example Walkthrough: `root = [5, 3, 6, 2, 4, None, None, 1]`, `k = 3`

The tree is:

```text
        5
       / \
      3   6
     / \
    2   4
   /
  1
```

The sorted order is:

```text
1, 2, 3, 4, 5, 6
```

So the third smallest value should be `3`.

Trace the iterative traversal:

```text
current = 5, stack = [], k = 3
```

Move left:

```text
push 5
push 3
push 2
push 1
current = None
stack = [5, 3, 2, 1]
```

Pop and visit `1`:

```text
k = 2
current = 1.right = None
```

Pop and visit `2`:

```text
k = 1
current = 2.right = None
```

Pop and visit `3`:

```text
k = 0
```

Now `3` is the third node visited in sorted order, so return:

```text
3
```

Notice that the traversal stops before visiting `4`, `5`, or `6`.

---

### 9. Code

Iterative in-order traversal:

```python
def kthSmallest(root, k):
    stack = []
    current = root

    while current is not None or stack:
        while current is not None:
            stack.append(current)
            current = current.left

        current = stack.pop()
        k -= 1

        if k == 0:
            return current.val

        current = current.right
```

Recursive in-order traversal expresses the same ordering idea:

```python
def kthSmallest(root, k):
    values_seen = 0
    answer = None

    def inorder(node):
        nonlocal values_seen, answer
        if node is None or answer is not None:
            return

        inorder(node.left)

        values_seen += 1
        if values_seen == k:
            answer = node.val
            return

        inorder(node.right)

    inorder(root)
    return answer
```

The iterative version avoids shared recursive state, but both rely on the same invariant: in-order traversal of a BST visits values in increasing order.

---

### 10. Correctness

We prove that the algorithm returns the `k`th smallest value.

First, in a BST, every value in a node's left subtree is smaller than the node's value, and every value in its right subtree is larger than the node's value.

An in-order traversal visits:

```text
left subtree -> node -> right subtree
```

By the BST property, this means it visits all smaller values before the node and all larger values after the node. Applying this reasoning recursively to every subtree, the full in-order traversal visits all tree values in increasing order.

The iterative algorithm exactly simulates this in-order traversal:

- The inner loop pushes a node and all reachable left descendants, postponing each node until its left subtree has been handled.
- Popping from the stack visits the next node whose left side is complete.
- Moving to the right child then handles values larger than the popped node but still within its subtree.

Therefore, each time the algorithm pops a node, it has visited the next value in increasing order.

The algorithm decreases `k` once per popped node. When `k` becomes `0`, exactly the original `k` nodes have been visited in increasing order. The current node is therefore the original `k`th smallest value, and returning its value is correct.

---

### 11. Complexity

Let `h` be the height of the tree.

- Time: `O(h + k)` if the traversal stops at the `k`th node, because it only walks down paths and visits the first `k` sorted nodes. In the worst case, this is `O(n)`.
- Space: `O(h)` for the stack. A balanced tree has `h = O(log n)`, while a completely skewed tree has `h = O(n)`.

The baseline collect-and-sort approach would take `O(n log n)` time and `O(n)` extra space, so using the BST ordering is a real improvement.

---

### 12. Common Pitfalls

- Using pre-order traversal and counting visits. Pre-order does not produce sorted values in a BST.
- Forgetting that `k` is 1-indexed. The first smallest value corresponds to `k == 1`, not index `0` during traversal.
- Continuing the traversal after finding the answer. Once `k == 0`, the answer is known.
- Assuming the tree is balanced. The stack can grow to `O(n)` in a skewed tree.
- Sorting unnecessarily. Sorting works, but it ignores the BST property and costs more.
- Mutating shared recursive state without resetting it between calls. This is especially easy to do if `count` or `answer` is stored on `self`.
- Confusing the BST rule with heap order. A BST does not guarantee the root is the smallest value; the smallest value is the leftmost node.

---

### 13. First-Principles Summary

The BST property gives each node a local ordering guarantee:

```text
left values < node value < right values
```

In-order traversal turns that local guarantee into a global sorted sequence:

```text
smallest, second smallest, third smallest, ...
```

Once the tree can be read as a sorted stream, the problem becomes simple counting:

```text
visit next sorted value
decrease k
return when k reaches 0
```

So the solution is not “search for the value” in the usual binary-search sense. It is “generate the BST values in sorted order, but stop as soon as the requested rank is reached.”

## Implementation
See `solutions/binary_search_tree/p230_kth_smallest_element_in_a_bst.py`.

## Tests
See `tests/binary_search_tree/test_p230_kth_smallest_element_in_a_bst.py`.

## Examples

### Example 1
- Input: `{'root': [3, 1, 4, None, 2], 'k': 1}`
- Output: `1`

### Example 2
- Input: `{'root': [5, 3, 6, 2, 4, None, None, 1], 'k': 3}`
- Output: `3`

## Follow-up Practice
- Solve the same task recursively and iteratively.
- Trace a case where the answer is in a right subtree, such as `k = 4` in the second example.
- Compare the early-stopping traversal with the collect-and-sort baseline.
