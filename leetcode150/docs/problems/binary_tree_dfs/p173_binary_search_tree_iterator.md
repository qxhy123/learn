# 173. Binary Search Tree Iterator

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/binary-search-tree-iterator/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given the root of a binary search tree, and you must design an object called `BSTIterator` with two operations:

```text
next()    -> return the next smallest value
hasNext() -> return whether another value exists
```

The important detail is that this is an **iterator**, not a one-time function.

A normal traversal function could return all values at once:

```text
[3, 7, 9, 15, 20]
```

But an iterator must reveal those values one by one:

```text
next() -> 3
next() -> 7
hasNext() -> true
next() -> 9
...
```

So the problem is asking:

> How can we output the BST values in sorted order, one call at a time, while remembering exactly where the traversal should continue?

Because the input is a **binary search tree**, its values are ordered by structure:

```text
all values in left subtree < node.val < all values in right subtree
```

For a BST, an **in-order traversal** visits values in ascending order:

```text
left subtree -> node -> right subtree
```

Therefore, the iterator does not need to invent a new ordering rule. It needs to simulate in-order traversal across many separate method calls.

### 2. Baseline: Flatten the Whole Tree First

The simplest correct idea is to perform a full in-order traversal in the constructor and store every value in an array.

For example, for this tree:

```text
      7
     / \
    3   15
       /  \
      9   20
```

The constructor could build:

```text
values = [3, 7, 9, 15, 20]
index = 0
```

Then:

```python
next():
    value = values[index]
    index += 1
    return value

hasNext():
    return index < len(values)
```

This is easy to reason about:

- `next()` is `O(1)`.
- `hasNext()` is `O(1)`.
- The constructor is `O(n)`.
- The memory usage is `O(n)` because every value is stored.

That baseline is often accepted in spirit, but it misses the main iterator idea. If the caller only asks for the first two values of a huge tree, flattening the entire tree did unnecessary work and used unnecessary memory.

The deeper question is:

> Can we remember only the unfinished part of the in-order traversal?

### 3. The Key Observation: The Next Value Is the Leftmost Unvisited Node

In-order traversal says:

```text
left subtree -> node -> right subtree
```

So before visiting a node, we must visit everything smaller than it in its left subtree.

That means the smallest unvisited value is found by repeatedly going left from the current area of the tree.

At the very beginning, the next value is the leftmost node from `root`.

For the example tree:

```text
      7
     / \
    3   15
       /  \
      9   20
```

Start at `7`, go left to `3`, and stop. The first value is `3`.

But after returning `3`, the traversal must continue back to `7`. After returning `7`, it must enter `7`'s right subtree and again find the leftmost node there, which is `9`.

This is exactly what recursion normally remembers for us: the path of ancestors whose left sides have been explored but whose own values or right sides may still be pending.

Because an iterator cannot keep a recursive call stack alive between `next()` calls, we store that stack explicitly.

### 4. The Stack / Traversal Invariant

The iterator maintains a stack of tree nodes.

The invariant is:

```text
The top of the stack is always the next node that should be returned.
Every node below it is an ancestor that must be returned later,
after the nodes above it have been processed.
```

Equivalently:

```text
The stack contains a path of pending nodes.
For each node on the stack, its left subtree has already been fully scheduled or visited,
but the node itself has not been returned yet.
```

The helper operation is:

```text
push all left descendants from a node
```

If we call this helper on `root`, the stack becomes the path to the smallest value.

For the example tree:

```text
pushLeft(7):
    push 7
    move to 3
    push 3
    move to null
```

The stack is:

```text
bottom [7, 3] top
```

The top is `3`, which is the next smallest value.

This invariant gives both iterator operations their meaning:

- `hasNext()` checks whether the stack is non-empty.
- `next()` pops the top node, returns it, and then schedules that node's right subtree by pushing its left spine.

### 5. Why the Right Subtree Is Scheduled After Popping

Suppose `next()` pops a node `x`.

At that moment, `x` is the smallest unvisited node, so returning `x.val` is correct.

What values can come after `x`?

There are two possibilities:

1. Values inside `x.right`.
2. Ancestors already waiting in the stack.

In in-order traversal, after visiting `x`, we must visit the leftmost values of `x.right` before returning to any larger ancestor.

So if `x.right` exists, we do:

```text
pushLeft(x.right)
```

That places the smallest node in `x.right` on top of the stack. Since every value in `x.right` is greater than `x` but still less than the first ancestor for which `x` was in the left side, this preserves sorted order.

If `x.right` does not exist, there is nothing new to schedule. The next value is simply the previous ancestor already on the stack.

### 6. Detailed Algorithm

The iterator stores one field:

```text
stack
```

The constructor initializes the iterator by pushing the left spine from the root:

```text
BSTIterator(root):
    stack = []
    pushLeft(root)
```

The helper is:

```text
pushLeft(node):
    while node is not null:
        stack.push(node)
        node = node.left
```

The `next()` operation is:

```text
next():
    node = stack.pop()
    answer = node.val
    pushLeft(node.right)
    return answer
```

The `hasNext()` operation is:

```text
hasNext():
    return stack is not empty
```

The algorithm is lazy because it does not traverse a subtree until the iterator reaches the point where that subtree can contain the next output value.

### 7. Walkthrough on the Official Example

Input operations:

```text
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7,3,15,null,null,9,20]], [], [], [], [], [], [], [], [], []]
```

The tree is:

```text
      7
     / \
    3   15
       /  \
      9   20
```

#### Constructor

Call `pushLeft(7)`:

```text
push 7
push 3
```

Stack:

```text
bottom [7, 3] top
```

#### First `next()`

Pop `3`.

```text
return 3
```

Node `3` has no right child, so nothing is pushed.

Stack:

```text
bottom [7] top
```

#### Second `next()`

Pop `7`.

```text
return 7
```

Node `7` has right child `15`, so call `pushLeft(15)`:

```text
push 15
push 9
```

Stack:

```text
bottom [15, 9] top
```

#### First `hasNext()`

The stack is not empty.

```text
return true
```

#### Third `next()`

Pop `9`.

```text
return 9
```

Node `9` has no right child.

Stack:

```text
bottom [15] top
```

#### Second `hasNext()`

The stack is not empty.

```text
return true
```

#### Fourth `next()`

Pop `15`.

```text
return 15
```

Node `15` has right child `20`, so call `pushLeft(20)`:

```text
push 20
```

Stack:

```text
bottom [20] top
```

#### Third `hasNext()`

The stack is not empty.

```text
return true
```

#### Fifth `next()`

Pop `20`.

```text
return 20
```

Node `20` has no right child.

Stack:

```text
[]
```

#### Final `hasNext()`

The stack is empty.

```text
return false
```

The returned outputs are therefore:

```text
[null, 3, 7, true, 9, true, 15, true, 20, false]
```

The constructor output is `null` because object construction itself does not return a tree value.

### 8. Python-Style Code

LeetCode's interface defines a class rather than a single `Solution` method. The core logic looks like this:

```python
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        self._push_left(root)

    def _push_left(self, node: Optional[TreeNode]) -> None:
        while node is not None:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        node = self.stack.pop()
        self._push_left(node.right)
        return node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0
```

The order inside `next()` matters only in the sense that the returned value must come from the popped node. You may save `node.val`, push the right subtree, and then return the saved value; or push first and then return `node.val`. The popped node remains available in the local variable either way.

### 9. Correctness

We prove that the iterator returns exactly the BST values in ascending order.

#### Lemma 1: After `pushLeft(node)`, the top of the stack is the smallest node in that subtree that has just been scheduled.

`pushLeft(node)` pushes `node`, then `node.left`, then `node.left.left`, and so on until there is no left child. In a BST, repeatedly moving left always moves to a smaller value. The final pushed node is therefore the leftmost node in that subtree, which is the smallest value in that subtree. Since it was pushed last, it is on top of the stack.

#### Lemma 2: Before every call to `next()`, if the stack is non-empty, the top of the stack is the smallest unvisited node in the whole tree.

Initially, the constructor calls `pushLeft(root)`, so by Lemma 1 the smallest node in the tree is on top.

Now assume the lemma is true before a call to `next()`. Let `x` be the node popped from the stack. By the assumption, `x` is the smallest unvisited node, so returning `x.val` is correct.

After `x` is returned, the only newly available nodes are in `x.right`, because in-order traversal visits a node's right subtree immediately after the node itself. If `x.right` exists, `pushLeft(x.right)` places the smallest node of that right subtree on top of the stack. If `x.right` does not exist, the next candidate is the nearest pending ancestor already on the stack.

In both cases, the stack top becomes the smallest remaining unvisited node. Therefore the invariant is preserved.

#### Lemma 3: `hasNext()` returns true exactly when an unvisited node remains.

The stack stores precisely the pending nodes needed to continue traversal. If it is non-empty, the top node can be returned next. If it is empty, there is no pending node and no unscheduled subtree left, because every right subtree is scheduled immediately after its parent is popped.

#### Theorem: The iterator returns all values in ascending order and stops exactly after the last value.

By Lemma 2, every `next()` call returns the smallest currently unvisited node. Repeating this produces values in ascending order. By Lemma 3, `hasNext()` is true exactly while such a node exists. Therefore the iterator is correct.

### 10. Complexity

Let `n` be the number of nodes in the tree and `h` be the tree height.

#### Constructor

The constructor pushes the left spine from the root.

```text
Time:  O(h)
Space: O(h)
```

#### `next()`

One `next()` call pops one node and may push the left spine of that node's right subtree. A single call can therefore cost `O(h)` in the worst case.

However, across the full lifetime of the iterator, each node is pushed once and popped once.

So the amortized cost is:

```text
Amortized time per next(): O(1)
Worst-case time per next(): O(h)
```

#### `hasNext()`

`hasNext()` only checks whether the stack is empty.

```text
Time: O(1)
```

#### Total Space

At any moment, the stack contains a path of ancestors plus possibly a left spine from a right subtree. Its size is bounded by the tree height.

```text
Space: O(h)
```

For a balanced tree, `h = O(log n)`. For a completely skewed tree, `h = O(n)`.

### 11. Common Pitfalls

- **Flattening when the interviewer expects laziness.** Flattening is simple, but it uses `O(n)` memory instead of `O(h)`.
- **Pushing right children directly without their left descendants.** After visiting a node, the next value in its right subtree is not necessarily the right child; it is the leftmost node inside that right subtree.
- **Forgetting to initialize with the root's left spine.** If the constructor pushes only `root`, then `next()` might return the root before smaller left-subtree values.
- **Using pre-order or post-order by accident.** BST sorted order comes specifically from in-order traversal: left, node, right.
- **Making `hasNext()` advance the iterator.** `hasNext()` should only inspect state. It should not pop, push, or consume values.
- **Assuming the tree is balanced.** The height `h` can be `n`, so worst-case stack space can be linear.
- **Calling `next()` when no next value exists.** LeetCode usually guarantees calls are valid, but production code would guard against an empty stack.

### 12. First-Principles Summary

A BST iterator is a paused in-order traversal.

In-order traversal gives sorted values because every node sits between its smaller left subtree and larger right subtree. The only challenge is that an iterator must pause after each returned value and later resume from the exact right place.

The explicit stack is the paused recursion stack. It stores the path of pending nodes whose left sides have already been handled but whose own values still need to be returned. The top of the stack is always the next smallest unvisited node.

The whole algorithm follows from one rule:

```text
When you arrive at a subtree, push its entire left spine.
```

Do that once in the constructor for the root. Then, every time `next()` pops a node, do it again for that node's right subtree. This lazily schedules exactly the nodes that can become next, no more and no less.

## Implementation
See `solutions/binary_tree_dfs/p173_binary_search_tree_iterator.py`.

## Tests
See `tests/binary_tree_dfs/test_p173_binary_search_tree_iterator.py`.

## Examples

### Example 1
- Input: `{'raw': '["BSTIterator","next","next","hasNext","next","hasNext","next","hasNext","next","hasNext"]\n[[[7,3,15,null,null,9,20]],[],[],[],[],[],[],[],[],[]]'}`
- Output: `'See official examples'`

The official operation sequence corresponds to this result:

```text
[null, 3, 7, true, 9, true, 15, true, 20, false]
```
