# 105. Construct Binary Tree from Preorder and Inorder Traversal

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
- Official Group: Binary Tree General
- Pattern Group: Binary Tree DFS
- Patterns: binary-tree-dfs, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given two arrays that describe the same binary tree:

```text
preorder = root, then everything in the left subtree, then everything in the right subtree
inorder  = everything in the left subtree, then root, then everything in the right subtree
```

The values are unique, so each value identifies exactly one node.

The task is to rebuild the original binary tree and return its root.

For example:

```text
preorder = [3, 9, 20, 15, 7]
inorder  = [9, 3, 15, 20, 7]
```

The first preorder value is `3`, so the root must be `3`.

In the inorder array, `3` appears here:

```text
[9] 3 [15, 20, 7]
```

Everything left of `3` belongs to the left subtree, and everything right of `3` belongs to the right subtree.

So the tree starts as:

```text
    3
   / \
  ?   ?
```

The left subtree uses inorder values `[9]`, so it must be the single node `9`.

The right subtree uses inorder values `[15, 20, 7]`. The next unused preorder value after finishing the left subtree is `20`, so `20` is the root of the right subtree:

```text
[15] 20 [7]
```

That gives:

```text
    3
   / \
  9   20
     /  \
    15   7
```

The real problem is:

> Use preorder to discover each subtree root, and use inorder to determine which values belong to that root's left and right subtrees.

---

### 2. Why One Traversal Is Not Enough

A preorder traversal alone tells us the order in which roots are visited:

```text
root -> left subtree -> right subtree
```

But it does not tell us where the left subtree ends.

For example, this preorder traversal:

```text
[1, 2, 3]
```

could represent different trees:

```text
    1          1          1
   /            \        /
  2              2      2
 /                \      \
3                  3      3
```

All of them have preorder `[1, 2, 3]`.

An inorder traversal alone has the opposite problem. It tells us relative left-root-right ordering, but not which value should be chosen as the top root first.

Together, the two traversals remove the ambiguity:

```text
preorder gives the next root
inorder gives the boundary between left and right subtrees
```

---

### 3. Start From the Brute Force Idea

A direct recursive approach is:

1. The first element of `preorder` is the root.
2. Find that root inside `inorder` by scanning.
3. Split `inorder` into left and right parts.
4. Split the remaining preorder values into left and right parts of matching sizes.
5. Recursively build both subtrees.

Conceptually:

```python
def build(preorder, inorder):
    if not preorder:
        return None

    root_value = preorder[0]
    root_index = inorder.index(root_value)

    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index + 1:]

    left_size = len(left_inorder)
    left_preorder = preorder[1:1 + left_size]
    right_preorder = preorder[1 + left_size:]

    root = TreeNode(root_value)
    root.left = build(left_preorder, left_inorder)
    root.right = build(right_preorder, right_inorder)
    return root
```

This is easy to understand and correct, but it is inefficient.

The expensive parts are:

```text
inorder.index(root_value)   # scans a list
inorder[:root_index]        # copies a list
preorder[1:1 + left_size]   # copies a list
```

Across all recursive calls, repeated scans and slices can degrade to `O(n^2)` time and `O(n^2)` extra copying in skewed trees.

The first-principles improvement is not to change the logic. The logic is already right.

The improvement is:

> Keep the same subtree boundaries, but represent them with indices instead of copied arrays.

---

### 4. The Key Observation

For any subtree, two facts fully determine how to build it:

```text
1. Which part of inorder belongs to this subtree.
2. Which preorder value is the root of this subtree.
```

The inorder range gives the set of values in the subtree.

If a subtree occupies:

```text
inorder[in_left : in_right + 1]
```

then its root is the next root encountered in preorder.

Once we know the root value, we can find its position in inorder:

```text
root_index = position of root_value in inorder
```

Then:

```text
inorder[in_left : root_index]       belongs to the left subtree
inorder[root_index + 1 : in_right]  belongs to the right subtree
```

The number of left-subtree nodes is:

```text
left_size = root_index - in_left
```

That size tells us how to split preorder:

```text
preorder[pre_left] is the root
preorder[pre_left + 1 : pre_left + left_size] are the left subtree roots/descendants
preorder[pre_left + left_size + 1 : ...] are the right subtree roots/descendants
```

So we never need to copy arrays. We only need to pass index ranges.

---

### 5. The Recursive Contract

Define a recursive function with this contract:

```text
build(pre_left, in_left, size)
```

returns the root of the subtree that:

```text
uses size nodes,
starts at preorder index pre_left,
and uses inorder indices in_left through in_left + size - 1.
```

This contract is the heart of the solution.

It says exactly which nodes the recursive call owns. It also prevents accidental overlap between subtrees.

If `size == 0`, the subtree has no nodes:

```text
return None
```

Otherwise:

```text
root_value = preorder[pre_left]
```

Find the root in inorder:

```text
root_index = inorder_index[root_value]
```

Compute how many nodes belong to the left subtree:

```text
left_size = root_index - in_left
```

Compute how many nodes belong to the right subtree:

```text
right_size = size - left_size - 1
```

Then the recursive calls are forced:

```text
left subtree:
  preorder starts at pre_left + 1
  inorder starts at in_left
  size is left_size

right subtree:
  preorder starts at pre_left + 1 + left_size
  inorder starts at root_index + 1
  size is right_size
```

There is no guessing. The traversal definitions determine every index.

---

### 6. Why the Right Subtree Starts After the Left Subtree in Preorder

This is the most common place to make an off-by-one mistake.

Preorder for one subtree is ordered like this:

```text
[root] [all left subtree nodes] [all right subtree nodes]
```

If the current root is at:

```text
pre_left
```

then the left subtree starts immediately after it:

```text
pre_left + 1
```

If the left subtree has `left_size` nodes, those nodes occupy:

```text
pre_left + 1 through pre_left + left_size
```

Therefore the right subtree starts at:

```text
pre_left + left_size + 1
```

That is why the right call uses:

```text
build(pre_left + left_size + 1, root_index + 1, right_size)
```

---

### 7. Detailed Algorithm

1. Build a dictionary from value to inorder index:

```python
inorder_index = {value: index for index, value in enumerate(inorder)}
```

This changes root lookup from `O(n)` scanning to `O(1)` dictionary access.

2. Define a helper:

```text
build(pre_left, in_left, size)
```

3. If `size == 0`, return `None`.

4. Read the root value from preorder:

```python
root_value = preorder[pre_left]
```

5. Create the root node.

6. Find the root's inorder index:

```python
root_index = inorder_index[root_value]
```

7. Compute subtree sizes:

```python
left_size = root_index - in_left
right_size = size - left_size - 1
```

8. Recursively build the left subtree:

```python
root.left = build(pre_left + 1, in_left, left_size)
```

9. Recursively build the right subtree:

```python
root.right = build(pre_left + left_size + 1, root_index + 1, right_size)
```

10. Return the root.

The initial call is:

```python
build(0, 0, len(preorder))
```

---

### 8. Pseudocode

```python
def buildTree(preorder, inorder):
    inorder_index = {}
    for index, value in enumerate(inorder):
        inorder_index[value] = index

    def build(pre_left, in_left, size):
        if size == 0:
            return None

        root_value = preorder[pre_left]
        root = TreeNode(root_value)

        root_index = inorder_index[root_value]
        left_size = root_index - in_left
        right_size = size - left_size - 1

        root.left = build(
            pre_left + 1,
            in_left,
            left_size,
        )

        root.right = build(
            pre_left + left_size + 1,
            root_index + 1,
            right_size,
        )

        return root

    return build(0, 0, len(preorder))
```

Some implementations instead pass four boundaries:

```text
pre_left, pre_right, in_left, in_right
```

That is also valid. The `size` version is often easier to reason about because each call owns exactly `size` nodes.

---

### 9. Walkthrough: `preorder = [3, 9, 20, 15, 7]`, `inorder = [9, 3, 15, 20, 7]`

First build the index map:

```text
value -> inorder index
9  -> 0
3  -> 1
15 -> 2
20 -> 3
7  -> 4
```

Initial call:

```text
build(pre_left = 0, in_left = 0, size = 5)
```

This call owns all nodes:

```text
preorder segment starts at: [3, 9, 20, 15, 7]
inorder segment:           [9, 3, 15, 20, 7]
```

The root is:

```text
preorder[0] = 3
```

In inorder, `3` is at index `1`:

```text
[9] 3 [15, 20, 7]
```

So:

```text
left_size = 1 - 0 = 1
right_size = 5 - 1 - 1 = 3
```

Create:

```text
3
```

Build the left subtree:

```text
build(pre_left = 1, in_left = 0, size = 1)
```

This call owns:

```text
preorder starts at: [9, ...]
inorder segment:   [9]
```

Root:

```text
preorder[1] = 9
```

In inorder, `9` is at index `0`:

```text
[] 9 []
```

So:

```text
left_size = 0
right_size = 0
```

The node `9` has no children.

Build the right subtree of `3`:

```text
build(pre_left = 2, in_left = 2, size = 3)
```

This call owns:

```text
preorder starts at: [20, 15, 7]
inorder segment:   [15, 20, 7]
```

Root:

```text
preorder[2] = 20
```

In inorder, `20` is at index `3`:

```text
[15] 20 [7]
```

So:

```text
left_size = 3 - 2 = 1
right_size = 3 - 1 - 1 = 1
```

Build the left child of `20`:

```text
build(pre_left = 3, in_left = 2, size = 1)
```

Root:

```text
preorder[3] = 15
```

It has no children.

Build the right child of `20`:

```text
build(pre_left = 4, in_left = 4, size = 1)
```

Root:

```text
preorder[4] = 7
```

It has no children.

The final tree is:

```text
    3
   / \
  9   20
     /  \
    15   7
```

Serialized in level-order form, this is:

```text
[3, 9, 20, None, None, 15, 7]
```

---

### 10. Correctness

We prove that `build(pre_left, in_left, size)` returns exactly the subtree described by its contract.

#### Base Case

If `size == 0`, the subtree contains no values.

The algorithm returns `None`, which is exactly the correct tree for an empty subtree.

#### Recursive Step

Assume `size > 0`.

By preorder definition, the first value in a subtree's preorder segment is the root of that subtree. The helper reads:

```text
root_value = preorder[pre_left]
```

So it chooses the correct root.

By inorder definition, all values left of the root inside the subtree's inorder segment belong to the left subtree, and all values right of the root belong to the right subtree.

The algorithm finds `root_index` in inorder and computes:

```text
left_size = root_index - in_left
right_size = size - left_size - 1
```

Therefore the left recursive call receives exactly the values that must form the left subtree, and the right recursive call receives exactly the values that must form the right subtree.

The preorder start for the left subtree is `pre_left + 1` because preorder visits the root before the left subtree.

The preorder start for the right subtree is `pre_left + left_size + 1` because preorder lists all `left_size` left-subtree nodes before any right-subtree node.

By the induction hypothesis, both recursive calls correctly construct their assigned subtrees.

Attaching those two correct subtrees to the correct root produces the correct subtree for the current call.

#### Whole Tree

The initial call:

```text
build(0, 0, len(preorder))
```

owns every node in both traversals. Therefore it returns the root of the entire reconstructed tree.

---

### 11. Complexity

Let `n` be the number of nodes.

Building the inorder index map takes:

```text
O(n)
```

Each recursive call creates exactly one tree node, and there is one call per node. The root lookup is `O(1)` because of the dictionary.

So the total time complexity is:

```text
O(n)
```

The extra space is:

```text
O(n)
```

for the inorder index map.

The recursion stack uses:

```text
O(h)
```

where `h` is the height of the tree.

In the best case, the tree is balanced and `h = O(log n)`. In the worst case, the tree is completely skewed and `h = O(n)`.

If the output tree itself is counted, storing the tree also takes `O(n)` space.

---

### 12. Common Pitfalls

#### Pitfall 1: Forgetting That Preorder Chooses the Root

The root of the current subtree is not the first element of the current inorder range.

It is the first element of the current preorder range:

```text
root_value = preorder[pre_left]
```

Inorder is used only to split left from right.

#### Pitfall 2: Scanning Inorder Every Time

Using:

```python
inorder.index(root_value)
```

inside every recursive call can make the solution `O(n^2)`.

Build a map once:

```python
inorder_index = {value: index for index, value in enumerate(inorder)}
```

#### Pitfall 3: Copying Slices Recursively

Code with slices is readable, but every slice allocates a new list.

For learning, slicing is fine. For an efficient solution, pass boundaries or sizes.

#### Pitfall 4: Starting the Right Subtree Too Early

The right subtree does not start at `pre_left + 2` in general.

It starts after all left-subtree nodes:

```text
pre_left + left_size + 1
```

This matters whenever the left subtree has more than one node.

#### Pitfall 5: Using Inclusive and Exclusive Boundaries Inconsistently

If using four boundaries, decide whether ranges are inclusive or half-open and stay consistent.

For example, both are valid styles:

```text
inclusive: inorder[in_left ... in_right]
half-open: inorder[in_left ... in_right)
```

Mixing the two styles causes off-by-one errors.

#### Pitfall 6: Ignoring the Uniqueness Assumption

This reconstruction relies on each value having one position in inorder.

If duplicate values were allowed, `value -> inorder index` would not identify a unique split. The standard LeetCode problem assumes all values are unique.

---

### 13. First-Principles Summary

A binary tree traversal is a compressed description of recursive structure.

Preorder says:

```text
Here is the next root.
```

Inorder says:

```text
Here are the nodes left of that root, and here are the nodes right of that root.
```

The algorithm repeatedly applies those two facts:

```text
choose root from preorder
split subtree by root position in inorder
recursively build left and right subtrees
```

The recursive invariant is:

```text
build(pre_left, in_left, size) constructs exactly the subtree represented by
preorder[pre_left ... pre_left + size - 1] and
inorder[in_left ... in_left + size - 1].
```

Once that invariant is trusted, the implementation is just careful index arithmetic.

## Implementation
See `solutions/binary_tree_dfs/p105_construct_binary_tree_from_preorder_and_inorder_traversal.py`.

## Tests
See `tests/binary_tree_dfs/test_p105_construct_binary_tree_from_preorder_and_inorder_traversal.py`.

## Examples

### Example 1
- Input: `{'preorder': [3, 9, 20, 15, 7], 'inorder': [9, 3, 15, 20, 7]}`
- Output: `[3, 9, 20, None, None, 15, 7]`

### Example 2
- Input: `{'preorder': [-1], 'inorder': [-1]}`
- Output: `[-1]`

## Follow-up Practice
- Write the recursive contract in one sentence before writing code.
- Trace a tree with only a right child and verify the right subtree's preorder start.
- Implement both versions: one with slices for clarity and one with index boundaries for efficiency.
