# 108. Convert Sorted Array to Binary Search Tree

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
- Official Group: Divide & Conquer
- Pattern Group: Divide & Conquer
- Patterns: divide-conquer, tree-traversal

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given a sorted integer array:

```text
nums = [-10, -3, 0, 5, 9]
```

construct a binary search tree whose inorder traversal would visit the values in the same sorted order.

A binary search tree, or BST, has this rule at every node:

```text
all values in the left subtree  < node.val < all values in the right subtree
```

The problem adds one more requirement: the tree must be height-balanced.

Height-balanced means that, for every node, the heights of its left and right subtrees differ by no more than `1`.

So the task is not merely:

> Put the sorted values into any BST.

It is more specific:

> Use all array values exactly once, preserve BST ordering, and choose the tree shape so no side becomes much deeper than the other.

For the array:

```text
[-10, -3, 0, 5, 9]
```

one valid answer is:

```text
        0
      /   \
    -3     9
    /     /
 -10     5
```

The exact serialized output may differ because multiple balanced BSTs can be valid. For example, when an interval has an even number of elements, either middle element can be chosen as the root depending on the implementation. The important properties are BST ordering and height balance.

---

### 2. Start From the Baseline Idea

A sorted array already tells us the order in which values should appear in an inorder traversal.

A direct but poor idea is:

1. Start with an empty BST.
2. Insert each number from left to right.

For example, inserting:

```text
[-10, -3, 0, 5, 9]
```

in that order produces:

```text
-10
  \
  -3
    \
     0
      \
       5
        \
         9
```

This is a valid BST, because every right child is larger than its parent.

But it is not height-balanced. It is essentially a linked list, with height `n`.

This baseline shows the central danger:

> The sorted order is useful for BST correctness, but blindly inserting sorted values creates the worst possible shape.

We need to choose roots intentionally.

---

### 3. Key Observation: The Middle Value Should Be the Root

In a BST, the root splits the values into two groups:

```text
values smaller than root -> left subtree
values larger than root  -> right subtree
```

Because `nums` is sorted, if we choose index `mid` as the root:

```text
nums[left:mid]      are all smaller than nums[mid]
nums[mid + 1:right] are all larger than nums[mid]
```

So choosing a value automatically gives us the correct left and right subarrays.

Now consider balance.

To make the whole tree height-balanced, the root should leave roughly the same number of values on both sides. The best way to do that in a sorted array is to choose the middle element.

For:

```text
[-10, -3, 0, 5, 9]
```

choose `0`:

```text
left side:  [-10, -3]
root:       0
right side: [5, 9]
```

Both sides have two values, so neither side starts with a size advantage.

This is the first-principles reason the algorithm works:

> A sorted array gives the BST partition for free, and choosing the middle element keeps the partition sizes as equal as possible.

---

### 4. The Recursive Invariant

The problem repeats inside each half.

After choosing the root from the whole array, we still need to build:

```text
a balanced BST from the left half
a balanced BST from the right half
```

That suggests a recursive helper over an index interval.

Define:

```text
build(left, right)
```

to mean:

> Build a height-balanced BST containing exactly `nums[left]` through `nums[right]`, inclusive.

The invariant of this helper is:

```text
build(left, right) returns a balanced BST whose inorder traversal is nums[left:right + 1]
```

This invariant contains everything we need:

- `exactly nums[left:right + 1]` prevents dropping or duplicating values.
- `inorder traversal` preserves sorted order, which proves the BST property.
- `balanced` preserves the height requirement.

The recursive decision is local:

```text
mid = the middle index of [left, right]
nums[mid] becomes the root
```

Then:

```text
root.left  = build(left, mid - 1)
root.right = build(mid + 1, right)
```

The base case is the empty interval:

```text
if left > right:
    return None
```

An empty interval contains no values, so it corresponds to an empty child pointer.

---

### 5. Why This Constructs a BST

Suppose the helper is building the interval:

```text
nums[left:right + 1]
```

It chooses:

```text
mid = (left + right) // 2
root.val = nums[mid]
```

Because the array is sorted:

```text
for every i in [left, mid - 1], nums[i] < nums[mid]
for every i in [mid + 1, right], nums[i] > nums[mid]
```

So every value sent to the left recursive call is smaller than the root, and every value sent to the right recursive call is larger than the root.

If the recursive calls also build valid BSTs for their intervals, then connecting them under `root` preserves the BST rule at `root` and inside both subtrees.

This is why no comparison-based insertion is needed. The array indices already encode the correct ordering.

---

### 6. Why This Keeps the Tree Balanced

At each interval, the algorithm chooses a middle index.

That means the two recursive subproblems have sizes that differ by at most one:

```text
left size  = mid - left
right size = right - mid
```

When an interval has odd length, the two sides have exactly the same size.

When an interval has even length, one side has one more node than the other.

Since this same middle-choice rule is applied recursively to every subarray, every node is built from two subtrees whose sizes are as equal as possible for that interval.

For a sorted array, this produces a height-balanced BST.

The key is that balance is not fixed after the tree is built. Balance is built into every root choice.

---

### 7. Detailed Algorithm

1. If `nums` is empty, return `None`.

2. Define a recursive helper:

```text
build(left, right)
```

3. If the interval is empty:

```text
left > right
```

return `None`.

4. Choose the middle index:

```text
mid = (left + right) // 2
```

This chooses the left middle for even-length intervals. Choosing the right middle also works if used consistently.

5. Create a tree node:

```text
root = TreeNode(nums[mid])
```

6. Recursively build the smaller values as the left subtree:

```text
root.left = build(left, mid - 1)
```

7. Recursively build the larger values as the right subtree:

```text
root.right = build(mid + 1, right)
```

8. Return `root`.

9. The final answer is:

```text
build(0, len(nums) - 1)
```

---

### 8. Pseudocode

```python
def sortedArrayToBST(nums):
    def build(left, right):
        if left > right:
            return None

        mid = (left + right) // 2

        root = TreeNode(nums[mid])
        root.left = build(left, mid - 1)
        root.right = build(mid + 1, right)

        return root

    return build(0, len(nums) - 1)
```

The important detail is that the helper passes index boundaries, not copied subarrays.

Using slices like:

```python
nums[:mid]
nums[mid + 1:]
```

is conceptually simple, but it repeatedly copies array portions. Index boundaries keep the construction linear.

---

### 9. Walkthrough: `nums = [-10, -3, 0, 5, 9]`

Start with the full interval:

```text
build(0, 4)
nums[0:5] = [-10, -3, 0, 5, 9]
```

Choose the middle:

```text
mid = (0 + 4) // 2 = 2
nums[2] = 0
```

Create:

```text
0
```

Now build the left subtree from indices `0..1`:

```text
build(0, 1)
nums[0:2] = [-10, -3]
```

Choose:

```text
mid = (0 + 1) // 2 = 0
nums[0] = -10
```

So the left subtree root is `-10`.

Its left interval is empty:

```text
build(0, -1) -> None
```

Its right interval contains one value:

```text
build(1, 1)
```

For `build(1, 1)`:

```text
mid = 1
nums[1] = -3
```

This creates a leaf node `-3`.

So the left side becomes:

```text
-10
   \
   -3
```

Now build the right subtree from indices `3..4`:

```text
build(3, 4)
nums[3:5] = [5, 9]
```

Choose:

```text
mid = (3 + 4) // 2 = 3
nums[3] = 5
```

The right subtree root is `5`.

Its left interval is empty:

```text
build(3, 2) -> None
```

Its right interval contains one value:

```text
build(4, 4)
```

For `build(4, 4)`:

```text
mid = 4
nums[4] = 9
```

This creates a leaf node `9`.

So the right side becomes:

```text
5
 \
  9
```

Putting everything together:

```text
        0
      /   \
   -10     5
      \     \
      -3     9
```

This is a valid balanced BST.

LeetCode's sample output uses a different but also valid tree:

```text
        0
      /   \
    -3     9
    /     /
 -10     5
```

That version comes from choosing the right middle in two-element intervals. Both trees satisfy the requirements.

---

### 10. Walkthrough: `nums = [1, 3]`

There are two valid balanced answers.

If we choose the left middle:

```text
mid = (0 + 1) // 2 = 0
nums[0] = 1
```

Then:

```text
1
 \
  3
```

If we choose the right middle:

```text
mid = (0 + 1 + 1) // 2 = 1
nums[1] = 3
```

Then:

```text
  3
 /
1
```

The official example output is:

```text
[3, 1]
```

That corresponds to choosing the right middle. The left-middle version is still accepted by LeetCode because the judge checks whether the returned tree is a valid height-balanced BST, not whether it matches one exact serialization.

---

### 11. Correctness

We prove that the algorithm returns a height-balanced BST containing exactly all values in `nums`.

#### Lemma 1: `build(left, right)` uses exactly the values in `nums[left:right + 1]`.

If `left > right`, the interval is empty and the function returns `None`, so it uses no values.

Otherwise, the function chooses one index `mid` in the interval. It creates exactly one node for `nums[mid]`. The left recursive call uses only indices `left..mid - 1`, and the right recursive call uses only indices `mid + 1..right`.

These three parts are disjoint and together cover exactly `left..right`.

Therefore, `build(left, right)` uses exactly the values in that interval.

#### Lemma 2: `build(left, right)` returns a BST.

If the interval is empty, `None` is a valid empty BST.

Otherwise, the root value is `nums[mid]`.

Because `nums` is sorted, every value in `nums[left:mid]` is smaller than `nums[mid]`, and every value in `nums[mid + 1:right + 1]` is larger than `nums[mid]`.

By the recursive assumption, the left and right calls return BSTs for those intervals. Connecting them to the root preserves the BST rule at the root and inside both subtrees.

Therefore, `build(left, right)` returns a BST.

#### Lemma 3: `build(left, right)` returns a height-balanced tree.

The algorithm chooses a middle index, so the left and right intervals differ in size by at most one.

The same rule is applied recursively inside both intervals. Therefore, each subtree is built by repeatedly splitting its interval as evenly as possible.

By induction on interval length, both recursive subtrees are height-balanced, and their heights differ by at most one at the current root.

Therefore, the returned tree is height-balanced.

#### Theorem: `sortedArrayToBST(nums)` returns a height-balanced BST containing exactly all values from `nums`.

The main function calls:

```text
build(0, len(nums) - 1)
```

By Lemma 1, this uses exactly every array value once.

By Lemma 2, the result is a BST.

By Lemma 3, the result is height-balanced.

Therefore, the algorithm is correct.

---

### 12. Complexity

Let `n = len(nums)`.

Each array element becomes exactly one tree node.

Creating a node and assigning its children takes constant work, so the total time is:

```text
O(n)
```

The recursion depth is the height of the balanced tree:

```text
O(log n)
```

So auxiliary space, excluding the output tree itself, is:

```text
O(log n)
```

The output tree contains `n` nodes, so if output space is counted, total space is:

```text
O(n)
```

---

### 13. Common Pitfalls

#### Pitfall 1: Inserting Values in Sorted Order

Sequential insertion produces a valid BST but not a balanced one:

```text
1 -> 2 -> 3 -> 4 -> 5
```

The problem requires height balance, so root choice matters.

#### Pitfall 2: Forgetting the Empty-Interval Base Case

The base case must be:

```python
if left > right:
    return None
```

Without it, leaf children recurse forever or attempt invalid array access.

#### Pitfall 3: Thinking the Output Must Match the Example Exactly

For even-length intervals, left-middle and right-middle choices can produce different serialized trees.

Both are valid if the result is a height-balanced BST containing the same values.

#### Pitfall 4: Copying Subarrays at Every Recursive Call

This version is easy to read:

```python
root.left = sortedArrayToBST(nums[:mid])
root.right = sortedArrayToBST(nums[mid + 1:])
```

But slicing copies elements. Across all recursive calls, that adds avoidable overhead.

Passing `left` and `right` boundaries avoids repeated copying.

#### Pitfall 5: Choosing an Endpoint as the Root

Choosing `left` or `right` as the root recreates the skewed-tree problem.

The middle is what keeps the recursive halves close in size.

---

### 14. First-Principles Summary

A sorted array is already the inorder traversal of some BST.

To build a BST from it, choose a root, put smaller values on the left, and larger values on the right.

To make the BST balanced, choose the root so the left and right sides have nearly equal sizes.

In a sorted array, the middle element is exactly that choice.

Then the same reasoning applies recursively to the left half and the right half.

So the whole solution is the repeated application of one idea:

```text
middle value becomes root; left half becomes left subtree; right half becomes right subtree
```

That local decision simultaneously preserves ordering and creates balance.

## Implementation
See `solutions/divide_conquer/p108_convert_sorted_array_to_binary_search_tree.py`.

## Tests
See `tests/divide_conquer/test_p108_convert_sorted_array_to_binary_search_tree.py`.

## Examples

### Example 1
- Input: `{'nums': [-10, -3, 0, 5, 9]}`
- Output: `[0, -3, 9, -10, None, 5]`

### Example 2
- Input: `{'nums': [1, 3]}`
- Output: `[3, 1]`

## Follow-up Practice
- Draw the recursion tree for arrays of lengths `1`, `2`, `3`, `4`, and `5`.
- Implement both left-middle and right-middle versions, then compare their serialized trees.
- Verify the inorder traversal of your returned tree equals the original array.
- Check the height difference at every node to confirm the tree is balanced.
