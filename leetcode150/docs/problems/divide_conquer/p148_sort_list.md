# 148. Sort List

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/sort-list/
- Official Group: Divide & Conquer
- Pattern Group: Divide & Conquer
- Patterns: divide-conquer, linked-list

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given the head of a singly linked list.

Each node contains:

```text
value
next pointer
```

The task is to return the head of a linked list containing the same nodes, but ordered by node value in nondecreasing order.

For example:

```text
4 -> 2 -> 1 -> 3
```

should become:

```text
1 -> 2 -> 3 -> 4
```

The important detail is that this is a linked list, not an array.

In an array, random access is cheap:

```text
nums[mid]
```

is available in `O(1)` time.

In a singly linked list, random access is not cheap. To reach the middle, you must walk node by node. You also cannot swap distant elements as conveniently as in an array, because the structure is controlled by `next` pointers.

So the real problem is:

> Reorder the `next` pointers of a singly linked list so that the nodes appear in sorted order.

The values determine the ordering, but the pointers determine the result.

A correct solution must preserve every original node exactly once. It must not lose nodes, create cycles, or accidentally leave part of the old list attached in the wrong place.

---

### 2. The Baseline: Extract, Sort, Rebuild

The simplest way to think about the problem is to ignore the linked-list structure temporarily.

A baseline approach is:

1. Traverse the list and collect all values into an array.
2. Sort the array.
3. Write the sorted values back into the linked-list nodes, or create a new list from the sorted values.

Conceptually:

```python
values = []
node = head

while node:
    values.append(node.val)
    node = node.next

values.sort()

node = head
for value in values:
    node.val = value
    node = node.next

return head
```

This is easy to reason about:

- The array sort gives sorted values.
- Writing them back makes the list appear sorted.

But it avoids the main linked-list challenge.

It uses `O(n)` extra space for the array. If we create a brand-new list, it also does not reuse the original nodes as a pointer-based linked-list sorting solution normally would.

There is another brute-force linked-list idea: repeatedly find the minimum remaining node and append it to the sorted result. That resembles selection sort.

```text
Repeatedly remove the smallest node from the unsorted part.
Append it to the sorted part.
```

This uses little extra space, but it is slow. Finding the smallest node in a linked list takes a full scan, and doing that for every node costs:

```text
n + (n - 1) + (n - 2) + ... + 1 = O(n^2)
```

For a medium-sized list, `O(n^2)` can be too slow.

We want the sorting quality of an efficient array sort, but with operations that fit a singly linked list.

---

### 3. Why Merge Sort Fits a Linked List

The usual efficient comparison sorts are built around a few primitive operations.

For arrays:

- Quicksort is attractive because partitioning and swapping by index are cheap.
- Heapsort is attractive because parent/child index arithmetic is cheap.
- Merge sort is attractive because merging sorted sequences is systematic.

For singly linked lists, merge sort is especially natural because of one simple fact:

> Merging two already-sorted linked lists only needs forward pointer movement.

Suppose we have:

```text
left:  1 -> 4 -> 7
right: 2 -> 3 -> 9
```

To merge them, we never need to jump backward or access an index. We only compare the current heads:

```text
compare 1 and 2 -> take 1
compare 4 and 2 -> take 2
compare 4 and 3 -> take 3
compare 4 and 9 -> take 4
compare 7 and 9 -> take 7
append remaining 9
```

Result:

```text
1 -> 2 -> 3 -> 4 -> 7 -> 9
```

Every step consumes one node from the front of either list and appends it to the output.

That is exactly what singly linked lists are good at.

So the core idea is:

```text
Sort List = split the list into smaller lists + sort them + merge the sorted lists
```

This is merge sort.

---

### 4. The Key Observation

Sorting a linked list directly feels hard because the list may be long and disordered.

But a list of length `0` or `1` is already sorted.

That gives us a base case:

```text
empty list      -> sorted
single node     -> sorted
```

For a longer list, we can split it into two smaller lists:

```text
original: 4 -> 2 -> 1 -> 3

left:     4 -> 2
right:    1 -> 3
```

If we can sort each half, then we only need to merge two sorted lists:

```text
sort(left):  2 -> 4
sort(right): 1 -> 3

merge:       1 -> 2 -> 3 -> 4
```

The difficult problem becomes two smaller versions of the same problem.

The first-principles insight is:

> A linked list can be sorted efficiently if we can repeatedly cut it into independent halves and then merge sorted halves by rewiring `next` pointers.

The word “independent” matters. After splitting, the left half must actually end. If the left half still points into the right half, recursion and merging become incorrect.

---

### 5. The Split Invariant

To apply merge sort, we need a reliable way to split a singly linked list into two smaller lists.

Because the list has no length stored on each node, we can find the middle using two pointers:

```text
slow moves one step at a time
fast moves two steps at a time
```

When `fast` reaches the end, `slow` is near the middle.

However, to cut the list into two pieces, we also need the node before `slow`. Call it `prev`.

For example:

```text
4 -> 2 -> 1 -> 3
```

Walk with `slow` and `fast`:

```text
start:
slow = 4
fast = 4
prev = None

step 1:
prev = 4
slow = 2
fast = 1

step 2:
prev = 2
slow = 1
fast = None
```

Now:

```text
left starts at head = 4
right starts at slow = 1
prev is 2, the last node of the left half
```

Cut the link:

```python
prev.next = None
```

Now the original list becomes two independent lists:

```text
left:  4 -> 2 -> None
right: 1 -> 3 -> None
```

The split invariant is:

> After splitting, every original node belongs to exactly one of the two halves, and the left half's final node points to `None`.

This invariant prevents accidental overlap between recursive calls.

If we forget the cut, the “left half” is not really a half. It still contains the right half too.

---

### 6. The Merge Invariant

The merge step combines two sorted linked lists into one sorted linked list.

At any moment, we maintain:

```text
merged prefix: already sorted
left:          remaining sorted nodes from the left list
right:         remaining sorted nodes from the right list
```

The invariant is:

> The merged prefix is sorted and contains exactly the smallest nodes already chosen from the two input lists.

To extend the prefix, compare `left.val` and `right.val`.

If `left.val <= right.val`, the left node is no larger than the first remaining right node. Because the entire right list is sorted, that left node is also no larger than every later right node. Because the left list is sorted, it is also the smallest remaining left node.

So it is safe to append `left` next.

Otherwise, append `right` next.

This is the local decision that powers the whole merge:

```text
The smaller current head is the next globally smallest remaining node.
```

A dummy node often makes the implementation simpler:

```text
dummy -> merged nodes...
          ^
          tail
```

The dummy node is not part of the answer. It just gives `tail` a stable starting point so every append operation looks the same.

---

### 7. Detailed Algorithm

The recursive top-down merge sort algorithm is:

1. If the list is empty or has one node, return it.
2. Find the middle using slow/fast pointers.
3. Cut the list into two independent halves.
4. Recursively sort the left half.
5. Recursively sort the right half.
6. Merge the two sorted halves.
7. Return the merged head.

In pointer terms:

```text
sortList(head):
    if head is None or head.next is None:
        return head

    left_head = head
    right_head = split list around the middle

    sorted_left = sortList(left_head)
    sorted_right = sortList(right_head)

    return merge(sorted_left, sorted_right)
```

The algorithm never needs random indexing.

It only needs:

- forward traversal to find the middle;
- pointer cuts to separate halves;
- pointer rewiring to merge sorted halves.

Those are natural operations for a singly linked list.

---

### 8. Pseudocode

Here is Python-style pseudocode using the standard LeetCode `ListNode` shape.

```python
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head

        # Find the middle and keep the node before it.
        prev = None
        slow = head
        fast = head

        while fast is not None and fast.next is not None:
            prev = slow
            slow = slow.next
            fast = fast.next.next

        # Cut the list into two independent halves.
        prev.next = None

        left = self.sortList(head)
        right = self.sortList(slow)

        return self.merge(left, right)

    def merge(
        self,
        left: Optional[ListNode],
        right: Optional[ListNode],
    ) -> Optional[ListNode]:
        dummy = ListNode(0)
        tail = dummy

        while left is not None and right is not None:
            if left.val <= right.val:
                tail.next = left
                left = left.next
            else:
                tail.next = right
                right = right.next
            tail = tail.next

        if left is not None:
            tail.next = left
        else:
            tail.next = right

        return dummy.next
```

A few details are worth calling out.

First, the base case must include both empty and one-node lists:

```python
if head is None or head.next is None:
    return head
```

Without this, recursion has no stopping point.

Second, `prev.next = None` is the actual split. Finding the middle is not enough.

Third, after attaching a node during merge, `tail` must move forward:

```python
tail = tail.next
```

Otherwise every append overwrites the same pointer.

---

### 9. Walkthrough: Example 1

Input:

```text
4 -> 2 -> 1 -> 3
```

The first call sees more than one node, so it splits the list.

Using slow/fast pointers:

```text
left:  4 -> 2
right: 1 -> 3
```

Now recursively sort the left half:

```text
4 -> 2
```

Split it:

```text
left:  4
right: 2
```

Both halves have one node, so both are already sorted.

Merge them:

```text
compare 4 and 2 -> take 2
append remaining 4
```

Sorted left side:

```text
2 -> 4
```

Now recursively sort the right half from the original split:

```text
1 -> 3
```

Split it:

```text
left:  1
right: 3
```

Both are already sorted.

Merge them:

```text
compare 1 and 3 -> take 1
append remaining 3
```

Sorted right side:

```text
1 -> 3
```

Now merge the two sorted halves:

```text
left:  2 -> 4
right: 1 -> 3
```

Step by step:

```text
compare 2 and 1 -> take 1
merged: 1

compare 2 and 3 -> take 2
merged: 1 -> 2

compare 4 and 3 -> take 3
merged: 1 -> 2 -> 3

right is empty, append remaining left
merged: 1 -> 2 -> 3 -> 4
```

Final output:

```text
1 -> 2 -> 3 -> 4
```

The recursion tree looks like:

```text
            4 -> 2 -> 1 -> 3
             /              \
        4 -> 2              1 -> 3
        /    \              /    \
       4      2            1      3
        \    /              \    /
        2 -> 4              1 -> 3
             \              /
          1 -> 2 -> 3 -> 4
```

Each merge works only because its two inputs have already been sorted by smaller recursive calls.

---

### 10. Walkthrough: Example 2 With Negative Values

Input:

```text
-1 -> 5 -> 3 -> 4 -> 0
```

Negative values do not change the algorithm. Comparisons still work normally.

A possible first split is:

```text
left:  -1 -> 5
right: 3 -> 4 -> 0
```

Sort the left side:

```text
-1 -> 5
```

It splits into `-1` and `5`, then merges back as:

```text
-1 -> 5
```

Sort the right side:

```text
3 -> 4 -> 0
```

It can split into:

```text
left:  3
right: 4 -> 0
```

Sort `4 -> 0`:

```text
left:  4
right: 0
merge: 0 -> 4
```

Now merge `3` with `0 -> 4`:

```text
compare 3 and 0 -> take 0
compare 3 and 4 -> take 3
append remaining 4
```

Sorted right side:

```text
0 -> 3 -> 4
```

Now merge the two sorted halves:

```text
left:  -1 -> 5
right: 0 -> 3 -> 4
```

Step by step:

```text
compare -1 and 0 -> take -1
compare 5 and 0  -> take 0
compare 5 and 3  -> take 3
compare 5 and 4  -> take 4
append remaining 5
```

Final output:

```text
-1 -> 0 -> 3 -> 4 -> 5
```

---

### 11. Why the Algorithm Is Correct

We can prove correctness by induction on the number of nodes in the list.

#### Base Case

If the list has zero nodes, it is empty and therefore sorted.

If the list has one node, there are no other nodes that could be out of order, so it is sorted.

The algorithm returns the list unchanged in both cases.

#### Inductive Hypothesis

Assume `sortList` correctly sorts every linked list with fewer than `n` nodes.

That means for any smaller list, it returns a list that:

1. contains exactly the original nodes from that smaller list;
2. is sorted in nondecreasing order;
3. has no missing nodes, duplicate nodes, or extra nodes.

#### Inductive Step

Consider a list with `n` nodes, where `n >= 2`.

The algorithm splits it into two independent nonempty halves.

Because the split only cuts one `next` pointer:

- every original node remains in one of the two halves;
- no node belongs to both halves;
- the two recursive calls receive smaller lists than the original.

By the inductive hypothesis:

```text
sorted_left  is a sorted list containing exactly the nodes from the left half
sorted_right is a sorted list containing exactly the nodes from the right half
```

Now the algorithm merges `sorted_left` and `sorted_right`.

During merge, the maintained invariant is:

```text
The output prefix is sorted and contains exactly the smallest already-chosen nodes from both lists.
```

At each step, the smaller current head must be the smallest remaining node overall, because each input list is already sorted. Appending it preserves sorted order.

When one list becomes empty, all remaining nodes in the other list are already sorted and are greater than or equal to the last appended node, so appending the rest preserves sorted order.

Therefore, the merged list is sorted and contains exactly all nodes from both halves.

Since the halves together contain exactly the original `n` nodes, the returned list is a sorted ordering of exactly the original list.

By induction, the algorithm is correct for all list lengths.

---

### 12. Complexity Analysis

Let `n` be the number of nodes.

#### Time Complexity

At each recursion level:

- splitting all lists at that level requires walking through their nodes;
- merging all lists at that level also touches each node once.

So each level costs `O(n)` total work.

Because the list is split roughly in half each time, there are `O(log n)` levels.

Therefore:

```text
Time = O(n log n)
```

This is a large improvement over selection-sort-style linked-list sorting, which costs `O(n^2)`.

#### Space Complexity

The merge itself rewires existing nodes and uses only a few pointers, so the merge operation is `O(1)` auxiliary space.

The recursive top-down version uses call stack space. Because the split is balanced, the recursion depth is `O(log n)`.

Therefore:

```text
Auxiliary space = O(log n)
```

Some versions of this problem ask for constant extra space. A bottom-up iterative merge sort can achieve `O(1)` auxiliary space by avoiding recursion. The same split/merge idea remains, but the implementation merges runs of length `1`, then `2`, then `4`, and so on.

For understanding the problem from first principles, the recursive version is the clearest expression of the core idea.

---

### 13. Common Pitfalls

#### Forgetting to Cut the List

This is the most common bug.

Finding the middle gives you a pointer to the right half, but it does not split the list by itself.

Wrong idea:

```python
right = slow
left = head
# missing cut
```

Correct idea:

```python
right = slow
prev.next = None
left = head
```

Without the cut, the left recursive call may still see the full original list, causing infinite recursion or duplicated structure.

#### Losing the Right Half

If you overwrite pointers before saving the next starting point, you can lose access to nodes.

The safe order is:

```python
right = slow
prev.next = None
```

Keep a pointer to the beginning of the right half before or while cutting.

#### Bad Base Case

The base case must stop both empty and one-node lists:

```python
if head is None or head.next is None:
    return head
```

Checking only `head is None` is not enough. A single-node list would keep trying to split.

#### Not Moving `tail` During Merge

After attaching a node, advance the output tail:

```python
tail.next = left
tail = tail.next
```

If `tail` does not move, the merged list will be corrupted because every new attachment rewrites the same `next` pointer.

#### Creating Cycles Accidentally

Linked-list sorting is pointer surgery. A cycle can appear if an old `next` pointer is left connected in the wrong direction or if a node is reattached without advancing the source pointer.

The merge loop should always do both:

```python
tail.next = chosen_node
chosen_list = chosen_list.next
tail = tail.next
```

#### Expecting Array-Like Access

Do not try to repeatedly access the middle by index unless you first convert to an array. In a linked list, reaching index `i` costs `O(i)`, so array-style algorithms can become much slower than expected.

#### Assuming Values Must Be Unique

The list may contain duplicate values. The comparison should handle equality correctly.

Using:

```python
if left.val <= right.val:
```

keeps equal-valued nodes from the left half before equal-valued nodes from the right half, making the merge stable. Stability is not required by the problem, but this comparison is simple and safe.

---

### 14. First-Principles Summary

A singly linked list is hard to sort with array-style techniques because it does not support cheap random access or cheap distant swaps.

But it is very good at one operation:

```text
Move forward through nodes and change `next` pointers.
```

Merge sort fits that structure perfectly.

The reasoning chain is:

```text
A 0-node or 1-node list is already sorted.
A longer list can be split into two smaller independent lists.
If both smaller lists are sorted, they can be merged in linear time.
Merging sorted linked lists only requires forward pointer movement.
Balanced splitting gives log n levels.
Each level touches all nodes a constant number of times.
Therefore the total time is O(n log n).
```

The two central invariants are:

```text
Split invariant:
After cutting, each original node belongs to exactly one half.

Merge invariant:
The merged prefix is sorted and contains exactly the smallest chosen nodes so far.
```

Once those invariants are protected, the algorithm becomes straightforward:

```text
base case -> split -> recursively sort -> merge -> return merged head
```

For this problem, the key is not a clever formula. It is respecting the physical structure of a linked list. The sorted order is produced by carefully rewiring nodes, one local comparison and one safe pointer update at a time.

## Implementation
See `solutions/divide_conquer/p148_sort_list.py`.

## Tests
See `tests/divide_conquer/test_p148_sort_list.py`.

## Examples

### Example 1
- Input: `{'head': [4, 2, 1, 3]}`
- Output: `[1, 2, 3, 4]`

### Example 2
- Input: `{'head': [-1, 5, 3, 4, 0]}`
- Output: `[-1, 0, 3, 4, 5]`

### Example 3
- Input: `{'head': []}`
- Output: `[]`

## Follow-up Practice
- Trace the slow/fast pointer split on lists of length `2`, `3`, `4`, and `5`.
- Write the `merge` helper before writing the recursive `sortList` function.
- Test empty, one-element, two-element, duplicate-value, negative-value, and already-sorted inputs.
- For an extra challenge, implement the bottom-up iterative merge sort version to avoid recursion stack space.
