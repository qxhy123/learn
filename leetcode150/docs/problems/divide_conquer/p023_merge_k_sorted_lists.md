# 23. Merge k Sorted Lists

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/merge-k-sorted-lists/
- Official Group: Divide & Conquer
- Pattern Group: Divide & Conquer
- Patterns: divide-conquer, linked-list

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given `k` linked lists, each individual list is already sorted in nondecreasing order.

For example:

```text
lists = [
  1 -> 4 -> 5,
  1 -> 3 -> 4,
  2 -> 6
]
```

We need to combine all nodes into one sorted linked list:

```text
1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 5 -> 6
```

The important facts are:

```text
Each input list is sorted.
The lists are separate linked lists.
The final output must also be one sorted linked list.
```

So the real problem is:

> Given many already-sorted streams of nodes, repeatedly choose the smallest available next value until every node has been placed into one sorted output chain.

The linked-list detail means we are not just returning values. In a normal LeetCode implementation, the output is a `ListNode` head. Conceptually, though, the same ordering problem appears whether we imagine nodes or arrays.

---

### 2. Start From the Simplest Case: Merge Two Sorted Lists

Before thinking about `k` lists, ask the smallest useful question:

> If there are only two sorted lists, how do we merge them?

Suppose:

```text
A = 1 -> 4 -> 5
B = 1 -> 3 -> 4
```

Because both lists are sorted, the smallest remaining node must be at the head of either `A` or `B`.

It cannot be hidden later inside a list, because every later node in a sorted list is at least as large as that list's head.

So we compare only two nodes:

```text
A head = 1
B head = 1
```

Choose one `1`, append it to the output, and advance that list. Then compare the two current heads again.

This gives the standard two-list merge:

```python
def merge_two(a, b):
    dummy = ListNode(0)
    tail = dummy

    while a and b:
        if a.val <= b.val:
            tail.next = a
            a = a.next
        else:
            tail.next = b
            b = b.next
        tail = tail.next

    tail.next = a if a else b
    return dummy.next
```

The dummy node is not part of the answer. It is only a convenient fixed starting point so appending the first real node does not need special-case code.

---

### 3. Brute Force and Baseline Approaches

There are several correct ways to solve the problem. The divide-and-conquer solution is easier to appreciate after seeing the baselines.

#### Baseline A: Collect, Sort, Rebuild

The most direct approach is:

1. Traverse every linked list.
2. Put every value into an array.
3. Sort the array.
4. Build a new linked list from the sorted values.

Conceptually:

```python
values = []

for head in lists:
    while head:
        values.append(head.val)
        head = head.next

values.sort()
return build_linked_list(values)
```

If there are `N` total nodes across all lists, this costs:

```text
Time:  O(N log N)
Space: O(N)
```

It is correct, but it ignores the strongest property of the input: each list is already sorted.

#### Baseline B: Repeatedly Scan All List Heads

Another direct approach keeps all lists as streams:

1. Look at the current head of every non-empty list.
2. Pick the smallest head.
3. Append it to the answer.
4. Advance that list.
5. Repeat until all lists are empty.

This uses the sorted-list property, but the selection step is expensive.

For each of `N` output nodes, scanning up to `k` heads costs `O(k)`, so:

```text
Time:  O(Nk)
Space: O(1) extra, ignoring the output chain
```

This can be much worse than sorting when `k` is large.

#### Baseline C: Merge Lists One by One

A more natural linked-list baseline is:

```text
merged = lists[0]
merged = merge_two(merged, lists[1])
merged = merge_two(merged, lists[2])
...
```

This is correct, but it can become unbalanced.

If all lists are roughly the same size, the first merge is small, the next is bigger, the next is bigger again, and nodes from early lists are repeatedly reprocessed.

For `k` lists with `N` total nodes, this can cost about:

```text
O(Nk)
```

The problem is not the two-list merge. The problem is the order in which we combine the lists.

---

### 4. Key Observation: Merging Is Associative for Sorted Lists

The final sorted order does not depend on whether we merge from left to right or in balanced groups.

For three sorted lists `A`, `B`, and `C`:

```text
merge_two(merge_two(A, B), C)
```

produces the same sorted multiset of nodes as:

```text
merge_two(A, merge_two(B, C))
```

or:

```text
merge_two(merge_two(A, C), B)
```

The exact identity of equal-valued nodes may differ if there are ties, but the required output values are the same sorted sequence.

This means we are free to choose a merge order that is efficient.

The best merge order is balanced:

```text
Merge small groups into medium groups.
Merge medium groups into large groups.
Keep doubling the amount of data per merged result.
```

That is the divide-and-conquer idea.

---

### 5. Divide-and-Conquer View

Instead of merging all `k` lists one by one, split the list collection in half:

```text
lists[left ... right]
```

Solve the left half:

```text
lists[left ... mid]
```

Solve the right half:

```text
lists[mid + 1 ... right]
```

Then merge the two sorted results with `merge_two`.

The recursive meaning is:

```text
merge_range(left, right) returns one sorted linked list containing
exactly all nodes from lists[left], lists[left + 1], ..., lists[right].
```

Base cases:

```text
If left > right: there are no lists, so return None.
If left == right: there is one list, already sorted, so return lists[left].
```

Recursive step:

```text
mid = (left + right) // 2
left_sorted = merge_range(left, mid)
right_sorted = merge_range(mid + 1, right)
return merge_two(left_sorted, right_sorted)
```

This is exactly the same idea as merge sort, except the inputs are already sorted linked lists instead of unsorted array elements.

---

### 6. The Divide/Merge Invariant

A divide-and-conquer algorithm is only safe if every recursive result has a precise meaning.

For this problem, the invariant is:

> For any range `lists[left:right + 1]`, `merge_range(left, right)` returns a sorted linked list containing every node from that range exactly once and no nodes from outside that range.

This invariant has three parts.

#### Sorted

The returned list must be in nondecreasing order.

This is guaranteed because:

- a single input list is already sorted;
- merging two sorted lists with `merge_two` produces another sorted list.

#### Complete

Every node from every list in the range must appear in the result.

This is guaranteed because:

- the base case returns the one list it represents;
- the recursive step covers the left half and right half;
- those halves together cover the whole range;
- `merge_two` appends all nodes from both halves.

#### No Duplication

No node should be lost or used twice.

This is guaranteed because:

- the left and right recursive ranges do not overlap;
- each recursive call owns a disjoint set of lists;
- `merge_two` advances through each input list exactly once.

Once this invariant is true for smaller ranges, the merge step makes it true for the larger range.

---

### 7. Detailed Algorithm

Assume the LeetCode type signature is:

```python
def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
```

The algorithm is:

1. If `lists` is empty, return `None`.
2. Define a helper `merge_two(a, b)` that merges two sorted linked lists.
3. Define a helper `merge_range(left, right)`:
   - if `left == right`, return `lists[left]`;
   - split at `mid`;
   - recursively merge the left half;
   - recursively merge the right half;
   - merge those two sorted results.
4. Return `merge_range(0, len(lists) - 1)`.

Pseudocode:

```python
def mergeKLists(lists):
    if not lists:
        return None

    def merge_two(a, b):
        dummy = ListNode(0)
        tail = dummy

        while a and b:
            if a.val <= b.val:
                tail.next = a
                a = a.next
            else:
                tail.next = b
                b = b.next
            tail = tail.next

        if a:
            tail.next = a
        else:
            tail.next = b

        return dummy.next

    def merge_range(left, right):
        if left == right:
            return lists[left]

        mid = (left + right) // 2
        left_list = merge_range(left, mid)
        right_list = merge_range(mid + 1, right)
        return merge_two(left_list, right_list)

    return merge_range(0, len(lists) - 1)
```

An iterative pairwise version expresses the same invariant level by level:

```python
def mergeKLists(lists):
    if not lists:
        return None

    interval = 1
    while interval < len(lists):
        for i in range(0, len(lists) - interval, interval * 2):
            lists[i] = merge_two(lists[i], lists[i + interval])
        interval *= 2

    return lists[0]
```

Both versions merge in balanced rounds. The recursive version is often easier to explain; the iterative version avoids recursion over the list index range.

---

### 8. Detailed Example Walkthrough

Use the official example:

```text
lists = [
  L0: 1 -> 4 -> 5,
  L1: 1 -> 3 -> 4,
  L2: 2 -> 6
]
```

Call:

```text
merge_range(0, 2)
```

Split around `mid = 1`:

```text
left half:  merge_range(0, 1)
right half: merge_range(2, 2)
```

The right half has only one list:

```text
merge_range(2, 2) = 2 -> 6
```

Now solve the left half:

```text
merge_range(0, 1)
```

Split into:

```text
merge_range(0, 0) = 1 -> 4 -> 5
merge_range(1, 1) = 1 -> 3 -> 4
```

Merge those two lists:

```text
A: 1 -> 4 -> 5
B: 1 -> 3 -> 4
```

Step by step:

```text
Compare 1 and 1: take A's 1
output: 1
A: 4 -> 5
B: 1 -> 3 -> 4

Compare 4 and 1: take B's 1
output: 1 -> 1
A: 4 -> 5
B: 3 -> 4

Compare 4 and 3: take B's 3
output: 1 -> 1 -> 3
A: 4 -> 5
B: 4

Compare 4 and 4: take A's 4
output: 1 -> 1 -> 3 -> 4
A: 5
B: 4

Compare 5 and 4: take B's 4
output: 1 -> 1 -> 3 -> 4 -> 4
A: 5
B: empty

Append the rest of A:
output: 1 -> 1 -> 3 -> 4 -> 4 -> 5
```

So:

```text
merge_range(0, 1) = 1 -> 1 -> 3 -> 4 -> 4 -> 5
```

Now merge the two half-results:

```text
left result:  1 -> 1 -> 3 -> 4 -> 4 -> 5
right result: 2 -> 6
```

Step by step:

```text
Compare 1 and 2: take 1
output: 1

Compare 1 and 2: take 1
output: 1 -> 1

Compare 3 and 2: take 2
output: 1 -> 1 -> 2

Compare 3 and 6: take 3
output: 1 -> 1 -> 2 -> 3

Compare 4 and 6: take 4
output: 1 -> 1 -> 2 -> 3 -> 4

Compare 4 and 6: take 4
output: 1 -> 1 -> 2 -> 3 -> 4 -> 4

Compare 5 and 6: take 5
output: 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 5

Left side is empty, append 6:
output: 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 5 -> 6
```

Final answer:

```text
1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 5 -> 6
```

---

### 9. Correctness Argument

We prove that the algorithm returns one sorted list containing exactly all input nodes.

#### Lemma 1: `merge_two(a, b)` correctly merges two sorted linked lists.

At every step, `a` and `b` point to the smallest not-yet-output nodes of their respective lists.

Because each list is sorted, the smallest node among all remaining nodes in both lists must be either `a` or `b`.

`merge_two` compares those two heads and appends the smaller one. Therefore, every appended node is the smallest remaining node globally, so the output order stays sorted.

When one list becomes empty, all remaining nodes in the other list are already sorted and are greater than or equal to the last appended node, so appending the rest preserves sorted order.

Also, each step advances exactly one input pointer, so every input node is appended exactly once.

Therefore, `merge_two` returns a sorted list containing exactly the nodes from its two inputs.

#### Lemma 2: `merge_range(left, right)` satisfies the range invariant.

We prove this by induction on the number of lists in the range.

Base case: if `left == right`, the range contains one list. That list is already sorted by the problem statement and contains exactly the nodes from that range.

Inductive step: assume the invariant holds for smaller ranges. For `merge_range(left, right)`, the algorithm splits the range into two disjoint smaller ranges:

```text
left ... mid
mid + 1 ... right
```

By the induction hypothesis, the recursive calls return sorted lists containing exactly the nodes from those two ranges.

By Lemma 1, merging those two sorted results returns one sorted list containing exactly the nodes from both halves.

The two halves together are exactly the original range, so the invariant holds for `merge_range(left, right)`.

#### Theorem: `mergeKLists(lists)` returns the correct answer.

If `lists` is empty, there are no nodes, so returning `None` is correct.

Otherwise, the algorithm returns `merge_range(0, len(lists) - 1)`. By Lemma 2, this is a sorted linked list containing exactly all nodes from every input list.

That is precisely what the problem asks for.

---

### 10. Complexity

Let:

```text
k = number of linked lists
N = total number of nodes across all lists
```

Each call to `merge_two` costs linear time in the number of nodes being merged.

In balanced divide-and-conquer, each node participates in one merge per level of the recursion tree.

The number of levels is:

```text
O(log k)
```

So the total time is:

```text
O(N log k)
```

This improves over repeated scanning or unbalanced one-by-one merging when `k` is large.

Extra space:

```text
O(log k)
```

for the recursion stack in the recursive version.

The merge itself can relink existing nodes and use only `O(1)` extra pointer space. If an implementation creates brand-new nodes instead of reusing existing ones, then it uses `O(N)` additional space for the copied output nodes.

The iterative pairwise version can reduce recursion stack space to:

```text
O(1)
```

besides the input list array and output nodes.

---

### 11. Common Pitfalls

#### Forgetting the Empty Input Case

This input is valid:

```text
lists = []
```

There is no `lists[0]`. Return `None` immediately.

#### Confusing an Empty List With an Empty List Collection

These are different:

```text
lists = []      # no lists at all
lists = [[]]    # one list, but that list has no nodes
```

Both produce an empty output, but they reach it differently. In linked-list form, an empty list is represented by `None`.

#### Losing the Rest of a List

After the main two-list merge loop ends, one input may still contain nodes:

```python
tail.next = a if a else b
```

Without this step, the algorithm drops the remaining suffix.

#### Advancing Pointers in the Wrong Order

A safe pattern is:

```python
tail.next = a
a = a.next
tail = tail.next
```

If you overwrite `a.next` before saving or advancing correctly, you can accidentally disconnect the rest of the list.

#### Returning the Dummy Node Instead of `dummy.next`

The dummy node is a helper. The actual answer starts at:

```python
dummy.next
```

Returning `dummy` adds an extra fake value to the list.

#### Merging One by One and Calling It Divide-and-Conquer

This is not balanced:

```text
(((L0 + L1) + L2) + L3) + ...
```

Balanced merging looks like:

```text
(L0 + L1), (L2 + L3), (L4 + L5), ...
then merge those results again
```

The balanced structure is what gives `O(N log k)` time.

#### Assuming Values Are Unique

The lists may contain duplicate values. The output must preserve all copies:

```text
[1, 1, 2, 3, 4, 4, 5, 6]
```

Do not use a set or any structure that removes duplicates.

---

### 12. First-Principles Summary

The problem looks difficult because there are `k` lists, but the essential operation is only this:

> Merge two sorted linked lists by repeatedly taking the smaller head.

Once that operation is correct, the remaining question is merge order.

Merging one list at a time can repeatedly move the same nodes through larger and larger intermediate lists. Divide-and-conquer avoids that by keeping merges balanced.

The invariant is:

```text
Each recursive range returns one sorted list containing exactly the nodes from that range.
```

The base case is a single already-sorted list. The merge step combines two already-correct sorted lists into a larger correct sorted list.

That is why the whole algorithm is correct, and why its time complexity is `O(N log k)` instead of `O(Nk)`.

## Implementation
See `solutions/divide_conquer/p023_merge_k_sorted_lists.py`.

## Tests
See `tests/divide_conquer/test_p023_merge_k_sorted_lists.py`.

## Examples

### Example 1
- Input: `{'lists': [[1, 4, 5], [1, 3, 4], [2, 6]]}`
- Output: `[1, 1, 2, 3, 4, 4, 5, 6]`

### Example 2
- Input: `{'lists': []}`
- Output: `[]`

### Example 3
- Input: `{'lists': [[]]}`
- Output: `[]`

## Follow-up Practice
- Manually merge two sorted linked lists until the pointer movement feels mechanical.
- Draw the divide-and-conquer recursion tree for `k = 5` lists.
- Compare left-to-right merging with balanced pairwise merging on equal-sized lists.
- Test empty input, one empty list, duplicate values, negative values, and lists of very different lengths.
