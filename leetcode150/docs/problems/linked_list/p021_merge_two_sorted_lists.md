# 21. Merge Two Sorted Lists

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/merge-two-sorted-lists/
- Official Group: Linked List
- Pattern Group: Linked List
- Patterns: linked-list

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given the heads of two singly linked lists:

```text
list1
list2
```

Each list is already sorted in nondecreasing order.

Your job is to combine all nodes from both lists into one sorted linked list, then return the head of that merged list.

For example:

```text
list1: 1 -> 2 -> 4
list2: 1 -> 3 -> 4
```

The merged result should be:

```text
1 -> 1 -> 2 -> 3 -> 4 -> 4
```

The important detail is that this is a linked-list problem, not an array problem.

In an array, you can freely write values into positions. In a linked list, the structure is built out of nodes and `next` pointers:

```text
node.val
node.next
```

So the real problem is:

> Repeatedly choose the smaller available front node from two sorted chains, connect it to the output chain, and preserve a sorted order until both chains are exhausted.

The sorted order of the input lists is the main source of power. We do not need to inspect every possible ordering of all nodes, because the smallest remaining node of each list is always sitting at that list's current head.

---

### 2. Start From the Brute-Force Baseline

A simple way to solve the problem is to ignore the linked-list structure at first:

1. Traverse `list1` and collect all values.
2. Traverse `list2` and collect all values.
3. Sort the collected values.
4. Build a new linked list from the sorted values.

Conceptually:

```python
values = []

while list1 is not None:
    values.append(list1.val)
    list1 = list1.next

while list2 is not None:
    values.append(list2.val)
    list2 = list2.next

values.sort()

return build_linked_list(values)
```

This is correct because sorting all values gives the required global order.

But it wastes information.

The input lists are already sorted. Sorting all values again costs extra time:

```text
O((m + n) log(m + n))
```

where `m` and `n` are the lengths of the two lists.

It may also allocate a new array and possibly new nodes, which is unnecessary if we are allowed to reuse the existing nodes by changing pointers.

The first-principles question is:

> If both inputs are already sorted, what is the least information we need to decide the next output node?

---

### 3. Key Observation: Only the Two Front Nodes Matter

Suppose the current unmerged parts are:

```text
list1: a -> ...
list2: b -> ...
```

Because `list1` is sorted, every node after `a` has value at least `a`.

Because `list2` is sorted, every node after `b` has value at least `b`.

Therefore, the smallest node among all remaining nodes must be either:

```text
a
```

or:

```text
b
```

It cannot be hidden deeper inside either list, because a deeper node is never smaller than that list's current head.

So each step has a local decision:

```text
if list1.val <= list2.val:
    append list1's current node
else:
    append list2's current node
```

After appending one node, advance only the pointer for the list that supplied that node.

This turns the problem into the same merge step used by merge sort, except the output is a linked chain instead of an array.

---

### 4. Why a Dummy Node Helps

Building a linked list has an annoying special case: the first node determines the head.

Without a dummy node, code often has to say:

```python
if head is None:
    head = chosen
    tail = chosen
else:
    tail.next = chosen
    tail = tail.next
```

That conditional is not about the algorithm's idea. It is only bookkeeping for the first insertion.

A dummy node removes that special case.

Create a temporary node before the real answer:

```text
dummy -> nothing yet
 tail
```

`tail` always points to the last node in the merged list built so far.

At the beginning, no real node has been appended, so `tail` points to `dummy`.

When we append a node, we do:

```python
tail.next = chosen
tail = tail.next
```

At the end, the real merged list starts after the dummy:

```python
return dummy.next
```

The dummy node is not part of the answer. It is just a stable anchor that lets every append use the same pointer operation.

---

### 5. The Pointer Invariant

The heart of the algorithm is this invariant:

```text
dummy.next through tail is a sorted merged chain containing exactly the nodes already chosen.
list1 points to the first unchosen node from the original first list.
list2 points to the first unchosen node from the original second list.
tail.next is where the next chosen node will be attached.
```

This invariant tells us what every pointer means.

Before the loop:

```text
merged chain: empty
list1: first node of the first list, if any
list2: first node of the second list, if any
tail: dummy
```

The invariant is true because no nodes have been chosen yet.

During each loop iteration, both `list1` and `list2` are non-null. We compare their values and choose the smaller front node.

If we choose `list1`, we attach it to `tail.next`, move `list1` forward, and then move `tail` forward:

```python
tail.next = list1
list1 = list1.next
tail = tail.next
```

If we choose `list2`, we do the symmetric operation:

```python
tail.next = list2
list2 = list2.next
tail = tail.next
```

After either operation, the chosen node becomes the new last node of the merged chain, and the source list pointer advances to the next unchosen node.

The invariant remains true.

---

### 6. Detailed Algorithm

Use two moving input pointers and one output-tail pointer.

1. Create a dummy node.
2. Set `tail = dummy`.
3. While both lists still contain nodes:
   - Compare `list1.val` and `list2.val`.
   - Attach the smaller front node to `tail.next`.
   - Advance the pointer of the list that supplied that node.
   - Advance `tail` to the newly attached node.
4. Once one list is empty, attach the other list directly to `tail.next`.
5. Return `dummy.next`.

The final attachment works because of sortedness.

If `list1` is empty, every remaining node in `list2` is already sorted relative to itself, and all previously chosen nodes are no greater than the next node we are attaching. So we can connect the whole remainder at once:

```python
tail.next = list2
```

Likewise, if `list2` is empty:

```python
tail.next = list1
```

There is no need to copy the remainder node by node.

---

### 7. Pseudocode

```python
def mergeTwoLists(list1, list2):
    dummy = ListNode(0)
    tail = dummy

    while list1 is not None and list2 is not None:
        if list1.val <= list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next

        tail = tail.next

    if list1 is not None:
        tail.next = list1
    else:
        tail.next = list2

    return dummy.next
```

Some implementations write the final attachment more compactly:

```python
tail.next = list1 if list1 is not None else list2
```

Both forms mean the same thing.

---

### 8. Example Walkthrough

Use the first official example:

```text
list1: 1 -> 2 -> 4
list2: 1 -> 3 -> 4
```

Start with an empty merged chain:

```text
dummy
tail = dummy

list1: 1 -> 2 -> 4
list2: 1 -> 3 -> 4
```

#### Step 1

Compare the two front values:

```text
list1.val = 1
list2.val = 1
```

They are equal. If the code uses `<=`, choose from `list1`.

Attach that node:

```text
merged: 1
tail:   ^

list1: 2 -> 4
list2: 1 -> 3 -> 4
```

#### Step 2

Compare:

```text
list1.val = 2
list2.val = 1
```

Choose `list2`'s `1`:

```text
merged: 1 -> 1
          tail

list1: 2 -> 4
list2: 3 -> 4
```

#### Step 3

Compare:

```text
list1.val = 2
list2.val = 3
```

Choose `list1`'s `2`:

```text
merged: 1 -> 1 -> 2
               tail

list1: 4
list2: 3 -> 4
```

#### Step 4

Compare:

```text
list1.val = 4
list2.val = 3
```

Choose `list2`'s `3`:

```text
merged: 1 -> 1 -> 2 -> 3
                    tail

list1: 4
list2: 4
```

#### Step 5

Compare:

```text
list1.val = 4
list2.val = 4
```

They are equal. Choose from `list1` because of `<=`:

```text
merged: 1 -> 1 -> 2 -> 3 -> 4
                         tail

list1: empty
list2: 4
```

Now `list1` is empty, so the loop stops.

Attach the remaining `list2` directly:

```text
merged: 1 -> 1 -> 2 -> 3 -> 4 -> 4
```

Return `dummy.next`, which points to the first real node.

---

### 9. Correctness

We prove the algorithm returns a sorted list containing exactly all nodes from the two input lists.

#### Invariant

At the start of every loop iteration:

```text
dummy.next through tail is sorted and contains exactly the nodes already chosen from the original inputs.
list1 and list2 point to the first unchosen nodes of their respective lists.
Every unchosen node is still reachable from either list1 or list2.
```

#### Initialization

Before the loop, no input node has been chosen.

The chain from `dummy.next` through `tail` is empty, so it is sorted and contains exactly the chosen nodes: none.

`list1` and `list2` still point to the heads of the original lists, so all unchosen nodes are reachable.

Therefore, the invariant holds initially.

#### Maintenance

During a loop iteration, both `list1` and `list2` are non-null.

Because each remaining list is sorted, the smallest unchosen node must be one of the two current front nodes.

The algorithm compares those two front nodes and appends the smaller one to the merged chain.

Appending the smallest remaining node preserves sorted order: all previous merged nodes are no larger than it, and it is no larger than every still-unchosen node that could come before the other list's front.

Then the algorithm advances the pointer of the list that supplied the node. This removes exactly that node from the unchosen portion while leaving every other unchosen node reachable.

Finally, `tail` moves to the newly appended node, so `dummy.next` through `tail` again describes exactly the merged chosen chain.

Therefore, the invariant is preserved.

#### Termination

The loop stops when at least one list is empty.

At that point, the invariant says the merged chain is sorted and contains exactly the nodes already chosen.

The non-empty remainder, if any, is already sorted. Also, every node in that remainder is at least as large as the last appended node; otherwise the algorithm would have chosen a smaller front node earlier while both lists were non-empty.

So attaching the entire remaining chain preserves sorted order and includes all remaining nodes exactly once.

The algorithm returns `dummy.next`, the head of the real merged list. Thus the returned list is sorted and contains exactly all nodes from the two input lists.

---

### 10. Complexity

Let:

```text
m = length of list1
n = length of list2
```

Each loop iteration attaches one node and advances one input pointer.

No node is processed more than once.

So the time complexity is:

```text
O(m + n)
```

The auxiliary space complexity is:

```text
O(1)
```

The algorithm uses only a few pointers and one dummy node. It does not allocate an array of all values. If the implementation reuses the existing nodes, the output list is formed by rewiring `next` pointers rather than creating a full second copy of the input.

---

### 11. Common Pitfalls

#### Forgetting the Head Special Case

If you do not use a dummy node, you must handle the first appended node separately.

A dummy node makes the append operation uniform from the beginning:

```python
tail.next = chosen
tail = tail.next
```

#### Advancing the Wrong Pointer

After attaching a node from `list1`, advance `list1`, not `list2`.

After attaching a node from `list2`, advance `list2`, not `list1`.

The source pointer must move because that node has now been consumed.

#### Forgetting to Move `tail`

This is a common bug:

```python
tail.next = list1
list1 = list1.next
# forgot: tail = tail.next
```

If `tail` does not move, the next append overwrites `tail.next` again and breaks the merged chain.

#### Losing the Rest of a List

When rewiring linked lists, order matters.

This safe version works because `list1` is advanced using the chosen node's existing `next` before `tail` is moved away from the appended node:

```python
tail.next = list1
list1 = list1.next
tail = tail.next
```

Another safe style is to save the next pointer explicitly:

```python
chosen = list1
list1 = list1.next
tail.next = chosen
tail = chosen
```

Both are fine as long as the unprocessed remainder remains reachable.

#### Not Attaching the Remainder

When one list becomes empty, the other list may still contain many nodes.

Do not stop immediately and return the partial result. Connect the leftover chain:

```python
tail.next = list1 if list1 is not None else list2
```

#### Confusing Values With Nodes

The problem is usually expected to return a linked list, not a Python list of values.

The examples are shown as arrays only because arrays are easier to display. Internally, `[1, 2, 4]` represents:

```text
1 -> 2 -> 4
```

---

### 12. First-Principles Summary

The sorted input lists give a local guarantee:

```text
The next smallest remaining node must be one of the two current heads.
```

That guarantee is enough to build the entire merged list one node at a time.

The dummy node gives the output chain a stable anchor.

The `tail` pointer marks the end of the part already built.

The `list1` and `list2` pointers mark the unprocessed parts.

At every step, choose the smaller front node, attach it after `tail`, advance the source list, and move `tail` forward.

When one list runs out, attach the other list because it is already sorted.

So the problem is not about sorting from scratch. It is about preserving this pointer invariant:

```text
processed prefix is sorted, unprocessed suffixes remain reachable, and the next decision only depends on the two current heads.
```

## Implementation
See `solutions/linked_list/p021_merge_two_sorted_lists.py`.

## Tests
See `tests/linked_list/test_p021_merge_two_sorted_lists.py`.

## Examples

### Example 1
- Input: `{'list1': [1, 2, 4], 'list2': [1, 3, 4]}`
- Output: `[1, 1, 2, 3, 4, 4]`

### Example 2
- Input: `{'list1': [], 'list2': []}`
- Output: `[]`

### Example 3
- Input: `{'list1': [], 'list2': [0]}`
- Output: `[0]`

## Follow-up Practice
- Merge lists where one input is empty from the start.
- Merge lists with equal values and decide whether you want stable tie handling using `<=`.
- Trace the dummy and tail pointers on paper before coding.
