# 19. Remove Nth Node From End of List

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/remove-nth-node-from-end-of-list/
- Official Group: Linked List
- Pattern Group: Linked List
- Patterns: linked-list

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given the head of a singly linked list and an integer `n`.

The task is to remove the `n`th node from the end of the list and return the head of the resulting list.

For example:

```text
head = [1, 2, 3, 4, 5]
n = 2
```

Counting from the end:

```text
5 is 1st from the end
4 is 2nd from the end
3 is 3rd from the end
2 is 4th from the end
1 is 5th from the end
```

So we remove `4`:

```text
[1, 2, 3, 5]
```

The important linked-list detail is that you do not remove a node by deleting an array slot. In a singly linked list, each node only knows its `next` node. To remove a node, you must change the pointer of the node immediately before it.

If the list is:

```text
1 -> 2 -> 3 -> 4 -> 5 -> null
```

and we want to remove `4`, the actual operation is:

```text
3.next = 5
```

The real problem is therefore:

> Find the node immediately before the `n`th node from the end, then make it skip the node being removed.

There is one special case: if the node to remove is the original head, there is no previous real node. A clean solution handles that case without separate branching.

---

### 2. Start From the Brute Force Baseline

The most direct way to know which node is `n`th from the end is to first know the length of the list.

Suppose the list length is `L`.

The `n`th node from the end is the:

```text
(L - n + 1)th node from the beginning     using 1-based indexing
```

The node before it is the:

```text
(L - n)th node from the beginning
```

For example:

```text
head = [1, 2, 3, 4, 5]
n = 2
L = 5
```

The node to remove is position:

```text
L - n + 1 = 5 - 2 + 1 = 4
```

So we remove the 4th node, value `4`.

A two-pass baseline is:

1. Walk the list once to count `L`.
2. Compute the position to remove.
3. Walk again to the node just before that position.
4. Rewire its `next` pointer.

Pseudocode:

```python
length = 0
current = head
while current:
    length += 1
    current = current.next

position_before_removed = length - n

if position_before_removed == 0:
    return head.next

current = head
for _ in range(position_before_removed - 1):
    current = current.next

current.next = current.next.next
return head
```

This is correct and uses `O(1)` extra space, but it makes two passes over the list. The follow-up usually asks for a one-pass solution.

---

### 3. Key Observation: Distance From the End Can Be Converted Into a Gap

The difficulty is that a singly linked list does not tell us its length up front.

But we do not actually need the length. We only need to identify the node immediately before the node to remove.

Think about two pointers walking at the same speed:

```text
fast pointer
slow pointer
```

If `fast` starts `n + 1` links ahead of `slow`, and both move one step at a time, then that gap stays constant.

When `fast` reaches the end, `slow` must be exactly one node before the node that is `n`th from the end.

Why `n + 1` links instead of `n`?

Because removal needs the predecessor.

If `slow` should stop before the removed node, then there must be exactly `n` nodes after `slow.next` through the end, and `fast` can be used as the marker that proves this distance.

A dummy node makes this especially clean.

---

### 4. Why Use a Dummy Node?

Create a new node before the original head:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
```

The dummy is not part of the input list values. It is a temporary helper whose `next` points to the real head.

The dummy solves the head-removal case.

For example:

```text
head = [1]
n = 1
```

The node to remove is the head itself. Without a dummy, there is no real predecessor node whose `next` can be changed.

With a dummy:

```text
dummy -> 1 -> null
```

Removing `1` is just:

```text
dummy.next = dummy.next.next
```

which becomes:

```text
dummy -> null
```

Then the answer is always:

```text
dummy.next
```

This is the same return expression whether the original head was removed or not.

---

### 5. The Two-Pointer/Dummy Invariant

Initialize both pointers at `dummy`:

```text
slow = dummy
fast = dummy
```

First move `fast` forward `n + 1` times.

After that setup, maintain this invariant:

```text
fast is exactly n + 1 links ahead of slow
```

Then repeatedly move both pointers together:

```text
slow = slow.next
fast = fast.next
```

Because they move at the same speed, the gap stays the same.

When `fast` reaches `null`, the invariant tells us:

```text
slow is immediately before the node to remove
```

Then the removal is one pointer rewrite:

```text
slow.next = slow.next.next
```

Finally return:

```text
dummy.next
```

This invariant is the whole solution. The pointers are not magic; they are just a way to preserve the exact distance needed to locate the predecessor in one pass.

---

### 6. Detailed Algorithm

1. Create a `dummy` node whose `next` is `head`.

```text
dummy.next = head
```

2. Set both pointers to `dummy`.

```text
fast = dummy
slow = dummy
```

3. Advance `fast` exactly `n + 1` links.

```text
repeat n + 1 times:
    fast = fast.next
```

After this, `fast` is ahead of `slow` by one more than the number of nodes that should remain after the predecessor.

4. Move both pointers until `fast` falls off the list.

```text
while fast is not null:
    fast = fast.next
    slow = slow.next
```

5. Now `slow.next` is the node to remove. Skip it.

```text
slow.next = slow.next.next
```

6. Return `dummy.next`.

This works because `dummy.next` is the correct head after the rewrite, even if the original head was removed.

---

### 7. Example Walkthrough: `head = [1, 2, 3, 4, 5]`, `n = 2`

Start with a dummy node:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
slow
fast
```

Both `slow` and `fast` start at `dummy`.

Because `n = 2`, advance `fast` by `n + 1 = 3` links.

After 1 step:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
slow    fast
```

After 2 steps:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
slow         fast
```

After 3 steps:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
slow              fast
```

Now the gap is fixed. Move both pointers together.

Move 1:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
         slow              fast
```

Move 2:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
              slow              fast
```

Move 3:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
                   slow              fast
```

Now `fast` is `null`, so stop.

`slow` is at node `3`, immediately before node `4`, which is the node to remove.

Before rewiring:

```text
3 -> 4 -> 5
```

Rewrite:

```text
slow.next = slow.next.next
```

After rewiring:

```text
3 -> 5
```

The full list is now:

```text
dummy -> 1 -> 2 -> 3 -> 5 -> null
```

Return:

```text
dummy.next
```

which is:

```text
[1, 2, 3, 5]
```

---

### 8. Edge Case Walkthrough: Removing the Head

Consider:

```text
head = [1, 2]
n = 2
```

The 2nd node from the end is `1`, the head.

With a dummy:

```text
dummy -> 1 -> 2 -> null
slow
fast
```

Advance `fast` by `n + 1 = 3` links:

```text
dummy -> 1 -> 2 -> null
slow                   fast
```

`fast` is already `null`, so the simultaneous-move loop does not run.

`slow` is still `dummy`, and `slow.next` is the original head `1`.

Remove it:

```text
dummy.next = dummy.next.next
```

Now:

```text
dummy -> 2 -> null
```

Return `dummy.next`, giving:

```text
[2]
```

No special-case branch was needed.

---

### 9. Python-Style Pseudocode

```python
def removeNthFromEnd(head, n):
    dummy = ListNode(0)
    dummy.next = head

    slow = dummy
    fast = dummy

    for _ in range(n + 1):
        fast = fast.next

    while fast is not None:
        fast = fast.next
        slow = slow.next

    slow.next = slow.next.next
    return dummy.next
```

Depending on the local `ListNode` definition, the dummy can also be created as:

```python
dummy = ListNode(0, head)
```

The essential idea is not the dummy value. The value is ignored. The important part is that `dummy.next` points to `head`.

---

### 10. Correctness Argument

We prove that the algorithm returns the list after removing exactly the `n`th node from the end.

#### Lemma 1: After the initial advancement, `fast` is `n + 1` links ahead of `slow`.

Both pointers start at `dummy`. The algorithm advances only `fast` exactly `n + 1` times before moving `slow` at all. Therefore, after this setup, the distance from `slow` to `fast` is exactly `n + 1` links.

#### Lemma 2: During the simultaneous movement loop, the distance between `slow` and `fast` remains `n + 1` links.

Each loop iteration advances both pointers by exactly one link. Advancing both endpoints by the same amount preserves their distance. Since the distance was `n + 1` before the loop, it remains `n + 1` after every iteration.

#### Lemma 3: When the loop ends, `slow.next` is the `n`th node from the end.

The loop ends when `fast` is `null`, meaning `fast` has moved one link past the last node. By Lemma 2, `fast` is still `n + 1` links ahead of `slow` at that moment.

So from `slow` to `null` there are `n + 1` links:

```text
slow -> node_to_remove -> ... n total real nodes ... -> null
```

That means there are exactly `n` real nodes from `slow.next` through the tail. Therefore, `slow.next` is exactly the `n`th node from the end.

#### Lemma 4: The pointer rewrite removes exactly that node and preserves all other nodes in order.

The statement:

```text
slow.next = slow.next.next
```

changes the predecessor `slow` to point to the node after `slow.next`. Thus `slow.next` is skipped. No other `next` pointer is changed, so every other node remains reachable in the same relative order.

#### Theorem: The algorithm returns the correct resulting list.

By Lemma 3, `slow.next` is exactly the node that must be removed. By Lemma 4, the rewrite removes exactly that node and preserves all others. Because the returned head is `dummy.next`, the return value is correct both when the original head remains and when the original head is removed. Therefore, the algorithm is correct.

---

### 11. Complexity

Let `L` be the number of nodes in the list.

- Time: `O(L)`
  - `fast` advances at most `L + 1` times including the dummy-to-null movement.
  - `slow` advances at most `L` times.
  - The total work is linear.

- Space: `O(1)`
  - Only a dummy node and two pointers are used.
  - No array, stack, hash map, or recursion is required.

---

### 12. Common Pitfalls

- Advancing `fast` only `n` steps instead of `n + 1` steps when `slow` starts at `dummy`.
  - That makes `slow` stop on the node to remove, not the predecessor.

- Forgetting the dummy node.
  - Removing the original head then needs a special case, and many bugs happen there.

- Returning `head` instead of `dummy.next`.
  - If the head was removed, `head` still points to the old removed node.

- Rewiring the wrong pointer.
  - The removal must be done from the predecessor:

```text
slow.next = slow.next.next
```

- Losing track of what `n` means.
  - `n = 1` removes the tail.
  - `n = length` removes the head.

- Mixing array indexing with linked-list movement.
  - There is no direct access to the `k`th node; every position is reached by following `next` links.

---

### 13. First-Principles Summary

A singly linked list removal is not about the node being removed first. It is about the node before it, because that predecessor owns the pointer that must change.

The brute-force way finds the predecessor by computing the list length. The one-pass way avoids computing length by maintaining a fixed gap:

```text
fast is n + 1 links ahead of slow
```

When `fast` reaches `null`, the fixed gap forces `slow` to be the predecessor of the `n`th node from the end.

The dummy node turns head removal into the same operation as every other removal:

```text
slow.next = slow.next.next
```

Then `dummy.next` is always the correct new head.

## Implementation
See `solutions/linked_list/p019_remove_nth_node_from_end_of_list.py`.

## Tests
See `tests/linked_list/test_p019_remove_nth_node_from_end_of_list.py`.

## Examples

### Example 1
- Input: `{'head': [1, 2, 3, 4, 5], 'n': 2}`
- Output: `[1, 2, 3, 5]`

### Example 2
- Input: `{'head': [1], 'n': 1}`
- Output: `[]`

### Example 3
- Input: `{'head': [1, 2], 'n': 1}`
- Output: `[1]`

## Follow-up Practice
- Redraw the list with a dummy node before writing code.
- Mark the exact predecessor of the removed node, not just the removed node.
- Test removing the tail, removing the head, and removing from a one-node list.
