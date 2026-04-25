# 92. Reverse Linked List II

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/reverse-linked-list-ii/
- Official Group: Linked List
- Pattern Group: Linked List
- Patterns: linked-list

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given the head of a singly linked list and two 1-indexed positions:

```text
left
right
```

Reverse only the nodes from position `left` through position `right`, inclusive.

Everything before `left` must stay in the same order.
Everything after `right` must stay in the same order.
Only the middle segment is reversed.

For example:

```text
head  = 1 -> 2 -> 3 -> 4 -> 5
left  = 2
right = 4
```

The segment from position `2` to position `4` is:

```text
2 -> 3 -> 4
```

Reversing that segment gives:

```text
4 -> 3 -> 2
```

So the final list is:

```text
1 -> 4 -> 3 -> 2 -> 5
```

The problem is not asking us to create a new list. It asks us to modify the links of the existing list so that exactly one contiguous sublist is reversed.

The real question is:

> How can we reverse a middle portion of a singly linked list without losing access to the rest of the list?

---

### 2. Start From the Brute Force Idea

The most direct way to avoid pointer mistakes is to stop thinking about pointers temporarily.

A brute-force approach would be:

1. Traverse the linked list and copy all node values into an array.
2. Reverse the array slice from `left - 1` through `right - 1`.
3. Traverse the linked list again and write the values back into the nodes.

Conceptually:

```python
values = []
node = head

while node:
    values.append(node.val)
    node = node.next

values[left - 1:right] = reversed(values[left - 1:right])

node = head
for value in values:
    node.val = value
    node = node.next
```

This produces the right sequence of values, and it is easy to reason about.

But it does not solve the linked-list problem in the strongest sense:

- It uses `O(n)` extra space.
- It rewrites values instead of changing node links.
- It avoids the central challenge: reversing part of a singly linked list in place.

A better solution should use the structure of the list directly.

---

### 3. The Key Observation

A singly linked list gives each node only one outgoing pointer:

```text
node.next
```

To reverse a sublist, we must redirect some of these `next` pointers.

For the example:

```text
1 -> 2 -> 3 -> 4 -> 5
     ^         ^
    left     right
```

we want:

```text
1 -> 4 -> 3 -> 2 -> 5
```

Only four boundary facts matter:

```text
node before the sublist: 1
first node in sublist:   2
last node in sublist:    4
node after the sublist:  5
```

After reversal:

```text
node before the sublist should point to the old last node
old first node should point to the node after the sublist
```

So the subproblem is not “reverse the whole list.”
It is:

> Keep the outside boundaries stable while repeatedly moving nodes from inside the sublist to the front of that sublist.

That leads to a clean in-place method.

---

### 4. Why a Dummy Node Helps

The sublist may start at the head.

For example:

```text
head  = 1 -> 2 -> 3
left  = 1
right = 2
```

The answer is:

```text
2 -> 1 -> 3
```

Here the head changes from `1` to `2`.

Head changes are a common source of special cases. A dummy node removes that special case.

Create:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5
```

Now every sublist has a node before it:

- If `left = 1`, the node before the sublist is `dummy`.
- If `left > 1`, the node before the sublist is a real list node.

At the end, return:

```text
dummy.next
```

This works whether the original head changed or not.

---

### 5. Pointer Roles

After positioning ourselves just before the sublist, keep two important pointers:

```text
before = node immediately before position left
start  = first node in the sublist
```

For:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5
left = 2, right = 4
```

we have:

```text
before = 1
start  = 2
```

Visually:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5
         ^    ^
      before start
```

The node `start` is important because after reversal it becomes the tail of the reversed sublist.

In the final result:

```text
1 -> 4 -> 3 -> 2 -> 5
              ^
            start
```

So `start.next` must eventually point to the first node after the reversed region.

---

### 6. Sublist-Reversal Pointer Invariant

The in-place trick is to repeatedly take the node immediately after `start` and move it to the front of the sublist, right after `before`.

Maintain this invariant:

```text
before.next is the head of the already-reversed prefix of the target sublist.
start is the tail of that reversed prefix.
start.next is the first node not yet moved from the remaining part of the target sublist.
All nodes before `before` and after the not-yet-processed suffix remain reachable.
```

Initially, the reversed prefix contains only `start`:

```text
before -> start -> next_node -> ...
```

That one-node prefix is already reversed.

Each operation moves `start.next` to the front:

```text
node_to_move = start.next
```

Before moving:

```text
before -> reversed_prefix_head -> ... -> start -> node_to_move -> rest
```

After moving:

```text
before -> node_to_move -> reversed_prefix_head -> ... -> start -> rest
```

This grows the reversed prefix by one node while keeping `start` as the tail.

The order of pointer assignments matters because a singly linked list has no backward links. If we overwrite a `next` pointer too early, we may lose access to the rest of the list.

The safe rewiring sequence is:

```python
node_to_move = start.next
start.next = node_to_move.next
node_to_move.next = before.next
before.next = node_to_move
```

Read it as four concrete actions:

1. Remember the node being moved.
2. Detach it from after `start`.
3. Point it to the current front of the reversed prefix.
4. Make it the new front of the reversed prefix.

---

### 7. Detailed Algorithm

1. Create a dummy node whose `next` points to `head`.
2. Move `before` so that it points to the node immediately before position `left`.
3. Set `start = before.next`.
4. Repeat `right - left` times:
   - Let `node_to_move = start.next`.
   - Detach `node_to_move` by setting `start.next = node_to_move.next`.
   - Insert `node_to_move` immediately after `before`.
5. Return `dummy.next`.

Why repeat `right - left` times?

The sublist length is:

```text
right - left + 1
```

A sublist of length `1` needs `0` moves.
A sublist of length `3` needs `2` moves:

```text
2 -> 3 -> 4
```

Move `3` to the front:

```text
3 -> 2 -> 4
```

Move `4` to the front:

```text
4 -> 3 -> 2
```

So the number of front-insertion moves is one less than the sublist length:

```text
right - left
```

---

### 8. Detailed Example Walkthrough

Use the official example:

```text
head  = 1 -> 2 -> 3 -> 4 -> 5
left  = 2
right = 4
```

Add a dummy node:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5
```

Move `before` to the node before position `2`:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5
         ^    ^
      before start
```

The target sublist has length `3`, so we perform `2` moves.

#### Move 1

`start` is `2`, so `node_to_move` is `start.next`, which is `3`:

```text
before -> 2 -> 3 -> 4 -> 5
          ^    ^
        start move
```

Detach `3` from after `2`:

```text
2 -> 4 -> 5
```

Insert `3` after `before`:

```text
dummy -> 1 -> 3 -> 2 -> 4 -> 5
         ^         ^
      before     start
```

Now the reversed prefix is:

```text
3 -> 2
```

`start` is still `2`, the tail of the reversed prefix.

#### Move 2

`start.next` is now `4`, so move `4` to the front of the sublist:

Before:

```text
dummy -> 1 -> 3 -> 2 -> 4 -> 5
         ^         ^    ^
      before     start move
```

Detach `4` from after `2`:

```text
2 -> 5
```

Insert `4` after `before`:

```text
dummy -> 1 -> 4 -> 3 -> 2 -> 5
         ^              ^
      before          start
```

The sublist from positions `2` to `4` is now reversed:

```text
4 -> 3 -> 2
```

Return `dummy.next`:

```text
1 -> 4 -> 3 -> 2 -> 5
```

---

### 9. Pseudocode

```python
def reverseBetween(head, left, right):
    dummy = ListNode(0)
    dummy.next = head

    before = dummy
    for _ in range(left - 1):
        before = before.next

    start = before.next

    for _ in range(right - left):
        node_to_move = start.next
        start.next = node_to_move.next
        node_to_move.next = before.next
        before.next = node_to_move

    return dummy.next
```

This is the head-insertion version of partial linked-list reversal.

It does not need a separate pointer to the node after `right` because `start.next` naturally becomes that node after the required number of moves.

---

### 10. Correctness

We prove that the algorithm returns the list with exactly the nodes from position `left` to position `right` reversed.

#### Boundary setup is correct

The dummy node points to the original head, and `before` advances `left - 1` times from the dummy node.

Therefore:

```text
before
```

is the node immediately before the target sublist, even when `left = 1`.

Also:

```text
start = before.next
```

is the first node of the target sublist.

#### The loop invariant is maintained

Before each loop iteration:

```text
before.next
```

points to the head of the reversed prefix of the target sublist, and:

```text
start
```

is the tail of that reversed prefix.

The next node to add to the reversed prefix is:

```text
start.next
```

The algorithm removes that node from after `start` and inserts it immediately after `before`.

This makes the moved node the new head of the reversed prefix. The previous reversed prefix remains after it, and `start` remains the tail. The unprocessed suffix stays reachable through `start.next`.

So the invariant remains true after each iteration.

#### The loop reverses exactly the requested segment

The target segment contains:

```text
right - left + 1
```

nodes.

Initially, the reversed prefix contains the first target node, so it has length `1`.

Each iteration moves one additional target node into the front of the reversed prefix. After:

```text
right - left
```

iterations, the reversed prefix has length:

```text
1 + (right - left) = right - left + 1
```

So it contains the entire target segment, reversed.

No node before `left` is moved because all insertions happen after `before`.
No node after `right` is moved because the loop performs exactly enough moves to consume the target segment and then stops.

#### The final head is correct

Because the dummy node always points to the true head of the resulting list, returning:

```text
dummy.next
```

returns the correct head whether or not the original head was part of the reversed sublist.

Therefore, the algorithm is correct.

---

### 11. Complexity

Let `n` be the number of nodes in the list.

The algorithm first walks to the node before `left`, then performs one constant-time rewiring operation for each node moved inside the sublist.

So the total time is:

```text
O(n)
```

More precisely, it touches only the prefix up to `left` and the reversed segment, but in the worst case that is the whole list.

The algorithm uses only a fixed number of pointers:

```text
dummy, before, start, node_to_move
```

So the auxiliary space is:

```text
O(1)
```

---

### 12. Common Pitfalls

- **Forgetting the dummy node.** If `left = 1`, the head may change. A dummy node makes this case identical to every other case.
- **Moving `before` too far.** `before` must stop at the node immediately before `left`, not at `left` itself.
- **Using the wrong loop count.** The number of moves is `right - left`, not `right - left + 1`.
- **Losing the rest of the list.** Store `node_to_move = start.next` before changing links.
- **Rewiring in an unsafe order.** Detach the moved node first, then insert it at the front.
- **Returning `head` instead of `dummy.next`.** If the reversal starts at position `1`, the original `head` is no longer the answer.
- **Accidentally creating a cycle.** The assignment `start.next = node_to_move.next` must happen before `node_to_move.next = before.next`.

---

### 13. First-Principles Summary

A singly linked list can only be changed by rewiring local `next` pointers.

For this problem, the important boundary is the node immediately before the sublist. Once that node is fixed, the reversal can be built by repeatedly moving the node after the sublist tail to the front of the sublist.

The invariant is:

```text
before.next is the head of the reversed prefix,
start is the tail of that reversed prefix,
start.next is the first unprocessed node.
```

Each iteration grows the reversed prefix by one node and preserves reachability of the rest of the list.

That is why the entire partial reversal can be done in one pass with constant extra space.

## Implementation
See `solutions/linked_list/p092_reverse_linked_list_ii.py`.

## Tests
See `tests/linked_list/test_p092_reverse_linked_list_ii.py`.

## Examples

### Example 1
- Input: `{'head': [1, 2, 3, 4, 5], 'left': 2, 'right': 4}`
- Output: `[1, 4, 3, 2, 5]`

### Example 2
- Input: `{'head': [5], 'left': 1, 'right': 1}`
- Output: `[5]`

## Follow-up Practice
- Reverse a sublist that starts at the head, such as `left = 1, right = 3`.
- Reverse a two-node sublist to check the pointer order.
- Trace `before`, `start`, and `node_to_move` on paper before coding.
