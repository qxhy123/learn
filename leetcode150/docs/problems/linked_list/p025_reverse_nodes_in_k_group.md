# 25. Reverse Nodes in k-Group

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/reverse-nodes-in-k-group/
- Official Group: Linked List
- Pattern Group: Linked List
- Patterns: linked-list

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given the head of a singly linked list and an integer `k`, reverse the nodes of the list in consecutive blocks of exactly `k` nodes.

For example, if:

```text
head = 1 -> 2 -> 3 -> 4 -> 5
k = 2
```

then the list is split into groups of size `2`:

```text
[1, 2] [3, 4] [5]
```

Each complete group is reversed:

```text
[2, 1] [4, 3] [5]
```

So the answer is:

```text
2 -> 1 -> 4 -> 3 -> 5
```

The final group is different. If fewer than `k` nodes remain, those nodes must stay in their original order.

For `k = 3`:

```text
head = 1 -> 2 -> 3 -> 4 -> 5
```

The groups are:

```text
[1, 2, 3] [4, 5]
```

Only the first group has exactly `3` nodes, so the answer is:

```text
3 -> 2 -> 1 -> 4 -> 5
```

The operation is not asking us to change node values. It asks us to change the links between existing nodes.

That distinction matters because in a linked list, the list order is defined entirely by `next` pointers:

```text
node.next
```

So the real task is:

> Rewire the list so that every full block of `k` existing nodes points backward, while all blocks remain connected in the correct overall order.

### 2. Start From the Brute Force Idea

The easiest way to think about the problem is to temporarily ignore the pointer constraints.

A brute-force approach could be:

1. Walk through the list and copy all nodes, or all node values, into an array.
2. Reverse every complete slice of length `k` in that array.
3. Rebuild a linked list, or rewrite values in the original nodes.

Conceptually:

```python
values = []
node = head
while node:
    values.append(node.val)
    node = node.next

for start in range(0, len(values), k):
    if start + k <= len(values):
        values[start:start + k] = reversed(values[start:start + k])

return build_list(values)
```

This captures the output rule correctly, but it avoids the core linked-list challenge.

There are two problems with using this as the final solution:

- It uses `O(n)` extra space for the array.
- If we rebuild or rewrite values, we are no longer solving the pointer problem directly.

The problem is designed to be solved in-place by changing links. Therefore, the first-principles question is:

> Can we reverse one complete group using only a constant number of pointers, then attach it to the already processed part of the list?

If we can do that, then repeating the same local operation group by group solves the whole list.

### 3. The Key Observation: A Complete Group Is a Closed Local Problem

Suppose the list currently looks like this:

```text
processed part -> A -> B -> C -> D -> remaining part
```

and `k = 3`.

The next group to reverse is:

```text
A -> B -> C
```

After reversal, that local group should become:

```text
C -> B -> A
```

But the whole list must still be connected:

```text
processed part -> C -> B -> A -> D -> remaining part
```

Only a few boundary pointers matter:

```text
group_prev -> A -> B -> C -> group_next
```

where:

- `group_prev` is the node immediately before the group.
- `A` is the original first node of the group.
- `C` is the original last node of the group.
- `group_next` is the node immediately after the group.

After reversing the group:

```text
group_prev -> C -> B -> A -> group_next
```

The inner nodes are reversed, and the two outside connections are restored.

So each group reversal has two jobs:

1. Reverse the `k` links inside the group.
2. Reconnect the reversed group to the previous processed part and the next unprocessed part.

### 4. Why a Dummy Node Helps

The first group may include the original `head`.

For example:

```text
1 -> 2 -> 3 -> 4 -> 5
k = 2
```

After reversing the first group, the head changes from `1` to `2`:

```text
2 -> 1 -> 3 -> 4 -> 5
```

Head changes are a common source of special cases in linked-list problems. A dummy node removes that special case.

Create:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5
```

Now every group, including the first one, has a node before it:

```text
group_prev
```

For the first group, `group_prev` is `dummy`.

At the end, the real answer is:

```text
dummy.next
```

### 5. The Group-Reversal Pointer Invariant

The central invariant is:

```text
All nodes before group_prev are already in final order.
group_prev.next is the first node of the next candidate group.
No nodes after group_prev have been lost.
```

Before processing a group:

```text
final processed part -> group_prev -> first -> ...
```

We then ask:

> Are there at least `k` nodes after `group_prev`?

If not, the remaining nodes are fewer than `k`, so they must stay unchanged and the algorithm is done.

If yes, let:

```text
kth = the kth node after group_prev
group_next = kth.next
```

The candidate group is exactly:

```text
group_prev.next ... kth
```

The node after the group is:

```text
group_next
```

During the reversal, maintain this inner invariant:

```text
prev points to the already reversed suffix of the group, followed by group_next.
curr points to the first not-yet-reversed node inside the group.
```

Initialize:

```text
prev = group_next
curr = group_prev.next
```

Why does `prev` start at `group_next`?

Because the original first node of the group will become the last node after reversal. Its `next` pointer should eventually point to `group_next`.

Then repeatedly move `curr` from the unreversed part to the front of the reversed part:

```text
nxt = curr.next
curr.next = prev
prev = curr
curr = nxt
```

Stop when `curr == group_next`.

At that exact moment, every node in the group has been reversed, and `prev` points to the new head of the reversed group.

### 6. Detailed Algorithm

Use `group_prev` to mark the node before the next group to process.

1. Create a dummy node whose `next` is `head`.
2. Set `group_prev = dummy`.
3. Repeatedly find the kth node after `group_prev`.
4. If there is no kth node, return `dummy.next` because the remaining suffix is too short to reverse.
5. Save `group_next = kth.next`.
6. Reverse the nodes from `group_prev.next` up to `kth`, stopping before `group_next`.
7. Reconnect the reversed group:
   - `group_prev.next` should point to the new group head.
   - The old group head becomes the tail of the reversed group.
8. Move `group_prev` to that new tail.
9. Continue with the next group.

The reconnection step is the easiest place to make a mistake, so name the old group head before rewiring:

```text
group_tail = group_prev.next
```

After reversal:

```text
prev = new head of reversed group
group_tail = old head, now tail
```

Reconnect:

```text
group_prev.next = prev
group_prev = group_tail
```

The tail already points to `group_next` because the reversal initialized `prev = group_next`.

### 7. Pseudocode

```python
def reverseKGroup(head, k):
    dummy = ListNode(0, head)
    group_prev = dummy

    while True:
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if kth is None:
                return dummy.next

        group_next = kth.next
        group_tail = group_prev.next

        prev = group_next
        curr = group_prev.next

        while curr is not group_next:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt

        group_prev.next = prev
        group_prev = group_tail
```

This is a pointer-only solution. It does not allocate an array, does not rebuild the list, and does not swap values.

### 8. Detailed Example Walkthrough

Use:

```text
head = 1 -> 2 -> 3 -> 4 -> 5
k = 2
```

Add the dummy node:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5
```

Initially:

```text
group_prev = dummy
```

#### First Group

Find the kth node after `group_prev`.

For `k = 2`, the group is:

```text
1 -> 2
```

So:

```text
kth = 2
group_next = 3
group_tail = 1
```

Before reversal:

```text
dummy -> 1 -> 2 -> 3 -> 4 -> 5
         ^         ^
         curr      group_next is after kth
```

Initialize:

```text
prev = 3
curr = 1
```

Reverse node `1`:

```text
nxt = 2
1.next = 3
prev = 1
curr = 2
```

Now the partially reversed structure is:

```text
1 -> 3 -> 4 -> 5
2 -> 1 -> 3 -> 4 -> 5   after the next step completes
```

Reverse node `2`:

```text
nxt = 3
2.next = 1
prev = 2
curr = 3
```

Now `curr == group_next`, so the group is fully reversed.

The new group head is `prev = 2`.

Reconnect from `group_prev`:

```text
dummy.next = 2
```

Move `group_prev` to the tail of the reversed group:

```text
group_prev = 1
```

The list is now:

```text
dummy -> 2 -> 1 -> 3 -> 4 -> 5
```

Everything up through node `1` is final.

#### Second Group

Now `group_prev` is node `1`, so the next candidate group starts at node `3`.

Find two nodes:

```text
3 -> 4
```

So:

```text
kth = 4
group_next = 5
group_tail = 3
```

Reverse with:

```text
prev = 5
curr = 3
```

Reverse node `3`:

```text
3.next = 5
prev = 3
curr = 4
```

Reverse node `4`:

```text
4.next = 3
prev = 4
curr = 5
```

Reconnect:

```text
1.next = 4
group_prev = 3
```

The list is now:

```text
dummy -> 2 -> 1 -> 4 -> 3 -> 5
```

Everything up through node `3` is final.

#### Remaining Suffix

The next candidate group starts at node `5`.

There is only one node left, but `k = 2`, so there is no complete group.

The remaining suffix stays unchanged.

Return:

```text
2 -> 1 -> 4 -> 3 -> 5
```

### 9. Correctness

We prove the algorithm returns the list obtained by reversing every complete group of `k` nodes and leaving the final incomplete group unchanged.

#### Lemma 1: The algorithm only reverses complete groups.

Before reversing any group, the algorithm searches for the kth node after `group_prev`. If that kth node does not exist, fewer than `k` nodes remain, and the algorithm returns immediately without changing the suffix. Therefore, every reversed group contains exactly `k` nodes.

#### Lemma 2: Reversing one group preserves all nodes and connects the group to the following suffix.

For a complete group, the algorithm saves `group_next = kth.next`. It initializes `prev = group_next` and then processes nodes from the group head until `curr == group_next`. On each iteration, it saves `curr.next` in `nxt` before changing any pointer, sets `curr.next = prev`, and advances `prev` and `curr`. Thus no unprocessed node is lost. Because `prev` initially equals `group_next`, the original group head becomes the final tail and points to the suffix after reversal.

#### Lemma 3: After each successful group reversal, all nodes up to `group_prev` are in final order.

Before a group is processed, the invariant states that all nodes before `group_prev` are already final and `group_prev.next` begins the next candidate group. After reversing the next complete group, the algorithm connects `group_prev.next` to the new group head and moves `group_prev` to the old group head, which is now the group tail. The processed prefix is extended by exactly one correctly reversed group, so all nodes up to the new `group_prev` are final.

#### Lemma 4: The algorithm preserves the relative order of the final incomplete group.

If fewer than `k` nodes remain, the algorithm detects that no kth node exists and returns without rewiring any node in that suffix. Therefore, the final incomplete group remains in its original order.

#### Theorem: The algorithm is correct.

By Lemma 1, only complete groups are reversed. By Lemma 2, each reversed group has exactly the correct internal order and remains connected to the rest of the list. By Lemma 3, the processed prefix is correct after every iteration. By Lemma 4, the final incomplete suffix is unchanged. When the algorithm terminates, every complete group has been reversed, the incomplete suffix is unchanged, and the returned head is `dummy.next`, which is the correct head of the transformed list.

### 10. Complexity

Let `n` be the number of nodes in the list.

Each node is visited a constant number of times:

- Once while checking whether its group has `k` nodes.
- Once while reversing its group, if it belongs to a complete group.

So the time complexity is:

```text
O(n)
```

The algorithm uses only a fixed number of pointers:

```text
dummy, group_prev, kth, group_next, group_tail, prev, curr, nxt
```

So the auxiliary space complexity is:

```text
O(1)
```

### 11. Common Pitfalls

- **Reversing an incomplete final group.** Always find the kth node before changing pointers.
- **Losing the rest of the list.** Save `group_next = kth.next` before reversing the group.
- **Losing the next node inside reversal.** Save `nxt = curr.next` before assigning `curr.next = prev`.
- **Forgetting that the head can change.** Use a dummy node and return `dummy.next`.
- **Moving `group_prev` to the wrong node.** After reversal, `group_prev` must become the old group head, because that node is now the tail of the reversed group.
- **Using `while curr` instead of `while curr is not group_next`.** The reversal must stop exactly after the current group, not at the end of the entire list.
- **Swapping values instead of nodes.** The intended linked-list solution rewires nodes; value swapping can hide pointer mistakes and may violate stricter interpretations of the problem.
- **Creating a cycle.** Cycles usually happen when the old group head is not connected to `group_next` or when the stop condition crosses the group boundary.

### 12. First-Principles Summary

A singly linked list gives access only through local `next` pointers, so the solution must protect reachability before every rewrite.

The problem becomes simple once one group is isolated by its boundaries:

```text
group_prev -> group head ... kth -> group_next
```

If the kth node exists, the group is complete and may be reversed. If it does not exist, the remaining suffix is too short and must be left alone.

The core trick is initializing:

```text
prev = group_next
curr = group_prev.next
```

Then each pointer rewrite moves one node from the unreversed group into the reversed prefix. When `curr == group_next`, the whole group has been reversed, `prev` is the new group head, and the old group head is the new tail.

Finally, `group_prev.next = prev` attaches the processed prefix to the reversed group, and moving `group_prev` to the old group head prepares the invariant for the next group.

So the whole algorithm is just this repeated local transformation:

```text
before: group_prev -> A -> B -> C -> group_next
reverse k nodes
 after: group_prev -> C -> B -> A -> group_next
```

with the invariant that everything before `group_prev` is already final.

## Implementation
See `solutions/linked_list/p025_reverse_nodes_in_k_group.py`.

## Tests
See `tests/linked_list/test_p025_reverse_nodes_in_k_group.py`.

## Examples

### Example 1
- Input: `{'head': [1, 2, 3, 4, 5], 'k': 2}`
- Output: `[2, 1, 4, 3, 5]`

### Example 2
- Input: `{'head': [1, 2, 3, 4, 5], 'k': 3}`
- Output: `[3, 2, 1, 4, 5]`

## Follow-up Practice
- Trace `k = 1`; the algorithm should leave the list unchanged while still satisfying the invariant.
- Trace a list whose length is exactly divisible by `k`.
- Trace a list whose final group has `k - 1` nodes.
- Draw `group_prev`, `kth`, `group_next`, `prev`, and `curr` before writing code.
