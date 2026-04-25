# 61. Rotate List

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/rotate-list/
- Official Group: Linked List
- Pattern Group: Linked List
- Patterns: linked-list

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given the head of a singly linked list and an integer `k`, rotate the list to the right by `k` places.

A right rotation by one place means:

```text
last node moves to the front
all other nodes shift one position to the right
```

For example:

```text
1 -> 2 -> 3 -> 4 -> 5
```

After one right rotation:

```text
5 -> 1 -> 2 -> 3 -> 4
```

After two right rotations:

```text
4 -> 5 -> 1 -> 2 -> 3
```

So for:

```text
head = [1, 2, 3, 4, 5]
k = 2
```

we return:

```text
[4, 5, 1, 2, 3]
```

The important detail is that the values do not change. We are not sorting, reversing, or creating a new order from scratch. We are cutting the existing list at one position and reconnecting the two pieces in the opposite order:

```text
original:  A -> B
rotated:   B -> A
```

For `1 -> 2 -> 3 -> 4 -> 5` rotated right by `2`:

```text
A = 1 -> 2 -> 3
B = 4 -> 5

B -> A = 4 -> 5 -> 1 -> 2 -> 3
```

So the real problem is:

> Find the cut point that separates the final suffix from the final prefix, then rewire only a few links.

---

### 2. Start From the Brute Force Baseline

The most literal way to rotate right by `k` is to perform one rotation at a time.

One right rotation does this:

1. Walk to the tail.
2. Also remember the node before the tail.
3. Detach the tail from the end.
4. Put that tail before the old head.
5. Update `head`.

Conceptually:

```python
for _ in range(k):
    if head is None or head.next is None:
        return head

    previous = None
    tail = head

    while tail.next:
        previous = tail
        tail = tail.next

    previous.next = None
    tail.next = head
    head = tail
```

This is correct, but it is wasteful.

If the list has `n` nodes, each single rotation costs `O(n)` because we must walk to the tail. Doing that `k` times costs:

```text
O(k * n)
```

That is especially bad when `k` is very large. For example, rotating a 5-node list by `1,000,000,002` places should not require one billion list traversals.

The brute-force baseline teaches the shape of the operation, though:

```text
right rotation = move a suffix to the front
```

The optimized solution keeps that idea but finds the final suffix directly.

---

### 3. Key Observation: Rotations Repeat Every List Length

If a list has `n` nodes, rotating it right by exactly `n` places returns the same list.

Example with `n = 5`:

```text
start:      1 -> 2 -> 3 -> 4 -> 5
rotate 1:   5 -> 1 -> 2 -> 3 -> 4
rotate 2:   4 -> 5 -> 1 -> 2 -> 3
rotate 3:   3 -> 4 -> 5 -> 1 -> 2
rotate 4:   2 -> 3 -> 4 -> 5 -> 1
rotate 5:   1 -> 2 -> 3 -> 4 -> 5
```

So only the remainder matters:

```text
effective_rotation = k % n
```

If `effective_rotation == 0`, the list does not change.

This immediately handles huge `k` values. For a 5-node list:

```text
k = 1,000,000,002
k % 5 = 2
```

So rotating by `1,000,000,002` places is exactly the same as rotating by `2` places.

---

### 4. Convert Right Rotation Into a Cut Position

Suppose the list length is `n`, and the effective right rotation is `r`.

The last `r` nodes become the front of the answer.

That means:

```text
original list = first n - r nodes + last r nodes
rotated list  = last r nodes + first n - r nodes
```

So we need to cut after the first `n - r` nodes.

For:

```text
1 -> 2 -> 3 -> 4 -> 5
n = 5
r = 2
```

The first `n - r = 3` nodes are:

```text
1 -> 2 -> 3
```

The last `r = 2` nodes are:

```text
4 -> 5
```

Cut after node `3`:

```text
1 -> 2 -> 3    4 -> 5
```

Then reconnect suffix before prefix:

```text
4 -> 5 -> 1 -> 2 -> 3
```

The new head is the node immediately after the cut.

The new tail is the node immediately before the cut.

---

### 5. Circular-List Pointer Invariant

A clean way to rotate the list is to temporarily make it circular.

While counting the nodes, we can also find the tail. Once we have the tail, connect it back to the head:

```text
tail.next = head
```

Now the list is a cycle:

```text
1 -> 2 -> 3 -> 4 -> 5
^                   |
|___________________|
```

This temporary cycle is useful because a rotation is just choosing a new place to break the circle.

The invariant is:

```text
After tail.next = head, every original node is still reachable by following next pointers,
and the relative order around the circle is exactly the original list order.
```

No node is duplicated. No node is lost. We have only added one link from the old tail to the old head.

Then we walk to the new tail and break the circle there:

```text
new_head = new_tail.next
new_tail.next = None
```

The invariant before the final break is:

```text
new_tail is the last node of the rotated linear list.
new_tail.next is the first node of the rotated linear list.
```

Once we set `new_tail.next = None`, the circular order becomes the required linear order.

This avoids repeatedly moving nodes and reduces the operation to:

```text
count nodes
make one temporary cycle
advance to the new tail
break the cycle
```

---

### 6. Detailed Algorithm

Handle small cases first:

1. If `head` is `None`, return `head`.
2. If `head.next` is `None`, return `head`.
3. If `k == 0`, return `head`.

Then process the general case:

1. Traverse the list once to find:
   - `length`, the number of nodes.
   - `tail`, the last node.
2. Compute:

   ```text
   rotation = k % length
   ```

3. If `rotation == 0`, return the original `head`.
4. Link the list into a circle:

   ```text
   tail.next = head
   ```

5. The new tail is at position:

   ```text
   length - rotation
   ```

   if positions are counted as the number of nodes from the old head to include in the prefix.

   More directly, starting at `head`, move `length - rotation - 1` steps to land on the new tail.

6. Set:

   ```text
   new_head = new_tail.next
   new_tail.next = None
   ```

7. Return `new_head`.

The `-1` in the walk is the common off-by-one point.

For `1 -> 2 -> 3 -> 4 -> 5`, `length = 5`, `rotation = 2`:

```text
length - rotation = 3
```

The new tail is the 3rd node, value `3`.

Starting at node `1`, moving `2` steps reaches node `3`:

```text
start at 1
step 1 -> 2
step 2 -> 3
```

So the loop uses:

```text
length - rotation - 1
```

---

### 7. Pseudocode

```python
def rotateRight(head, k):
    if head is None or head.next is None or k == 0:
        return head

    length = 1
    tail = head

    while tail.next is not None:
        tail = tail.next
        length += 1

    rotation = k % length
    if rotation == 0:
        return head

    tail.next = head

    steps_to_new_tail = length - rotation - 1
    new_tail = head

    for _ in range(steps_to_new_tail):
        new_tail = new_tail.next

    new_head = new_tail.next
    new_tail.next = None

    return new_head
```

The implementation only changes two `next` links:

```text
tail.next = head        # make temporary circle
new_tail.next = None    # break circle at the right place
```

Everything else is pointer movement.

---

### 8. Detailed Example Walkthrough

Use the first official example:

```text
head = [1, 2, 3, 4, 5]
k = 2
```

Initial list:

```text
1 -> 2 -> 3 -> 4 -> 5 -> None
```

First, count nodes and find the tail:

```text
length = 5
tail = node 5
```

Compute the effective rotation:

```text
rotation = k % length
rotation = 2 % 5
rotation = 2
```

Since `rotation` is not zero, the list changes.

Make the temporary circle:

```text
5.next = 1
```

Now the order around the cycle is:

```text
1 -> 2 -> 3 -> 4 -> 5 -> 1 -> ...
```

The last `2` nodes should move to the front:

```text
4 -> 5
```

Therefore the new head should be node `4`, and the new tail should be node `3`.

Compute the number of steps from the old head to the new tail:

```text
steps_to_new_tail = length - rotation - 1
steps_to_new_tail = 5 - 2 - 1
steps_to_new_tail = 2
```

Walk from old head:

```text
new_tail = 1

after 1 step: new_tail = 2
after 2 steps: new_tail = 3
```

So:

```text
new_tail = 3
new_head = new_tail.next = 4
```

Break the circle:

```text
3.next = None
```

The resulting list is:

```text
4 -> 5 -> 1 -> 2 -> 3 -> None
```

Return node `4`.

---

### 9. Walkthrough With `k` Larger Than the Length

Use the second official example:

```text
head = [0, 1, 2]
k = 4
```

The list length is:

```text
length = 3
```

Rotating by `3` would return the same list, so:

```text
rotation = 4 % 3 = 1
```

We only need one right rotation.

The last `1` node becomes the new front:

```text
2
```

The first `length - rotation = 2` nodes move after it:

```text
0 -> 1
```

So the expected result is:

```text
2 -> 0 -> 1
```

Using the circular method:

```text
0 -> 1 -> 2 -> 0 -> ...
```

The new tail is after the first `length - rotation = 2` nodes, so it is node `1`.

Break after node `1`:

```text
2 -> 0 -> 1 -> None
```

---

### 10. Correctness Argument

We prove that the algorithm returns the list rotated right by `k` places.

First, the algorithm computes the list length `n` and the original tail by traversing every node exactly once. Therefore `n` is the true number of nodes, and `tail` is the final node of the original list.

Second, the algorithm replaces `k` with `k % n`. Rotating a list of length `n` by `n` places restores the original list because every node returns to its original position. Therefore rotations that differ by a multiple of `n` produce the same final list. So `k % n` is the only rotation count that matters.

Let:

```text
r = k % n
```

If `r == 0`, no node changes position, so returning the original head is correct.

Now assume `r > 0`.

A right rotation by `r` moves exactly the final `r` nodes of the original list to the front, preserving their internal order, and moves the first `n - r` nodes after them, also preserving their internal order. Thus the desired output has the form:

```text
last r nodes -> first n - r nodes
```

The algorithm creates a temporary circle by setting:

```text
tail.next = head
```

This preserves the original relative order of all nodes around the circle. In that circle, choosing any node as the head and breaking the link before it produces a linear list whose order is the same as walking forward around the original list from that chosen head.

The algorithm chooses `new_tail` as the `(n - r)`-th node of the original list. Therefore `new_tail.next` is the first node among the original last `r` nodes. This node is exactly the required new head.

Finally, setting:

```text
new_tail.next = None
```

breaks the circle after the first `n - r` original nodes. The resulting linear list starts at the original last `r` nodes and then continues through the original first `n - r` nodes. That is exactly the definition of rotating right by `r` places.

Therefore the algorithm is correct.

---

### 11. Complexity

Let `n` be the number of nodes in the list.

The algorithm traverses the list once to compute `length` and find `tail`. It may then walk up to `n - 1` more steps to find the new tail.

So the time complexity is:

```text
O(n)
```

The algorithm uses only a fixed number of pointer variables:

```text
head, tail, new_tail, new_head, length, rotation
```

It does not allocate another list or an array of nodes.

So the auxiliary space complexity is:

```text
O(1)
```

---

### 12. Common Pitfalls

#### Forgetting `k % length`

Without reducing `k`, the algorithm may do unnecessary work or compute the wrong cut position for very large `k`.

Always reduce:

```text
rotation = k % length
```

#### Breaking at the wrong node

For a right rotation by `r`, the new tail is not the `r`-th node. It is the `(length - r)`-th node from the old head.

For example, with:

```text
1 -> 2 -> 3 -> 4 -> 5
r = 2
```

The new tail is `3`, not `2` or `4`.

#### Missing the `-1` when walking from `head`

If the new tail is the `(length - rotation)`-th node, and you start with `new_tail = head`, you already stand on the first node.

So the number of pointer moves is:

```text
length - rotation - 1
```

not:

```text
length - rotation
```

#### Returning before breaking the cycle

After:

```text
tail.next = head
```

there is a cycle. If you return without setting `new_tail.next = None`, any code that traverses the list can loop forever.

#### Mishandling empty or one-node lists

For an empty list or a single-node list, rotation changes nothing. Return immediately before doing tail logic.

#### Rewiring before saving `new_head`

Do this order:

```text
new_head = new_tail.next
new_tail.next = None
```

If you break first, you lose the pointer to the new head.

---

### 13. First-Principles Summary

A right rotation does not require moving nodes one by one. It only changes where the list starts and where it ends.

The list can be viewed as two consecutive pieces:

```text
prefix = first length - rotation nodes
suffix = last rotation nodes
```

The rotated result is:

```text
suffix -> prefix
```

A singly linked list makes it awkward to jump directly to the suffix, so we first count the nodes. Once we know the length, the cut position is determined exactly.

The circular-list trick makes the pointer logic simple:

```text
old tail points to old head
walk to the new tail
break after the new tail
```

The invariant is that the temporary circle preserves every original node in the original order. Rotation is just choosing a different node as the first node of that preserved circular order.

## Implementation
See `solutions/linked_list/p061_rotate_list.py`.

## Tests
See `tests/linked_list/test_p061_rotate_list.py`.

## Examples

### Example 1
- Input: `{'head': [1, 2, 3, 4, 5], 'k': 2}`
- Output: `[4, 5, 1, 2, 3]`

### Example 2
- Input: `{'head': [0, 1, 2], 'k': 4}`
- Output: `[2, 0, 1]`

## Follow-up Practice
- Trace the algorithm on a 2-node list with `k = 1`.
- Trace the algorithm when `k` equals the list length.
- Explain why the temporary cycle must be broken before returning.
