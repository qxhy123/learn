# 155. Min Stack

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/min-stack/
- Official Group: Stack
- Pattern Group: Stack
- Patterns: stack

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Design a stack that supports four operations:

```text
push(val)  = put val on top of the stack
pop()      = remove the top value
top()      = return the top value
getMin()   = return the minimum value currently in the stack
```

The unusual operation is `getMin()`.

A normal stack already gives us `push`, `pop`, and `top` naturally, because it only cares about the most recently inserted value. But `getMin()` asks about the smallest value anywhere in the stack, not necessarily at the top.

For example, after:

```text
push(-2)
push(0)
push(-3)
```

the stack from bottom to top is:

```text
[-2, 0, -3]
```

The top is:

```text
-3
```

The minimum is also:

```text
-3
```

After `pop()`, the stack becomes:

```text
[-2, 0]
```

Now the top is:

```text
0
```

but the minimum is:

```text
-2
```

So the real problem is:

> Maintain a normal LIFO stack while also being able to answer the current minimum after every push and pop in constant time.

The words "current minimum" are important. We do not need the minimum over all values ever pushed. We need the minimum among only the values that are still inside the stack.

---

### 2. Start From the Brute Force Baseline

The simplest design is to store the stack values in a list.

```python
stack = []
```

Then:

```python
push(val):
    stack.append(val)

pop():
    stack.pop()

top():
    return stack[-1]

getMin():
    return min(stack)
```

This is correct.

The problem is `getMin()`.

If the stack contains:

```text
[5, 2, 7, 1, 9, 3]
```

then `getMin()` must inspect every value to discover that `1` is the smallest. That costs `O(n)` time.

The LeetCode problem expects every operation to run in `O(1)` time, so recomputing the minimum from scratch is too slow.

The deeper question is:

> What information would let us answer `getMin()` immediately, without scanning the whole stack?

The answer is: at every stack depth, remember the minimum value seen up to that depth.

---

### 3. Why One Global Minimum Is Not Enough

A tempting idea is to keep one variable:

```python
current_min
```

When pushing a value:

```python
current_min = min(current_min, val)
```

This works while values are only being added.

But stacks also remove values.

Consider:

```text
push(5)   current_min = 5
push(2)   current_min = 2
push(7)   current_min = 2
pop()     removes 7, current_min is still 2
pop()     removes 2
```

After the second `pop()`, the stack contains:

```text
[5]
```

The minimum should now be:

```text
5
```

But if we only stored `current_min = 2`, we have lost the previous minimum. We know that `2` left the stack, but we do not know what minimum should be restored.

So the first-principles need is not just:

```text
the current minimum
```

It is:

```text
the current minimum at every historical stack depth
```

When a `pop()` returns the stack to an earlier depth, the minimum should return to the value that was true at that earlier depth.

That is exactly the kind of history a stack is good at storing.

---

### 4. The Key Observation

When we push a new value, the new minimum depends only on two things:

```text
1. the value being pushed
2. the previous minimum before the push
```

If the old stack minimum was `old_min`, then after pushing `val`:

```text
new_min = min(old_min, val)
```

That means each stack position can store not only the value, but also the minimum of the stack at the moment that value became the top.

For example:

```text
push(-2)
```

The stack has one value, so the minimum is `-2`.

Store:

```text
(value = -2, min_so_far = -2)
```

Then:

```text
push(0)
```

The previous minimum is `-2`, and:

```text
min(-2, 0) = -2
```

Store:

```text
(value = 0, min_so_far = -2)
```

Then:

```text
push(-3)
```

The previous minimum is `-2`, and:

```text
min(-2, -3) = -3
```

Store:

```text
(value = -3, min_so_far = -3)
```

Now the internal stack is:

```text
bottom                              top
(-2, -2), (0, -2), (-3, -3)
```

The top entry tells us both:

```text
top value       = -3
current minimum = -3
```

If we pop the top entry, the stack becomes:

```text
bottom                    top
(-2, -2), (0, -2)
```

The new top entry immediately tells us:

```text
top value       = 0
current minimum = -2
```

No recomputation is needed.

---

### 5. The Stack/Minimum Invariant

The central invariant is:

> For every entry `(value, min_so_far)` in the stack, `min_so_far` is the minimum value among all stack values from the bottom up to that entry.

Equivalently, if an entry is at index `i`, then:

```text
entry.min_so_far = min(stack[0].value, stack[1].value, ..., stack[i].value)
```

This invariant gives every operation a simple meaning.

For the top entry:

```text
stack[-1] = (top_value, current_min)
```

So:

```text
top()    returns stack[-1].value
getMin() returns stack[-1].min_so_far
```

The invariant also explains why `pop()` is easy. Removing the top entry removes both:

```text
1. the value that is leaving the stack
2. the minimum state that belonged only to that stack depth
```

After the pop, the new top entry contains the previous depth's correct minimum.

This is the main idea of the problem.

---

### 6. Detailed Algorithm

Maintain one internal list called `stack`.

Each element of `stack` is a pair:

```text
(value, min_so_far)
```

#### `push(val)`

If the stack is empty, then `val` is the only value, so it is also the minimum:

```python
stack.append((val, val))
```

If the stack is not empty, the previous minimum is stored on the current top entry:

```python
previous_min = stack[-1][1]
```

The new minimum is:

```python
new_min = min(previous_min, val)
```

Push both pieces of information:

```python
stack.append((val, new_min))
```

#### `pop()`

Remove the top pair:

```python
stack.pop()
```

LeetCode guarantees `pop()` is called only when the stack is non-empty.

#### `top()`

Return the value part of the top pair:

```python
return stack[-1][0]
```

#### `getMin()`

Return the minimum part of the top pair:

```python
return stack[-1][1]
```

LeetCode guarantees `top()` and `getMin()` are called only when the stack is non-empty.

---

### 7. Pseudocode

```python
class MinStack:
    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            previous_min = self.stack[-1][1]
            self.stack.append((val, min(val, previous_min)))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

This uses one physical stack whose entries carry extra information.

Another common implementation uses two stacks:

```text
values stack = all pushed values
mins stack   = minimum value at each depth
```

That is the same invariant represented with two parallel lists instead of one list of pairs. The pair-based version is often easier to reason about because the value and the minimum for the same depth live together.

---

### 8. Detailed Example Walkthrough

Use the official-style sequence:

```text
operations = ["MinStack", "push", "push", "push", "getMin", "pop", "top", "getMin"]
arguments  = [[],         [-2],   [0],    [-3],   [],       [],    [],    []]
```

Start with an empty stack:

```text
[]
```

#### Operation 1: `push(-2)`

The stack is empty.

The value is `-2`, and the minimum is also `-2`.

```text
[(-2, -2)]
```

#### Operation 2: `push(0)`

Previous minimum:

```text
-2
```

New value:

```text
0
```

New minimum:

```text
min(-2, 0) = -2
```

Stack:

```text
[(-2, -2), (0, -2)]
```

#### Operation 3: `push(-3)`

Previous minimum:

```text
-2
```

New value:

```text
-3
```

New minimum:

```text
min(-2, -3) = -3
```

Stack:

```text
[(-2, -2), (0, -2), (-3, -3)]
```

#### Operation 4: `getMin()`

Look at the top entry:

```text
(-3, -3)
```

The second component is the current minimum.

Return:

```text
-3
```

#### Operation 5: `pop()`

Remove the top entry:

```text
(-3, -3)
```

Stack becomes:

```text
[(-2, -2), (0, -2)]
```

Notice what happened: because the entry containing `-3` was removed, its minimum state was removed too. The new top entry restores the previous minimum, `-2`.

#### Operation 6: `top()`

Look at the top entry:

```text
(0, -2)
```

The first component is the top value.

Return:

```text
0
```

#### Operation 7: `getMin()`

Look at the same top entry:

```text
(0, -2)
```

The second component is the current minimum.

Return:

```text
-2
```

So the outputs for the value-returning operations are:

```text
getMin() -> -3
top()    -> 0
getMin() -> -2
```

---

### 9. Why Duplicate Minimum Values Matter

Duplicates are a common source of bugs.

Suppose the operations are:

```text
push(2)
push(1)
push(1)
pop()
getMin()
```

After the first two pushes, the minimum is `1`.

After pushing another `1`, the minimum is still `1`.

After popping one `1`, the stack still contains another `1`, so `getMin()` must still return:

```text
1
```

The pair-based approach handles this automatically:

```text
(2, 2), (1, 1), (1, 1)
```

After one pop:

```text
(2, 2), (1, 1)
```

The top entry still records `1` as the minimum.

This is why designs that store minimum values separately must be careful with equality. If using a separate min stack that stores only changes in the minimum, then pushing a value equal to the current minimum should usually be recorded too, or the implementation must keep counts.

---

### 10. Correctness

We prove that the data structure returns the correct values for all operations.

#### Invariant

After every operation, for every stored pair `(value, min_so_far)`, `min_so_far` equals the minimum of all values from the bottom of the stack through that pair.

#### Initialization

After construction, the stack is empty.

There are no entries, so the invariant holds vacuously.

#### Push Preserves the Invariant

When pushing the first value `val`, the stack contains only `val`, so the minimum through the new top is `val`. The algorithm stores `(val, val)`, so the invariant holds.

When pushing onto a non-empty stack, assume the invariant already holds. The previous top stores the minimum of all old stack values. Call it `previous_min`.

After pushing `val`, the minimum of all values through the new top is exactly:

```text
min(previous_min, val)
```

The algorithm stores that value as the new pair's `min_so_far`. Existing pairs are not changed, so their invariant remains true. Therefore the invariant still holds after `push`.

#### Pop Preserves the Invariant

`pop()` removes only the top pair.

All remaining pairs keep the same values and the same prefixes beneath them. Since the invariant was true for those pairs before the pop, it remains true after the pop.

#### `top()` Is Correct

The top of the abstract stack is the value stored in the top pair. `top()` returns the first component of that pair, so it returns the correct top value.

#### `getMin()` Is Correct

By the invariant, the second component of the top pair is the minimum of all values from the bottom through the top. That is the minimum of the entire current stack. Therefore `getMin()` returns the correct minimum.

Because every operation preserves the invariant and the query operations read exactly the value guaranteed by the invariant, the data structure is correct.

---

### 11. Complexity

Let `n` be the number of values currently in the stack.

Each operation does constant work:

- `push`: one comparison and one append
- `pop`: one removal from the end
- `top`: one indexed read
- `getMin`: one indexed read

So the time complexity is:

```text
push    O(1)
pop     O(1)
top     O(1)
getMin  O(1)
```

The stack stores one pair per pushed value that has not been popped.

So the space complexity is:

```text
O(n)
```

This is optimal up to constant factors, because the stack must remember the current values in order to support `pop()` and `top()` correctly.

---

### 12. Common Pitfalls

- **Recomputing `min(stack)` inside `getMin()`:** this is correct but costs `O(n)`, which misses the point of the problem.
- **Keeping only one `current_min`:** this fails when the current minimum is popped and the previous minimum must be restored.
- **Ignoring duplicate minima:** if two equal minimum values are pushed, popping one of them should not lose the minimum.
- **Updating the minimum after pushing but not storing history:** the previous minimum must be recoverable after a pop.
- **Mixing up tuple fields:** `top()` returns the stored value, while `getMin()` returns the stored minimum.
- **Trying to sort values:** sorting destroys stack order and is unnecessary. This is not a global ordering problem; it is a history-of-prefix-minimums problem.
- **Overcomplicating empty-stack behavior:** LeetCode guarantees invalid operations are not called, so the core solution does not need special error handling unless building a production API.

---

### 13. First-Principles Summary

A stack is about history. Every push creates a new stack depth, and every pop returns to the previous depth.

`getMin()` is hard only if we think of the minimum as something to search for. Instead, treat the minimum as part of the state of each stack depth.

At depth `i`, store:

```text
the value pushed at depth i
the minimum among all values up to depth i
```

Then the top stack entry completely describes the current state:

```text
current top value
current minimum value
```

When we push, we compute one new minimum from the old minimum and the new value. When we pop, we discard the current depth and automatically reveal the previous depth's minimum.

That is the whole trick:

> Make the stack remember not only values, but also the minimum that was true when each value was on top.

## Implementation
See `solutions/stack/p155_min_stack.py`.

## Tests
See `tests/stack/test_p155_min_stack.py`.

## Examples

### Example 1
- Input: `{'raw': '["MinStack","push","push","push","getMin","pop","top","getMin"]\n[[],[-2],[0],[-3],[],[],[],[]]'}`
- Output: `'See official examples'`

## Follow-up Practice
- Trace the stored pair `(value, min_so_far)` after every operation.
- Test duplicate minimum values, such as `push(1), push(1), pop(), getMin()`.
- Explain why one global `current_min` is insufficient after `pop()`.
- Implement the equivalent two-stack version and identify the same invariant.
