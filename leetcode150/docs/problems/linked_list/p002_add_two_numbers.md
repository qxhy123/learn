# 2. Add Two Numbers

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/add-two-numbers/
- Official Group: Linked List
- Pattern Group: Linked List
- Patterns: linked-list

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given two non-empty linked lists, `l1` and `l2`.

Each linked list stores a non-negative integer, but the digits are stored in **reverse order**:

```text
l1 = 2 -> 4 -> 3
```

This does not represent `243`.

It represents:

```text
342
```

because the head node is the ones digit:

```text
2 is the ones digit
4 is the tens digit
3 is the hundreds digit
```

Similarly:

```text
l2 = 5 -> 6 -> 4
```

represents:

```text
465
```

The problem asks us to add the two numbers:

```text
342 + 465 = 807
```

and return the sum in the same reversed linked-list format:

```text
7 -> 0 -> 8
```

The output is not an integer or an array. It is a linked list whose nodes contain the digits of the result from least significant to most significant.

So the real task is:

> Add two numbers digit by digit from right to left, except the linked lists already give us those digits in the exact order we need to process them.

That last point is the key. Normal addition starts at the ones digit. Because the lists are reversed, the head node is already the ones digit.

### 2. Why Reverse Order Makes This Natural

If the number were written normally, like `342`, addition would start at the rightmost digit:

```text
  342
+ 465
-----
```

We would add:

```text
2 + 5 first
4 + 6 second
3 + 4 third
```

A singly linked list cannot move backward from the tail to the head. If the digits were stored in normal order, reaching the ones digit first would be awkward.

But the problem stores digits as:

```text
342 -> 2 -> 4 -> 3
465 -> 5 -> 6 -> 4
```

So walking forward through both lists gives exactly the grade-school addition order:

```text
ones, tens, hundreds, thousands, ...
```

This means we do not need stacks, reversing, or integer conversion. We only need to simulate column addition while advancing pointers.

### 3. Start From the Brute Force Idea

A tempting baseline is:

1. Traverse `l1` and convert it to an integer.
2. Traverse `l2` and convert it to an integer.
3. Add the integers.
4. Convert the sum back into a reversed linked list.

For example:

```python
value1 = 0
place = 1
while l1:
    value1 += l1.val * place
    place *= 10
    l1 = l1.next

value2 = 0
place = 1
while l2:
    value2 += l2.val * place
    place *= 10
    l2 = l2.next

sum_value = value1 + value2
```

This approach matches the definition of the input, so it is easy to understand.

But it misses the point of the linked-list representation.

The problem may contain very long numbers. In many languages, converting the whole linked list into a built-in integer can overflow. Even in Python, where integers can grow arbitrarily large, this approach builds a large number that is unnecessary.

More importantly, grade-school addition never requires knowing the whole number at once. To decide the current output digit, we only need:

```text
current digit from l1
current digit from l2
carry from the previous column
```

So the brute force approach stores far more information than needed.

### 4. The Key Observation

Addition is local by digit.

At any column, suppose we are adding:

```text
digit1 + digit2 + carry
```

The result determines two things:

```text
output digit = total % 10
next carry   = total // 10
```

For example:

```text
4 + 6 + 0 = 10
```

So:

```text
output digit = 10 % 10 = 0
next carry   = 10 // 10 = 1
```

That is exactly what happens in normal addition:

```text
  342
+ 465
-----
   7    ones column: 2 + 5 = 7
  0     tens column: 4 + 6 = 10, write 0 carry 1
 8      hundreds column: 3 + 4 + 1 = 8
```

Because each linked list gives digits from least significant to most significant, we can produce the answer in the same order we compute it.

This is the core observation:

> The next result node depends only on the current two input nodes and the carry. Once that result node is created, earlier nodes never need to be changed.

### 5. Pointer and Carry Invariant

The algorithm maintains three pieces of state:

```text
p1    -> current unprocessed node in l1
p2    -> current unprocessed node in l2
carry -> carry into the current digit column
```

It also maintains an output list:

```text
dummy -> result nodes already built
```

A useful invariant is:

> Before each loop iteration, the output list contains the correct reversed digits for all columns already processed, and `carry` is exactly the carry that must be added to the next unprocessed column.

For example, after processing the ones digit of:

```text
l1 = 2 -> 4 -> 3
l2 = 5 -> 6 -> 4
```

we have:

```text
output = 7
carry  = 0
p1 points to 4
p2 points to 6
```

The invariant says:

```text
7 is the correct result for the ones column,
and carry = 0 is exactly what the tens column should receive.
```

After processing the tens digit:

```text
4 + 6 + 0 = 10
```

we have:

```text
output = 7 -> 0
carry  = 1
p1 points to 3
p2 points to 4
```

Now the invariant says:

```text
7 -> 0 is correct for the ones and tens columns,
and carry = 1 must be included in the hundreds column.
```

This invariant is what makes the solution safe. We never guess, backtrack, or inspect future digits to fix past digits.

### 6. Why Use a Dummy Head?

The result list is built one node at a time.

Without a dummy head, the first node needs special handling:

```python
if head is None:
    head = new_node
    tail = new_node
else:
    tail.next = new_node
    tail = tail.next
```

A dummy head removes that special case.

We create a placeholder node before the real answer:

```text
dummy -> None
 tail
```

Whenever we compute a result digit, we append it after `tail`:

```text
dummy -> 7
          tail
```

then:

```text
dummy -> 7 -> 0
               tail
```

At the end, the actual answer starts at:

```text
dummy.next
```

The dummy node is not part of the number. It is just a stable anchor that makes list construction uniform.

### 7. Detailed Algorithm

Use two pointers to walk through the input lists:

```text
p1 = l1
p2 = l2
```

Use one integer to remember the carry:

```text
carry = 0
```

Use a dummy node and a tail pointer to build the result:

```text
dummy = ListNode(0)
tail = dummy
```

Then repeat while there is still something left to process:

```text
while p1 exists or p2 exists or carry is nonzero
```

The `or carry is nonzero` part is important. Sometimes the final addition creates a new most significant digit.

For example:

```text
5 + 5 = 10
```

The result is:

```text
0 -> 1
```

Even after both input lists are exhausted, the carry `1` still needs to become a node.

Inside each loop iteration:

1. Read the current digit from `p1`, or use `0` if `p1` is already exhausted.
2. Read the current digit from `p2`, or use `0` if `p2` is already exhausted.
3. Add both digits plus `carry`.
4. The new output digit is `total % 10`.
5. The next carry is `total // 10`.
6. Append a new node containing the output digit.
7. Advance any input pointer that still exists.

In pseudocode:

```python
def addTwoNumbers(l1, l2):
    dummy = ListNode(0)
    tail = dummy
    carry = 0

    p1 = l1
    p2 = l2

    while p1 is not None or p2 is not None or carry != 0:
        digit1 = p1.val if p1 is not None else 0
        digit2 = p2.val if p2 is not None else 0

        total = digit1 + digit2 + carry
        digit = total % 10
        carry = total // 10

        tail.next = ListNode(digit)
        tail = tail.next

        if p1 is not None:
            p1 = p1.next
        if p2 is not None:
            p2 = p2.next

    return dummy.next
```

### 8. Detailed Walkthrough of Example 1

Input:

```text
l1 = 2 -> 4 -> 3
l2 = 5 -> 6 -> 4
```

These represent:

```text
342 + 465
```

Initialize:

```text
p1 = 2
p2 = 5
carry = 0
result = empty
```

#### Column 1: ones digit

Read:

```text
digit1 = 2
digit2 = 5
carry  = 0
```

Compute:

```text
total = 2 + 5 + 0 = 7
digit = 7 % 10 = 7
carry = 7 // 10 = 0
```

Append `7`:

```text
result = 7
```

Advance both pointers:

```text
p1 = 4
p2 = 6
```

#### Column 2: tens digit

Read:

```text
digit1 = 4
digit2 = 6
carry  = 0
```

Compute:

```text
total = 4 + 6 + 0 = 10
digit = 10 % 10 = 0
carry = 10 // 10 = 1
```

Append `0`:

```text
result = 7 -> 0
```

Advance both pointers:

```text
p1 = 3
p2 = 4
```

#### Column 3: hundreds digit

Read:

```text
digit1 = 3
digit2 = 4
carry  = 1
```

Compute:

```text
total = 3 + 4 + 1 = 8
digit = 8 % 10 = 8
carry = 8 // 10 = 0
```

Append `8`:

```text
result = 7 -> 0 -> 8
```

Advance both pointers:

```text
p1 = None
p2 = None
```

Now both lists are exhausted and `carry = 0`, so the loop stops.

Return:

```text
7 -> 0 -> 8
```

which represents `807`.

### 9. Walkthrough of Unequal Lengths and Final Carry

Consider the third official example:

```text
l1 = 9 -> 9 -> 9 -> 9 -> 9 -> 9 -> 9
l2 = 9 -> 9 -> 9 -> 9
```

These represent:

```text
9,999,999 + 9,999 = 10,009,998
```

The output should be:

```text
8 -> 9 -> 9 -> 9 -> 0 -> 0 -> 0 -> 1
```

Column by column:

```text
9 + 9 + 0 = 18  -> write 8, carry 1
9 + 9 + 1 = 19  -> write 9, carry 1
9 + 9 + 1 = 19  -> write 9, carry 1
9 + 9 + 1 = 19  -> write 9, carry 1
9 + 0 + 1 = 10  -> write 0, carry 1
9 + 0 + 1 = 10  -> write 0, carry 1
9 + 0 + 1 = 10  -> write 0, carry 1
0 + 0 + 1 = 1   -> write 1, carry 0
```

There are two important details here.

First, after `l2` runs out, its missing digits behave like zeroes. That is the same as normal arithmetic:

```text
  9999999
+    9999
---------
```

The shorter number has implicit leading zeroes.

Second, after both lists run out, the final carry still matters. The last `1` in the output exists only because the loop continues while `carry != 0`.

### 10. Correctness Argument

We prove that the algorithm returns the correct reversed linked list for the sum of the two input numbers.

#### Invariant

Before each loop iteration:

1. The result list built after `dummy` contains exactly the correct digits for all columns already processed.
2. `p1` and `p2` point to the next unprocessed digits of `l1` and `l2`, or are `None` if that list has no more digits.
3. `carry` is exactly the carry produced by the last processed column and required by the next column.

#### Initialization

Before the first iteration, no columns have been processed.

The result list is empty, which is correct for zero processed columns. `p1` and `p2` point to the heads of the two input lists, which are the ones digits. `carry = 0`, which is the correct carry before any addition happens.

So the invariant holds initially.

#### Maintenance

Assume the invariant holds at the start of an iteration.

The algorithm reads the next digit from each list, using `0` when a list is exhausted. This matches arithmetic with implicit leading zeroes for the shorter number.

It computes:

```text
total = digit1 + digit2 + carry
```

The current result digit must be the ones digit of `total`, which is:

```text
total % 10
```

The carry into the next column must be:

```text
total // 10
```

The algorithm appends exactly that result digit and updates `carry` to exactly that next carry. Then it advances each non-exhausted input pointer to the following digit.

Therefore, after the iteration, the result list is correct for one more column, the pointers identify the next unprocessed column, and `carry` is correct for that next column.

So the invariant is preserved.

#### Termination

The loop stops only when:

```text
p1 is None
p2 is None
carry == 0
```

At that moment, there are no unprocessed digits in either input list and no remaining carry to append.

By the invariant, the result list contains exactly the correct digits for every column of the sum, in least-significant-to-most-significant order.

That is precisely the linked-list format required by the problem.

So returning `dummy.next` is correct.

### 11. Complexity

Let:

```text
m = length of l1
n = length of l2
```

The algorithm processes each node of each input list once.

It may perform one additional iteration if there is a final carry.

So the time complexity is:

```text
O(max(m, n))
```

The output list contains one node per result digit. Its length is at most:

```text
max(m, n) + 1
```

If we do not count the required output list as auxiliary space, the extra working space is:

```text
O(1)
```

because the algorithm only stores a few pointers and the carry.

Including the returned linked list, the space used for the output is:

```text
O(max(m, n))
```

### 12. Common Pitfalls

#### Forgetting the Final Carry

If the loop condition is only:

```python
while p1 or p2:
```

then this case fails:

```text
l1 = 5
l2 = 5
```

The first column produces digit `0` and carry `1`. Both lists are exhausted, but the carry still needs to become a node:

```text
0 -> 1
```

Use:

```python
while p1 or p2 or carry:
```

#### Treating Missing Digits as Missing Work

When one list is shorter, do not stop immediately.

For example:

```text
l1 = 9 -> 9 -> 9
l2 = 1
```

After the first node of `l2`, the remaining digits of `l2` are implicit zeroes:

```text
9 + 1 = 10
9 + 0 + 1 = 10
9 + 0 + 1 = 10
0 + 0 + 1 = 1
```

The correct output is:

```text
0 -> 0 -> 0 -> 1
```

#### Appending the Carry Instead of Splitting the Total

Do not append `total` directly.

If:

```text
total = 17
```

then the current node must store only one digit:

```text
7
```

and the next carry is:

```text
1
```

Each node stores a single decimal digit from `0` to `9`.

#### Returning the Dummy Node

The dummy node is only a construction helper.

Return:

```python
return dummy.next
```

not:

```python
return dummy
```

Returning `dummy` would add an extra leading `0` at the front of the reversed result.

#### Advancing Pointers Before Reading Values

Read the current node values before moving to `.next`.

The safe order is:

```python
digit1 = p1.val if p1 else 0
# then later:
if p1:
    p1 = p1.next
```

If you advance first, you skip the head digit, which is the ones digit and must be included.

### 13. First-Principles Summary

The problem looks like a linked-list problem, but the heart of it is column addition.

The linked list matters because it gives us the digits one at a time. The reverse order matters because it gives them in exactly the order addition needs:

```text
ones -> tens -> hundreds -> ...
```

At every step, all necessary information fits into one local calculation:

```text
current digit from l1
current digit from l2
carry from previous column
```

From that calculation, we get:

```text
result digit for this column
carry for the next column
```

The pointer invariant keeps the implementation honest:

```text
processed result nodes are final,
p1 and p2 point to the next column,
carry is the only information that crosses from one column to the next.
```

Once that invariant is clear, the algorithm is just grade-school addition translated into linked-list construction.

## Implementation
See `solutions/linked_list/p002_add_two_numbers.py`.

## Tests
See `tests/linked_list/test_p002_add_two_numbers.py`.

## Examples

### Example 1
- Input: `{'l1': [2, 4, 3], 'l2': [5, 6, 4]}`
- Output: `[7, 0, 8]`

### Example 2
- Input: `{'l1': [0], 'l2': [0]}`
- Output: `[0]`

### Example 3
- Input: `{'l1': [9, 9, 9, 9, 9, 9, 9], 'l2': [9, 9, 9, 9]}`
- Output: `[8, 9, 9, 9, 0, 0, 0, 1]`

## Follow-up Practice

- Trace the algorithm on `l1 = [5]`, `l2 = [5]` to see why the final carry condition is necessary.
- Trace unequal lengths, such as `l1 = [9, 9, 9]`, `l2 = [1]`.
- Re-derive the invariant before coding: result built so far is final, pointers mark the next column, and carry is the only cross-column state.
