# 150. Evaluate Reverse Polish Notation

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/evaluate-reverse-polish-notation/
- Official Group: Stack
- Pattern Group: Stack
- Patterns: stack

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a list of string tokens that form an arithmetic expression in **Reverse Polish Notation**.

Reverse Polish Notation is also called **postfix notation**. The operator comes **after** its operands:

```text
infix:   2 + 1
postfix: 2 1 +
```

The problem asks us to evaluate the expression and return the final integer result.

The allowed operators are:

```text
+   addition
-   subtraction
*   multiplication
/   integer division truncated toward zero
```

Every other token is an integer, possibly negative:

```text
"2"
"13"
"-11"
```

So the input:

```text
["2", "1", "+", "3", "*"]
```

means:

```text
(2 + 1) * 3 = 9
```

The important restriction is that the expression is already valid. We do not need to decide whether it is malformed; we only need to evaluate it correctly.

### 2. Why Infix Intuition Is the Wrong Starting Point

Most arithmetic expressions people write are infix expressions:

```text
2 + 1 * 3
```

In infix notation, evaluation is complicated because we must understand precedence and grouping:

```text
* happens before +
parentheses may override precedence
```

Reverse Polish Notation removes that ambiguity.

Instead of asking:

```text
Which operator should I evaluate first?
```

RPN makes the answer local:

```text
When an operator appears, its operands are the two most recent values that have not yet been consumed.
```

That sentence is the whole problem.

We need a data structure that can remember values in “most recent first” order.

That data structure is a stack.

### 3. Start From the Brute Force Idea

A direct but clumsy way to evaluate RPN is to repeatedly search the token list for the first operator whose two operands are immediately before it.

For example:

```text
["2", "1", "+", "3", "*"]
```

Find `+`:

```text
"2", "1", "+" -> 3
```

Replace those three tokens with the result:

```text
["3", "3", "*"]
```

Find `*`:

```text
"3", "3", "*" -> 9
```

Replace again:

```text
["9"]
```

This works conceptually, but it is inefficient and awkward:

1. We repeatedly scan or modify the token list.
2. Removing and inserting list elements costs extra time.
3. We are not using the fact that tokens can be processed naturally from left to right.

The repeated work is that every time we reduce part of the expression, we are really just saying:

```text
This completed subexpression is now a single value.
```

So instead of rewriting the token list, we can keep completed values separately.

### 4. The Key Observation

In RPN, every token has one of two roles.

If the token is a number:

```text
It is a value that may be used by a future operator.
```

If the token is an operator:

```text
It immediately consumes the two most recent available values.
It produces one new value.
That new value may be used by a future operator.
```

This is exactly a stack process:

```text
number   -> push it
operator -> pop right operand, pop left operand, compute, push result
```

Why pop the right operand first?

Because the right operand appears closer to the operator.

For the expression:

```text
4 13 5 / +
```

When we reach `/`, the two most recent values are:

```text
13, 5
```

The intended operation is:

```text
13 / 5
```

But a stack pops `5` first, then `13`:

```text
right = pop()  # 5
left = pop()   # 13
```

Then compute:

```text
left / right
```

This order matters for subtraction and division.

### 5. The Stack Invariant

After processing some prefix of the token list, the stack contains exactly the values of all completed subexpressions from that prefix that have not yet been consumed by a later operator.

More concretely:

```text
stack bottom ---------------- stack top
older available values        newest available value
```

The top of the stack is the most recent unresolved value.

That invariant explains every operation:

- When we see a number, it is a completed expression by itself, so we push it.
- When we see an operator, the valid RPN grammar guarantees that the top two stack values are the operands for that operator.
- After computing the operator result, the whole local expression has become one completed value, so we push that result back.

The stack does not store operators. It stores values that operators can consume.

### 6. Detailed Algorithm

Initialize an empty stack.

Then scan `tokens` from left to right.

For each token:

1. If the token is not one of `+`, `-`, `*`, `/`:
   - Convert it to an integer.
   - Push it onto the stack.
2. Otherwise, the token is an operator:
   - Pop the top value as `right`.
   - Pop the next value as `left`.
   - Compute `left operator right`.
   - Push the computed result back onto the stack.

At the end, the valid expression has been fully reduced to one value.

Return that value.

### 7. Pseudocode

```python
def evalRPN(tokens):
    stack = []

    for token in tokens:
        if token is a number:
            stack.append(int(token))
            continue

        right = stack.pop()
        left = stack.pop()

        if token == "+":
            stack.append(left + right)
        elif token == "-":
            stack.append(left - right)
        elif token == "*":
            stack.append(left * right)
        else:
            stack.append(truncate_toward_zero(left / right))

    return stack[-1]
```

In Python, there is one detail worth being explicit about.

LeetCode requires division to truncate toward zero:

```text
 6 /  5 ->  1
-6 /  5 -> -1
 6 / -5 -> -1
```

Python's `//` operator floors toward negative infinity, not toward zero:

```python
-6 // 5 == -2
```

So a simple implementation can use:

```python
int(left / right)
```

because converting a floating-point quotient to `int` truncates toward zero.

For LeetCode's usual constraints this is accepted. If avoiding floating point entirely, compute the sign separately and divide absolute values.

### 8. Walkthrough: Example 1

Input:

```text
tokens = ["2", "1", "+", "3", "*"]
```

Start:

```text
stack = []
```

Read `"2"`:

```text
push 2
stack = [2]
```

Read `"1"`:

```text
push 1
stack = [2, 1]
```

Read `"+"`:

```text
right = 1
left  = 2
left + right = 3
push 3
stack = [3]
```

The prefix `2 1 +` has now collapsed into the single value `3`.

Read `"3"`:

```text
push 3
stack = [3, 3]
```

Read `"*"`:

```text
right = 3
left  = 3
left * right = 9
push 9
stack = [9]
```

The final answer is:

```text
9
```

### 9. Walkthrough: Example 2

Input:

```text
tokens = ["4", "13", "5", "/", "+"]
```

Process the numbers first:

```text
push 4  -> stack = [4]
push 13 -> stack = [4, 13]
push 5  -> stack = [4, 13, 5]
```

Read `"/"`:

```text
right = 5
left  = 13
13 / 5 truncates toward zero -> 2
push 2
stack = [4, 2]
```

Read `"+"`:

```text
right = 2
left  = 4
4 + 2 = 6
push 6
stack = [6]
```

The final answer is:

```text
6
```

### 10. Walkthrough: Example 3

Input:

```text
tokens = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
```

Track only the stack after each token:

```text
token  action                         stack
-----  -----------------------------  ----------------
10     push 10                        [10]
6      push 6                         [10, 6]
9      push 9                         [10, 6, 9]
3      push 3                         [10, 6, 9, 3]
+      9 + 3 = 12                    [10, 6, 12]
-11    push -11                       [10, 6, 12, -11]
*      12 * -11 = -132               [10, 6, -132]
/      6 / -132 truncates to 0        [10, 0]
*      10 * 0 = 0                    [0]
17     push 17                        [0, 17]
+      0 + 17 = 17                   [17]
5      push 5                         [17, 5]
+      17 + 5 = 22                   [22]
```

The final answer is:

```text
22
```

This example is useful because it includes a negative divisor case. The operation `6 / -132` must become `0`, not `-1`, because division truncates toward zero.

### 11. Correctness

We prove that the algorithm returns the value of the Reverse Polish expression.

The main invariant is:

```text
After processing any prefix of tokens, the stack contains exactly the values of the completed subexpressions in that prefix that are still available to be used by future operators, in their original left-to-right order.
```

At the beginning, no tokens have been processed and the stack is empty, so the invariant is true.

Now consider the next token.

If the token is a number, that number is a complete subexpression by itself. It has not been consumed by any operator yet, and it appears after all previously available values, so pushing it onto the stack preserves the invariant.

If the token is an operator, valid RPN guarantees that the operator applies to the two most recent available subexpression values. By the invariant, those values are exactly the top two stack entries. The algorithm pops them, applies the operator in left-then-right order, and pushes the result. This replaces two consumed available values with the single value of the newly completed larger subexpression, so the invariant is preserved.

After all tokens have been processed, the entire valid expression has been reduced. There are no future operators left, so exactly one available value remains: the value of the whole expression. The algorithm returns that value.

Therefore, the algorithm is correct.

### 12. Complexity

Let `n` be the number of tokens.

Each token is processed once.

For every token, the algorithm does a constant amount of work:

- one push for a number
- two pops, one arithmetic operation, and one push for an operator

So the time complexity is:

```text
O(n)
```

The stack can hold many numbers before operators consume them. In the worst case, it can grow proportional to the number of tokens.

So the space complexity is:

```text
O(n)
```

### 13. Common Pitfalls

#### Reversing Operand Order

For `+` and `*`, operand order does not change the result.

For `-` and `/`, it does.

When reading an operator:

```python
right = stack.pop()
left = stack.pop()
```

Then compute:

```python
left - right
left / right
```

Do not compute `right - left` or `right / left`.

#### Using Python Floor Division

Python's `//` is tempting, but it is wrong for negative quotients in this problem.

```python
-1 // 2 == -1
```

But truncation toward zero should produce:

```text
-1 / 2 -> 0
```

Use `int(left / right)` or an integer-only truncation helper.

#### Treating Negative Numbers as Operators

The token `"-"` is an operator.

The token `"-11"` is a number.

So do not check only whether `"-" in token`.

A safe test is:

```python
if token in {"+", "-", "*", "/"}:
    # operator
else:
    # integer
```

#### Forgetting That Intermediate Results Are Values

After evaluating `2 1 +`, the result `3` should be pushed back onto the stack because it may be an operand for a later operator.

The stack is not just for original input numbers. It holds all currently available expression values.

#### Overcomplicating With Precedence

There is no precedence table in RPN evaluation.

The token order already encodes the grouping. Evaluate exactly when an operator appears.

### 14. First-Principles Summary

Reverse Polish Notation is easy to evaluate because each operator tells us that its operands are already known.

The only question is where those operands are.

They are the two most recent values that have not been consumed yet.

A stack is the natural structure for that rule:

```text
numbers wait on the stack
operators consume the top two values
operator results become new stack values
```

The invariant is that the stack always represents the unresolved completed subexpressions of the processed prefix. Once all tokens are processed, the whole expression has collapsed into one value, which is the answer.

## Implementation
See `solutions/stack/p150_evaluate_reverse_polish_notation.py`.

## Tests
See `tests/stack/test_p150_evaluate_reverse_polish_notation.py`.

## Examples

### Example 1
- Input: `{'tokens': ['2', '1', '+', '3', '*']}`
- Output: `9`

### Example 2
- Input: `{'tokens': ['4', '13', '5', '/', '+']}`
- Output: `6`

### Example 3
- Input: `{'tokens': ['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', '5', '+']}`
- Output: `22`

## Follow-up Practice
- Trace every push and pop.
- Test single-token, negative-number, and negative-division inputs.
- State what each stack entry means after every processed token.
