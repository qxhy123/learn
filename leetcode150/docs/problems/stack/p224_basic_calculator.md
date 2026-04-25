# 224. Basic Calculator

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/basic-calculator/
- Official Group: Stack
- Pattern Group: Stack
- Patterns: stack

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a string `s` representing an arithmetic expression. The expression may contain:

- non-negative integers,
- plus signs `+`,
- minus signs `-`,
- opening parentheses `(`,
- closing parentheses `)`,
- spaces.

You must evaluate the expression and return its integer value.

For this problem, there is no multiplication or division. That matters a lot. With only `+` and `-`, expression evaluation is mostly about one question:

```text
What sign should each number have in the final sum?
```

For example:

```text
1 - (2 + 3)
```

is not hard because of arithmetic precedence. It is hard because the minus sign before the parentheses flips everything inside:

```text
1 - 2 - 3
```

So the problem is really asking us to scan a string and correctly account for the sign context created by nested parentheses.

### 2. Start From the Brute-Force Baseline

The most literal solution is to build a full expression parser.

A parser would:

1. Convert the string into tokens.
2. Parse nested parenthesized subexpressions.
3. Evaluate each subexpression recursively.
4. Return the final value.

Conceptually:

```python
def eval_expr(tokens):
    total = 0
    sign = +1

    while tokens remain:
        token = next token

        if token is a number:
            total += sign * token
        elif token == '+':
            sign = +1
        elif token == '-':
            sign = -1
        elif token == '(':
            total += sign * eval_expr(tokens until matching ')')
        elif token == ')':
            return total

    return total
```

This is a valid way to think about the problem. It is also close to the clean recursive solution.

But if implemented naively, the brute-force version can become awkward:

- finding the matching `)` may require extra scans,
- slicing substrings creates unnecessary work,
- recursive calls need careful index management,
- every nested expression still depends on the sign outside it.

The first-principles question is:

> Can we evaluate the expression in one left-to-right pass without repeatedly searching for matching parentheses?

Yes. Parentheses create temporary context, and temporary nested context is exactly what a stack stores.

### 3. Key Observation: Addition and Subtraction Are Signed Summation

Because the only binary operators are `+` and `-`, every expression can be viewed as a sum of signed numbers.

For example:

```text
7 - 4 + 12 - 3
```

means:

```text
(+7) + (-4) + (+12) + (-3)
```

So while scanning the expression, we can keep:

```text
result = sum of fully processed signed numbers in the current parenthesis level
sign   = sign to apply to the next number or parenthesized expression
```

When we read a number, we add:

```text
result += sign * number
```

Then `+` and `-` only change the sign for the next thing:

```text
+ means next sign is +1
- means next sign is -1
```

If the expression had no parentheses, this would be enough.

Example:

```text
s = "2-1+2"
```

Scan it as signed numbers:

```text
+2, -1, +2
```

The answer is:

```text
2 - 1 + 2 = 3
```

Parentheses are the only remaining complication.

### 4. What Parentheses Actually Do

A parenthesized expression is evaluated as a unit, then inserted into the outer expression.

For example:

```text
10 - (3 + 4)
```

The outer expression is:

```text
10 - [value of (3 + 4)]
```

When we enter the parenthesis, we need to temporarily forget the outer running total and evaluate the inside from zero:

```text
inside result = 3 + 4 = 7
```

Then we return to the outer context and apply the sign that preceded the parenthesis:

```text
outer result = 10 + (-1) * 7 = 3
```

So when we see `(`, two pieces of outer context must be saved:

```text
outer result
outer sign before this parenthesized expression
```

Then we reset for the inner expression:

```text
result = 0
sign = +1
```

When we see `)`, the current `result` is the completed value of the inner expression. We pop the saved context and combine:

```text
inner_value = result
previous_sign = popped sign
previous_result = popped result

result = previous_result + previous_sign * inner_value
```

That is the entire stack idea.

### 5. The Stack/Sign Invariant

At every point in the scan, we maintain this invariant:

```text
result = value of all completely processed terms in the current parenthesis level
sign   = sign that should be applied to the next number or parenthesized expression in the current level
stack  = saved outer contexts, each containing:
         (result before the parenthesis, sign before the parenthesis)
```

A stack entry is not an arbitrary previous value. It has a precise meaning:

```text
(saved_result, saved_sign)
```

where:

- `saved_result` is the value accumulated before the `(` in the outer level,
- `saved_sign` is the sign that should multiply the entire parenthesized expression.

For example, in:

```text
8 - (1 + 2)
```

right before scanning inside the parentheses:

```text
saved_result = 8
saved_sign = -1
```

The inner expression computes:

```text
1 + 2 = 3
```

Then closing the parenthesis gives:

```text
8 + (-1) * 3 = 5
```

The invariant is useful because it tells us exactly what each character must do.

### 6. Detailed Algorithm

Scan the string from left to right with index `i`.

Maintain:

```text
result = 0
sign = +1
stack = []
```

For each character:

#### Case 1: Space

Ignore it.

Spaces do not affect the expression.

#### Case 2: Digit

Read the full multi-digit number, not just one character.

For example, in:

```text
"123 + 4"
```

we must read `123` as one number.

Then add it with the current sign:

```text
result += sign * number
```

#### Case 3: Plus Sign

The next number or parenthesized expression should be positive relative to the current level:

```text
sign = +1
```

#### Case 4: Minus Sign

The next number or parenthesized expression should be negative relative to the current level:

```text
sign = -1
```

#### Case 5: Opening Parenthesis

The current `result` and `sign` belong to the outer level, so save them:

```text
push result
push sign
```

or equivalently:

```text
push (result, sign)
```

Then start a new inner level:

```text
result = 0
sign = +1
```

The inner expression begins fresh because its value should be computed independently before being combined with the outer context.

#### Case 6: Closing Parenthesis

The current `result` is now the value of the completed inner expression.

Pop the saved context and combine:

```text
previous_sign = pop()
previous_result = pop()
result = previous_result + previous_sign * result
```

If using pair entries:

```text
previous_result, previous_sign = stack.pop()
result = previous_result + previous_sign * result
```

After this, `result` again represents the processed value in the outer level.

### 7. Pseudocode

```python
def calculate(s):
    result = 0
    sign = 1
    stack = []
    i = 0

    while i < len(s):
        ch = s[i]

        if ch == ' ':
            i += 1

        elif ch.isdigit():
            number = 0
            while i < len(s) and s[i].isdigit():
                number = number * 10 + int(s[i])
                i += 1
            result += sign * number

        elif ch == '+':
            sign = 1
            i += 1

        elif ch == '-':
            sign = -1
            i += 1

        elif ch == '(':
            stack.append((result, sign))
            result = 0
            sign = 1
            i += 1

        elif ch == ')':
            previous_result, previous_sign = stack.pop()
            result = previous_result + previous_sign * result
            i += 1

    return result
```

The important detail is that the digit branch advances `i` through the entire number. Because that branch already moves `i`, it should not also blindly increment `i` at the bottom of the loop unless the loop is written carefully.

### 8. Detailed Walkthrough

Consider the official example:

```text
s = "(1+(4+5+2)-3)+(6+8)"
```

We track:

```text
result, sign, stack
```

Start:

```text
result = 0
sign = +1
stack = []
```

Read `(`:

```text
push (0, +1)
result = 0
sign = +1
stack = [(0, +1)]
```

Read `1`:

```text
result = 0 + (+1) * 1 = 1
```

Read `+`:

```text
sign = +1
```

Read `(`:

The outer level currently has value `1`, and the sign before this inner parenthesis is `+1`.

```text
push (1, +1)
result = 0
sign = +1
stack = [(0, +1), (1, +1)]
```

Read `4`:

```text
result = 4
```

Read `+`:

```text
sign = +1
```

Read `5`:

```text
result = 4 + 5 = 9
```

Read `+`:

```text
sign = +1
```

Read `2`:

```text
result = 9 + 2 = 11
```

Read `)`:

The inner expression `(4+5+2)` is complete and equals `11`.

Pop `(1, +1)` and combine:

```text
result = 1 + (+1) * 11 = 12
stack = [(0, +1)]
```

Read `-`:

```text
sign = -1
```

Read `3`:

```text
result = 12 + (-1) * 3 = 9
```

Read `)`:

The larger expression `(1+(4+5+2)-3)` is complete and equals `9`.

Pop `(0, +1)` and combine:

```text
result = 0 + (+1) * 9 = 9
stack = []
```

Read `+`:

```text
sign = +1
```

Read `(`:

Save the outer context before evaluating `(6+8)`:

```text
push (9, +1)
result = 0
sign = +1
stack = [(9, +1)]
```

Read `6`:

```text
result = 6
```

Read `+`:

```text
sign = +1
```

Read `8`:

```text
result = 14
```

Read `)`:

Pop `(9, +1)` and combine:

```text
result = 9 + (+1) * 14 = 23
stack = []
```

End of string:

```text
answer = 23
```

### 9. Why the Minus Before Parentheses Works

The most common source of mistakes is an expression like:

```text
1 - (4 + 5 - 2)
```

The correct answer is:

```text
1 - 7 = -6
```

The algorithm handles this naturally.

Before `(`:

```text
result = 1
sign = -1
```

On `(`, save:

```text
stack.append((1, -1))
```

Inside the parentheses, compute normally:

```text
4 + 5 - 2 = 7
```

On `)`, combine:

```text
result = 1 + (-1) * 7 = -6
```

Notice that we did not need to manually flip every sign inside the parentheses. The entire inner expression is multiplied once by the saved outer sign.

This is the central reason the stack approach stays simple.

### 10. Correctness Argument

We prove that the algorithm returns the value of the expression.

#### Invariant

After processing any prefix of the input, within the current parenthesis level:

```text
result
```

is the exact value of all complete numbers and complete parenthesized expressions already seen in that level, and:

```text
sign
```

is the sign that should be applied to the next number or parenthesized expression in that same level.

The stack stores all unfinished outer levels. Each stack entry `(saved_result, saved_sign)` means:

```text
When the current parenthesized expression is complete,
its value should be combined as:

saved_result + saved_sign * current_value
```

#### Initialization

Before scanning any characters:

```text
result = 0
sign = +1
stack = []
```

No terms have been processed, so the current value is correctly `0`. The first term is positive unless preceded by a minus sign, so `+1` is the correct initial sign. There are no unfinished outer parentheses, so the stack is empty.

#### Maintenance

Each possible token preserves the invariant:

- A space changes nothing.
- A number is a complete term in the current level, so adding `sign * number` makes `result` include exactly that term.
- A `+` sets the sign for the next term to positive.
- A `-` sets the sign for the next term to negative.
- A `(` starts a nested expression. The current outer `result` and the sign that applies to the whole nested expression are saved on the stack, then the inner level starts with value `0` and sign `+1`.
- A `)` finishes the current nested expression. By the invariant, `result` is its exact value. Popping `(saved_result, saved_sign)` and computing `saved_result + saved_sign * result` gives the exact value of the outer level after replacing the parenthesized expression by its value.

Thus the invariant remains true after every processed token.

#### Termination

When the scan finishes, there is no remaining unprocessed token. The input expression is valid, so all opened parentheses have been closed and combined. By the invariant, `result` is the value of the entire expression.

Therefore, the algorithm returns the correct answer.

### 11. Complexity

Let `n` be the length of `s`.

Each character is processed a constant number of times:

- spaces are skipped once,
- operators are read once,
- parentheses are pushed or popped once,
- digits are consumed once as part of their full number.

So the time complexity is:

```text
O(n)
```

The stack stores one context per currently open parenthesis. In the worst case, the expression can be deeply nested:

```text
((((1))))
```

So the space complexity is:

```text
O(n)
```

More precisely, the stack space is proportional to the maximum parenthesis nesting depth.

### 12. Common Pitfalls

#### Pitfall 1: Reading only one digit

The expression can contain multi-digit numbers.

Wrong mental model:

```text
"123" means 1, then 2, then 3
```

Correct mental model:

```text
"123" means one number: 123
```

Build the number with:

```text
number = number * 10 + digit
```

#### Pitfall 2: Forgetting to reset after `(`

After pushing the outer context, the inner expression must start fresh:

```text
result = 0
sign = +1
```

If you do not reset `result`, the outer value leaks into the inner expression.

#### Pitfall 3: Saving only `result` but not `sign`

For:

```text
1 - (2 + 3)
```

saving only `1` is not enough. You also need the `-1` that applies to the whole parenthesized expression.

The stack entry must preserve both:

```text
(saved_result, saved_sign)
```

#### Pitfall 4: Applying the outer sign too early

Inside:

```text
1 - (2 + 3)
```

it is tempting to flip every sign inside the parentheses. That is unnecessary and error-prone.

Compute the inside normally:

```text
2 + 3 = 5
```

Then apply the saved outer sign once:

```text
1 + (-1) * 5
```

#### Pitfall 5: Mishandling the index after a number

If the digit-reading loop advances `i` to the first non-digit character, do not increment `i` again at the end of the same branch unless your loop structure accounts for it. Otherwise, you may skip an operator or parenthesis.

#### Pitfall 6: Treating this like a full precedence problem

This problem has no `*` or `/`. You do not need an operator-precedence stack. The only nesting structure is parentheses, and the only operator state needed is the sign for the next term.

### 13. First-Principles Summary

The expression can be reduced to a signed sum because it contains only addition and subtraction.

Without parentheses, the algorithm is just:

```text
read number -> add sign * number
read operator -> update sign
```

Parentheses introduce nested expression contexts. When entering a parenthesis, save the outer running value and the sign that applies to the entire parenthesized expression. When leaving it, combine the saved outer value with the completed inner value.

So the core invariant is:

```text
current result = completed value of the current level
current sign   = sign for the next term in the current level
stack          = outer levels waiting for the current level to finish
```

Once this invariant is clear, the algorithm is a direct one-pass translation of arithmetic meaning rather than a memorized stack trick.

## Implementation
See `solutions/stack/p224_basic_calculator.py`.

## Tests
See `tests/stack/test_p224_basic_calculator.py`.

## Examples

### Example 1
- Input: `{'s': '1 + 1'}`
- Output: `2`

### Example 2
- Input: `{'s': ' 2-1 + 2 '}`
- Output: `3`

### Example 3
- Input: `{'s': '(1+(4+5+2)-3)+(6+8)'}`
- Output: `23`

## Follow-up Practice

- Trace `1-(2+3)` and write down the stack contents at each parenthesis.
- Trace `(7)-(0)+(4)` and confirm that parentheses after `+` and `-` are handled the same way.
- Test multi-digit numbers such as `12-(3+40)`.
- State exactly what each stack entry means before writing code.
