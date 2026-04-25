# 20. Valid Parentheses

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/valid-parentheses/
- Official Group: Stack
- Pattern Group: Stack
- Patterns: stack

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given a string `s` containing only these six characters:

```text
( ) [ ] { }
```

decide whether the parentheses, square brackets, and curly braces form a valid bracket expression.

A string is valid when every closing bracket matches the most recent still-open bracket of the same type.

For example:

```text
"()[]{}"
```

is valid because each pair opens and closes independently:

```text
()  []  {}
```

This is also valid:

```text
"([])"
```

The outer parentheses enclose a complete square-bracket pair:

```text
(
  []
)
```

But this is not valid:

```text
"([)]"
```

The string starts by opening `(` and then `[`. Since `[` was opened most recently, it must be closed before `(` can close. The next closing bracket is `)`, so the nesting order is broken.

So the real problem is:

> While reading the string from left to right, can every closing bracket correctly close the most recent unmatched opening bracket?

---

### 2. Start From the Brute Force Baseline

A very direct way to think about the problem is to repeatedly remove adjacent matching pairs:

```text
()
[]
{}
```

For example:

```text
s = "({[]})"
```

We can remove the innermost adjacent pair first:

```text
"({[]})" -> "({})"
```

Then remove the next pair:

```text
"({})" -> "()"
```

Then remove the final pair:

```text
"()" -> ""
```

If the whole string disappears, it was valid.

Conceptually:

```python
while s contains "()" or "[]" or "{}":
    remove one such adjacent pair

return s == ""
```

This idea is correct because a valid bracket expression must contain at least one innermost pair that appears as adjacent characters.

But it is inefficient. Each removal can require scanning and rebuilding the string, and there may be many removals. In the worst case, this can become `O(n^2)` time.

The deeper question is:

> Can we simulate the same innermost-pair logic in one left-to-right pass?

Yes. The data structure that does exactly this is a stack.

---

### 3. The Key Observation

When a closing bracket appears, it does not get to choose any earlier opening bracket.

It must match the most recent opening bracket that has not already been closed.

For example, in:

```text
"({[]})"
```

after reading:

```text
"({["
```

the currently open brackets are:

```text
(
{
[
```

The next valid close must be `]`, because `[` is the latest unresolved opening bracket.

This is a last-in, first-out rule:

```text
The last bracket opened must be the first one closed.
```

That is exactly what a stack represents.

---

### 4. The Stack Invariant

The stack should contain the opening brackets that have been seen but not yet matched.

More precisely, after processing `s[0:i]`, the stack contains exactly the unmatched opening brackets from that prefix, in the order they were opened.

The top of the stack is the most recent unmatched opening bracket.

This invariant gives every character a simple meaning:

- If the character is an opening bracket, it becomes unresolved, so push it.
- If the character is a closing bracket, it must resolve the stack top.
- If the stack is empty when a closing bracket appears, there is nothing to close, so the string is invalid.
- If the stack top is the wrong opening bracket, the nesting is invalid.
- If the stack top matches, pop it because that pair is now fully resolved.

At the end, all opened brackets must have been closed. Therefore the stack must be empty.

---

### 5. Detailed Algorithm

Create a map from closing brackets to the opening brackets they require:

```text
')' -> '('
']' -> '['
'}' -> '{'
```

Then scan the string from left to right.

For each character `ch`:

1. If `ch` is an opening bracket, push it onto the stack.
2. Otherwise, `ch` is a closing bracket.
3. If the stack is empty, return `False`.
4. Pop or inspect the top opening bracket.
5. If that opening bracket is not the one required by `ch`, return `False`.

After the loop:

1. If the stack is empty, return `True`.
2. Otherwise, return `False` because some opening brackets were never closed.

The important point is that the algorithm never needs to remember the entire processed prefix. It only needs the unresolved openings, because already matched pairs can no longer affect future validity.

---

### 6. Detailed Example Walkthrough

Consider:

```text
s = "({[]})"
```

Start with an empty stack:

```text
stack = []
```

Read `(`:

```text
opening bracket -> push
stack = ['(']
```

Read `{`:

```text
opening bracket -> push
stack = ['(', '{']
```

Read `[`:

```text
opening bracket -> push
stack = ['(', '{', '[']
```

Read `]`:

```text
closing bracket -> requires '['
top of stack is '[' -> match
pop
stack = ['(', '{']
```

Read `}`:

```text
closing bracket -> requires '{'
top of stack is '{' -> match
pop
stack = ['(']
```

Read `)`:

```text
closing bracket -> requires '('
top of stack is '(' -> match
pop
stack = []
```

The scan is finished and the stack is empty, so the string is valid.

Now compare that with:

```text
s = "([)]"
```

Process `(` and `[`:

```text
stack = ['(', '[']
```

Read `)`:

```text
closing bracket -> requires '('
top of stack is '['
```

The most recent unmatched opening bracket is `[`, but `)` wants to close `(`. That would cross the nesting boundary, so the answer is immediately `False`.

---

### 7. Code / Pseudocode

Python-style implementation:

```python
def isValid(s: str) -> bool:
    matching_open = {
        ")": "(",
        "]": "[",
        "}": "{",
    }

    stack = []

    for ch in s:
        if ch in "([{":
            stack.append(ch)
        else:
            if not stack:
                return False

            top = stack.pop()
            if top != matching_open[ch]:
                return False

    return not stack
```

An equivalent version checks the closing-bracket map first:

```python
def isValid(s: str) -> bool:
    matching_open = {
        ")": "(",
        "]": "[",
        "}": "{",
    }

    stack = []

    for ch in s:
        if ch not in matching_open:
            stack.append(ch)
            continue

        if not stack or stack.pop() != matching_open[ch]:
            return False

    return len(stack) == 0
```

Because LeetCode defines the input as containing only bracket characters, either style is fine. If a broader input alphabet were allowed, you would need to decide explicitly whether non-bracket characters should be ignored or rejected.

---

### 8. Correctness

We prove that the algorithm returns `True` exactly when the input string is valid.

#### Invariant

After processing any prefix of the string, the stack contains exactly the unmatched opening brackets in that prefix, ordered from earliest at the bottom to latest at the top.

#### Why The Invariant Holds

Initially, before reading any characters, there are no unmatched opening brackets and the stack is empty.

When the algorithm reads an opening bracket, that bracket has not been matched yet, so pushing it adds exactly one new unmatched opening bracket to the stack.

When the algorithm reads a closing bracket, a valid expression requires it to match the most recent unmatched opening bracket. The most recent unmatched opening bracket is exactly the stack top by the invariant. If the stack is empty, no such opening bracket exists, so the string cannot be valid. If the stack top has the wrong type, the closing bracket would close an older bracket before closing the newer one, so the nesting order is invalid. If the stack top has the correct type, popping it removes exactly the pair that has now been matched.

Thus every step preserves the invariant unless the algorithm correctly detects an invalid string.

#### Why Empty Stack At The End Means Valid

If the algorithm finishes with an empty stack, then every opening bracket was matched by a later closing bracket of the correct type, and no closing bracket ever appeared without the correct most recent opening bracket. Therefore the entire string is valid.

#### Why Non-Empty Stack At The End Means Invalid

If the algorithm finishes with a non-empty stack, then at least one opening bracket was never closed. A valid bracket string cannot leave unmatched openings, so the string is invalid.

Therefore, the algorithm is correct.

---

### 9. Complexity

- Time: `O(n)`, where `n` is the length of `s`. Each character is processed once, and each bracket is pushed or popped at most once.
- Space: `O(n)` in the worst case. For example, `"((((((("` pushes every character before the algorithm reaches the end.

---

### 10. Common Pitfalls

- Forgetting to check for an empty stack before popping. A string like `")"` must return `False`, not crash.
- Returning `True` too early after all closes match so far. A string like `"(("` has no mismatch during the scan, but it is still invalid because the stack is not empty at the end.
- Matching against any previous opening bracket instead of the most recent one. This incorrectly accepts crossing patterns like `"([)]"`.
- Only checking counts of each bracket type. The counts in `"([)]"` are balanced, but the order is invalid.
- Pushing closing brackets as well as opening brackets without a clear invariant. The stack should represent unresolved openings, not every character seen.
- Reversing the mapping direction accidentally. For this scan, it is usually easiest to map each closing bracket to the opening bracket it requires.

---

### 11. First-Principles Summary

This problem is about nested obligations.

Every opening bracket creates an obligation that must be satisfied later. But obligations created later must be satisfied earlier, because nested structures close from the inside out.

That creates a LIFO rule:

```text
last opened = first closed
```

A stack is the direct data structure for that rule.

The algorithm is not a memorized trick. It follows from the grammar of valid brackets:

```text
An opening bracket waits.
A closing bracket must close the latest thing still waiting.
At the end, nothing may still be waiting.
```

Once that invariant is clear, the implementation is just one pass over the string.

## Implementation
See `solutions/stack/p020_valid_parentheses.py`.

## Tests
See `tests/stack/test_p020_valid_parentheses.py`.

## Examples

### Example 1
- Input: `{'raw': '"()"\n"()[]{}"\n"(]"\n"([])"\n"([)]"'}`
- Output: `'See official examples'`

### Additional Examples

```text
Input: s = "()"
Output: true
Reason: '(' is immediately closed by ')'.
```

```text
Input: s = "()[]{}"
Output: true
Reason: each independent pair is correctly closed.
```

```text
Input: s = "(]"
Output: false
Reason: ')' would be needed to close '(', but the string gives ']'.
```

```text
Input: s = "([])"
Output: true
Reason: '[' closes before the surrounding '(' closes.
```

```text
Input: s = "([)]"
Output: false
Reason: ')' tries to close '(' while '[' is still the most recent unmatched opening bracket.
```

## Follow-up Practice

- Trace the stack for `"{[()()]}"` and find where validity succeeds or fails.
- Explain why checking only bracket counts is insufficient.
- Write down the stack invariant before writing code.
- Test empty input if the platform allows it; under the usual definition, an empty bracket string is valid because there are no unmatched brackets.
