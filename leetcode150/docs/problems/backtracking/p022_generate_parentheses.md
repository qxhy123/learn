# 22. Generate Parentheses

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/generate-parentheses/
- Official Group: Backtracking
- Pattern Group: Backtracking
- Patterns: backtracking, constraint-pruning

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an integer `n`, generate every string that contains:

```text
n opening parentheses: "("
n closing parentheses: ")"
```

and is a valid parentheses expression.

For example, when:

```text
n = 3
```

we must use exactly six characters:

```text
3 copies of "("
3 copies of ")"
```

Some arrangements are valid:

```text
((()))
(()())
(())()
()(())
()()()
```

Some arrangements are not valid:

```text
())(()
())()(
)((())
```

The key rule is not just that the final counts match. A valid parentheses string must also be valid at every prefix.

For example:

```text
())(()
```

has three opening and three closing parentheses, but it fails early:

```text
()
())  <- this prefix has more closing parentheses than opening parentheses
```

Once a prefix has more `)` than `(`, no later character can repair it. The string has already tried to close something that was never opened.

So the problem is really asking:

> List every length-`2n` string over `(` and `)` such that every prefix has at least as many `(` as `)`, and the final string has exactly `n` of each.

### 2. Start From the Brute Force Idea

The most direct baseline is to generate every possible string of length `2n` made from two characters.

At each of the `2n` positions, choose either:

```text
"("
")"
```

Then validate the completed string.

Conceptually:

```python
answers = []

for every string candidate of length 2 * n using '(' and ')':
    if candidate is valid and uses n '(' and n ')':
        answers.append(candidate)
```

This is correct, but wasteful.

There are:

```text
2^(2n)
```

raw strings of length `2n`.

Most of them are impossible answers. For `n = 3`, the brute-force tree contains strings such as:

```text
))))))
((((((
())())
```

Some use the wrong counts. Some become invalid immediately. But brute force still spends time constructing them before throwing them away.

The deeper question is:

> Can we avoid building partial strings that are already impossible to finish validly?

### 3. The Key Observation: Validity Is a Prefix Property

A parentheses string is valid if two conditions hold:

```text
1. The total number of '(' equals the total number of ')'.
2. In every prefix, count('(') >= count(')').
```

The second condition is the one that lets us prune early.

Suppose a partial string is:

```text
())
```

It has:

```text
open_count  = 1
close_count = 2
```

This is already invalid because there are more closing parentheses than opening parentheses.

No suffix can fix it:

```text
())...
```

The problem happened at the third character. Adding more characters later cannot change the fact that a prefix was invalid.

So when constructing the string from left to right, we should never take a step that makes:

```text
close_count > open_count
```

That one inequality removes all branches that try to close more groups than have been opened.

### 4. What Choices Are Actually Safe?

At any point, we have a partial string, such as:

```text
(()
```

We need decide whether the next character can be `(`, `)`, or both.

Let:

```text
open_count  = number of '(' already used
close_count = number of ')' already used
```

There are only two possible next moves.

#### Choice A: Add `(`

Adding `(` is safe if we still have opening parentheses available:

```text
open_count < n
```

Why?

Because the final answer must use exactly `n` opening parentheses. If we have already used `n`, adding another one would make the string impossible.

#### Choice B: Add `)`

Adding `)` is safe if it does not close more groups than are currently open:

```text
close_count < open_count
```

Why not `close_count < n`?

Because having closing parentheses remaining is not enough.

For example, at the beginning:

```text
open_count  = 0
close_count = 0
```

There are certainly `n` closing parentheses remaining, but we cannot start with `)` because there is nothing to close.

So the correct rule is:

```text
You may add ')' only when there is an unmatched '(' before it.
```

That is exactly:

```text
close_count < open_count
```

### 5. The Recursion State and Invariant

We build the answer one character at a time.

A recursive call needs to know three things:

```text
path        = the current partial string
open_count  = how many '(' have been used
close_count = how many ')' have been used
```

The invariant is:

```text
path is a valid prefix of at least one possible answer.
```

More concretely, every recursive state must satisfy:

```text
0 <= open_count <= n
0 <= close_count <= open_count
len(path) = open_count + close_count
```

These facts mean:

```text
- We have not used too many opening parentheses.
- We have not used too many closing parentheses.
- The prefix has never closed more groups than it opened.
```

The recursion only explores states that satisfy this invariant.

When the path reaches length `2n`, there is no room left to add characters. If the invariant still holds, then the string must have exactly `n` openings and `n` closings.

Why?

At length `2n`:

```text
open_count + close_count = 2n
open_count <= n
close_count <= open_count
```

The only way to fill all `2n` positions without exceeding `n` openings and without closings exceeding openings is:

```text
open_count = n
close_count = n
```

So a complete invariant-preserving path is a valid answer.

### 6. Detailed Algorithm

Use depth-first search.

At each call:

1. If the current string has length `2n`, add it to the result.
2. If `open_count < n`, append `(` and recurse.
3. If `close_count < open_count`, append `)` and recurse.
4. Return to the caller so the other branch can be explored.

With a mutable list of characters, the backtracking shape is:

```text
append a character
recurse
pop that character
```

The `pop` matters because the same `path` list is reused for sibling branches.

For example, if one branch explores:

```text
((()))
```

then the algorithm must remove characters as it returns so it can later explore:

```text
(()())
```

Without undoing choices, different branches would leak into each other.

### 7. Code

```python
from typing import List


class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result = []
        path = []

        def dfs(open_count: int, close_count: int) -> None:
            if len(path) == 2 * n:
                result.append("".join(path))
                return

            if open_count < n:
                path.append("(")
                dfs(open_count + 1, close_count)
                path.pop()

            if close_count < open_count:
                path.append(")")
                dfs(open_count, close_count + 1)
                path.pop()

        dfs(0, 0)
        return result
```

The same idea can also be written without mutating `path`:

```python
def dfs(path: str, open_count: int, close_count: int) -> None:
    if len(path) == 2 * n:
        result.append(path)
        return

    if open_count < n:
        dfs(path + "(", open_count + 1, close_count)

    if close_count < open_count:
        dfs(path + ")", open_count, close_count + 1)
```

The mutable-list version avoids creating a new string at every recursive step, but both versions express the same state transition.

### 8. Walkthrough for `n = 3`

Start with an empty path:

```text
path = ""
open = 0
close = 0
```

Can we add `(`?

```text
open < n
0 < 3  yes
```

Can we add `)`?

```text
close < open
0 < 0  no
```

So the first character must be `(`.

#### Step 1: Choose `(`

```text
path = "("
open = 1
close = 0
```

Now both choices are possible:

```text
add '(' because open < 3
add ')' because close < open
```

Depth-first search tries `(` first.

#### Step 2: Choose another `(`

```text
path = "(("
open = 2
close = 0
```

Again, we can add either `(` or `)`.

Choose `(` first.

#### Step 3: Choose another `(`

```text
path = "((("
open = 3
close = 0
```

Now we cannot add `(` anymore:

```text
open < n
3 < 3  no
```

The only legal move is `)`.

#### Step 4: Close until complete

```text
path = "((()"
open = 3
close = 1
```

Still only `)` is possible, because no openings remain.

```text
path = "((())"
open = 3
close = 2
```

Again add `)`:

```text
path = "((()))"
open = 3
close = 3
```

The path length is `6`, so record:

```text
((()))
```

Now backtracking begins.

The algorithm returns to the most recent state that still has an unexplored legal choice.

#### Backtrack to Explore a Different Shape

After recording `((()))`, the recursion unwinds to:

```text
path = "(("
open = 2
close = 0
```

The branch that added `(` has been fully explored. Now try `)`:

```text
path = "(()"
open = 2
close = 1
```

From here, try `(`:

```text
path = "(()("
open = 3
close = 1
```

Then only closings remain:

```text
path = "(()()"
open = 3
close = 2
```

```text
path = "(()())"
open = 3
close = 3
```

Record:

```text
(()())
```

Backtracking continues and eventually records all valid strings in this order:

```text
((()))
(()())
(())()
()(())
()()()
```

Notice what never appears in the search:

```text
())...
)(...
```

Those prefixes are pruned immediately because `close_count < open_count` would be false before adding the illegal `)`.

### 9. Why This Algorithm Is Correct

We need prove two things:

```text
1. Every string the algorithm returns is valid.
2. Every valid string is returned by the algorithm.
```

#### Every Returned String Is Valid

The algorithm only appends `(` when:

```text
open_count < n
```

So no path can contain more than `n` opening parentheses.

The algorithm only appends `)` when:

```text
close_count < open_count
```

So after adding a closing parenthesis, the number of closings is still at most the number of openings in the prefix.

Therefore every prefix of every explored path satisfies:

```text
close_count <= open_count
```

When the algorithm records a string, its length is `2n`. Since it never used more than `n` openings and never allowed closings to exceed openings, the completed string must contain exactly `n` openings and `n` closings.

So every returned string is a valid parentheses string.

#### Every Valid String Is Returned

Take any valid answer string.

Read it from left to right.

At each position, the string contains either `(` or `)`.

If the next character is `(`, then the valid answer has not yet used all `n` openings. Otherwise it would contain more than `n` openings. So the algorithm allows that `(` branch.

If the next character is `)`, then the prefix before it has more openings than closings. Otherwise adding `)` would make some prefix invalid. So the algorithm allows that `)` branch.

Thus every character of the valid answer corresponds to a branch the algorithm will take.

Because depth-first search explores all legal branches, it eventually follows the exact sequence of choices for that valid answer and records it.

So every valid string is returned.

Together, these prove that the algorithm returns exactly the required set of strings.

### 10. Complexity

The output size is not polynomial in `n`. The number of valid parentheses strings with `n` pairs is the `n`th Catalan number:

```text
C_n = (1 / (n + 1)) * binomial(2n, n)
```

The algorithm outputs exactly `C_n` strings.

Each string has length:

```text
2n
```

So just constructing the output costs:

```text
O(C_n * n)
```

More explicitly:

```text
Time:  O(C_n * n)
Space: O(n) recursion/path space, excluding the output
Output: O(C_n * n)
```

If output storage is included, total space is:

```text
O(C_n * n)
```

The backtracking recursion depth is at most `2n`, because each recursive step appends exactly one character.

### 11. Common Pitfalls

#### Pitfall 1: Allowing `)` When `close_count < n`

This rule is too weak:

```python
if close_count < n:
    add ")"
```

It would allow invalid prefixes like:

```text
)
())
```

The correct condition is:

```python
if close_count < open_count:
    add ")"
```

You may close only if there is an unmatched opening parenthesis.

#### Pitfall 2: Waiting Until the End to Check Validity

Generating all `2^(2n)` strings and validating afterward works conceptually, but it ignores the most important structure of the problem.

Once a prefix is invalid, the entire branch is doomed.

Backtracking should enforce the prefix invariant during construction, not after construction.

#### Pitfall 3: Forgetting to Undo a Mutable Path

If using a list, every append in a branch must be paired with a pop after recursion:

```python
path.append("(")
dfs(...)
path.pop()
```

Without `pop`, sibling branches start from the wrong partial string.

#### Pitfall 4: Recording the Mutable List Directly

If `path` is a list, do not append the list itself to `result`:

```python
result.append(path)      # wrong shape and wrong mutability
```

Join or copy it:

```python
result.append("".join(path))
```

The result should contain strings, not references to the changing work buffer.

#### Pitfall 5: Thinking the Order Is the Main Point

The examples show one common order:

```text
((()))
(()())
(())()
()(())
()()()
```

This order naturally appears when DFS tries `(` before `)`. The mathematical requirement is to generate all valid strings; online judges often accept any order unless the problem statement or tests require a specific ordering.

### 12. First-Principles Summary

This problem follows from a small set of facts:

```text
1. A final answer has exactly n '(' and n ')'.
2. A string is invalid forever once any prefix has more ')' than '('.
3. Therefore, build from left to right and keep only valid prefixes.
4. Add '(' only while fewer than n openings have been used.
5. Add ')' only when there is an unmatched '(' to close.
6. When the path length reaches 2n, the valid prefix is a complete answer.
```

So the whole algorithm is:

> Grow a parentheses string one character at a time, maintain the invariant that the current path is always a valid prefix, recursively explore every safe next character, and record each complete path.

## Implementation

See `solutions/backtracking/p022_generate_parentheses.py`.

## Tests

See `tests/backtracking/test_p022_generate_parentheses.py`.

## Examples

### Example 1
- Input: `{'n': 3}`
- Output: `['((()))', '(()())', '(())()', '()(())', '()()()']`

### Example 2
- Input: `{'n': 1}`
- Output: `['()']`

## Follow-up Practice
- Draw the recursion tree for `n = 2` and mark where `)` is disallowed.
- Rewrite the solution using remaining counts instead of used counts.
- Explain why the invalid prefix `())` can never be repaired by adding more characters.
