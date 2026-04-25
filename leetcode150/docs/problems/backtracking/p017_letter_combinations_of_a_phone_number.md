# 17. Letter Combinations of a Phone Number

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/letter-combinations-of-a-phone-number/
- Official Group: Backtracking
- Pattern Group: Backtracking
- Patterns: backtracking

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given a string `digits`, return every possible letter string that those digits could represent on a classic telephone keypad.

The keypad mapping is fixed:

```text
2 -> abc
3 -> def
4 -> ghi
5 -> jkl
6 -> mno
7 -> pqrs
8 -> tuv
9 -> wxyz
```

Each input digit contributes exactly one letter to each output string.

For example, if:

```text
digits = "23"
```

then the first output character must come from digit `2`:

```text
a, b, or c
```

and the second output character must come from digit `3`:

```text
d, e, or f
```

So valid answers include:

```text
ad, ae, af, bd, be, bf, cd, ce, cf
```

The problem is not asking for one best combination. It is asking for **all** combinations.

Every result must satisfy two rules:

1. Its length equals `len(digits)`.
2. At index `i`, its character must be one of the letters mapped from `digits[i]`.

If `digits` is empty, there are no digit positions to fill, so the expected answer is an empty list:

```text
[]
```

### 2. Start From the Brute-Force Idea

The most direct way to think about the problem is:

1. Pick a letter for the first digit.
2. For each such pick, pick a letter for the second digit.
3. Continue until every digit has a letter.
4. Record the completed string.

For `"23"`, this looks like nested loops:

```python
answers = []

for first in "abc":
    for second in "def":
        answers.append(first + second)
```

That works for exactly two digits.

For three digits, we would need three nested loops:

```python
for first in letters_for_digit_0:
    for second in letters_for_digit_1:
        for third in letters_for_digit_2:
            ...
```

For four digits, four loops.

This reveals the core problem:

> The number of loop levels depends on the input length.

A fixed set of hand-written loops cannot naturally handle arbitrary `digits` length. We need a way to express:

```text
Choose one letter for the current digit, then solve the same problem for the next digit.
```

That is exactly what recursion gives us.

### 3. The Key Observation

A partial combination has a simple meaning.

If we have processed the first `i` digits, then our current path contains exactly `i` letters:

```text
path[0] came from digits[0]
path[1] came from digits[1]
...
path[i - 1] came from digits[i - 1]
```

The next decision is forced by position:

```text
Use digits[i] to decide which letters are available next.
```

There is no need to guess where a letter belongs. The recursion depth tells us the digit index.

So the search space is a decision tree:

```text
level 0: choose a letter for digits[0]
level 1: choose a letter for digits[1]
level 2: choose a letter for digits[2]
...
```

A root-to-leaf path is one completed phone-number letter combination.

For `digits = "23"`:

```text
                 ""
          /       |       \
        a         b         c
      / | \     / | \     / | \
    ad ae af  bd be bf  cd ce cf
```

The answer is the list of all leaves.

### 4. Recursion State and Invariant

A clean recursive state is:

```text
index = which digit we are about to process
path  = letters chosen so far
```

The invariant is:

```text
Before each recursive call, path has length index,
and path[j] is a valid letter for digits[j] for every j < index.
```

This invariant is the reason the algorithm is safe.

If the invariant holds at `index`, then choosing any letter from `phone[digits[index]]` creates a new path that is valid for the first `index + 1` digits. After that, the same reasoning applies to the next digit.

The base case is when:

```text
index == len(digits)
```

At that point, the path has one valid letter for every digit, so it is a complete answer and should be recorded.

### 5. Detailed Algorithm

First handle the empty input:

```text
If digits is empty, return [].
```

This avoids treating the empty path as a real phone-number combination.

Then create the keypad mapping:

```text
2 -> abc
3 -> def
4 -> ghi
5 -> jkl
6 -> mno
7 -> pqrs
8 -> tuv
9 -> wxyz
```

Maintain:

```text
answers = completed combinations
path    = current partial combination
```

Define a recursive function `backtrack(index)`:

1. If `index == len(digits)`, join the current `path` into a string and append it to `answers`.
2. Otherwise, read the current digit: `digit = digits[index]`.
3. Look up all letters for that digit.
4. For each possible letter:
   - append the letter to `path`;
   - recurse to `index + 1`;
   - remove the letter from `path` so the next branch starts from the same earlier state.

The append/recurse/pop sequence is important:

```text
choose
explore
undo
```

The undo step is not cleanup for aesthetics. It restores the invariant for the caller, so sibling branches do not accidentally inherit letters from previous branches.

### 6. Detailed Example Walkthrough

Take:

```text
digits = "23"
```

The mapping gives:

```text
2 -> abc
3 -> def
```

Start with:

```text
index = 0
path = []
answers = []
```

#### Choose for digit `2`

At `index = 0`, the current digit is `2`, so the choices are:

```text
a, b, c
```

Choose `a`:

```text
path = ["a"]
index = 1
```

Now the invariant says:

```text
path has length 1,
and path[0] is valid for digits[0].
```

That is true because `a` belongs to `2`.

#### Choose for digit `3`

At `index = 1`, the current digit is `3`, so the choices are:

```text
d, e, f
```

Choose `d`:

```text
path = ["a", "d"]
index = 2
```

Now `index == len(digits)`, so the path is complete:

```text
"ad"
```

Append it:

```text
answers = ["ad"]
```

Then undo the last choice:

```text
path = ["a"]
```

Still under the branch where the first letter is `a`, try the next letter for digit `3`.

Choose `e`:

```text
path = ["a", "e"]
answers = ["ad", "ae"]
```

Undo:

```text
path = ["a"]
```

Choose `f`:

```text
path = ["a", "f"]
answers = ["ad", "ae", "af"]
```

Undo:

```text
path = ["a"]
```

There are no more choices for digit `3`, so return to the previous level and undo `a`:

```text
path = []
```

#### Try the next first letters

Now choose `b` for digit `2`, and repeat the same process for digit `3`:

```text
bd, be, bf
```

Then choose `c` for digit `2`:

```text
cd, ce, cf
```

The final answer is:

```text
["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]
```

### 7. Code / Pseudocode

```python
def letterCombinations(digits: str) -> list[str]:
    if not digits:
        return []

    phone = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }

    answers = []
    path = []

    def backtrack(index: int) -> None:
        if index == len(digits):
            answers.append("".join(path))
            return

        digit = digits[index]
        for letter in phone[digit]:
            path.append(letter)
            backtrack(index + 1)
            path.pop()

    backtrack(0)
    return answers
```

The same idea can also be written by passing strings instead of mutating a list:

```python
def backtrack(index, prefix):
    if index == len(digits):
        answers.append(prefix)
        return

    for letter in phone[digits[index]]:
        backtrack(index + 1, prefix + letter)
```

Both versions are based on the same state:

```text
current digit index + letters chosen so far
```

The list version makes the backtracking mechanics explicit; the string-prefix version avoids manual `pop`, but creates a new string at each recursive edge.

### 8. Correctness

We prove that the algorithm returns exactly all valid phone-number letter combinations.

#### Lemma 1: Every recorded string is valid.

A string is recorded only when `index == len(digits)`. By the invariant, at that moment `path` has length `len(digits)`, and for every position `j`, `path[j]` is a letter mapped from `digits[j]`. Therefore every recorded string has exactly one valid letter for each input digit, so every recorded string is a valid combination.

#### Lemma 2: Every valid combination is recorded.

Take any valid combination `combo`. For each position `i`, `combo[i]` is one of the letters mapped from `digits[i]`. When the recursion reaches level `i`, it iterates over all letters mapped from `digits[i]`, including `combo[i]`. Therefore there is a branch that chooses `combo[0]`, then `combo[1]`, and so on until all positions are chosen. That branch reaches the base case and records `combo`.

#### Lemma 3: No combination is recorded more than once.

Each recursive level chooses the letter for one fixed digit position. A completed path is determined by exactly one sequence of choices, one choice per digit. The algorithm visits each such sequence once because each loop iterates through the available letters once for that digit position. Therefore no completed combination is recorded more than once.

#### Theorem: The algorithm returns exactly the required answer.

By Lemma 1, everything returned is valid. By Lemma 2, every valid combination is returned. By Lemma 3, the algorithm does not introduce duplicate records for the same sequence of choices. Therefore the returned list is exactly the set of all letter combinations represented by `digits`.

### 9. Complexity

Let `n = len(digits)`.

Each digit has either 3 or 4 possible letters:

```text
2, 3, 4, 5, 6, 8 -> 3 letters
7, 9             -> 4 letters
```

If the input contains `k` digits that map to 4 letters and `n - k` digits that map to 3 letters, then the number of output strings is:

```text
4^k * 3^(n - k)
```

For each completed output, joining the path into a string costs `O(n)`.

So the precise output-sensitive time complexity is:

```text
O(n * 4^k * 3^(n - k))
```

The common worst-case bound is:

```text
O(n * 4^n)
```

because no digit has more than 4 letters.

Auxiliary space, not counting the returned output list, is:

```text
O(n)
```

for the recursion stack and current path.

The output list itself stores:

```text
O(n * 4^k * 3^(n - k))
```

characters in the worst case for the produced strings.

### 10. Common Pitfalls

#### Returning `[""]` for empty input

The recursive base case naturally records the empty path if called on an empty string. But this problem expects:

```text
[]
```

not:

```text
[""]
```

So handle `if not digits: return []` before starting recursion.

#### Forgetting to undo the choice

If using a mutable `path`, every `append` must be paired with a `pop` after the recursive call.

Wrong shape:

```python
path.append(letter)
backtrack(index + 1)
```

Correct shape:

```python
path.append(letter)
backtrack(index + 1)
path.pop()
```

Without the `pop`, later branches contain letters chosen by earlier branches.

#### Advancing by the wrong amount

Each recursion level corresponds to exactly one digit. After choosing a letter for `digits[index]`, the next call must use:

```text
index + 1
```

not the next letter position inside the keypad string.

#### Mixing up digits and letters

The input contains digit characters like `"2"`, not integer `2`. If the mapping keys are strings, lookups must use string digits:

```python
phone[digits[index]]
```

#### Trying to prune branches

There is no invalid partial branch to prune here as long as each chosen letter comes from the current digit's mapping. Every partial path can be extended until the end. The task is pure enumeration.

### 11. First-Principles Summary

The problem is a product of independent choices: one letter for each digit. The only challenge is that the number of choices is not fixed in the source code; it depends on `len(digits)`.

Recursion solves that by making each level responsible for one digit position.

The invariant is:

```text
path already contains valid choices for exactly the digits before index.
```

At each level, we try every letter for the current digit, recurse to fill the rest, and then undo the choice so the next letter starts from the same clean state.

A completed path is a root-to-leaf path through the decision tree, and the answer is the list of all such leaves.

## Implementation

See `solutions/backtracking/p017_letter_combinations_of_a_phone_number.py`.

## Tests

See `tests/backtracking/test_p017_letter_combinations_of_a_phone_number.py`.

## Examples

### Example 1
- Input: `{'digits': '23'}`
- Output: `['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']`

### Example 2
- Input: `{'digits': '2'}`
- Output: `['a', 'b', 'c']`

## Follow-up Practice
- Draw the decision tree for `"79"`, where both digits have four choices.
- State the recursion invariant before writing code.
- Trace exactly when `path.append(...)` and `path.pop()` happen for input `"23"`.
