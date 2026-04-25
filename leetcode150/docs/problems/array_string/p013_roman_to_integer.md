# 13. Roman to Integer

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/roman-to-integer/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: parsing, lookahead

## First-Principles Explanation

### What The Problem Asks
You are given a Roman numeral string like `"III"`, `"LVIII"`, or `"MCMXCIV"`, and you must convert it into the ordinary integer it represents.

The main difficulty is that Roman numerals are not purely additive.

Most of the time, a symbol simply contributes its value:

- `I = 1`
- `V = 5`
- `X = 10`
- `L = 50`
- `C = 100`
- `D = 500`
- `M = 1000`

So `"VIII"` is easy: `5 + 1 + 1 + 1 = 8`.

But Roman numerals also use a subtractive rule. When a smaller symbol appears immediately before a larger one, the smaller value should be subtracted instead of added:

- `IV = 4` because `5 - 1`
- `IX = 9` because `10 - 1`
- `XL = 40` because `50 - 10`
- `XC = 90` because `100 - 10`
- `CD = 400` because `500 - 100`
- `CM = 900` because `1000 - 100`

So the problem is really:

> While reading the string from left to right, decide whether each symbol should be added normally or treated as a subtraction because of the symbol immediately after it.

That is the whole problem. Once that local decision is correct at every position, the total is correct.

### Brute-Force / Baseline Thinking
The most direct baseline is to parse the string using special two-character cases.

You could keep a table like:

- `"IV" -> 4`
- `"IX" -> 9`
- `"XL" -> 40`
- `"XC" -> 90`
- `"CD" -> 400`
- `"CM" -> 900`

Then scan the string:

1. If the next two characters form one of those special subtractive pairs, add that pair's value and jump ahead by two.
2. Otherwise add the value of the single current symbol and move ahead by one.

This works, and it is still `O(n)`.

So why look for something simpler?

- It requires remembering a separate list of special pairs.
- It treats subtractive notation as six exceptional cases instead of a general rule.
- It hides the real structure of the problem: a symbol is subtractive exactly when it is smaller than the symbol to its right.

The better solution comes from expressing that rule directly.

### Key Observation
For any position `i`, the meaning of `s[i]` depends only on one question:

> Is `value(s[i])` smaller than `value(s[i + 1])`?

If yes, then `s[i]` is being used subtractively, so we should subtract it.

If no, then `s[i]` is just part of the sum, so we should add it.

That means we do not need to recognize `"IV"` as a magical object.
We only need to notice:

- `I < V`, so subtract `I`
- `X < C`, so subtract `X`
- `C < M`, so subtract `C`

Everything else is addition.

This converts the problem from "parse many Roman numeral cases" into "compare adjacent values while scanning once."

### Invariant / State
The useful invariant is:

> After processing positions `0..i`, `total` equals the correct integer value contributed by that processed prefix.

To preserve that invariant, when we stand at index `i`:

- subtract `value(s[i])` if a larger symbol is immediately to the right
- otherwise add `value(s[i])`

Why is that safe?

Because Roman notation only changes the role of a symbol through the next symbol.
There is no long-range dependency where a character three steps later changes the meaning of the current character.

So at index `i`, a one-character lookahead is enough to decide the current symbol permanently.

That is the whole state:

- a map from Roman symbols to integer values
- the running total
- the current index

No stack, no backtracking, no dynamic programming.

### Detailed Algorithm
1. Build a value map for the seven Roman symbols.
2. Initialize `total = 0`.
3. Scan the string from left to right.
4. For each character:
   - let `curr` be its value
   - if there is a next character, let `next` be the next value
   - if `curr < next`, subtract `curr` from `total`
   - otherwise add `curr` to `total`
5. Return `total`.

In pseudocode:

```text
values = {
  'I': 1, 'V': 5, 'X': 10, 'L': 50,
  'C': 100, 'D': 500, 'M': 1000
}

total = 0

for i from 0 to len(s) - 1:
    curr = values[s[i]]

    if i + 1 < len(s) and curr < values[s[i + 1]]:
        total -= curr
    else:
        total += curr

return total
```

### Why This Algorithm Matches Roman Numerals
Think of a subtractive pair such as `"IV"`.

If we process one symbol at a time:

- at `I`, since `1 < 5`, we subtract `1`
- at `V`, there is no larger symbol to its right, so we add `5`

Net effect: `-1 + 5 = 4`

That is exactly what `"IV"` means.

The same logic works for every subtractive pair:

- `"IX"` becomes `-1 + 10 = 9`
- `"XL"` becomes `-10 + 50 = 40`
- `"CM"` becomes `-100 + 1000 = 900`

And ordinary additive runs still work naturally:

- `"VIII"` becomes `5 + 1 + 1 + 1 = 8`
- `"XXX"` becomes `10 + 10 + 10 = 30`

So one local rule covers both additive and subtractive notation.

### Detailed Walkthrough
Take the hardest official example: `"MCMXCIV"`.

Character values:

- `M = 1000`
- `C = 100`
- `M = 1000`
- `X = 10`
- `C = 100`
- `I = 1`
- `V = 5`

Now scan left to right.

#### Step 1: index 0, `'M'`
- current value = `1000`
- next value = `100`
- `1000 < 100` is false
- add `1000`

`total = 1000`

#### Step 2: index 1, `'C'`
- current value = `100`
- next value = `1000`
- `100 < 1000` is true
- subtract `100`

`total = 900`

This captures the `"CM"` part: the `C` is subtractive.

#### Step 3: index 2, `'M'`
- current value = `1000`
- next value = `10`
- `1000 < 10` is false
- add `1000`

`total = 1900`

#### Step 4: index 3, `'X'`
- current value = `10`
- next value = `100`
- `10 < 100` is true
- subtract `10`

`total = 1890`

This captures the `"XC"` part.

#### Step 5: index 4, `'C'`
- current value = `100`
- next value = `1`
- `100 < 1` is false
- add `100`

`total = 1990`

#### Step 6: index 5, `'I'`
- current value = `1`
- next value = `5`
- `1 < 5` is true
- subtract `1`

`total = 1989`

This captures the `"IV"` part.

#### Step 7: index 6, `'V'`
- current value = `5`
- there is no next symbol
- add `5`

`total = 1994`

Final answer: `1994`

Notice what happened:

- `"CM"` was handled automatically by subtracting `C`
- `"XC"` was handled automatically by subtracting `X`
- `"IV"` was handled automatically by subtracting `I`

We never needed special-case parsing logic beyond "smaller before larger means subtract."

### Reference Code
Python version of the idea:

```python
def romanToInt(s: str) -> int:
    values = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }

    total = 0

    for i, ch in enumerate(s):
        curr = values[ch]
        if i + 1 < len(s) and curr < values[s[i + 1]]:
            total -= curr
        else:
            total += curr

    return total
```

### Correctness
We can justify the algorithm with a simple local argument.

For each symbol `s[i]`, exactly one of two situations is true:

1. `s[i]` is followed by a larger-valued symbol.
2. `s[i]` is not followed by a larger-valued symbol.

If situation 1 holds, Roman numeral rules say `s[i]` is subtractive, so its contribution to the final number is negative. The algorithm subtracts it, which is correct.

If situation 2 holds, `s[i]` is additive, so its contribution is positive. The algorithm adds it, which is correct.

Because every symbol's contribution is decided correctly, and because the final integer is just the sum of all symbol contributions, the algorithm returns the correct integer for the whole string.

Another way to say it:

- every character contributes exactly once
- it is contributed with the right sign
- therefore the total is correct

### Complexity
- Time: `O(n)`, where `n` is the length of the string, because we scan once.
- Space: `O(1)` extra space, because the value map has constant size and the running total uses constant memory.

### Common Pitfalls
- Forgetting the boundary check for the last character. The last symbol has no next symbol, so it must be added.
- Using `<=` instead of `<`. Equal symbols like `"II"` or `"XX"` are additive, not subtractive.
- Overcomplicating the solution by hardcoding subtractive pairs when a general adjacent comparison is enough.
- Subtracting both characters of a subtractive pair. Only the smaller left symbol is subtracted; the larger right symbol is still added normally.
- Thinking a symbol can be affected by something farther than one position away. In this problem, immediate lookahead is sufficient.

### First-Principles Summary
Roman numerals look tricky only because some symbols are not always additive. But the rule is local:

> A symbol should be subtracted exactly when it is smaller than the symbol immediately after it.

Once you state the rule that way, the algorithm becomes inevitable:

- read left to right
- compare current with next
- subtract on an upward step
- add otherwise

This is why the problem is an array/string parsing problem rather than a memorization problem.
The solution comes from identifying the exact local condition that changes a symbol's sign.

## Implementation

See `solutions/array_string/p013_roman_to_integer.py`.

## Tests

See `tests/array_string/test_p013_roman_to_integer.py`.

## Examples

### Example 1
- Input: `{'s': 'III'}`
- Output: `3`

Why:

- `I` is not before a larger value, so add `1`
- next `I`, add `1`
- next `I`, add `1`
- total = `3`

### Example 2
- Input: `{'s': 'LVIII'}`
- Output: `58`

Why:

- `L = 50`
- `V = 5`
- `I + I + I = 3`
- total = `50 + 5 + 3 = 58`

### Example 3
- Input: `{'s': 'MCMXCIV'}`
- Output: `1994`

Why:

- `M = 1000`
- `CM = 900`
- `XC = 90`
- `IV = 4`
- total = `1994`

## Follow-up Practice
- Re-derive the solution without memorizing subtractive pairs; use only the smaller-before-larger rule.
- Trace the running total for `"IX"`, `"XLII"`, and `"CDXLIV"`.
- Explain why one-character lookahead is enough to decide the contribution of the current symbol.
