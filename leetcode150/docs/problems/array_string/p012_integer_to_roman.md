# 12. Integer to Roman

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/integer-to-roman/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: greedy, lookup-table, string-building

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an integer `num`, convert it into its Roman numeral representation.

The LeetCode constraints are important:

```text
1 <= num <= 3999
```

So we do not need to handle zero, negative numbers, or numbers that require non-standard overline notation.

Roman numerals are built from symbols:

```text
I = 1
V = 5
X = 10
L = 50
C = 100
D = 500
M = 1000
```

But the representation is not just "repeat symbols until the sum matches." Some values use subtraction:

```text
IV = 4
IX = 9
XL = 40
XC = 90
CD = 400
CM = 900
```

For example:

```text
58 = 50 + 5 + 3
   = L  + V + III
   = LVIII
```

and:

```text
1994 = 1000 + 900 + 90 + 4
     = M    + CM  + XC + IV
     = MCMXCIV
```

The task is therefore:

> Write `num` as a sequence of Roman tokens whose numeric values add to `num`, using the standard canonical Roman representation.

The word **canonical** matters. Many strings can sum to the same value if we allow arbitrary repetition:

```text
IIII    also sums to 4
VIIII   also sums to 9
DCCCC   also sums to 900
```

But these are not the standard Roman forms. The standard forms are:

```text
IV
IX
CM
```

So the problem is not just arithmetic. It is arithmetic under Roman numeral formatting rules.

### 2. Start From a Baseline Idea

A direct baseline is to handle each digit place separately:

```text
thousands place
hundreds place
tens place
ones place
```

For each place, we could write special cases:

```text
1, 2, 3 -> repeat the one-symbol
4       -> one-symbol before five-symbol
5       -> five-symbol
6, 7, 8 -> five-symbol plus repeated one-symbol
9       -> one-symbol before next-place one-symbol
```

For example, in the tens place:

```text
10 = X
20 = XX
30 = XXX
40 = XL
50 = L
60 = LX
70 = LXX
80 = LXXX
90 = XC
```

This works, and it is still constant time because `num <= 3999`.

But it creates a lot of place-specific branching. The code has to separately remember the rules for ones, tens, hundreds, and thousands.

The deeper question is:

> Can we express all Roman formatting rules as one ordered list of reusable pieces?

Yes. Treat every valid Roman symbol or subtractive pair as a token with a value.

### 3. The Key Observation: Subtractive Pairs Are Tokens

The usual symbols are tokens:

```text
M  = 1000
D  = 500
C  = 100
L  = 50
X  = 10
V  = 5
I  = 1
```

The subtractive forms should also be treated as tokens:

```text
CM = 900
CD = 400
XC = 90
XL = 40
IX = 9
IV = 4
```

Now sort all tokens from largest value to smallest value:

```text
1000 -> M
 900 -> CM
 500 -> D
 400 -> CD
 100 -> C
  90 -> XC
  50 -> L
  40 -> XL
  10 -> X
   9 -> IX
   5 -> V
   4 -> IV
   1 -> I
```

This list is the whole problem.

Once subtractive pairs are included, Roman conversion becomes:

> Repeatedly take the largest token whose value fits in the remaining number.

For example, for `1994`:

```text
remaining = 1994

largest token <= 1994 is M  (1000) -> output M,  remaining = 994
largest token <= 994  is CM (900)  -> output CM, remaining = 94
largest token <= 94   is XC (90)   -> output XC, remaining = 4
largest token <= 4    is IV (4)    -> output IV, remaining = 0

answer = MCMXCIV
```

The subtractive cases do not need special `if num == 4` or `if num == 9` logic. They are already present in the table.

### 4. Why Greedy Is Safe Here

Greedy means:

```text
At each step, choose the largest Roman token that does not exceed the remaining value.
```

This is safe because standard Roman numerals are written from larger values to smaller values, with the only exceptions being the six subtractive pairs. After we include those six pairs in the token list, there are no hidden exceptions left.

For each place value, the canonical representation is exactly the greedy decomposition:

```text
3000 -> MMM
900  -> CM
400  -> CD
90   -> XC
40   -> XL
9    -> IX
4    -> IV
```

Consider a remaining value `r`.

If `r >= 1000`, the canonical representation must begin with `M`, because no smaller token can replace a leading thousand without using many lower tokens, and standard Roman numerals use thousands first.

If `900 <= r < 1000`, the canonical representation must begin with `CM`, not `DCCCC`, because `CM` is the standard form for nine hundreds.

If `500 <= r < 900`, it begins with `D`.

If `400 <= r < 500`, it begins with `CD`.

The same reasoning repeats for hundreds, tens, and ones.

So choosing the largest fitting token does not accidentally block the correct answer later. It chooses exactly the next prefix that a canonical Roman numeral would have.

### 5. The State and Invariant

We maintain two pieces of state:

```text
remaining = the part of the original number not yet represented
result    = Roman tokens already emitted
```

The invariant is:

```text
value(result) + remaining == original num
```

and:

```text
result is the canonical Roman prefix for the part already consumed
```

At the beginning:

```text
result = ""
remaining = num
```

The invariant is true because:

```text
value("") + num == num
```

At each step, we choose a token `(value, symbol)` such that:

```text
value <= remaining
```

Then we append `symbol` and subtract `value`:

```text
result += symbol
remaining -= value
```

The sum is preserved:

```text
new value(result) + new remaining
= old value(result) + value + (old remaining - value)
= old value(result) + old remaining
= original num
```

The canonical-prefix part is also preserved because the token list is ordered exactly as Roman numerals are ordered, including subtractive pairs.

When `remaining == 0`, the invariant says:

```text
value(result) == original num
```

and the prefix is now the whole canonical Roman numeral.

### 6. Detailed Algorithm

Build a list of `(value, symbol)` pairs in descending value order:

```text
[
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]
```

Then:

1. Start with an empty list of output pieces.
2. For each `(value, symbol)` from largest to smallest:
3. While `num >= value`:
4. Append `symbol`.
5. Subtract `value` from `num`.
6. Join the output pieces into one string.

The loop uses `while`, not `if`, because some symbols can repeat.

For example:

```text
3000 -> M + M + M
30   -> X + X + X
3    -> I + I + I
```

But subtractive tokens naturally appear at most once per place because after subtracting one of them, the remaining value is below that token.

### 7. Code

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        values = [
            (1000, "M"),
            (900, "CM"),
            (500, "D"),
            (400, "CD"),
            (100, "C"),
            (90, "XC"),
            (50, "L"),
            (40, "XL"),
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I"),
        ]

        result = []

        for value, symbol in values:
            while num >= value:
                result.append(symbol)
                num -= value

        return "".join(result)
```

This version is intentionally direct. It mirrors the reasoning:

```text
largest fitting token -> append it -> remove its value
```

There is also a small optimization using `divmod`:

```python
for value, symbol in values:
    count, num = divmod(num, value)
    result.append(symbol * count)
```

Both approaches rely on the same invariant.

### 8. Detailed Example Walkthrough

Use:

```text
num = 3749
```

Start:

```text
remaining = 3749
result = ""
```

The largest token is `M = 1000`.

`3749 >= 1000`, so append `M`:

```text
remaining = 2749
result = "M"
```

Still `2749 >= 1000`, append another `M`:

```text
remaining = 1749
result = "MM"
```

Still `1749 >= 1000`, append another `M`:

```text
remaining = 749
result = "MMM"
```

Now `749 < 1000`, so move to the next token.

`900 = CM` does not fit:

```text
749 < 900
```

`500 = D` fits:

```text
remaining = 249
result = "MMMD"
```

`500` no longer fits.

`400 = CD` does not fit:

```text
249 < 400
```

`100 = C` fits:

```text
remaining = 149
result = "MMMDC"
```

`100 = C` fits again:

```text
remaining = 49
result = "MMMDCC"
```

`100` no longer fits.

`90 = XC` does not fit:

```text
49 < 90
```

`50 = L` does not fit:

```text
49 < 50
```

`40 = XL` fits:

```text
remaining = 9
result = "MMMDCCXL"
```

`10 = X` does not fit:

```text
9 < 10
```

`9 = IX` fits:

```text
remaining = 0
result = "MMMDCCXLIX"
```

The remaining value is zero, so the final answer is:

```text
MMMDCCXLIX
```

Notice how the subtractive cases appeared automatically:

```text
40 -> XL
9  -> IX
```

No digit-specific branch was needed.

### 9. Correctness

We prove that the algorithm returns the canonical Roman numeral for `num`.

#### Lemma 1: Each emitted token is a valid next Roman token.

The token list contains all standard Roman numeral pieces:

```text
M, CM, D, CD, C, XC, L, XL, X, IX, V, IV, I
```

in descending order. These include both ordinary symbols and all legal subtractive pairs.

When the algorithm appends a token, its value is no larger than the current remaining number. Therefore appending it never represents more value than is still needed.

#### Lemma 2: The greedy token matches the next token of the canonical representation.

For any remaining value, the canonical Roman numeral begins with the largest token that fits.

This follows from the Roman numeral rules by place:

```text
900 is represented by CM, not DCCCC
400 is represented by CD, not CCCC
90  is represented by XC, not LXXXX
40  is represented by XL, not XXXX
9   is represented by IX, not VIIII
4   is represented by IV, not IIII
```

All other ranges begin with the largest ordinary symbol that fits, such as `M`, `D`, `C`, `L`, `X`, `V`, or `I`.

Because the list includes these subtractive forms in the correct positions, the largest fitting token is exactly the canonical next token.

#### Lemma 3: The invariant is maintained.

The algorithm maintains:

```text
value(result) + remaining == original num
```

Initially this is true because `result` is empty and `remaining == original num`.

Each step appends a token of value `v` and subtracts `v` from `remaining`. The represented value increases by exactly the same amount that `remaining` decreases, so the sum stays unchanged.

#### Theorem: The algorithm returns the correct Roman numeral.

By Lemma 2, every token appended by the algorithm is the next token of the canonical Roman representation of the remaining value.

By Lemma 3, the algorithm always represents exactly the consumed portion of the original number and never loses or creates value.

The loop stops only when `remaining == 0`. At that point:

```text
value(result) == original num
```

and every prefix choice was canonical. Therefore `result` is the canonical Roman numeral for the input integer.

### 10. Complexity

The token list has constant size: `13`.

Because `num <= 3999`, the output length is also bounded by a small constant. The maximum standard Roman numeral length in this range is small.

So under the LeetCode constraints:

```text
Time:  O(1)
Space: O(1) extra space, excluding the returned string
```

If we describe the algorithm in terms of output length `L`, then:

```text
Time:  O(L)
Space: O(L) for the result string/list
```

Both descriptions are useful. For this exact problem, `L` is bounded, so the runtime is constant.

### 11. Common Pitfalls

#### Pitfall 1: Forgetting subtractive pairs

If the table contains only:

```text
1000, 500, 100, 50, 10, 5, 1
```

then `4` becomes:

```text
IIII
```

instead of:

```text
IV
```

The subtractive pairs must be included as tokens.

#### Pitfall 2: Putting tokens in the wrong order

The order matters.

For example, `900 = CM` must appear before `500 = D` and `100 = C`.

If `C` appears before `CM`, the algorithm may build:

```text
DCCCC
```

instead of:

```text
CM
```

Descending order is what makes the greedy step canonical.

#### Pitfall 3: Using `if` when a token can repeat

For `3000`, the answer needs three `M` symbols:

```text
MMM
```

If the code only checks `if num >= 1000`, it emits one `M` and leaves `2000` unprocessed until no later token can represent it correctly.

Use `while`, or compute the repetition count with `divmod`.

#### Pitfall 4: Treating the input as decimal digits but mishandling zeros

A digit-place solution can work, but it must handle zeros carefully.

For example:

```text
1004 = MIV
```

There is no symbol for the zero hundreds or zero tens places. The greedy token method avoids this issue because it only emits tokens that actually fit.

#### Pitfall 5: Returning a list instead of a joined string

Building with a list is efficient:

```python
result.append(symbol)
```

But the final answer must be:

```python
"".join(result)
```

not the list itself.

### 12. First-Principles Summary

Roman numerals look tricky because some values are written subtractively. The simplifying move is to stop treating subtraction as an exception and instead make each subtractive pair a normal token.

Once the token table contains:

```text
CM, CD, XC, XL, IX, IV
```

the conversion rule becomes straightforward:

> Keep taking the largest Roman token that fits in the remaining number.

The algorithm works because canonical Roman numerals are written from larger values to smaller values, and the token list captures every legal larger-before-smaller exception explicitly.

## Implementation

See `solutions/array_string/p012_integer_to_roman.py`.

## Tests

See `tests/array_string/test_p012_integer_to_roman.py`.

## Examples

### Example 1

- Input: `{'num': 3749}`
- Output: `'MMMDCCXLIX'`

### Example 2

- Input: `{'num': 58}`
- Output: `'LVIII'`

### Example 3

- Input: `{'num': 1994}`
- Output: `'MCMXCIV'`

## Follow-up Practice

- Trace `remaining` and `result` for `944`, `3999`, and `1004`.
- Rewrite the solution with `divmod` and compare it with the `while` version.
- Implement the digit-place baseline, then explain why the greedy token table removes most branching.
