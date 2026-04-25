# 6. Zigzag Conversion

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/zigzag-conversion/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: simulation, string-building

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a string `s` and an integer `numRows`.

Imagine writing the characters of `s` in a zigzag pattern:

1. Go straight down row by row.
2. Then go diagonally up until you reach the top row.
3. Repeat.

After all characters are placed, read the rows from top to bottom and concatenate them.

For example, with:

```text
s = "PAYPALISHIRING"
numRows = 4
```

the pattern is:

```text
P     I    N
A   L S  I G
Y A   H R
P     I
```

Reading row by row gives:

```text
"PINALSIGYAHRPI"
```

So the problem is not asking you to draw something nicely. It is asking:

> For each character, which row does it belong to in the zigzag walk?

Once you know that, the answer is just the characters of row `0`, then row `1`, and so on.

### 2. Start From a Baseline Idea

The most literal approach is:

1. Simulate writing characters into a 2D grid.
2. Move downward and then diagonally upward.
3. Fill unused grid cells with blanks.
4. Read the grid row by row, skipping blanks.

That works conceptually, but it wastes effort.

Why?

- The grid is mostly empty.
- We do not actually care about columns in the final answer.
- The final string only depends on the order of characters within each row.

So a full matrix stores much more information than the problem needs.

This is the first simplification:

> We do not need to know the exact picture. We only need to know which row each character joins.

### 3. The Key Observation

The row movement is completely deterministic.

If `numRows = 4`, the visited rows are:

```text
0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, ...
```

If `numRows = 3`, the visited rows are:

```text
0, 1, 2, 1, 0, 1, 2, 1, ...
```

So each new character has exactly one destination row, and that row can be tracked using:

- `current_row`
- `direction`

where `direction` is either:

- `+1` for moving downward
- `-1` for moving upward

When you reach:

- the top row, you must start moving down
- the bottom row, you must start moving up

That is the whole simulation.

### 4. The Right State to Maintain

Instead of building a grid, maintain:

```text
rows[i] = the characters that belong to row i, in left-to-right encounter order
```

Along with:

```text
current_row = row for the next character
step = +1 or -1
```

The invariant is:

> After processing the first `k` characters of `s`, each `rows[i]` contains exactly the characters that would appear in row `i` of the zigzag drawing for that prefix, in the correct order.

This is the key reason the simulation is enough.

We never need to fix earlier placements, because once a character is assigned to a row, its relative order inside that row is final.

### 5. Edge Cases Fall Out Naturally

There are two special cases where the zigzag does nothing:

1. `numRows == 1`
2. `numRows >= len(s)`

Why?

If there is only one row, every character stays in that row.

If there are at least as many rows as characters, you never bounce back upward; each character just occupies its own row position in the original order.

In both cases, the answer is simply:

```text
s
```

### 6. Detailed Algorithm

Use one string builder per row.

For each character:

1. Append it to `rows[current_row]`.
2. If `current_row == 0`, set direction to downward.
3. If `current_row == numRows - 1`, set direction to upward.
4. Move `current_row += step`.

At the end, join all row builders from top to bottom.

In compact pseudocode:

```text
if numRows == 1 or numRows >= len(s):
    return s

rows = ["", "", ..., ""]   # one per row
current_row = 0
step = 1

for ch in s:
    rows[current_row] += ch

    if current_row == 0:
        step = 1
    elif current_row == numRows - 1:
        step = -1

    current_row += step

return "".join(rows)
```

### 7. Walk Through the Main Example

Take:

```text
s = "PAYPALISHIRING"
numRows = 4
```

We will track:

```text
rows
current_row
step
```

Start with:

```text
rows = ["", "", "", ""]
current_row = 0
step = 1
```

#### Character 1: `P`

Append to row `0`:

```text
rows = ["P", "", "", ""]
```

We are at the top, so direction stays downward.
Next row is `1`.

#### Character 2: `A`

Append to row `1`:

```text
rows = ["P", "A", "", ""]
```

Continue downward to row `2`.

#### Character 3: `Y`

Append to row `2`:

```text
rows = ["P", "A", "Y", ""]
```

Continue downward to row `3`.

#### Character 4: `P`

Append to row `3`:

```text
rows = ["P", "A", "Y", "P"]
```

Now we are at the bottom, so reverse direction.
Next row is `2`.

#### Character 5: `A`

Append to row `2`:

```text
rows = ["P", "A", "YA", "P"]
```

Continue upward to row `1`.

#### Character 6: `L`

Append to row `1`:

```text
rows = ["P", "AL", "YA", "P"]
```

Continue upward to row `0`.

#### Character 7: `I`

Append to row `0`:

```text
rows = ["PI", "AL", "YA", "P"]
```

At the top again, reverse direction downward.
Next row is `1`.

#### Continue the same pattern

Process the remaining characters:

```text
S -> row 1
H -> row 2
I -> row 3
R -> row 2
I -> row 1
N -> row 0
G -> row 1
```

Final rows become:

```text
row 0: "PIN"
row 1: "ALSIG"
row 2: "YAHR"
row 3: "PI"
```

Join them:

```text
"PIN" + "ALSIG" + "YAHR" + "PI" = "PINALSIGYAHRPI"
```

That matches the expected answer.

### 8. Why This Works

The zigzag drawing process places each character in exactly one row, and characters appear within each row in the same order they are encountered while scanning `s` from left to right.

Our algorithm does exactly that:

- `current_row` identifies the row where the next character should go.
- `step` ensures the row walk follows the down-then-up zigzag path.
- appending to `rows[current_row]` preserves left-to-right order within that row

So after processing every character, each row builder matches the corresponding row of the true zigzag layout. Reading rows from top to bottom therefore produces exactly the required output string.

### 9. Reference Pseudocode

```python
def convert(s: str, numRows: int) -> str:
    if numRows == 1 or numRows >= len(s):
        return s

    rows = [list() for _ in range(numRows)]
    current_row = 0
    step = 1

    for ch in s:
        rows[current_row].append(ch)

        if current_row == 0:
            step = 1
        elif current_row == numRows - 1:
            step = -1

        current_row += step

    return "".join("".join(row) for row in rows)
```

You can also store each row as a string builder rather than a list of characters. The idea is the same.

### 10. Correctness Argument

We prove the algorithm is correct by maintaining the row-content invariant.

Before processing any characters, every row is empty, which matches the zigzag drawing of the empty prefix.

Assume after processing the first `k` characters:

- each `rows[i]` contains exactly the characters belonging to row `i` for that prefix
- `current_row` is the correct row for character `k + 1`
- `step` points in the correct movement direction of the zigzag walk

Now process the next character `s[k]`:

1. The algorithm appends it to `rows[current_row]`, which is exactly the row where the zigzag walk would place it.
2. If this row is the top or bottom boundary, the algorithm flips direction exactly when the zigzag path would bounce.
3. It then updates `current_row` by one step, making it the correct destination row for the next character.

Therefore the invariant still holds after processing one more character.

By induction, the invariant holds after all characters are processed. Since the final answer is defined as reading the zigzag rows from top to bottom, joining `rows[0]` through `rows[numRows - 1]` returns the correct string.

### 11. Complexity

Let `n = len(s)`.

- Time: `O(n)` because each character is processed once.
- Extra space: `O(n)` because the row builders store all output characters before joining.

This is optimal up to the space needed for the returned string itself.

### 12. Common Pitfalls

#### Forgetting the `numRows == 1` case

If you try to bounce between rows when there is only one row, the movement logic becomes invalid.

#### Updating the row before checking boundaries incorrectly

The order matters. A simple pattern is:

1. append current character
2. possibly flip direction at top/bottom
3. move to the next row

That avoids off-by-one mistakes.

#### Building a full matrix

It works, but it is unnecessary and obscures the real structure of the problem.

#### Confusing this with a math-indexing problem too early

There is a cycle-length formula approach, but the row-walk simulation is usually simpler, clearer, and less error-prone. The problem is fundamentally about following a repeating path.

### 13. First-Principles Summary

This problem becomes easy once you stop thinking about the picture and focus on the state transition.

- Each character belongs to exactly one row.
- The row sequence is a repeating down-then-up walk.
- The only state you need is the current row and the current direction.
- Once characters are grouped by row in encounter order, concatenating the rows gives the answer.

So the real solution is:

> Simulate the row path, not the whole drawing.

## Implementation

See `solutions/array_string/p006_zigzag_conversion.py`.

## Tests

See `tests/array_string/test_p006_zigzag_conversion.py`.

## Examples

### Example 1
- Input: `{'s': 'PAYPALISHIRING', 'numRows': 3}`
- Output: `'PAHNAPLSIIGYIR'`

### Example 2
- Input: `{'s': 'PAYPALISHIRING', 'numRows': 4}`
- Output: `'PINALSIGYAHRPI'`

### Example 3
- Input: `{'s': 'A', 'numRows': 1}`
- Output: `'A'`
