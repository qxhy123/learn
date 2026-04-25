# 72. Edit Distance

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/edit-distance/
- Official Group: Multidimensional DP
- Pattern Group: Dynamic Programming Multidimensional
- Patterns: dynamic-programming-multidimensional

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given two strings:

```text
word1
word2
```

You may transform `word1` into `word2` using three operations:

```text
insert one character
delete one character
replace one character
```

Each operation costs `1`.

The task is to return the minimum total cost needed to turn `word1` into `word2`.

For example:

```text
word1 = "horse"
word2 = "ros"
```

One optimal sequence is:

```text
horse
rorse   replace 'h' with 'r'
rose    delete 'r'
ros     delete 'e'
```

That uses `3` operations, so the answer is `3`.

The important phrase is **minimum number of operations**. We are not asked to output the operations themselves. We only need the smallest possible cost.

So the real problem is:

> Among all possible ways to align, remove, insert, and change characters so that the first string becomes the second string, what is the least number of single-character edits?

---

### 2. Start From the Brute-Force Recursion

A direct way to think about the problem is to compare the strings from the front.

Suppose we are currently trying to convert this suffix of `word1`:

```text
word1[i:]
```

into this suffix of `word2`:

```text
word2[j:]
```

If both suffixes are empty, no work remains.

If the first suffix is empty, then the only way to create `word2[j:]` is to insert every remaining character of `word2`.

If the second suffix is empty, then the only way to remove `word1[i:]` is to delete every remaining character of `word1`.

Otherwise, compare the current characters:

```text
word1[i]
word2[j]
```

If they are equal, there is no reason to edit this character. We can keep it and solve the smaller problem:

```text
word1[i + 1:] -> word2[j + 1:]
```

If they are different, an optimal sequence must begin with one of the three allowed edit operations:

1. **Insert** `word2[j]` into `word1` before `word1[i]`.
   - Now `word2[j]` has been matched.
   - `word1[i]` is still waiting.
   - Remaining problem: `word1[i:] -> word2[j + 1:]`.

2. **Delete** `word1[i]`.
   - Now `word1[i]` is gone.
   - `word2[j]` is still waiting.
   - Remaining problem: `word1[i + 1:] -> word2[j:]`.

3. **Replace** `word1[i]` with `word2[j]`.
   - Now both current characters are handled.
   - Remaining problem: `word1[i + 1:] -> word2[j + 1:]`.

So a brute-force recursive definition is:

```python
def edit(i, j):
    if i == len(word1):
        return len(word2) - j
    if j == len(word2):
        return len(word1) - i

    if word1[i] == word2[j]:
        return edit(i + 1, j + 1)

    insert_cost = 1 + edit(i, j + 1)
    delete_cost = 1 + edit(i + 1, j)
    replace_cost = 1 + edit(i + 1, j + 1)

    return min(insert_cost, delete_cost, replace_cost)
```

This is correct as a mathematical description, but it is too slow if implemented naively.

Why?

Because the same `(i, j)` subproblem appears through many different edit sequences.

For example, one path might insert then delete, while another path might delete then insert, and both can eventually ask for the same conversion:

```text
word1[i:] -> word2[j:]
```

The brute-force recursion branches up to three ways at many mismatches, so it can grow exponentially.

---

### 3. The Key Observation

The future only depends on two positions:

```text
i = how much of word1 has already been handled
j = how much of word2 has already been handled
```

It does not matter how we reached `(i, j)`.

If two different edit sequences both leave us with the task:

```text
convert word1[i:] into word2[j:]
```

then the minimum remaining cost is identical for both paths.

That is the overlapping-subproblem structure.

Instead of recomputing the same answer again and again, store the answer for each pair of prefixes or suffixes.

There are only:

```text
(len(word1) + 1) * (len(word2) + 1)
```

such pairs.

This turns the exponential search tree into a polynomial-size table.

---

### 4. DP State and Invariant

A common bottom-up state is:

```text
dp[i][j] = minimum edits needed to convert word1[:i] into word2[:j]
```

This means:

```text
word1[:i] = first i characters of word1
word2[:j] = first j characters of word2
```

The invariant is precise:

> After `dp[i][j]` is computed, it stores the true minimum number of insertions, deletions, and replacements needed to transform the first `i` characters of `word1` into the first `j` characters of `word2`.

The answer to the original problem is therefore:

```text
dp[len(word1)][len(word2)]
```

because that is the cost to convert all of `word1` into all of `word2`.

---

### 5. Boundary Conditions

Before comparing real characters, handle empty prefixes.

If `word2[:0]` is empty, then converting `word1[:i]` into it requires deleting all `i` characters:

```text
dp[i][0] = i
```

For example:

```text
"hors" -> "" costs 4 deletes
```

If `word1[:0]` is empty, then converting it into `word2[:j]` requires inserting all `j` characters:

```text
dp[0][j] = j
```

For example:

```text
"" -> "ros" costs 3 inserts
```

The top-left cell is:

```text
dp[0][0] = 0
```

because empty string already equals empty string.

---

### 6. Deriving the Transition From the Last Step

To compute `dp[i][j]`, look at the last characters of the prefixes:

```text
word1[i - 1]
word2[j - 1]
```

There are two cases.

#### Case 1: The Last Characters Match

If:

```text
word1[i - 1] == word2[j - 1]
```

then those last characters can be kept for free.

The cost is just the cost of converting the earlier prefixes:

```text
dp[i][j] = dp[i - 1][j - 1]
```

Example:

```text
"ho" -> "ro"
```

The last characters are both `'o'`, so the final `'o'` needs no edit. The problem reduces to:

```text
"h" -> "r"
```

#### Case 2: The Last Characters Differ

If:

```text
word1[i - 1] != word2[j - 1]
```

then the optimal transformation must end with exactly one of the allowed operations.

Think backward from the finished target prefix `word2[:j]`.

##### Insert

The final operation could insert `word2[j - 1]` at the end.

Before that insertion, `word1[:i]` must have already been converted into `word2[:j - 1]`.

So:

```text
insert cost = dp[i][j - 1] + 1
```

##### Delete

The final operation could delete `word1[i - 1]`.

Before that deletion, `word1[:i - 1]` must have already been converted into `word2[:j]`.

So:

```text
delete cost = dp[i - 1][j] + 1
```

##### Replace

The final operation could replace `word1[i - 1]` with `word2[j - 1]`.

Before that replacement, the earlier prefixes must already match optimally:

```text
word1[:i - 1] -> word2[:j - 1]
```

So:

```text
replace cost = dp[i - 1][j - 1] + 1
```

Take the cheapest possible final operation:

```text
dp[i][j] = 1 + min(
    dp[i][j - 1],      # insert
    dp[i - 1][j],      # delete
    dp[i - 1][j - 1],  # replace
)
```

---

### 7. Detailed Algorithm

Let:

```text
m = len(word1)
n = len(word2)
```

Create a table with `m + 1` rows and `n + 1` columns.

Rows represent prefixes of `word1`.
Columns represent prefixes of `word2`.

Algorithm:

1. Create `dp` filled with zeroes.
2. Initialize the first column:
   - `dp[i][0] = i`
   - this means delete all characters from `word1[:i]`.
3. Initialize the first row:
   - `dp[0][j] = j`
   - this means insert all characters from `word2[:j]`.
4. For each `i` from `1` to `m`:
   - For each `j` from `1` to `n`:
     - If `word1[i - 1] == word2[j - 1]`, copy `dp[i - 1][j - 1]`.
     - Otherwise, take `1 + min(insert, delete, replace)`.
5. Return `dp[m][n]`.

The fill order works because every cell depends only on:

```text
dp[i][j - 1]      left
dp[i - 1][j]      above
dp[i - 1][j - 1]  diagonal upper-left
```

When filling row by row from top to bottom and left to right, all three dependencies have already been computed.

---

### 8. Code

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i

        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    insert_cost = dp[i][j - 1]
                    delete_cost = dp[i - 1][j]
                    replace_cost = dp[i - 1][j - 1]
                    dp[i][j] = 1 + min(insert_cost, delete_cost, replace_cost)

        return dp[m][n]
```

This table-based solution is the clearest form of the idea.

A space-optimized version can keep only the previous row and current row, because each state only needs the row above and the cell to the left. But the full table is usually better for first understanding the recurrence.

---

### 9. Detailed Example Walkthrough: `horse` -> `ros`

Use:

```text
word1 = "horse"
word2 = "ros"
```

Rows are prefixes of `word1`:

```text
"", "h", "ho", "hor", "hors", "horse"
```

Columns are prefixes of `word2`:

```text
"", "r", "ro", "ros"
```

The initialized table is:

```text
        ""  r  o  s
""      0   1  2  3
h       1
ho      2
hor     3
hors    4
horse   5
```

The first row says: from empty string, insert characters.
The first column says: to reach empty string, delete characters.

Now fill the inner cells.

For `dp[1][1]`, convert `"h"` to `"r"`.
The characters differ, so choose the best among insert, delete, and replace:

```text
insert  = dp[1][0] + 1 = 2
delete  = dp[0][1] + 1 = 2
replace = dp[0][0] + 1 = 1
```

So:

```text
dp[1][1] = 1
```

That corresponds to replacing `'h'` with `'r'`.

For `dp[2][2]`, convert `"ho"` to `"ro"`.
The last characters both equal `'o'`, so:

```text
dp[2][2] = dp[1][1] = 1
```

The best way is still just replace `'h'` with `'r'`; the `'o'` is kept.

For `dp[3][1]`, convert `"hor"` to `"r"`.
The last characters both equal `'r'`, so:

```text
dp[3][1] = dp[2][0] = 2
```

This corresponds to deleting `'h'` and `'o'`, then keeping `'r'`.

For `dp[4][3]`, convert `"hors"` to `"ros"`.
The last characters both equal `'s'`, so:

```text
dp[4][3] = dp[3][2]
```

The table has already computed the best way to convert `"hor"` to `"ro"`, so keeping the final `'s'` costs nothing extra.

After filling all cells, the table is:

```text
          ""  r  o  s
""        0   1  2  3
h         1   1  2  3
ho        2   2  1  2
hor       3   2  2  2
hors      4   3  3  2
horse     5   4  4  3
```

The answer is the bottom-right cell:

```text
dp[5][3] = 3
```

So the minimum edit distance from `"horse"` to `"ros"` is `3`.

---

### 10. Another Example: `intention` -> `execution`

The official second example is:

```text
word1 = "intention"
word2 = "execution"
```

The answer is `5`.

One valid sequence is:

```text
intention
inention    delete 't'
enention    replace 'i' with 'e'
exention    replace 'n' with 'x'
exection    replace 'n' with 'c'
execution   insert 'u'
```

That proves the distance is at most `5`.

The DP proves it cannot be smaller, because each table cell records the cheapest possible way to transform its prefix pair. When the algorithm reaches the full prefixes, `dp[9][9]` is `5`.

This example is useful because many characters overlap between the two strings, especially the ending:

```text
...tion
```

The DP does not greedily match the first equal-looking characters. It considers all prefix alignments through insert/delete/replace transitions, so it can find the globally cheapest transformation.

---

### 11. Correctness

We prove that the algorithm returns the minimum edit distance between `word1` and `word2`.

#### Lemma 1: The boundary values are correct.

For every `i`, `dp[i][0] = i` is correct because converting `word1[:i]` into the empty string requires deleting all `i` characters, and no operation can remove more than one character.

For every `j`, `dp[0][j] = j` is correct because converting the empty string into `word2[:j]` requires inserting all `j` characters, and no operation can add more than one character.

#### Lemma 2: If the last characters match, `dp[i][j] = dp[i - 1][j - 1]` is correct.

Assume `word1[i - 1] == word2[j - 1]`.

Any optimal transformation from `word1[:i - 1]` to `word2[:j - 1]` can be extended to a transformation from `word1[:i]` to `word2[:j]` by leaving the matching last character unchanged. Therefore:

```text
dp[i][j] <= dp[i - 1][j - 1]
```

Also, because the final characters already match, editing one of them is never better than keeping it and optimally transforming the earlier prefixes. Any solution that needlessly changes a matching final character can be replaced by one that keeps it without increasing cost. Therefore:

```text
dp[i][j] >= dp[i - 1][j - 1]
```

So:

```text
dp[i][j] = dp[i - 1][j - 1]
```

#### Lemma 3: If the last characters differ, the transition considers every possible optimal final operation.

Assume `word1[i - 1] != word2[j - 1]`.

Any valid transformation from `word1[:i]` to `word2[:j]` must end by making the final target prefix correct. The final edit operation must be one of the three allowed operations:

- Insert `word2[j - 1]`, after converting `word1[:i]` to `word2[:j - 1]`.
- Delete `word1[i - 1]`, after converting `word1[:i - 1]` to `word2[:j]`.
- Replace `word1[i - 1]` with `word2[j - 1]`, after converting `word1[:i - 1]` to `word2[:j - 1]`.

These are exactly the three candidates:

```text
dp[i][j - 1] + 1
dp[i - 1][j] + 1
dp[i - 1][j - 1] + 1
```

Taking the minimum over them gives the cheapest possible valid final operation.

#### Theorem: The algorithm returns the correct answer.

The table is filled in an order where each cell's dependencies are already known. By Lemma 1, all boundary cells are correct. For each inner cell, Lemma 2 or Lemma 3 proves that the recurrence computes the correct value from smaller correct values.

By induction over the table fill order, every `dp[i][j]` is correct.

The returned value `dp[len(word1)][len(word2)]` is therefore the minimum number of edits needed to convert the entire `word1` into the entire `word2`.

---

### 12. Complexity

Let:

```text
m = len(word1)
n = len(word2)
```

The DP table has:

```text
(m + 1) * (n + 1)
```

cells.

Each cell is computed in constant time.

So the time complexity is:

```text
O(m * n)
```

The full table uses:

```text
O(m * n)
```

space.

If optimized to keep only two rows, the space can be reduced to:

```text
O(n)
```

where `n = len(word2)`.

---

### 13. Common Pitfalls

#### Pitfall 1: Forgetting the Empty Prefix Row and Column

The table must have size:

```text
(m + 1) by (n + 1)
```

not:

```text
m by n
```

The empty-prefix states are real subproblems. Without them, insertions from an empty string and deletions to an empty string are hard to represent cleanly.

#### Pitfall 2: Mixing Up Insert and Delete Transitions

For prefix DP:

```text
insert = dp[i][j - 1] + 1
delete = dp[i - 1][j] + 1
replace = dp[i - 1][j - 1] + 1
```

The names can feel backwards at first.

Remember the state meaning:

```text
dp[i][j] = word1[:i] -> word2[:j]
```

If you insert the final character of `word2[:j]`, then before insertion you had already formed `word2[:j - 1]`, so the previous state is `dp[i][j - 1]`.

If you delete the final character of `word1[:i]`, then before deletion you had `word1[:i - 1]`, so the previous state is `dp[i - 1][j]`.

#### Pitfall 3: Adding One When Characters Match

When:

```text
word1[i - 1] == word2[j - 1]
```

there is no edit cost for those characters.

Use:

```text
dp[i][j] = dp[i - 1][j - 1]
```

not:

```text
dp[i][j] = 1 + dp[i - 1][j - 1]
```

#### Pitfall 4: Using Greedy Matching

It is tempting to keep matching characters as soon as possible and edit only mismatches.

That is not reliable.

Insertions and deletions shift alignment. A character that looks useful now might be better matched later. The DP works because it considers all possible alignments through the three transitions.

#### Pitfall 5: Optimizing Space Too Early

The recurrence is easier to reason about with the full table.

Only optimize to one or two rows after the state meaning and dependency order are completely clear. In one-row versions, the diagonal value `dp[i - 1][j - 1]` must be saved carefully before it is overwritten.

---

### 14. First-Principles Summary

Edit distance is not about simulating one chosen sequence of edits.

It is about recognizing that every partial transformation is fully described by two prefix lengths:

```text
how many characters of word1 are being converted
how many characters of word2 are being produced
```

From that state, the final step into the state must be one of the allowed edit operations:

```text
insert
delete
replace
```

or, if the final characters already match, no operation is needed for them.

That gives the complete recurrence:

```text
if word1[i - 1] == word2[j - 1]:
    dp[i][j] = dp[i - 1][j - 1]
else:
    dp[i][j] = 1 + min(
        dp[i][j - 1],
        dp[i - 1][j],
        dp[i - 1][j - 1],
    )
```

The reason this works is that an optimal transformation has an optimal smaller transformation immediately before its final operation. Dynamic programming stores those smaller optimal answers once and reuses them.

## Implementation
See `solutions/dynamic_programming_multidimensional/p072_edit_distance.py`.

## Tests
See `tests/dynamic_programming_multidimensional/test_p072_edit_distance.py`.

## Examples

### Example 1
- Input: `{'word1': 'horse', 'word2': 'ros'}`
- Output: `3`

### Example 2
- Input: `{'word1': 'intention', 'word2': 'execution'}`
- Output: `5`

## Follow-up Practice
- Fill the `horse` by `ros` table by hand and label every insert, delete, and replace choice.
- Re-derive the recurrence using suffixes instead of prefixes, then compare the base cases.
- Implement the two-row space optimization only after the full-table version is correct.
