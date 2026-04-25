# 79. Word Search

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/word-search/
- Official Group: Backtracking
- Pattern Group: Backtracking
- Patterns: backtracking, grid-dfs

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given:

```text
board = a 2D grid of characters
word  = a target string
```

You need to decide whether `word` can be formed by walking through the grid.

The walk has three rules:

1. The first cell in the walk must equal `word[0]`.
2. Each next cell must be horizontally or vertically adjacent to the previous cell.
3. The same board cell cannot be used more than once in the same word path.

Diagonal movement is not allowed.

For example, in this board:

```text
A B C E
S F C S
A D E E
```

the word:

```text
ABCCED
```

exists because we can trace:

```text
(0,0) A
(0,1) B
(0,2) C
(1,2) C
(2,2) E
(2,1) D
```

That path spells `A -> B -> C -> C -> E -> D`.

So the real problem is:

> Does there exist at least one simple path through adjacent grid cells whose letters exactly match `word`?

"Simple path" means no cell appears twice in the current path.

---

### 2. Start From the Brute Force Idea

The most direct way to think about the problem is:

1. Try every cell as the starting point.
2. From that cell, try every possible adjacent path.
3. Stop when a path spells the whole word.

Conceptually:

```python
for each cell in board:
    explore every path starting from that cell
    if any path spells word:
        return True

return False
```

This is correct because the word, if it exists, must start somewhere in the grid.

But the brute-force search tree can grow quickly. From a cell, there are up to four neighboring cells. From each neighbor, there are again several choices, and so on.

The important question is not whether we can avoid search completely. We generally cannot, because the answer depends on which route through repeated letters works.

The important question is:

> How do we search only paths that still have a chance to spell the word?

That leads to depth-first search with backtracking.

---

### 3. The Key Observation

At any point in a partial path, we know exactly which character must come next.

If we have already matched:

```text
word[0], word[1], ..., word[index - 1]
```

then the current recursive call is responsible for matching:

```text
word[index]
```

So when we stand on cell `(row, col)`, there are only two useful possibilities:

1. `board[row][col] == word[index]`, so this cell can be part of the path.
2. `board[row][col] != word[index]`, so this path is impossible and should stop immediately.

This gives us a very strong pruning rule:

```text
Never continue from a cell whose letter does not match the next required character.
```

The second key observation is about revisiting cells.

The rule says a cell cannot be reused in the same path, but it can be used in a different attempted path. Therefore, "visited" is not a global property of a cell. It is a property of the current recursive path.

That is exactly what backtracking is for:

```text
mark this cell as used
search deeper
unmark this cell before trying a different path
```

---

### 4. Why This Is Backtracking, Not Just Grid Traversal

A normal grid DFS might mark a cell visited forever because it is exploring connected components.

This problem is different.

Suppose one attempted path uses a cell and later fails. That does not mean the cell is useless. It may still be part of a different path that reaches it from a different direction or at a different position in the word.

For example, if the board contains repeated `C` cells, choosing the wrong `C` early may block a valid route later. We need to undo that choice and try another route.

So the search is a decision tree:

```text
choose a matching cell
choose the next adjacent matching cell
choose the next adjacent matching cell
...
```

If a choice leads to a dead end, we back up to the previous choice and try a different neighbor.

The board itself is not being permanently consumed. Only the current path temporarily owns the cells it has chosen.

---

### 5. Recursive State

A clean recursive function can be defined as:

```text
dfs(row, col, index)
```

Meaning:

> Starting from cell `(row, col)`, can we match `word[index:]`, assuming all cells already used by earlier characters in this path are marked unavailable?

The parameters have specific jobs:

- `row`, `col`: the current cell we want to use.
- `index`: the position in `word` that the current cell must match.
- the board's temporary marks, or a separate `visited` set: which cells are already used in the current path.

The recursive state is not "the whole path string". We do not need to build the string because `index` already tells us how many characters have been matched.

If `index == len(word)`, then every character has been matched, so the search succeeded.

Many implementations instead test success right after matching the final character:

```python
if index == len(word) - 1:
    return True
```

Both versions are equivalent. The important idea is that the recursion advances exactly one character of `word` per chosen cell.

---

### 6. The Invariant

Maintain this invariant for every active recursive call:

```text
The cells marked visited are exactly the cells used to match word[0:index].
```

Then `dfs(row, col, index)` tries to decide whether `(row, col)` can become the cell for `word[index]`.

After it verifies that:

```text
board[row][col] == word[index]
```

it marks `(row, col)` visited. At that moment, the invariant becomes:

```text
The visited cells are exactly the cells used to match word[0:index + 1].
```

Then the recursion tries neighbors for `word[index + 1]`.

When the recursive call returns, the current cell must be unmarked before control goes back to the caller. That restores the previous invariant for sibling branches.

This restoration step is the heart of backtracking.

Without it, one failed attempted path would incorrectly prevent later attempted paths from using the same cell.

---

### 7. Detailed Algorithm

The algorithm has two levels.

The outer level chooses the starting cell:

1. Let `rows = len(board)` and `cols = len(board[0])`.
2. For every cell `(row, col)`:
   - If the cell can start a successful DFS, return `True`.
3. If no starting cell works, return `False`.

The recursive level verifies one path:

1. If `(row, col)` is outside the board, return `False`.
2. If `(row, col)` is already used in the current path, return `False`.
3. If `board[row][col] != word[index]`, return `False`.
4. If `index` is the last position in `word`, return `True`.
5. Mark `(row, col)` as used.
6. Recursively try the four neighbors for `index + 1`:
   - up: `(row - 1, col)`
   - down: `(row + 1, col)`
   - left: `(row, col - 1)`
   - right: `(row, col + 1)`
7. If any neighbor succeeds, unmark the current cell and return `True`.
8. If all neighbors fail, unmark the current cell and return `False`.

The unmarking must happen whether the branch succeeds or fails if the implementation uses a shared `visited` set or temporary board mutation. A common pattern is to store the recursive result, restore state, then return the result.

---

### 8. Python-Style Pseudocode

Using a `visited` set:

```python
def exist(board, word):
    rows = len(board)
    cols = len(board[0])
    visited = set()

    def dfs(row, col, index):
        if row < 0 or row == rows or col < 0 or col == cols:
            return False

        if (row, col) in visited:
            return False

        if board[row][col] != word[index]:
            return False

        if index == len(word) - 1:
            return True

        visited.add((row, col))

        found = (
            dfs(row + 1, col, index + 1)
            or dfs(row - 1, col, index + 1)
            or dfs(row, col + 1, index + 1)
            or dfs(row, col - 1, index + 1)
        )

        visited.remove((row, col))
        return found

    for row in range(rows):
        for col in range(cols):
            if dfs(row, col, 0):
                return True

    return False
```

The same idea can be implemented without a `visited` set by temporarily replacing the board cell with a sentinel value such as `"#"` and restoring the original character after recursion.

That version saves auxiliary `visited` storage, but the logic is identical:

```text
temporarily mark chosen cell unavailable
search neighbors
restore original character
```

---

### 9. Walkthrough: `ABCCED`

Use the board:

```text
A B C E
S F C S
A D E E
```

and the word:

```text
ABCCED
```

The outer loop starts checking cells.

#### Start at `(0,0)`

```text
board[0][0] = A
word[0]     = A
```

This matches, so mark `(0,0)` as visited.

Current path:

```text
A
```

Next we need `word[1] = B`.

Neighbors of `(0,0)` are:

```text
(1,0) = S
(0,1) = B
```

`S` does not match `B`, so that branch stops. `(0,1)` matches.

#### Move to `(0,1)`

Current path:

```text
A B
```

Next we need `word[2] = C`.

Neighbors of `(0,1)` include:

```text
(0,0) = A, but already visited
(1,1) = F
(0,2) = C
```

The visited `A` cannot be reused. `F` does not match. `(0,2)` matches.

#### Move to `(0,2)`

Current path:

```text
A B C
```

Next we need `word[3] = C`.

Neighbors include:

```text
(0,1) = B, already visited
(0,3) = E
(1,2) = C
```

Only `(1,2)` works.

#### Move to `(1,2)`

Current path:

```text
A B C C
```

Next we need `word[4] = E`.

Neighbors include:

```text
(0,2) = C, already visited
(1,1) = F
(1,3) = S
(2,2) = E
```

Only `(2,2)` works.

#### Move to `(2,2)`

Current path:

```text
A B C C E
```

Next we need `word[5] = D`.

Neighbors include:

```text
(2,1) = D
(2,3) = E
(1,2) = C, already visited
```

`(2,1)` matches.

#### Move to `(2,1)`

Current path:

```text
A B C C E D
```

This matches the full word. The DFS returns `True`, and that `True` propagates all the way back to the outer loop.

The final answer is:

```text
True
```

---

### 10. Walkthrough: Why `ABCB` Fails

Use the same board:

```text
A B C E
S F C S
A D E E
```

and the word:

```text
ABCB
```

The promising start is again:

```text
(0,0) A -> (0,1) B -> (0,2) C
```

Now the next required character is:

```text
word[3] = B
```

From `(0,2)`, the adjacent cells are:

```text
(0,1) = B
(0,3) = E
(1,2) = C
```

At first glance, `(0,1)` looks like it could supply `B`.

But `(0,1)` is already part of the current path. Reusing it would mean the path is:

```text
(0,0) -> (0,1) -> (0,2) -> (0,1)
```

That violates the rule that each cell may be used at most once.

The other adjacent cells do not match `B`, so this path fails.

The algorithm then backtracks and tries other choices, but no valid route spells `ABCB` without reusing a cell.

The final answer is:

```text
False
```

This example is the reason the visited state must be path-specific. The `B` cell is not globally forbidden forever; it is forbidden only while it is already in the current path.

---

### 11. Correctness

We prove that the algorithm returns `True` if and only if the word exists in the board.

#### Lemma 1: Every path accepted by `dfs` spells the corresponding suffix of `word`.

`dfs(row, col, index)` returns `True` only after confirming that `(row, col)` is inside the board, has not already been used in the current path, and contains `word[index]`.

If `index` is the last character, this single valid cell completes the suffix.

Otherwise, `dfs` returns `True` only if one of the four adjacent recursive calls returns `True` for `index + 1`.

By induction on the remaining suffix length, the recursive call spells `word[index + 1:]` using adjacent unused cells. Adding the current matching cell in front spells `word[index:]`.

Therefore, any successful DFS branch corresponds to a valid word path.

#### Lemma 2: `dfs` considers every valid continuation from its current state.

After matching the current cell, the next cell in any valid path must be one of the four horizontal or vertical neighbors. The recursive step tries exactly those four neighbors.

It rejects only moves that are outside the board, reuse a cell already in the current path, or contain the wrong character. Each of those rejected moves violates a problem rule and cannot be part of a valid continuation.

Therefore, no valid continuation is skipped.

#### Lemma 3: Backtracking restores the state needed for other branches.

Before exploring neighbors, the algorithm marks the current cell as visited. After all neighbor exploration for that call is finished, it removes that mark.

Thus, when control returns to the caller, the visited set again represents exactly the caller's path and does not contain choices from a completed child branch.

Therefore, sibling branches are explored independently and are not polluted by earlier failed choices.

#### Theorem: The algorithm returns the correct answer.

If the algorithm returns `True`, then some DFS call found a valid path by Lemma 1, so the word exists.

If the word exists, its first cell is somewhere in the board, and the outer loop tries every cell as a start. Starting from the first cell of that valid path, Lemma 2 ensures DFS follows all valid continuations, and Lemma 3 ensures failed alternatives do not block the correct route. Therefore the algorithm eventually finds the path and returns `True`.

So the algorithm returns `True` exactly when the word exists in the board.

---

### 12. Complexity

Let:

```text
m = number of rows
n = number of columns
L = len(word)
```

There are `m * n` possible starting cells.

From the first cell of a path, there are at most `4` directions. After that, there are at most `3` useful directions at each step, because immediately going back to the previous cell is forbidden by the visited rule.

So a common tight upper-bound explanation is:

```text
O(m * n * 4 * 3^(L - 1))
```

This is usually simplified to:

```text
O(m * n * 3^L)
```

The search is exponential in the word length because the algorithm may need to explore many possible paths.

Space depends on how visited cells are tracked:

- With a `visited` set: `O(L)` for the current recursion path plus `O(L)` recursion depth.
- With in-place temporary board marking: `O(L)` recursion depth, excluding the input board.

The recursion never needs to be deeper than `L`, because each recursive level matches exactly one character.

---

### 13. Common Pitfalls

- **Using a global visited set:** A cell should be unavailable only inside the current path, not across all attempted starts and branches.
- **Forgetting to unmark a cell:** If a failed branch leaves a cell marked, later valid branches may be incorrectly blocked.
- **Allowing diagonal movement:** Only up, down, left, and right are allowed.
- **Checking neighbors before checking the current character:** The current cell must match `word[index]` before it can contribute to the path.
- **Reusing a cell to satisfy repeated letters:** Repeated letters in the word require distinct cells unless they appear at different positions in different attempted paths.
- **Returning too early without restoring state:** If the implementation mutates shared state, restore the cell or remove it from `visited` before returning from the call.
- **Confusing path length with board area:** A path can use at most `len(word)` cells, and it cannot exist if `len(word) > rows * cols`.

---

### 14. First-Principles Summary

The problem is not asking for all paths. It asks whether at least one valid path spells the target word.

The first-principles model is:

```text
A state is a current cell plus the next word index to match.
```

The local rule is:

```text
Use this cell only if it is in bounds, unused in the current path, and equals word[index].
```

The invariant is:

```text
Visited cells are exactly the cells already chosen for the current prefix of word.
```

The search branches because the next character could be in any of four directions. It backtracks because a choice that fails should not permanently affect other choices.

So the whole algorithm is just this loop of reasoning:

```text
match one required character
temporarily reserve that cell
try all legal adjacent cells for the next character
release the cell when done
```

That is why depth-first search plus backtracking fits this problem exactly.

## Implementation

See `solutions/backtracking/p079_word_search.py`.

## Tests

See `tests/backtracking/test_p079_word_search.py`.

## Examples

### Example 1
- Input: `{'board': [['A', 'B', 'C', 'E'], ['S', 'F', 'C', 'S'], ['A', 'D', 'E', 'E']], 'word': 'ABCCED'}`
- Output: `True`

### Example 2
- Input: `{'board': [['A', 'B', 'C', 'E'], ['S', 'F', 'C', 'S'], ['A', 'D', 'E', 'E']], 'word': 'SEE'}`
- Output: `True`

### Example 3
- Input: `{'board': [['A', 'B', 'C', 'E'], ['S', 'F', 'C', 'S'], ['A', 'D', 'E', 'E']], 'word': 'ABCB'}`
- Output: `False`

## Follow-up Practice

- Trace `SEE` by hand and write down when each cell becomes visited and unvisited.
- Change the neighbor order and confirm that correctness does not depend on trying directions in a specific order.
- Implement both versions of visited tracking: a `set` and temporary in-place board marking.
- Add a quick precheck for `len(word) > rows * cols` and explain why it is safe.
