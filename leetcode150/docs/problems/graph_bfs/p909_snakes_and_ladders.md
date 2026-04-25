# 909. Snakes and Ladders

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/snakes-and-ladders/
- Official Group: Graph BFS
- Pattern Group: Graph BFS
- Patterns: graph-bfs

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an `n x n` Snakes and Ladders board.

The board is not numbered in ordinary row-major order. The square labels run from `1` to `n * n` in a **boustrophedon** pattern:

```text
Start at the bottom-left square.
Move left-to-right across the bottom row.
Then move right-to-left across the row above it.
Keep alternating directions until the top row.
```

You start on square `1`. On each move, you choose a die result from `1` to `6`, so from square `s` you may try to move to any square:

```text
s + 1, s + 2, ..., s + 6
```

as long as the target does not exceed `n * n`.

After landing on that target square, if the board contains a snake or ladder there, you must immediately move to the destination written in that cell. If the cell contains `-1`, you stay there.

The important rule is:

> A snake or ladder is followed at most once per die roll.

So if a ladder sends you to another square that also contains a ladder, you do **not** chain again during the same move.

The task is:

> Return the minimum number of die rolls needed to reach square `n * n`, or `-1` if it is impossible.

This is a shortest-path problem hidden inside a board game.

Each square is a state. Each die roll is one move. Every move has equal cost. Therefore, the answer is the length of the shortest path from square `1` to square `n * n` in an implicit unweighted graph.

### 2. Start From the Brute Force Idea

A direct way to think about the game is recursive search:

```python
try every die roll from the current square
follow any snake or ladder from the landing square
recursively solve from the resulting square
return the smallest number of rolls
```

Conceptually:

```python
def search(square):
    if square == target:
        return 0

    best = infinity

    for roll in range(1, 7):
        landing = square + roll
        if landing > target:
            continue

        next_square = destination_after_snake_or_ladder(landing)
        best = min(best, 1 + search(next_square))

    return best
```

This captures the rules, but it has serious problems.

First, the same square can be reached many different ways. If we solve from that square repeatedly, we redo the same work.

Second, the board can contain cycles. A snake may send you backward, and later rolls may bring you back to a square you have already visited. A plain recursive search can loop forever unless it is carefully guarded.

Third, even with memoization, the recursive formulation is awkward because we are not asking for "some path". We are asking for the **fewest number of rolls**. When all moves cost one roll, there is a simpler tool that explores positions exactly in increasing roll count.

That tool is BFS.

### 3. The Key Observation

Forget the physical board for a moment and look only at square labels.

From each square `s`, there are at most six possible die outcomes. Each outcome produces exactly one resulting square after applying the forced snake-or-ladder jump.

So every square has directed edges like this:

```text
s --one die roll--> next_square
```

For example, if rolling from square `2` lands on square `5`, and square `5` has a ladder to square `17`, then there is an edge:

```text
2 -> 17
```

That edge still costs exactly one move, because the ladder is part of the same die roll.

The board is therefore an implicit graph:

- Nodes are square numbers from `1` to `n * n`.
- Edges represent one legal die roll plus the mandatory snake/ladder effect.
- Every edge has equal weight `1`.

BFS is designed for exactly this situation:

> In an unweighted graph, BFS reaches every node for the first time using the minimum possible number of edges.

Translated back to this problem:

> The first time BFS reaches square `n * n`, the current BFS level is the minimum number of die rolls.

### 4. Converting a Square Number to Board Coordinates

The only board-specific difficulty is mapping a square label to its `(row, col)` cell in `board`.

The input board is indexed from the top row down:

```text
board[0]       is the top row
board[n - 1]   is the bottom row
```

But square numbering starts from the bottom-left.

It helps to convert the square number to zero-based progress from square `1`:

```python
index = square - 1
row_from_bottom = index // n
col_offset = index % n
```

`row_from_bottom` tells us how many rows above the bottom row the square lies.

The actual matrix row is:

```python
row = n - 1 - row_from_bottom
```

The column depends on the direction of that row in the numbering pattern.

If `row_from_bottom` is even, that numbered row goes left-to-right:

```python
col = col_offset
```

If `row_from_bottom` is odd, that numbered row goes right-to-left:

```python
col = n - 1 - col_offset
```

So the complete helper is:

```python
def square_to_position(square, n):
    index = square - 1
    row_from_bottom = index // n
    col_offset = index % n

    row = n - 1 - row_from_bottom

    if row_from_bottom % 2 == 0:
        col = col_offset
    else:
        col = n - 1 - col_offset

    return row, col
```

This helper is the bridge between the one-dimensional BFS state and the two-dimensional board representation.

### 5. BFS State and Invariant

The BFS state can be just one integer:

```text
current square number
```

We do not need to store the full path, the board position, or the die roll history. Once we know the current square, all future legal moves are determined by the board.

Maintain:

```text
queue   = squares discovered but not fully expanded
visited = squares already discovered
moves   = number of die rolls used to reach the current BFS level
```

The central invariant is:

> At the start of each BFS level, every square currently in the queue is reachable in exactly `moves` die rolls, and no unvisited square has been reached in fewer than `moves` die rolls.

This invariant is what makes the first encounter with the target reliable.

When we expand one square, we try all die rolls `1` through `6`. For each candidate landing square:

1. Ignore it if it is larger than `n * n`.
2. Convert it to `(row, col)`.
3. If `board[row][col] != -1`, replace the landing square with that destination.
4. If the resulting square has not been visited, enqueue it.

Marking a square as visited when it is enqueued is important. It means the first discovered route to that square is already the shortest one, because BFS discovers states level by level.

### 6. Detailed Algorithm

Let:

```text
target = n * n
```

If we reach `target`, return the number of moves used.

Algorithm:

1. Start BFS from square `1` with `0` moves.
2. Store square `1` in `visited`.
3. While the queue is not empty:
   - Process all squares currently in the queue. These are exactly the squares reachable in the current number of moves.
   - For each square, try die results `1` through `6`.
   - Convert each landing square to board coordinates.
   - Apply one snake or ladder if present.
   - If the resulting square is `target`, return `moves + 1`.
   - Otherwise, enqueue it if it has not been visited.
4. If BFS finishes without reaching `target`, return `-1`.

There are two equivalent implementation styles.

One style stores `(square, moves)` pairs in the queue:

```python
queue = deque([(1, 0)])
```

Another style processes BFS level by level and increments `moves` after each layer.

Both are correct. The level-by-level style makes the invariant especially visible.

### 7. Pseudocode

```python
from collections import deque


def snakes_and_ladders(board):
    n = len(board)
    target = n * n

    def position(square):
        index = square - 1
        row_from_bottom = index // n
        col_offset = index % n

        row = n - 1 - row_from_bottom
        if row_from_bottom % 2 == 0:
            col = col_offset
        else:
            col = n - 1 - col_offset

        return row, col

    queue = deque([1])
    visited = {1}
    moves = 0

    while queue:
        for _ in range(len(queue)):
            square = queue.popleft()

            if square == target:
                return moves

            for roll in range(1, 7):
                landing = square + roll
                if landing > target:
                    break

                row, col = position(landing)

                if board[row][col] == -1:
                    next_square = landing
                else:
                    next_square = board[row][col]

                if next_square in visited:
                    continue

                if next_square == target:
                    return moves + 1

                visited.add(next_square)
                queue.append(next_square)

        moves += 1

    return -1
```

The target check can appear either when popping a square or immediately before enqueueing/returning a newly discovered square. The important part is that the returned count matches the number of die rolls used to reach that square.

### 8. Walk Through Example 1

Input:

```python
board = [
    [-1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1],
    [-1, 35, -1, -1, 13, -1],
    [-1, -1, -1, -1, -1, -1],
    [-1, 15, -1, -1, -1, -1],
]
```

Here `n = 6`, so the target is square `36`.

The bottom row is numbered:

```text
1  2  3  4  5  6
```

Square `2` contains a ladder to `15`, because `board[5][1] = 15`.

The row above is numbered right-to-left:

```text
12 11 10 9 8 7
```

Later rows continue alternating.

Start:

```text
moves = 0
queue = [1]
```

From square `1`, one die roll can land on squares `2` through `7`.

- Landing on `2` sends us to `15`.
- Landing on `3` stays on `3`.
- Landing on `4` stays on `4`.
- Landing on `5` stays on `5`.
- Landing on `6` stays on `6`.
- Landing on `7` stays on `7`.

After one move, BFS has discovered:

```text
15, 3, 4, 5, 6, 7
```

Now BFS expands all positions reachable in one move before considering any position reachable in two moves. That is the key shortest-path discipline.

One shortest route that BFS can discover is:

```text
1  --roll 1--> 2  --ladder--> 15
15 --roll 2--> 17 --snake--> 13
13 --roll 1--> 14 --ladder--> 35
35 --roll 1--> 36
```

That is four die rolls. Notice that the ladder and snake movements are not counted as separate rolls; they are forced effects of the square landed on by the die.

BFS does not need us to guess this route in advance. It systematically explores every square reachable in `1` roll, then every square reachable in `2` rolls, then every square reachable in `3` rolls, and so on.

For this board, BFS first reaches square `36` at level `4`, so the answer is:

```text
4
```

The important lesson from the example is not the memorized path. It is that snakes and ladders are simply folded into the neighbor-generation step, while BFS still counts only die rolls.

### 9. Walk Through Example 2

Input:

```python
board = [
    [-1, -1],
    [-1, 3],
]
```

The square labels are:

```text
4 3
1 2
```

Start at square `1`; target is square `4`.

From square `1`, possible landings are only `2`, `3`, and `4` because the board has four squares.

- Landing on `2` uses the ladder to square `3`.
- Landing on `3` stays on `3`.
- Landing on `4` reaches the target directly.

A single die roll of `3` reaches square `4`, so the answer is:

```text
1
```

### 10. Correctness

We prove that the BFS algorithm returns the minimum number of die rolls needed to reach the final square.

#### Lemma 1: Neighbor generation exactly represents one legal move.

From a square `s`, the algorithm considers every die result from `1` to `6` that does not move beyond `n * n`. For each such result, it computes the landing square `s + roll`. If that landing square contains a snake or ladder, the algorithm moves to the destination written on the board; otherwise it stays on the landing square.

This is exactly the rule for one move in the problem. The algorithm applies at most one snake or ladder because it only checks the original landing square once.

Therefore, each generated neighbor is reachable from `s` in one legal die roll, and every legal one-roll outcome from `s` is generated.

#### Lemma 2: At BFS level `moves`, every queued square is reachable in exactly `moves` die rolls.

Initially, the queue contains only square `1`, which is reachable in `0` rolls.

Assume the claim is true for the current level. When the algorithm expands a queued square, every generated neighbor is reachable by taking one additional legal die roll. Therefore, every newly enqueued square is reachable in `moves + 1` rolls.

After the current level is fully processed, the next queue contains only such newly discovered squares. Thus the invariant holds for the next level.

#### Lemma 3: The first time BFS discovers a square, it has found a shortest path to that square.

BFS processes all squares reachable in `0` rolls before any square reachable in `1` roll, all squares reachable in `1` roll before any square reachable in `2` rolls, and so on.

If a square is first discovered at level `k`, then no path with fewer than `k` rolls could exist; otherwise BFS would have discovered it at an earlier level.

Marking squares visited when enqueued preserves this property and prevents duplicate work.

#### Theorem: The algorithm returns the minimum number of die rolls needed to reach square `n * n`, or `-1` if it cannot be reached.

If the algorithm returns a number `m`, it does so when square `n * n` is reached during BFS level `m`. By Lemma 2, that means the target is reachable in `m` die rolls. By Lemma 3, no smaller number of die rolls can reach it. So `m` is optimal.

If the algorithm returns `-1`, the BFS queue has become empty. By Lemma 1, BFS generated every legal move from every reachable square. Therefore, no reachable square remains unexplored. Since the target was never reached, square `n * n` is impossible to reach, so `-1` is correct.

### 11. Complexity

There are `n * n` squares.

Each square is enqueued at most once because of `visited`.

For each square, the algorithm tries at most six die rolls.

Therefore:

```text
Time:  O(n^2)
Space: O(n^2)
```

The constant factor for time is small because each square checks at most six outgoing moves.

### 12. Common Pitfalls

#### Misreading the board direction

The most common bug is converting square numbers to `(row, col)` as if every row were left-to-right.

Rows alternate direction from the bottom:

```text
bottom row:        left-to-right
one row above:     right-to-left
next row above:    left-to-right
```

Base the direction on `row_from_bottom`, not on the matrix row index alone unless you are very careful.

#### Forgetting that input rows are top-to-bottom

Square `1` is not at `board[0][0]`. It is at:

```text
board[n - 1][0]
```

The matrix row must be flipped:

```python
row = n - 1 - row_from_bottom
```

#### Chaining snakes or ladders

If you land on a ladder that sends you to another ladder, do not follow the second ladder in the same move.

The move is:

```text
dice roll -> landing square -> optional one forced jump
```

not:

```text
dice roll -> keep jumping until no jump exists
```

#### Counting ladder movement as an extra move

A snake or ladder does not consume another die roll. It is part of the same move.

So if square `2` has a ladder to `15`, then from square `1` with roll `1`, the resulting state is square `15` after exactly one move.

#### Marking visited too late

Mark a square visited when it is enqueued, not only when it is popped. Otherwise the same square can be enqueued many times from different parents in the same or later BFS levels.

The first enqueue already represents the shortest route to that square.

#### Visiting the landing square instead of the final square

Suppose rolling lands on square `2`, and square `2` has a ladder to `15`. The state after the move is `15`, not `2`.

The visited set should contain the final resulting square after applying the snake or ladder.

#### Using DFS for a minimum-roll question

DFS can find a path, but it does not naturally find the shortest path in an unweighted graph. BFS is the natural fit because it explores by number of moves.

### 13. First-Principles Summary

The board game looks two-dimensional, but the actual state is one-dimensional: the current square number.

A die roll creates up to six possible next square labels. A snake or ladder simply rewrites the landing square into a final square for that same move.

Once viewed this way, the problem becomes:

```text
Find the shortest path from square 1 to square n*n
in an implicit unweighted directed graph.
```

BFS solves that because its queue levels correspond exactly to die-roll counts.

The only special implementation work is the square-to-coordinate conversion caused by the boustrophedon numbering. After that, the algorithm is ordinary shortest-path BFS:

```text
state = square number
neighbors = legal die outcomes after one optional jump
answer = first BFS level that reaches target
```

## Implementation
See `solutions/graph_bfs/p909_snakes_and_ladders.py`.

## Tests
See `tests/graph_bfs/test_p909_snakes_and_ladders.py`.

## Examples

### Example 1
- Input: `{'board': [[-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, 35, -1, -1, 13, -1], [-1, -1, -1, -1, -1, -1], [-1, 15, -1, -1, -1, -1]]}`
- Output: `4`

### Example 2
- Input: `{'board': [[-1, -1], [-1, 3]]}`
- Output: `1`

## Follow-up Practice
- Draw the square labels for a `3 x 3` or `4 x 4` board before writing code.
- Trace BFS levels rather than individual recursive paths.
- Test the square-to-coordinate helper separately on the first row, second row, and final square.
- Check that snakes and ladders are applied once per die roll.
