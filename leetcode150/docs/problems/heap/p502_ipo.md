# 502. IPO

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/ipo/
- Official Group: Heap
- Pattern Group: Heap
- Patterns: heap

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given:

```text
k       = the maximum number of projects you may choose
w       = your starting capital
profits = profits[i] is the capital gained after finishing project i
capital = capital[i] is the minimum capital required to start project i
```

Each project can be chosen at most once.

A project is available only when your current capital is at least its required capital:

```text
current_capital >= capital[i]
```

If you choose that project, your capital increases by its profit:

```text
current_capital += profits[i]
```

You may choose at most `k` projects, and the goal is to maximize your final capital.

So the real problem is:

> Starting with capital `w`, repeatedly choose an affordable unchosen project. After at most `k` choices, make the final capital as large as possible.

The important detail is that choosing a project never spends capital in this problem. The capital requirement is only a threshold for eligibility. Once the project is completed, the profit is added to your current capital.

For example:

```text
k = 2
w = 0
profits = [1, 2, 3]
capital = [0, 1, 1]
```

At capital `0`, only project `0` is affordable because it requires `0` capital. Choosing it gives profit `1`, so capital becomes `1`.

At capital `1`, projects `1` and `2` are now affordable. You should choose project `2`, because it gives profit `3` instead of `2`.

Final capital:

```text
0 + 1 + 3 = 4
```

---

### 2. Start From the Brute Force Idea

The most direct way to think about the problem is a decision tree.

At each step:

1. Look at all projects not chosen yet.
2. Keep only the projects whose required capital is at most your current capital.
3. Try choosing each affordable project.
4. Recurse with one fewer remaining choice.
5. Return the best final capital among all possible paths.

Conceptually:

```python
def search(current_capital, remaining_choices, unused_projects):
    if remaining_choices == 0:
        return current_capital

    best = current_capital

    for project in unused_projects:
        if project.required_capital <= current_capital:
            after = search(
                current_capital + project.profit,
                remaining_choices - 1,
                unused_projects without project,
            )
            best = max(best, after)

    return best
```

This is correct because it tries every valid sequence of project choices.

But it is far too slow.

If many projects are affordable, the first choice has many possibilities, the second choice has almost as many, and so on. In the worst case, the number of possible sequences grows like:

```text
n * (n - 1) * (n - 2) * ...
```

for up to `k` levels.

Even if we optimize the search, repeatedly scanning all projects at every step is wasteful.

The brute-force view is still useful because it exposes the two questions we must answer at every choice:

```text
Which projects are affordable now?
Among those affordable projects, which one should we choose?
```

---

### 3. The Key Observation: Greedy Choice Is Safe

At any moment, suppose several projects are affordable.

Because all currently affordable projects can be started immediately, choosing one with smaller profit instead of a larger profit cannot make you better off.

Why?

If two projects `A` and `B` are both affordable now, and:

```text
profit[A] >= profit[B]
```

then choosing `A` gives at least as much capital as choosing `B`.

After choosing `A`, your capital is:

```text
current_capital + profit[A]
```

After choosing `B`, your capital is:

```text
current_capital + profit[B]
```

Since `profit[A] >= profit[B]`, the capital after `A` is at least the capital after `B`.

Having more capital is never harmful here:

```text
more capital can only make more projects affordable, never fewer
```

So among the projects that are already affordable, the best immediate choice is the one with the largest profit.

This is the heart of the problem.

The difficulty is not deciding what to do among affordable projects. The difficulty is efficiently maintaining the set of projects that have become affordable as capital grows.

---

### 4. Why Sorting by Required Capital Helps

Each project has two values:

```text
required capital
profit
```

Affordability depends only on required capital:

```text
capital[i] <= current_capital
```

If we sort projects by required capital, then as current capital increases, projects become affordable from left to right in that sorted order.

Example:

```text
projects sorted by required capital:
(required=0, profit=1)
(required=1, profit=2)
(required=1, profit=3)
(required=5, profit=10)
```

Starting with capital `0`, only the first project is affordable.

After capital grows to `1`, the next two projects become affordable.

The project requiring `5` is still not affordable.

This suggests maintaining a pointer:

```text
index = first project in sorted order not yet moved into the affordable set
```

Whenever capital increases, advance this pointer while projects are affordable.

---

### 5. The Heap/Affordability Invariant

We need two structures:

1. A list of projects sorted by required capital.
2. A max-heap of profits for projects that are currently affordable and not yet chosen.

The invariant is:

```text
After the affordability-loading step:

- every unchosen project with required capital <= current_capital is in the max-heap
- no project with required capital > current_capital is in the max-heap
- every project before the sorted pointer has already been considered for affordability
- every project at or after the sorted pointer is not yet affordable, or has not been reached yet
```

This invariant separates the problem into two clean jobs:

```text
sorted-by-capital list: discovers when projects become affordable
max-heap by profit: chooses the best affordable project
```

The heap does not need to know capital requirements, because a project is pushed into the heap only after it has passed the affordability test.

Once a project is inside the heap, it is known to be currently affordable. Since capital never decreases, it will remain affordable forever.

---

### 6. Detailed Algorithm

First combine each project's requirement and profit:

```text
projects = [(capital[i], profits[i]) for each i]
```

Sort by required capital:

```text
projects.sort()
```

Maintain:

```text
current_capital = w
index = 0
max_profit_heap = empty heap
```

Python's `heapq` is a min-heap, so to simulate a max-heap, store negative profits:

```text
push -profit
pop gives smallest negative number, which corresponds to largest profit
```

Then repeat at most `k` times:

1. Move every newly affordable project into the heap:

```text
while index < n and projects[index].required_capital <= current_capital:
    push projects[index].profit into max heap
    index += 1
```

2. If the heap is empty, there is no affordable project.

```text
break
```

No future move can help, because capital only increases when we complete a project, and there is no project we can complete now.

3. Pop the largest available profit and add it to capital:

```text
current_capital += largest_affordable_profit
```

4. Continue to the next project choice.

After at most `k` choices, return `current_capital`.

---

### 7. Example Walkthrough 1

Input:

```text
k = 2
w = 0
profits = [1, 2, 3]
capital = [0, 1, 1]
```

Pair and sort projects by required capital:

```text
(required=0, profit=1)
(required=1, profit=2)
(required=1, profit=3)
```

Start:

```text
current_capital = 0
index = 0
heap = []
choices_left = 2
```

#### Choice 1

Load affordable projects:

```text
(required=0, profit=1) is affordable because 0 <= 0
push profit 1
index moves to 1
```

The next project requires `1`, which is not affordable yet.

Heap contains:

```text
[1]
```

Choose the largest affordable profit:

```text
pop 1
current_capital = 0 + 1 = 1
```

#### Choice 2

Now capital is `1`, so load newly affordable projects:

```text
(required=1, profit=2) is affordable
push profit 2

(required=1, profit=3) is affordable
push profit 3
```

Heap contains:

```text
[3, 2]
```

Choose the largest affordable profit:

```text
pop 3
current_capital = 1 + 3 = 4
```

We have made `2` choices, which is the limit `k`.

Return:

```text
4
```

---

### 8. Example Walkthrough 2

Input:

```text
k = 3
w = 0
profits = [1, 2, 3]
capital = [0, 1, 2]
```

Sorted projects:

```text
(required=0, profit=1)
(required=1, profit=2)
(required=2, profit=3)
```

Start with capital `0`.

#### Choice 1

Only the first project is affordable:

```text
choose profit 1
current_capital = 1
```

#### Choice 2

Now the project requiring `1` is affordable:

```text
choose profit 2
current_capital = 3
```

#### Choice 3

Now the project requiring `2` is affordable:

```text
choose profit 3
current_capital = 6
```

Return:

```text
6
```

---

### 9. Pseudocode

```text
function findMaximizedCapital(k, w, profits, capital):
    projects = pairs of (capital[i], profits[i])
    sort projects by capital requirement

    current_capital = w
    index = 0
    max_heap = empty

    repeat k times:
        while index < number of projects
              and projects[index].required_capital <= current_capital:
            push projects[index].profit into max_heap
            index += 1

        if max_heap is empty:
            break

        current_capital += pop largest profit from max_heap

    return current_capital
```

---

### 10. Python Code

```python
from heapq import heappop, heappush
from typing import List


class Solution:
    def findMaximizedCapital(
        self,
        k: int,
        w: int,
        profits: List[int],
        capital: List[int],
    ) -> int:
        projects = sorted(zip(capital, profits))
        max_profit_heap: list[int] = []
        project_index = 0
        current_capital = w

        for _ in range(k):
            while (
                project_index < len(projects)
                and projects[project_index][0] <= current_capital
            ):
                _, profit = projects[project_index]
                heappush(max_profit_heap, -profit)
                project_index += 1

            if not max_profit_heap:
                break

            current_capital += -heappop(max_profit_heap)

        return current_capital
```

The negative sign is only a Python implementation detail. The algorithmic idea is still:

```text
always pop the largest profit among currently affordable projects
```

---

### 11. Why This Algorithm Is Correct

We prove that the algorithm returns the maximum possible final capital after at most `k` project choices.

#### The affordability invariant is correct

Projects are sorted by required capital.

At the beginning of each iteration, the algorithm advances `project_index` while the next project's required capital is at most `current_capital`.

Therefore, every sorted project before `project_index` has been checked and, if it has not already been chosen, is represented in the heap.

Every sorted project at or after `project_index` has required capital greater than `current_capital`, unless it simply has not been reached yet. Because the list is sorted, once the algorithm reaches a project that is too expensive, all later projects are at least as expensive.

So after the loading step, the heap contains exactly the unchosen projects that are currently affordable.

#### The greedy choice is safe

Consider any iteration where the heap is not empty.

By the invariant, the heap contains exactly the projects that can be chosen now.

Let `G` be the project with the largest profit among those affordable projects. The algorithm chooses `G`.

Take any optimal strategy from this same state. If that strategy also chooses `G` next, it agrees with the algorithm for this step.

Otherwise, suppose it chooses some other affordable project `X` first.

Since `G` has maximum profit among affordable projects:

```text
profit[G] >= profit[X]
```

If we swap the first choice of that optimal strategy from `X` to `G`, the capital after the first choice is at least as large as before.

Having at least as much capital cannot make any future project unaffordable, because affordability is based on a `<= current_capital` threshold. It can only preserve or expand the set of future choices.

So there exists an optimal strategy that chooses `G` first.

Thus choosing the largest currently affordable profit is always safe.

#### Repeating the safe choice gives an optimal result

After the algorithm chooses the largest affordable project, the problem has the same form again:

```text
new current capital
one fewer allowed choice
some projects already chosen
some projects not yet chosen
```

The same invariant and greedy-choice argument apply to the next iteration.

By induction over the number of remaining choices, the algorithm can follow an optimal sequence of choices.

#### Empty heap termination is correct

If the heap is empty after loading affordable projects, then no unchosen project is currently affordable.

The only way to increase capital is to complete an affordable project, but none exists.

Therefore no future progress is possible, and stopping early is correct.

Since the algorithm makes only safe greedy choices and stops only when no valid choice remains or `k` choices have been used, it returns the maximum possible final capital.

---

### 12. Complexity

Let `n` be the number of projects.

Sorting projects costs:

```text
O(n log n)
```

Each project is pushed into the heap at most once.

Each selected project is popped at most once, and there are at most `k` selected projects.

Each heap operation costs `O(log n)`, so heap work is:

```text
O(n log n + k log n)
```

Because `k` cannot usefully exceed `n` chosen projects, this is usually summarized as:

```text
Time: O(n log n)
```

Space usage comes from the sorted project list and heap:

```text
Space: O(n)
```

---

### 13. Common Pitfalls

#### Pitfall 1: Treating capital as money spent

The condition:

```text
capital[i] <= current_capital
```

only determines whether the project can be started.

You do not subtract `capital[i]` when choosing the project.

Wrong idea:

```text
current_capital = current_capital - capital[i] + profits[i]
```

Correct idea:

```text
current_capital = current_capital + profits[i]
```

#### Pitfall 2: Sorting by profit only

Sorting all projects by profit and taking the largest profits can choose projects you cannot afford yet.

Affordability must be handled first; profit ranking only applies inside the affordable set.

That is why the algorithm uses both:

```text
sort by required capital
heap by profit
```

#### Pitfall 3: Scanning all projects every round

A solution that loops over every unchosen project for each of `k` rounds may be correct but too slow.

Sorting by capital lets each project enter the affordable heap once.

#### Pitfall 4: Forgetting to load newly affordable projects after capital increases

After choosing a profitable project, current capital may unlock more projects.

The affordability-loading `while` loop must run at the start of every iteration, not just once at the beginning.

#### Pitfall 5: Using a min-heap of profits by accident

Python's `heapq` pops the smallest value.

If you push profits directly, you will choose the least profitable affordable project.

Use negative profits:

```python
heappush(heap, -profit)
profit = -heappop(heap)
```

#### Pitfall 6: Continuing when the heap is empty

If no project is affordable, the algorithm must stop early.

There is no operation that can increase capital without first choosing an affordable project.

---

### 14. First-Principles Summary

This problem follows from these basic facts:

```text
1. A project becomes eligible when current capital reaches its required capital.
2. Completing a project increases capital by its profit.
3. Capital never decreases.
4. Therefore, once a project becomes affordable, it stays affordable.
5. More capital can only unlock more choices; it never removes choices.
6. Among currently affordable projects, choosing the largest profit is safe.
7. Sorting by required capital tells us when projects become affordable.
8. A max-heap tells us the best project among the currently affordable ones.
```

In one sentence:

> Sort projects by required capital, keep all currently affordable profits in a max-heap, and up to `k` times choose the largest affordable profit to grow capital as quickly as possible.

## Implementation

See `solutions/heap/p502_ipo.py`.

## Tests

See `tests/heap/test_p502_ipo.py`.

## Examples

### Example 1
- Input: `{'k': 2, 'w': 0, 'profits': [1, 2, 3], 'capital': [0, 1, 1]}`
- Output: `4`

### Example 2
- Input: `{'k': 3, 'w': 0, 'profits': [1, 2, 3], 'capital': [0, 1, 2]}`
- Output: `6`

## Follow-up Practice
- Trace the heap contents after every capital increase.
- Explain why the largest affordable profit is a safe greedy choice.
- Compare this with the brute-force decision tree and identify which repeated work the heap removes.
