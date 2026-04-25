# 210. Course Schedule II

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/course-schedule-ii/
- Official Group: Graph General
- Pattern Group: Graph General
- Patterns: graph-general

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given:

```text
numCourses = number of courses, labeled 0 through numCourses - 1
prerequisites = pairs [course, prerequisite]
```

A pair:

```text
[1, 0]
```

means:

```text
To take course 1, you must first take course 0.
```

So course `0` must appear before course `1` in the returned order.

The task is to return one valid order in which all courses can be completed. If no such order exists, return an empty array.

For example:

```text
numCourses = 2
prerequisites = [[1, 0]]
```

Course `0` must come before course `1`, so a valid answer is:

```text
[0, 1]
```

For:

```text
numCourses = 4
prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
```

The constraints are:

```text
0 before 1
0 before 2
1 before 3
2 before 3
```

So both of these are valid:

```text
[0, 1, 2, 3]
[0, 2, 1, 3]
```

The examples use `[0, 2, 1, 3]`, but the problem accepts any order that satisfies every prerequisite.

The real problem is therefore:

> Arrange all course labels so that every prerequisite appears before the course that depends on it.

This is exactly a topological ordering problem.

---

### 2. Turn Courses Into a Graph

Each course is a node.

Each prerequisite pair creates a directed edge.

For a pair:

```text
[course, prerequisite]
```

we draw an edge:

```text
prerequisite -> course
```

Why this direction?

Because the prerequisite unlocks the course. If course `0` must be taken before course `1`, then the natural dependency flow is:

```text
0 -> 1
```

This edge means:

```text
After taking 0, course 1 is one step closer to being available.
```

For Example 2:

```text
numCourses = 4
prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
```

The graph is:

```text
0 -> 1
0 -> 2
1 -> 3
2 -> 3
```

Course `0` has no prerequisites. Courses `1` and `2` need `0`. Course `3` needs both `1` and `2`.

---

### 3. Start From the Brute Force Baseline

The most direct way to solve the problem is to try to build every possible course order.

Conceptually:

1. Generate a permutation of all courses.
2. Check every prerequisite pair.
3. If each prerequisite appears before its course, return that permutation.
4. If no permutation works, return `[]`.

Pseudocode:

```python
for order in all_permutations(range(numCourses)):
    position = {course: index for index, course in enumerate(order)}

    valid = True
    for course, prerequisite in prerequisites:
        if position[prerequisite] > position[course]:
            valid = False
            break

    if valid:
        return order

return []
```

This is correct, but it is far too slow.

There are:

```text
numCourses!
```

possible orders. Even for a modest number of courses, that is enormous.

The important observation is that we do not need to guess a complete order. At any moment, some courses are already safe to take.

---

### 4. Key Observation: A Course With No Remaining Prerequisites Is Safe

Suppose we have not taken all courses yet.

A course is currently available if every prerequisite for it has already been taken.

In graph terms, count how many prerequisites each course still has. This count is called the course's indegree:

```text
indegree[course] = number of incoming prerequisite edges
```

For the edge:

```text
prerequisite -> course
```

`course` has one more incoming edge because it depends on `prerequisite`.

A course with:

```text
indegree[course] == 0
```

has no unsatisfied prerequisites, so it can be safely placed next in the answer.

After we take that course, it may unlock other courses. For every outgoing edge:

```text
taken_course -> next_course
```

we reduce:

```text
indegree[next_course] -= 1
```

because one of `next_course`'s prerequisites has now been satisfied.

If `indegree[next_course]` becomes `0`, that course is now safe too.

This gives a constructive algorithm: repeatedly take any available course.

---

### 5. The Topological Invariant

The algorithm maintains a precise invariant:

```text
Every course in the answer has already been placed after all of its prerequisites.
```

And for every unplaced course:

```text
indegree[course] = number of prerequisites for that course that have not yet been placed in the answer.
```

This invariant is the whole reason the algorithm works.

When we choose a course with indegree `0`, the invariant says it has zero unplaced prerequisites. Therefore appending it to the answer cannot violate any dependency.

When we append it, we remove its outgoing edges by decrementing the indegrees of courses it points to. That keeps the invariant true for the remaining unplaced courses.

So every local decision is safe:

```text
Pick an unplaced course with no remaining prerequisites.
```

The only way this process can get stuck is if every remaining course still has at least one remaining prerequisite. That means the remaining courses depend on each other in a cycle.

For example:

```text
0 -> 1
1 -> 0
```

Neither course can be taken first. Each requires the other. No valid order exists.

---

### 6. Detailed Algorithm: Kahn's Topological Sort

This problem is commonly solved with Kahn's algorithm, a breadth-first topological sort.

Build two structures:

```text
graph[course] = list of courses unlocked by course
indegree[course] = number of prerequisites course still needs
```

For each prerequisite pair:

```text
[course, prerequisite]
```

add:

```text
graph[prerequisite].append(course)
indegree[course] += 1
```

Then initialize a queue with every course whose indegree is `0`:

```text
queue = all courses with no prerequisites
```

Then repeat:

1. Remove one available course from the queue.
2. Append it to the answer.
3. For each course it unlocks, decrement that course's indegree.
4. If a neighbor's indegree becomes `0`, add it to the queue.

At the end:

- If the answer contains all `numCourses` courses, return it.
- Otherwise, return `[]` because a cycle prevented some courses from ever becoming available.

The queue can be a normal FIFO queue. The problem does not require the lexicographically smallest order, so any order among currently available courses is fine.

---

### 7. Pseudocode

```python
from collections import deque


def findOrder(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    indegree = [0] * numCourses

    for course, prerequisite in prerequisites:
        graph[prerequisite].append(course)
        indegree[course] += 1

    queue = deque()
    for course in range(numCourses):
        if indegree[course] == 0:
            queue.append(course)

    order = []

    while queue:
        course = queue.popleft()
        order.append(course)

        for next_course in graph[course]:
            indegree[next_course] -= 1
            if indegree[next_course] == 0:
                queue.append(next_course)

    if len(order) == numCourses:
        return order

    return []
```

This code directly encodes the invariant:

```text
queue = courses whose prerequisites have all been satisfied
order = courses already safely scheduled
indegree = remaining unsatisfied prerequisite count
```

---

### 8. Detailed Example Walkthrough

Use Example 2:

```text
numCourses = 4
prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
```

Build the graph:

```text
0 -> [1, 2]
1 -> [3]
2 -> [3]
3 -> []
```

Build indegrees:

```text
course 0: 0 prerequisites
course 1: 1 prerequisite  (0)
course 2: 1 prerequisite  (0)
course 3: 2 prerequisites (1 and 2)
```

So:

```text
indegree = [0, 1, 1, 2]
```

Initialize the queue with courses whose indegree is `0`:

```text
queue = [0]
order = []
```

#### Step 1: Take course 0

Pop `0`:

```text
order = [0]
```

Course `0` unlocks courses `1` and `2`.

Decrement their indegrees:

```text
indegree[1] = 0
indegree[2] = 0
```

Both are now available:

```text
queue = [1, 2]
indegree = [0, 0, 0, 2]
```

#### Step 2: Take course 1

Pop `1`:

```text
order = [0, 1]
```

Course `1` unlocks course `3`.

Decrement `indegree[3]`:

```text
indegree[3] = 1
```

Course `3` is not available yet because it still needs course `2`.

```text
queue = [2]
indegree = [0, 0, 0, 1]
```

#### Step 3: Take course 2

Pop `2`:

```text
order = [0, 1, 2]
```

Course `2` also unlocks course `3`.

Decrement `indegree[3]`:

```text
indegree[3] = 0
```

Now all prerequisites for `3` have been satisfied, so add it to the queue:

```text
queue = [3]
indegree = [0, 0, 0, 0]
```

#### Step 4: Take course 3

Pop `3`:

```text
order = [0, 1, 2, 3]
```

Course `3` unlocks nothing.

The queue is empty and the order has 4 courses, so this is a valid answer.

Depending on the queue order, the algorithm may also produce:

```text
[0, 2, 1, 3]
```

That is also valid because course `0` still appears before `1` and `2`, and both `1` and `2` appear before `3`.

---

### 9. What Happens When There Is a Cycle?

Consider:

```text
numCourses = 2
prerequisites = [[0, 1], [1, 0]]
```

The edges are:

```text
1 -> 0
0 -> 1
```

Indegrees are:

```text
indegree[0] = 1
indegree[1] = 1
```

No course has indegree `0`, so the initial queue is empty.

That means there is no valid first course. Course `0` waits for course `1`, and course `1` waits for course `0`.

The algorithm returns `[]` because the final order length is less than `numCourses`.

This same logic works for larger cycles too. If a group of remaining courses depends circularly on itself, none of them can ever reach indegree `0`.

---

### 10. Correctness

We prove that the algorithm returns a valid course order if one exists, and returns `[]` if no valid order exists.

#### Lemma 1: Every course appended to `order` has all prerequisites already appended.

A course is appended only after it is removed from the queue. A course enters the queue only when its indegree is `0`. By the invariant, indegree counts the number of prerequisites not yet appended to `order`. Therefore, when a course is appended, it has no unappended prerequisites. All of its prerequisites already appear earlier in `order`.

#### Lemma 2: After appending a course and decrementing its outgoing neighbors, the indegree invariant remains true.

When a course is appended, it changes from unplaced to placed. The only remaining prerequisite counts affected are for courses that directly depend on it. Those are exactly the outgoing neighbors in `graph[course]`. Decrementing each neighbor's indegree by one removes exactly this newly satisfied prerequisite. No other course's remaining prerequisite count changes. Therefore the invariant remains true.

#### Lemma 3: If the algorithm appends all courses, the returned order is valid.

By Lemma 1, every appended course appears after all of its prerequisites. If all courses are appended, this property holds for every course in the returned list. Therefore every prerequisite pair is satisfied, so the returned order is valid.

#### Lemma 4: If the algorithm cannot append all courses, no valid order exists.

If the algorithm stops early, the queue is empty while some courses remain unappended. By the invariant, every remaining course has at least one unappended prerequisite. Starting from any remaining course and repeatedly following one of its unappended prerequisites must eventually revisit a course, because there are only finitely many remaining courses. That creates a directed cycle among remaining courses. In a cycle, each course requires another course in the same cycle to come first, so no course in the cycle can be scheduled first. Therefore no valid order exists.

#### Theorem

The algorithm returns a valid ordering of all courses exactly when such an ordering exists; otherwise it returns `[]`.

This follows directly from Lemma 3 and Lemma 4.

---

### 11. Complexity

Let:

```text
V = numCourses
E = len(prerequisites)
```

Building the graph and indegree array processes each prerequisite once:

```text
O(E)
```

Initializing the queue scans all courses once:

```text
O(V)
```

During the main loop, each course is enqueued and dequeued at most once, and each edge is processed once when its prerequisite course is taken:

```text
O(V + E)
```

So the total time complexity is:

```text
O(V + E)
```

The graph stores every course and every edge, the indegree array stores one count per course, the queue can hold up to all courses, and the answer stores all courses:

```text
O(V + E)
```

space.

---

### 12. Common Pitfalls

#### Reversing the edge direction

For a pair:

```text
[course, prerequisite]
```

the edge should be:

```text
prerequisite -> course
```

If you accidentally build:

```text
course -> prerequisite
```

you will produce the reverse dependency relationship.

#### Treating the output as unique

There may be many valid answers. For Example 2, both `[0, 1, 2, 3]` and `[0, 2, 1, 3]` satisfy the prerequisites.

Tests for this problem should usually validate the ordering constraints rather than require one exact order, unless the implementation's queue behavior is intentionally fixed.

#### Forgetting isolated courses

A course may not appear in `prerequisites` at all.

For example:

```text
numCourses = 3
prerequisites = [[1, 0]]
```

Course `2` has no dependencies and unlocks nothing, but it still must appear in the returned order. This is why the graph and indegree array are initialized for all courses from `0` to `numCourses - 1`.

#### Returning a partial order after a cycle

If a cycle exists, the algorithm may still append some courses that are outside the cycle. That partial list is not a valid answer because the problem asks for all courses.

Always check:

```text
len(order) == numCourses
```

before returning `order`.

#### Confusing indegree with outdegree

Indegree counts prerequisites needed by a course.

Outdegree counts courses unlocked by a course.

The queue is based on indegree, not outdegree. A course is ready when it needs nothing else, not when it unlocks nothing else.

---

### 13. First-Principles Summary

Course scheduling is not about trying schedules one by one. It is about repeatedly finding courses whose prerequisites have already been satisfied.

The graph model is:

```text
prerequisite -> course
```

The central state is:

```text
indegree[course] = how many prerequisites are still missing
```

A course with indegree `0` is safe to take next. Taking it removes one missing prerequisite from each course it unlocks. If this process schedules every course, the produced list is a valid topological order. If it gets stuck before scheduling every course, the unscheduled courses contain a cycle, so no valid order exists.

That is the first-principles reason topological sorting solves Course Schedule II.

## Implementation
See `solutions/graph_general/p210_course_schedule_ii.py`.

## Tests
See `tests/graph_general/test_p210_course_schedule_ii.py`.

## Examples

### Example 1
- Input: `{'numCourses': 2, 'prerequisites': [[1, 0]]}`
- Output: `[0, 1]`

### Example 2
- Input: `{'numCourses': 4, 'prerequisites': [[1, 0], [2, 0], [3, 1], [3, 2]]}`
- Output: `[0, 2, 1, 3]`

### Example 3
- Input: `{'numCourses': 1, 'prerequisites': []}`
- Output: `[0]`

## Follow-up Practice
- Given a prerequisite pair, say the edge direction out loud before coding it.
- Trace the indegree array after each course is removed from the queue.
- Create one cyclic example and explain why no course in the cycle can be first.
- Compare BFS topological sort with DFS cycle detection and postorder reversal.
