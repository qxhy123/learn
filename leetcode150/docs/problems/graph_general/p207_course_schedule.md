# 207. Course Schedule

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/course-schedule/
- Official Group: Graph General
- Pattern Group: Graph General
- Patterns: graph-general

## First-Principles Explanation

### What The Problem Asks
You are given `numCourses` courses labeled from `0` to `numCourses - 1` and a list called `prerequisites`.

Each pair `[a, b]` means:

```text
To take course a, you must first take course b.
```

The question is not asking for the actual semester plan. It asks only whether some valid order exists that lets you finish every course.

So the real question is:

```text
Can all prerequisite constraints be arranged in a line so that every prerequisite appears before the course that depends on it?
```

If yes, return `True`. If there is an impossible circular dependency, return `False`.

For example:

```text
numCourses = 2
prerequisites = [[1, 0]]
```

Course `0` must be taken before course `1`, so one valid order is:

```text
0 -> 1
```

Return `True`.

But:

```text
numCourses = 2
prerequisites = [[1, 0], [0, 1]]
```

Course `1` needs course `0`, and course `0` needs course `1`. There is no first course to take. Return `False`.

### Turning Courses Into A Graph
The labels `0..numCourses-1` are graph nodes.

A prerequisite pair `[course, prerequisite]` creates a directed edge:

```text
prerequisite -> course
```

That direction means:

```text
After finishing prerequisite, course becomes one step closer to available.
```

So `[[1, 0], [2, 0], [3, 1], [3, 2]]` becomes:

```text
0 -> 1 -> 3
|         ^
v         |
2 --------+
```

This graph asks for a topological ordering: an ordering of directed graph nodes where every edge points from earlier to later.

The important fact is:

```text
A directed graph has a topological order if and only if it has no directed cycle.
```

Therefore Course Schedule is a cycle-detection problem in a directed graph, or equivalently a topological-sort feasibility problem.

### Brute-Force Baseline
A direct but inefficient way to think about the problem is to try to build every possible course order.

1. Choose an untaken course whose prerequisites are already satisfied.
2. Append it to the schedule.
3. Recurse on the remaining courses.
4. If one full schedule is found, return `True`.
5. If all choices fail, return `False`.

This matches the problem statement, but it is far too expensive. In the worst case there can be `numCourses!` possible orders. Even checking each partial order repeatedly costs extra work.

Another naive variant is to repeatedly scan all courses and all prerequisite pairs looking for a course that can now be taken. That can work if carefully implemented, but repeated full scans can degrade toward `O(V * E)` or worse, where `V = numCourses` and `E = len(prerequisites)`.

The brute-force view is still useful because it reveals the central bottleneck:

```text
At every moment, we only need to know which courses currently have zero unmet prerequisites.
```

We do not need to try every order. We only need to prove that the process can keep making progress until all courses are taken.

### Key Observation
A course is immediately takeable exactly when it has no remaining prerequisites.

In graph language, that means the node has indegree `0`.

- `indegree[x]` is the number of incoming edges into `x`.
- For this problem, `indegree[x]` is the number of prerequisites course `x` still needs.

If a course has indegree `0`, we can take it now. After taking it, every course that depended on it has one fewer unmet prerequisite.

That local update is enough:

```text
Take a zero-indegree course -> remove its outgoing edges -> maybe create new zero-indegree courses.
```

This is Kahn's algorithm for topological sorting. We do not need to output the order for problem 207; we only need to count whether all courses can be removed.

### The Topological Invariant
The algorithm maintains this invariant:

```text
For every not-yet-taken course, indegree[course] equals the number of prerequisites for that course that have not yet been taken.
```

Initially this is true because every prerequisite edge is counted once.

When we take a course `u`, every outgoing edge `u -> v` represents one prerequisite of `v` that has just been satisfied. So we decrement `indegree[v]` by `1`.

If `indegree[v]` becomes `0`, then all prerequisites of `v` have been taken, so `v` is now safe to add to the queue.

The queue therefore contains exactly the courses that can be taken next.

The failure case is also described by the same invariant. If the queue becomes empty before all courses have been taken, then every remaining course still has at least one unmet prerequisite. Since all remaining prerequisites point among remaining courses, those courses are locked in a directed cycle. There is no valid starting point left.

### Detailed Algorithm
Build the graph in the direction of dependency release:

```text
prerequisite -> course
```

Then process all currently available courses.

1. Create an adjacency list `graph` with one list per course.
   - `graph[u]` stores all courses that become closer to available after taking `u`.
2. Create an `indegree` array of length `numCourses`, initially all zero.
3. For each pair `[course, prerequisite]`:
   - Add `course` to `graph[prerequisite]`.
   - Increment `indegree[course]`.
4. Put every course with `indegree == 0` into a queue.
5. Initialize `taken = 0`.
6. While the queue is not empty:
   - Pop one available course `u`.
   - Count it as taken.
   - For every `next_course` in `graph[u]`:
     - Decrement `indegree[next_course]`.
     - If it becomes zero, push `next_course` into the queue.
7. Return whether `taken == numCourses`.

The order in which zero-indegree courses are popped does not matter for this problem. Different valid orders may exist, but all that matters is whether the process can consume every node.

### Detailed Example Walkthrough
Consider:

```text
numCourses = 4
prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
```

Each pair means:

```text
0 before 1
0 before 2
1 before 3
2 before 3
```

Build the graph:

```text
graph[0] = [1, 2]
graph[1] = [3]
graph[2] = [3]
graph[3] = []
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
queue = [0]
taken = 0
```

Step 1: take course `0`.

```text
taken = 1
```

Course `0` unlocks progress toward courses `1` and `2`:

```text
indegree[1] becomes 0
indegree[2] becomes 0
queue = [1, 2]
```

Step 2: take course `1`.

```text
taken = 2
```

Course `1` satisfies one prerequisite of course `3`:

```text
indegree[3] becomes 1
queue = [2]
```

Course `3` is not ready yet because it still needs course `2`.

Step 3: take course `2`.

```text
taken = 3
```

Course `2` satisfies the final prerequisite of course `3`:

```text
indegree[3] becomes 0
queue = [3]
```

Step 4: take course `3`.

```text
taken = 4
queue = []
```

Now `taken == numCourses`, so every course can be finished. Return `True`.

Now compare a cycle:

```text
numCourses = 2
prerequisites = [[1, 0], [0, 1]]
```

Graph:

```text
0 -> 1
1 -> 0
```

Indegrees:

```text
indegree = [1, 1]
queue = []
```

No course has zero prerequisites. There is no legal first move. The loop never takes a course, so `taken = 0`, which is not `numCourses`. Return `False`.

### Code
One Python implementation of the topological approach is:

```python
from collections import deque
from typing import List


class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses

        for course, prerequisite in prerequisites:
            graph[prerequisite].append(course)
            indegree[course] += 1

        queue = deque(
            course for course in range(numCourses)
            if indegree[course] == 0
        )

        taken = 0

        while queue:
            course = queue.popleft()
            taken += 1

            for next_course in graph[course]:
                indegree[next_course] -= 1
                if indegree[next_course] == 0:
                    queue.append(next_course)

        return taken == numCourses
```

Equivalent pseudocode:

```text
canFinish(numCourses, prerequisites):
    graph = array of empty lists, one per course
    indegree = array of zeros, one per course

    for [course, prerequisite] in prerequisites:
        graph[prerequisite].append(course)
        indegree[course] += 1

    queue = all courses whose indegree is 0
    taken = 0

    while queue is not empty:
        course = queue.pop_front()
        taken += 1

        for next_course in graph[course]:
            indegree[next_course] -= 1
            if indegree[next_course] == 0:
                queue.push_back(next_course)

    return taken == numCourses
```

### Correctness
We prove that the algorithm returns `True` exactly when all courses can be finished.

#### Lemma 1: Every course added to the queue is safe to take.
A course is added to the queue only when its indegree is `0`. By the invariant, indegree counts the number of prerequisites that have not yet been taken. Therefore, when a course is added to the queue, it has no unmet prerequisites and is safe to take.

#### Lemma 2: Taking a queued course preserves the indegree invariant.
When the algorithm takes course `u`, the only courses affected are those with an edge `u -> v`, meaning `u` is a prerequisite of `v`. Since `u` has now been taken, `v` has exactly one fewer unmet prerequisite, so decrementing `indegree[v]` by `1` is correct. All other courses have the same unmet prerequisites as before. Thus the invariant remains true.

#### Lemma 3: If the algorithm takes all courses, then a valid schedule exists.
By Lemma 1, every course taken by the algorithm is taken only after all its prerequisites have already been taken. Therefore the order in which the algorithm took the courses is a valid course schedule. If all courses are taken, all courses can be finished.

#### Lemma 4: If the algorithm stops before taking all courses, no valid schedule exists.
If the algorithm stops early, the queue is empty while at least one course remains untaken. By the invariant, every remaining course has at least one unmet prerequisite. Those unmet prerequisites cannot come from already taken courses, because their outgoing edges have already been processed and removed from the indegree counts. Therefore each remaining course depends, directly or indirectly, on other remaining courses. In a finite directed graph where every remaining node has an incoming edge from within the remaining set, there must be a directed cycle. A directed cycle cannot be scheduled because every course in the cycle waits for another course in the same cycle. So no valid schedule exists.

#### Theorem
The algorithm returns `True` if and only if all courses can be finished.

- If it returns `True`, then `taken == numCourses`, so by Lemma 3 there is a valid schedule.
- If it returns `False`, then it stopped before taking all courses, so by Lemma 4 no valid schedule exists.

### Complexity
Let:

```text
V = numCourses
E = len(prerequisites)
```

Building the graph and indegree array processes every prerequisite pair once, so it costs `O(E)` time.

Initializing the queue scans all courses once, so it costs `O(V)` time.

During the BFS/topological process, each course is pushed and popped at most once, and each directed edge is processed once when its prerequisite course is taken. That costs `O(V + E)` time.

Total time complexity:

```text
O(V + E)
```

The adjacency list stores every course and every edge, and the indegree array and queue store at most `V` courses.

Total space complexity:

```text
O(V + E)
```

### Common Pitfalls
- Reversing the edge direction without adjusting the algorithm. For Kahn's algorithm here, `[course, prerequisite]` should usually become `prerequisite -> course` so taking the prerequisite reduces the course's indegree.
- Treating `prerequisites` as undirected edges. Dependency is directional; `0 -> 1` is very different from `1 -> 0`.
- Returning `True` just because the queue was non-empty at the start. The queue only proves some courses can be started; you must verify all `numCourses` courses are eventually taken.
- Forgetting isolated courses. A course with no prerequisites and no dependent courses is still finishable and should count toward `taken`.
- Decrementing indegree for the wrong node. When processing `u`, decrement the courses that depend on `u`, not the prerequisites of `u`.
- Using a single `visited` set as if this were an undirected graph traversal. Directed cycle detection requires either topological indegrees or a DFS state model with `unvisited / visiting / visited`.
- Assuming duplicate prerequisite pairs cannot appear unless the platform guarantees it. The indegree and adjacency-list method remains consistent if each listed pair is treated as one edge.

### DFS Alternative
The same first-principles idea can also be implemented with DFS cycle detection.

Use states:

```text
0 = unvisited
1 = currently visiting this recursion path
2 = fully checked and safe
```

When DFS reaches a node already in state `1`, it found a directed cycle, so finishing all courses is impossible. When DFS reaches a node in state `2`, that node has already been proven safe and does not need to be explored again.

This is also `O(V + E)`. The topological BFS version is often easier to reason about for this problem because indegree directly means "number of unmet prerequisites."

### First-Principles Summary
Course Schedule is about whether dependency constraints have a legal starting point, then another legal starting point, and so on until every course is taken.

The core insight is:

```text
A course can be taken exactly when its number of unmet prerequisites is zero.
```

Represent each prerequisite pair as a directed edge from prerequisite to dependent course. Track each course's indegree as its count of unmet prerequisites. Repeatedly take zero-indegree courses and remove their outgoing dependency effects. If all courses are removed, the prerequisites form an acyclic dependency graph and the answer is `True`. If some courses remain stuck, they are trapped in a cycle and the answer is `False`.

## Implementation
See `solutions/graph_general/p207_course_schedule.py`.

## Tests
See `tests/graph_general/test_p207_course_schedule.py`.

## Examples

### Example 1
- Input: `{'numCourses': 2, 'prerequisites': [[1, 0]]}`
- Output: `True`

### Example 2
- Input: `{'numCourses': 2, 'prerequisites': [[1, 0], [0, 1]]}`
- Output: `False`

## Follow-up Practice
- Trace Kahn's algorithm on a graph with two independent starting courses.
- Rewrite the solution with DFS states `0`, `1`, and `2`.
- Explain why a non-empty remaining graph with no zero-indegree node must contain a cycle.
