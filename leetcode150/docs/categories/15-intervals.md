# 15 — Intervals（融合版）

> **难度**：★★★☆☆
> **题数**：4
> **核心套路**：排序 + 合并、贪心插入、扫描线 / 差分数组、区间总结
> **本文件**：覆盖 intervals 4 题的算法套路总结 + 典型题精讲 + 自测

---

## 一例速记

> **排序 + 合并**：按区间左端点排序，遍历时若当前区间与结果集最后一个区间重叠（`cur[0] <= last[1]`），则合并（更新 `last[1] = max(last[1], cur[1])`）；否则直接追加（56 Merge Intervals）
> **贪心插入**：插入新区间时，先将所有"在新区间左边不重叠"的区间直接加入结果，然后将所有"与新区间重叠"的区间逐个合并进 `newInterval`，最后追加新区间，再追加所有"在新区间右边"的区间（57 Insert Interval）
> **贪心射箭（按右端点排序）**：按右端点升序排序，用贪心"一根箭尽量射穿更多气球"——箭放在当前气球右端点，若下一个气球左端点 > 当前箭位置，则需要新的箭（452）
> **区间总结**：线性扫描连续段，遇到断点（`nums[i] != nums[i-1] + 1`）则输出一段（228）
> **AI 关联**：时间窗口聚合（Flink / Spark 滑动窗口）/ GPU 显存碎片合并 / 任务调度区间冲突检测

---

## 思维路径还原

> "看到 **'合并区间'（56）** → 先按左端点排序（`intervals.sort(key=lambda x: x[0])`），
> 然后用结果栈 `merged`，遍历每个区间：
> 若 merged 为空或 `cur[0] > merged[-1][1]`（无重叠），直接 append；
> 否则更新 `merged[-1][1] = max(merged[-1][1], cur[1])`（合并，取右端点的较大值）。
> 时间 O(n log n)（排序主导），空间 O(n)（输出）。
>
> 看到 **'插入区间'（57）** → 不需要排序（已排序），三段处理：
> ① 左边不重叠（`intervals[i][1] < newInterval[0]`）→ 直接 append；
> ② 重叠（`intervals[i][0] <= newInterval[1]`）→ 合并到 newInterval：
>    `newInterval[0] = min(newInterval[0], intervals[i][0])`，
>    `newInterval[1] = max(newInterval[1], intervals[i][1])`；
> ③ 把合并后的 newInterval append，再 append 剩余右边区间。
> 时间 O(n)，空间 O(n)。
>
> 看到 **'最少箭射气球'（452）** → 等价于找最多不重叠区间的个数（贪心经典）：
> 按**右端点**升序排序，维护当前箭的位置 `end`（初始化为第一个气球右端点），
> 遍历时若 `balloon[0] > end`（当前箭射不到），则需要新箭，更新 `end = balloon[1]`；
> 否则当前箭仍然有效（不更新 end）。答案 = 箭的总数。
> 时间 O(n log n)，空间 O(1)（不含排序空间）。
>
> 注意：452 的判断条件是 `balloon[0] > end`（**严格大于**），因为题目说"触碰也算射穿"；
> 若题目说"触碰不算"则改为 `>=`。
>
> 看到 **'区间总结'（228）** → 一次线性扫描：若数组为空直接返回 []；
> 维护 `start = nums[0]`，遍历 i=1..n-1：若 `nums[i] != nums[i-1]+1`（断点），
> 输出 `[start, nums[i-1]]` 的字符串，更新 `start = nums[i]`；
> 最后循环结束后别忘了输出最后一段。时间 O(n)，空间 O(1)（不含输出）。"

---

## 学习目标

- 掌握"排序 + 双指针/栈合并"的区间合并模板（56）
- 理解三段式贪心插入（57）：左边直接加 → 重叠合并 → 追加新区间 → 右边直接加
- 熟练区分 56（需先排序）和 57（已排序，O(n) 即可）两类题的时间复杂度差异
- 掌握"按右端点贪心"（452），能与"按左端点贪心"（56）区分使用场景
- 理解扫描线 / 差分数组思路（会议室 II 类变形），以及区间总结的线性扫描写法

---

## 抽象成方法（标准模板代码）

### 套路 1：排序 + 合并（Merge Intervals）

适用题：56

```python
from typing import List


def merge(intervals: List[List[int]]) -> List[List[int]]:
    """时间 O(n log n)，空间 O(n)。排序后逐个合并。"""
    intervals.sort(key=lambda x: x[0])   # 按左端点排序
    merged: List[List[int]] = []
    for cur in intervals:
        if not merged or cur[0] > merged[-1][1]:
            # 无重叠：当前区间在结果集最后一个区间右边
            merged.append(cur)
        else:
            # 有重叠：合并，取右端点较大值
            merged[-1][1] = max(merged[-1][1], cur[1])
    return merged
```

> 重叠判断：`cur[0] <= merged[-1][1]` 表示有重叠（注意：`<=` 包含端点相接的情况，如 `[1,3]` 和 `[3,5]` 合并为 `[1,5]`）。
> 合并只需更新右端点（左端点由排序保证 `cur[0] >= merged[-1][0]`）。

---

### 套路 2：三段式贪心插入（Insert Interval）

适用题：57

```python
def insert(intervals: List[List[int]],
           newInterval: List[int]) -> List[List[int]]:
    """时间 O(n)，空间 O(n)。三段：左边不重叠 + 合并 + 右边不重叠。"""
    result: List[List[int]] = []
    i, n = 0, len(intervals)

    # 段 1：在 newInterval 左边，不重叠
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1

    # 段 2：与 newInterval 重叠，合并
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    result.append(newInterval)

    # 段 3：在 newInterval 右边，不重叠
    while i < n:
        result.append(intervals[i])
        i += 1

    return result
```

> 57 题的 intervals 已按左端点排好序，所以 O(n) 一次扫描即可，无需再排序。
> 三段的分界条件：`intervals[i][1] < newInterval[0]`（完全在左）和 `intervals[i][0] <= newInterval[1]`（有重叠）。

---

### 套路 3：按右端点贪心射击（最少箭）

适用题：452

```python
def findMinArrowShots(points: List[List[int]]) -> int:
    """时间 O(n log n)，空间 O(1)。按右端点排序，贪心射穿最多气球。"""
    if not points:
        return 0
    points.sort(key=lambda x: x[1])   # 按右端点升序排序
    arrows = 1
    end = points[0][1]   # 第一根箭放在第一个气球右端点
    for start, right in points[1:]:
        if start > end:           # 当前气球射不到，需要新箭
            arrows += 1
            end = right           # 新箭放在当前气球右端点
        # start <= end：当前箭能射到，不更新 end（贪心保留箭在尽量靠左的位置）
    return arrows
```

> 为何按右端点排序？贪心思路："尽量让一根箭射穿更多气球"等价于"在当前气球右边缘射箭，能顺带覆盖所有左端点 ≤ 当前箭的后续气球"。

---

### 套路 4：线性扫描连续段（Summary Ranges）

适用题：228

```python
def summaryRanges(nums: List[int]) -> List[str]:
    """时间 O(n)，空间 O(1)（不含输出）。扫描断点，输出区间字符串。"""
    if not nums:
        return []
    result: List[str] = []
    start = nums[0]
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1] + 1:   # 断点：连续性断裂
            # 输出 [start, nums[i-1]]
            if start == nums[i - 1]:
                result.append(str(start))
            else:
                result.append(f"{start}->{nums[i - 1]}")
            start = nums[i]   # 新段开始
    # 别忘了最后一段
    if start == nums[-1]:
        result.append(str(start))
    else:
        result.append(f"{start}->{nums[-1]}")
    return result
```

> 易错点：循环结束后，最后一段 `[start, nums[-1]]` 需要单独输出（循环内只在遇到"断点"时输出前一段）。

---

### 套路 5：扫描线 / 差分数组（会议室类变形）

> 虽然 leetcode150 中 intervals category 没有直接包含 252/253（会议室 I/II），
> 但扫描线思路是区间 category 的核心变形，常在变形题和面试中出现，值得掌握。

```python
# 变形：会议室 II（253）— 求同时进行的最多会议数，即需要多少个会议室
# 等价：最多有多少个区间在同一时刻重叠
def min_meeting_rooms(intervals: List[List[int]]) -> int:
    """扫描线：+1 表示会议开始，-1 表示会议结束，扫描事件找最大并发数。"""
    events: List[tuple] = []
    for start, end in intervals:
        events.append((start, 1))    # 开始：+1
        events.append((end, -1))     # 结束：-1（注意若 end==start 则 -1 先处理）
    events.sort(key=lambda x: (x[0], x[1]))  # 同时刻：结束（-1）优先于开始（+1）
    current = best = 0
    for _, delta in events:
        current += delta
        best = max(best, current)
    return best


# 差分数组版（适合端点为整数且范围不大时）
def min_meeting_rooms_diff(intervals: List[List[int]]) -> int:
    """差分数组：在 start 位置 +1，在 end 位置 -1，前缀和求最大值。"""
    if not intervals:
        return 0
    max_end = max(e for _, e in intervals)
    diff = [0] * (max_end + 2)
    for start, end in intervals:
        diff[start] += 1
        diff[end] -= 1       # 结束时 -1（闭区间则 end+1 处 -1）
    current = best = 0
    for delta in diff:
        current += delta
        best = max(best, current)
    return best
```

---

### 速查表

| 题型特征 | 套路 | 时间 | 空间 |
|---|---|---|---|
| 合并重叠区间（无序） | 排序 + 逐个合并 | O(n log n) | O(n) |
| 插入新区间（已排序） | 三段式贪心 | O(n) | O(n) |
| 最少覆盖数（最少箭） | 按右端点排序 + 贪心 | O(n log n) | O(1) |
| 连续段总结 | 线性扫描断点 | O(n) | O(1) |
| 最大并发数（会议室 II） | 扫描线 / 差分数组 | O(n log n) / O(n+R) | O(n) |

---

## 方法变形（4 类）

### 变形 1：合并区间系列

- **56**（合并区间，无序输入）→ 排序 O(n log n)，再线性合并。
- **57**（插入区间，已排序输入）→ 三段式 O(n)，无需再排序。
- **252**（会议室 I，能否参加所有会议）→ 按左端点排序后检查相邻区间是否重叠（`intervals[i][0] < intervals[i-1][1]`），若有重叠则返回 False。
- 区别：56 合并输出结果；252 仅判断有无重叠（返回 bool）。

### 变形 2：贪心覆盖 / 区间调度

- **452**（最少箭）→ 按右端点贪心，等价于"最多不重叠区间"计数（Interval Scheduling Maximization）。
- **435**（Non-overlapping Intervals）→ 最少需要移除多少个区间使剩余不重叠 = n - 最多不重叠区间数。
- **253**（会议室 II）→ 最少需要多少间会议室 = 最大重叠数，用扫描线。
- 关键区分：
  - 按**左端点**贪心 → 合并、插入、判断重叠。
  - 按**右端点**贪心 → 最少箭数、最多不重叠、最少移除。

### 变形 3：扫描线 / 差分数组

- **扫描线**：将每个区间拆成"开始事件 +1"和"结束事件 -1"，排序后计算前缀和，求最大值（会议室 II）。
- **差分数组**：适合端点是整数且范围有界的情况（如时间戳在 [0, T] 内）。
- **天际线问题（218）**：扫描线 + 最大堆，维护当前最高建筑，O(n log n)（非本 category 但同框架）。
- AI 场景：Flink / Spark 的 sliding window 时间聚合本质是扫描线；GPU 任务调度按区间管理显存使用。

### 变形 4：区间总结 / 编码

- **228**（Summary Ranges）→ 线性扫描，维护段开始点，遇断点输出。
- **163**（Missing Ranges，非 LC150 但同类）→ 在 228 基础上改为输出"缺失"的区间段。
- 输出格式多变（字符串 / 数组 / 计数），但核心扫描逻辑不变。

---

## 思考路标（条件反射）

1. 看到 **"merge intervals / 合并区间"** → 先按左端点排序，再线性合并，`merged[-1][1] = max(...)`
2. 看到 **"insert interval / 插入区间（已排序输入）"** → 三段式 O(n)，无需再排序
3. 看到 **"最少箭 / 最多不重叠"** → 按右端点排序，贪心射穿，`start > end` 时新箭
4. 看到 **"最多并发 / 最少会议室"** → 扫描线（开始 +1，结束 -1，前缀和最大值）
5. 看到 **"summary ranges / 连续段"** → 线性扫描，遇断点输出，别忘最后一段
6. 看到 **"最少移除使不重叠"** → 先求最多不重叠区间数（按右端点贪心），答案 = n - 最多不重叠数
7. 看到 **"端点相接算不算重叠"** → 读题！452 中 `[1,3]` 和 `[3,5]` 算重叠（触碰即射穿）；56 中 `[1,3]` 和 `[3,5]` 也合并为 `[1,5]`；只有明确说"开区间"时才不包含端点
8. 看到 **"区间端点为整数且范围有界"** → 考虑差分数组替代扫描线，代码更简单

---

## 易错点

1. **56 重叠判断用 `<=`**：`cur[0] <= merged[-1][1]` 才算重叠（包含端点相接），不要用 `<`，否则 `[1,3],[3,5]` 无法合并。
2. **57 三段分界条件**：段 1 的分界是 `intervals[i][1] < newInterval[0]`（旧区间完全在左），段 2 是 `intervals[i][0] <= newInterval[1]`（有重叠）；两个条件用的是不同端点，易搞混。
3. **452 按右端点排序**：经典错误是按左端点排序后贪心，会得到错误答案。记忆方法：射箭贪的是"右端点"（箭能覆盖到哪）。
4. **452 判断条件严格大于**：`start > end` 时需要新箭；若题目端点是整数，`start == end` 时当前箭仍能射到（触碰即穿），所以是严格 `>`。
5. **228 最后一段漏输出**：for 循环内只在"断点"时输出"上一段"，循环结束后 `[start, nums[-1]]` 这最后一段必须单独 append，初学者常漏。
6. **56 就地修改 intervals 的坑**：`intervals.sort()` 会修改输入，若题目要求不修改原数组则需先 `sorted(intervals, ...)`。
7. **扫描线事件排序优先级**：同一时刻有"开始"和"结束"两个事件时，"结束（-1）"应排在"开始（+1）"前面（若开区间），或后面（若闭区间）；根据题目语义调整 sort key 中的第二维。
8. **228 单点区间**：若 `start == nums[i-1]`（单点），输出 `str(start)` 而非 `"start->start"`，不要忘记这个特判。

---

## 典型应用例题

### 例 1：56. Merge Intervals

**题目**：给定区间列表，合并所有重叠区间，返回合并后的区间列表。

**思路**：按左端点排序，维护结果列表 `merged`。遍历时：若 `merged` 为空或当前区间左端点 > 结果最后区间右端点（无重叠），直接 append；否则更新结果最后区间右端点为两者右端点的最大值。

**解**：

```python
# 参考：solutions/intervals/p056_merge_intervals.py
def merge(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0])
    merged: List[List[int]] = []
    for cur in intervals:
        if not merged or cur[0] > merged[-1][1]:
            merged.append(cur)
        else:
            merged[-1][1] = max(merged[-1][1], cur[1])
    return merged
```

**分析**：排序 $O(n \log n)$，合并 $O(n)$，总体 $O(n \log n)$；空间 $O(n)$（输出）。合并时只更新右端点，因为排序已保证 `cur[0] >= merged[-1][0]`，左端点不可能缩小。

---

### 例 2：57. Insert Interval

**题目**：给定已排序（按左端点升序，不重叠）的区间列表，插入一个新区间，返回合并后结果。

**思路**：三段式处理：① 完全在新区间左边的旧区间直接追加；② 与新区间重叠的旧区间逐个与 newInterval 合并（更新新区间的左右端点）；③ 追加合并后的新区间；④ 完全在右边的旧区间直接追加。

**解**：

```python
# 参考：solutions/intervals/p057_insert_interval.py
def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    result: List[List[int]] = []
    i, n = 0, len(intervals)
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    result.append(newInterval)
    while i < n:
        result.append(intervals[i])
        i += 1
    return result
```

**分析**：$O(n)$ 时间（每个区间访问一次），$O(n)$ 空间。不需要排序，因为输入已排序，三段式保证输出依然有序。

---

### 例 3：452. Minimum Number of Arrows to Burst Balloons

**题目**：气球在坐标轴上用区间 `[x_start, x_end]` 表示，垂直向上射箭可以射穿该坐标范围内所有气球，求最少需要多少支箭。

**思路**：按右端点升序排序，贪心策略：将箭放在当前气球的右端点，能射穿所有"左端点 ≤ 当前箭位置"的气球。若下一个气球左端点 > 当前箭位置，则需要新箭。

**解**：

```python
# 参考：solutions/intervals/p452_minimum_number_of_arrows_to_burst_balloons.py
def findMinArrowShots(points: List[List[int]]) -> int:
    if not points:
        return 0
    points.sort(key=lambda x: x[1])
    arrows = 1
    end = points[0][1]
    for start, right in points[1:]:
        if start > end:
            arrows += 1
            end = right
    return arrows
```

**分析**：排序 $O(n \log n)$，线性扫描 $O(n)$，总体 $O(n \log n)$；额外空间 $O(1)$。正确性：每次新箭不可避免（下一个气球的左端点已超出当前箭），贪心将箭放在右端点使其覆盖面积最大。

---

## 自测题

**自测 1**（56 Merge Intervals）—— `intervals=[[1,3],[2,6],[8,10],[15,18]]` 返回 `[[1,6],[8,10],[15,18]]`；`intervals=[[1,4],[4,5]]` 返回 `[[1,5]]`。提示：先按 `x[0]` 排序，`merged = []`，遍历时若无重叠则 append，否则 `merged[-1][1] = max(...)`。参考 `solutions/intervals/p056_merge_intervals.py`。

**自测 2**（57 Insert Interval）—— `intervals=[[1,3],[6,9]], newInterval=[2,5]` 返回 `[[1,5],[6,9]]`；`intervals=[[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval=[4,8]` 返回 `[[1,2],[3,10],[12,16]]`。提示：三段式，注意段 1 条件 `intervals[i][1] < newInterval[0]`，段 2 条件 `intervals[i][0] <= newInterval[1]`。参考 `solutions/intervals/p057_insert_interval.py`。

**自测 3**（228 Summary Ranges）—— `nums=[0,1,2,4,5,7]` 返回 `['0->2','4->5','7']`；`nums=[0,2,3,4,6,8,9]` 返回 `['0','2->4','6','8->9']`。提示：`start = nums[0]`，遍历遇断点输出上一段，循环后输出最后一段；单点输出 `str(start)` 而非 `'start->start'`。参考 `solutions/intervals/p228_summary_ranges.py`。

**自测 4**（452 Min Arrows）—— `points=[[10,16],[2,8],[1,6],[7,12]]` 返回 `2`；`points=[[1,2],[3,4],[5,6],[7,8]]` 返回 `4`；`points=[[1,2],[2,3],[3,4],[4,5]]` 返回 `2`。提示：按右端点排序，`arrows=1, end=points[0][1]`，`start > end` 时 `arrows+=1, end=right`。参考 `solutions/intervals/p452_minimum_number_of_arrows_to_burst_balloons.py`。

---

## 题目全览（4 题）

| # | 题目 | 套路分类 | 难度 |
|---|---|---|---|
| 56 | Merge Intervals | 排序 + 逐个合并 | Medium |
| 57 | Insert Interval | 三段式贪心插入 | Medium |
| 228 | Summary Ranges | 线性扫描连续段 | Easy |
| 452 | Minimum Number of Arrows to Burst Balloons | 按右端点贪心 | Medium |

---

## 融合版说明

| 段 | 来源 | 价值 |
|---|---|---|
| 一例速记 | 本文件 | 4 大套路一览 + AI 场景关联（时间窗口/资源调度） |
| 思维路径还原 | 本文件 | 4 道题的解题内心独白，含关键条件判断 |
| 抽象成方法 | 本文件 | 5 个标准模板（合并/插入/贪心射箭/区间总结/扫描线）+ 速查表 |
| 方法变形 | 本文件 | 4 类变体（合并系列/贪心覆盖/扫描线/区间编码） |
| 思考路标 | 本文件 | 8 条题型识别条件反射 |
| 易错点 | 本文件 | 8 条高频踩坑（判断条件/排序方向/最后一段漏输出等） |
| 典型应用例题 | solutions/ | 3 道精讲（56、57、452），代码 + 正确性分析 |
| 自测题 | leetcode | 4 题带提示，链接 solutions 文件 |
| 题目全览 | 本文件 | 4 题完整列表，套路分类一览 |

---

> **跨 category 导航**：
> - 排序是区间题的前置操作 → 见 `04-binary-search.md`（排序 + 二分）
> - 扫描线 + 最大堆（天际线问题）→ 见 `16-heap.md`
> - 区间 DP（如戳气球 312）→ 见 `11-dp-multidim.md`
> - Flink / Spark 的时间窗口聚合、GPU 显存碎片整理均以区间合并为核心操作
