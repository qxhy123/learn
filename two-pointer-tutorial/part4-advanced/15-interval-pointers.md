# 第15章：区间问题与双指针

## 15.1 区间问题的基本形式

区间问题处理形如 `[start, end]` 的数据，常见操作：
- **合并**：重叠区间合并成一个
- **插入**：插入新区间并合并
- **交集**：找两个区间列表的公共部分
- **覆盖**：找最少区间覆盖目标范围
- **最大重叠**：同一时刻最多有多少区间重叠

**双指针在区间问题中的角色**：区间一般先排序，然后用指针遍历比较相邻/当前区间。

---

## 15.2 区间合并

**问题（LeetCode 56）**：给定若干区间，合并所有重叠的区间。

```python
def merge_intervals(intervals):
    if not intervals:
        return []

    # 按起点排序
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]

        if start <= last_end:
            # 重叠：合并（更新终点为较大值）
            merged[-1][1] = max(last_end, end)
        else:
            # 不重叠：直接加入
            merged.append([start, end])

    return merged

# 测试
print(merge_intervals([[1,3],[2,6],[8,10],[15,18]]))
# [[1,6],[8,10],[15,18]]

print(merge_intervals([[1,4],[4,5]]))
# [[1,5]]（端点相邻也算重叠）

print(merge_intervals([[1,4],[0,4]]))
# [[0,4]]
```

**关键判断**：`start <= last_end` 表示重叠（注意等号：端点相邻也合并）。

---

## 15.3 插入区间

**问题（LeetCode 57）**：在已排序且无重叠的区间列表中，插入一个新区间并合并。

```python
def insert_interval(intervals, new_interval):
    result = []
    i = 0
    n = len(intervals)

    # 阶段1：添加所有在 new_interval 左边的区间（不重叠）
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1

    # 阶段2：合并所有与 new_interval 重叠的区间
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    result.append(new_interval)

    # 阶段3：添加所有在 new_interval 右边的区间（不重叠）
    while i < n:
        result.append(intervals[i])
        i += 1

    return result

# 测试
print(insert_interval([[1,3],[6,9]], [2,5]))
# [[1,5],[6,9]]

print(insert_interval([[1,2],[3,5],[6,7],[8,10],[12,16]], [4,8]))
# [[1,2],[3,10],[12,16]]
```

**三阶段扫描**是处理有序区间插入的标准框架：左侧 → 重叠区 → 右侧。

---

## 15.4 区间交集

**问题（LeetCode 986）**：给定两个有序区间列表，找所有交集。

```python
def interval_intersection(first_list, second_list):
    """双指针：两个列表各一个指针，比较当前区间"""
    i, j = 0, 0
    result = []

    while i < len(first_list) and j < len(second_list):
        # 交集：max(start) <= min(end)
        lo = max(first_list[i][0], second_list[j][0])
        hi = min(first_list[i][1], second_list[j][1])

        if lo <= hi:
            result.append([lo, hi])

        # 移动终点较小的那个（它已经无法贡献更多交集）
        if first_list[i][1] < second_list[j][1]:
            i += 1
        else:
            j += 1

    return result

# 测试
print(interval_intersection(
    [[0,2],[5,10],[13,23],[24,25]],
    [[1,5],[8,12],[15,24],[25,26]]
))
# [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
```

**移动决策**：终点较小的区间已经"用完"，它与当前另一侧区间的交集已计算，且由于有序性，与另一侧之后的区间也不可能有交集（另一侧的起点只会更大）。

---

## 15.5 最少箭刺穿气球

**问题（LeetCode 452）**：气球在 x 轴上横向铺开，找最少需要多少支箭。

```python
def find_min_arrow_shots(points):
    """贪心 + 排序：按终点排序，尽可能用一支箭射多个气球"""
    if not points:
        return 0

    points.sort(key=lambda x: x[1])  # 按终点排序
    arrows = 1
    arrow_pos = points[0][1]  # 第一支箭射在第一个气球的终点

    for start, end in points[1:]:
        if start > arrow_pos:
            # 当前气球在箭的右边，需要新的箭
            arrows += 1
            arrow_pos = end

    return arrows

# 测试
print(find_min_arrow_shots([[10,16],[2,8],[1,6],[7,12]]))  # 2
print(find_min_arrow_shots([[1,2],[3,4],[5,6],[7,8]]))      # 4
print(find_min_arrow_shots([[1,2],[2,3],[3,4],[4,5]]))      # 2
```

---

## 15.6 会议室（区间重叠检测）

**问题（LeetCode 252）**：给定一系列会议时间区间，判断一个人能否参加所有会议（无重叠）。

```python
def can_attend_meetings(intervals):
    intervals.sort()
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:  # 注意：<，等于表示一个结束另一个开始
            return False
    return True
```

**进阶（LeetCode 253）**：最少需要多少个会议室？

```python
import heapq

def min_meeting_rooms(intervals):
    """最小堆：堆存储当前所有会议的结束时间"""
    if not intervals:
        return 0

    intervals.sort()
    heap = []  # 最小堆，存结束时间

    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heapreplace(heap, end)  # 复用最早结束的会议室
        else:
            heapq.heappush(heap, end)     # 分配新会议室

    return len(heap)

# 双指针版（等价但更清晰）
def min_meeting_rooms_two_pointers(intervals):
    """将开始和结束时间分别排序，用双指针扫描"""
    starts = sorted(s for s, e in intervals)
    ends = sorted(e for s, e in intervals)

    rooms = 0
    end_ptr = 0

    for start in starts:
        if start < ends[end_ptr]:
            rooms += 1  # 没有会议室结束，需要新房间
        else:
            end_ptr += 1  # 有会议室结束，复用

    return rooms

# 测试
intervals = [[0,30],[5,10],[15,20]]
print(min_meeting_rooms(intervals))               # 2
print(min_meeting_rooms_two_pointers(intervals))  # 2
```

**双指针版的直觉**：`starts` 指针扫描每个会议的开始，`end_ptr` 追踪最早结束的会议。若新会议开始时有会议已结束，复用；否则新增房间。

---

## 15.7 区间覆盖（贪心 + 双指针）

**问题**：给定若干区间，找最少的区间数覆盖 `[0, n]`。

```python
def min_taps_to_open(n, ranges):
    """花园水龙头：最少需要打开多少水龙头（转化为区间覆盖）"""
    # 构建区间
    intervals = []
    for i, r in enumerate(ranges):
        if r > 0:
            intervals.append([max(0, i - r), min(n, i + r)])

    # 按起点排序
    intervals.sort()

    # 贪心：每次在当前覆盖范围内选终点最远的区间
    covered = 0   # 当前已覆盖到的右端点
    max_reach = 0  # 在考虑的所有区间中，最远可到达的右端点
    count = 0
    i = 0

    while covered < n:
        # 找所有起点 <= covered 的区间中，终点最大的
        while i < len(intervals) and intervals[i][0] <= covered:
            max_reach = max(max_reach, intervals[i][1])
            i += 1

        if max_reach <= covered:
            return -1  # 无法继续覆盖

        covered = max_reach
        count += 1

    return count

# 测试
print(min_taps_to_open(5, [3,4,1,1,0,0]))  # 1
print(min_taps_to_open(3, [0,0,0,0]))      # -1
```

---

## 15.8 本章小结

区间双指针的核心操作：

| 操作 | 技巧 | 关键判断 |
|------|------|----------|
| 合并区间 | 排序+顺序扫描 | `start <= last_end` |
| 插入区间 | 三阶段扫描 | 左侧/重叠/右侧 |
| 区间交集 | 双列表双指针 | 移动终点较小的 |
| 最大重叠 | 开始/结束分离双指针 | 新开始 vs 最早结束 |
| 区间覆盖 | 贪心跳跃 | 每步选最远终点 |

**下一章：多指针与分区问题——荷兰国旗及其推广**

---

## LeetCode 推荐题目

- [56. 合并区间](https://leetcode.cn/problems/merge-intervals/) ⭐⭐
- [57. 插入区间](https://leetcode.cn/problems/insert-interval/) ⭐⭐
- [986. 区间列表的交集](https://leetcode.cn/problems/interval-list-intersections/) ⭐⭐
- [252. 会议室](https://leetcode.cn/problems/meeting-rooms/) ⭐
- [253. 会议室 II](https://leetcode.cn/problems/meeting-rooms-ii/) ⭐⭐⭐
- [452. 用最少数量的箭引爆气球](https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/) ⭐⭐
