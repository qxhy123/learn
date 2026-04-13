# 第2章：决策树思维模型

## 2.1 决策树是理解回溯的核心

回溯算法的每次执行都对应一棵**决策树**的遍历。理解决策树，就理解了回溯。

**决策树的组成**：
- **节点**：当前状态（已做出的选择序列）
- **边**：从当前状态做的一个选择
- **叶节点**：终止状态（找到解或确定无解）
- **根节点**：初始状态（未做任何选择）

## 2.2 三类决策树

回溯问题的决策树大致分三类，对应不同的选择策略：

### 类型一：选或不选（子集型）

每个元素只有两种选择：**加入**或**不加入**。

```
元素 [1, 2, 3]，生成所有子集：

              []
            /    \
          [1]    []
          /\     /\
       [1,2] [1] [2] []
       / \   /\  /\  /\
[1,2,3][1,2][1,3][1][2,3][2][3][]
```

树的高度 = 元素个数，每层对应一个元素的"选/不选"决策。

```python
def subsets(nums):
    result = []
    
    def backtrack(start, path):
        result.append(path[:])  # 每个节点都是一个有效子集
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result
```

### 类型二：选哪个（排列/组合型）

每步从候选列表中选一个元素，不同顺序视为不同（排列）或相同（组合）。

```
元素 [1, 2, 3]，生成所有排列：

                    []
         /          |          \
        1           2           3
       / \         / \         / \
      2   3       1   3       1   2
      |   |       |   |       |   |
      3   2       3   1       2   1
```

```python
def permutations(nums):
    result = []
    used = [False] * len(nums)
    
    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False
    
    backtrack([])
    return result
```

### 类型三：逐格填写（棋盘型）

在网格/棋盘上逐格/逐行填写，每格有多种选择。

```
4×4 棋盘，N 皇后（每行选一列）：

         行0: 选列 0,1,2,3
        /    |    |    \
   行1: ...（根据行0约束筛选）
```

## 2.3 决策树的"前序"与"后序"位置

这是理解回溯的关键概念：

```python
def backtrack(path, choices):
    # ★ 前序位置：进入节点时执行
    #   - 做出选择
    #   - 路径记录
    
    if is_end():
        save_result()
        return
    
    for choice in choices:
        make_choice(choice)    # 前序：做选择
        
        backtrack(new_path, new_choices)
        
        undo_choice(choice)    # 后序：撤销选择（回溯）
    
    # ★ 后序位置：离开节点时执行
    #   - 撤销操作（关键！）
    #   - 统计/收集信息
```

**前序位置**：进入节点时，做选择，更新状态。
**后序位置**：离开节点时，撤销选择，恢复状态。

这一"撤销"操作是回溯的灵魂。

## 2.4 决策树的"宽度"与"深度"

```
宽度 = 每个节点的选择数量（for 循环的迭代次数）
深度 = 递归的层数（路径长度）

时间复杂度 ≈ O(宽度^深度)（未剪枝时）
```

**示例分析**：
- 全排列 n 元素：宽度≈n，深度=n → O(n!)
- 子集：宽度=2（选/不选），深度=n → O(2^n)
- N皇后：宽度≈n，深度=n → O(n!)

## 2.5 剪枝在决策树上的体现

剪枝 = **提前截断某个子树**，不再遍历其子孙节点。

```
N 皇后的剪枝：

         列0  列1  列2  列3
行0:      Q
行1:      ×    ×    Q         ← 列0,1被同行/对角线排除
行2:      ×    ×    ×    ×    ← 所有列都冲突，剪枝！回溯到行1
```

```python
# 剪枝示例：组合问题
def combine(n, k):
    result = []
    
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        
        # 剪枝：剩余元素数量 = n - start + 1
        # 需要的元素数量 = k - len(path)
        # 若剩余 < 需要，无法凑满 k 个，提前返回
        if n - start + 1 < k - len(path):
            return
        
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(1, [])
    return result
```

## 2.6 用决策树分析问题

面对一道回溯题，按以下步骤思考：

**步骤 1：确定树的宽度（for 循环）**
- 子集型：每个元素选/不选（宽度=2）
- 组合型：从候选中选一个（宽度=候选数量）
- 排列型：从所有未选中选一个

**步骤 2：确定树的深度（递归终止）**
- 路径长度达到目标长度
- 候选集为空
- 达到某个约束边界

**步骤 3：确定剪枝条件**
- 当前路径已不可能产生有效解
- 重复状态（去重）
- 超出范围

## 2.7 完整案例：组合总和

**问题**：从 `candidates` 中选数字（可重复选），使总和 = `target`。

```
candidates = [2, 3, 6, 7], target = 7

决策树：
              []
        /     |     |     \
      [2]    [3]   [6]   [7✓]
     / | \   /\    /
  [2,2][2,3][2,6] ...
  / |
[2,2,2][2,2,3]
  |
[2,2,2,?] → 2+2+2=6 < 7, 继续
[2,2,2,2] → 8 > 7, 剪枝！
[2,2,3] → 7 = 7 ✓

剪枝条件：当前和 > target，截断
```

```python
def combination_sum(candidates, target):
    candidates.sort()  # 排序便于剪枝
    result = []
    
    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break  # 排序后，后面更大，全部剪枝
            
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])  # i 不加1，允许重复
            path.pop()
    
    backtrack(0, [], target)
    return result
```

## 小结

| 概念 | 对应代码 |
|------|---------|
| 决策树节点 | 函数调用栈帧 |
| 边 | 做选择 |
| 前序位置 | 做选择（修改状态） |
| 后序位置 | 撤销选择（回溯） |
| 剪枝 | 提前 return/break/continue |

## 练习

1. 画出 `combination_sum([2,3,6,7], 7)` 的完整决策树（包含剪枝位置）
2. 对比子集型和排列型决策树的结构差异
3. 思考：如果不撤销选择，会发生什么？（试着删掉 `path.pop()` 运行看看）

---

**上一章：** [什么是回溯](01-what-is-backtracking.md) | **下一章：** [通用模板与剪枝](03-template-and-pruning.md)
