# 回溯算法从零到高阶完全教程

> 系统掌握回溯算法：从核心思想到竞赛级优化

## 教程结构

```
backtracking-tutorial/
├── part1-foundations/           # 第一部分：基础概念
│   ├── 01-what-is-backtracking.md   # 什么是回溯算法
│   ├── 02-decision-tree.md          # 决策树思维模型
│   ├── 03-template-and-pruning.md   # 通用模板与剪枝
│   └── 04-complexity-analysis.md   # 复杂度分析
│
├── part2-combinations/          # 第二部分：组合类问题
│   ├── 05-subsets.md                # 子集问题
│   ├── 06-combinations.md           # 组合问题
│   ├── 07-permutations.md           # 排列问题
│   └── 08-combination-sum.md        # 组合总和
│
├── part3-board-problems/        # 第三部分：棋盘类问题
│   ├── 09-n-queens.md               # N 皇后
│   ├── 10-sudoku.md                 # 数独求解
│   ├── 11-word-search.md            # 单词搜索
│   └── 12-flood-fill.md             # 洪水填充与岛屿
│
├── part4-string-partition/      # 第四部分：字符串问题
│   ├── 13-palindrome-partition.md   # 分割回文串
│   ├── 14-restore-ip.md             # 复原 IP 地址
│   ├── 15-letter-combinations.md    # 电话号码字母组合
│   └── 16-generate-parentheses.md  # 括号生成
│
└── part5-advanced/              # 第五部分：高级技巧
    ├── 17-pruning-optimization.md   # 剪枝优化策略
    ├── 18-memoization.md            # 记忆化回溯
    ├── 19-backtracking-vs-dp.md     # 回溯 vs 动态规划
    └── 20-competitive-programming.md # 竞赛级应用
```

## 学习路径

### 入门路径（1周）
Part1 全部 → Part2 第5-7章

### 进阶路径（再1周）
Part2 第8章 → Part3 全部 → Part4 前两章

### 竞赛路径（再1-2周）
Part4 全部 → Part5 全部

## 核心算法模板（预览）

```python
def backtrack(路径, 选择列表):
    if 满足结束条件:
        result.append(路径)
        return
    
    for 选择 in 选择列表:
        做选择
        backtrack(路径, 新的选择列表)
        撤销选择  # 回溯！
```

## 前置知识

- 递归函数（理解调用栈）
- 基本数据结构（列表、集合）
- 时间/空间复杂度概念

---

*共 20 章，预计学习时间：3-4 周*
