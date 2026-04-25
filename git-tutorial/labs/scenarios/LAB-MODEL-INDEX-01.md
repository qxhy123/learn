# LAB-MODEL-INDEX-01: Model Index 01

## 目标

把章节中的 `LAB-MODEL-INDEX-01` 引用落地为一个可执行练习入口。

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-MODEL-INDEX-01 --force
cd workspaces/model_index_01
```

## 执行

1. 先运行默认观察面板：`git status -sb`、`git branch -vv`、`git log --oneline --graph --decorate --all --max-count=12`。
2. 根据对应章节要求完成一次小步操作。
3. 每一步后记录 working tree、index、HEAD、branch、remote 的变化。

## 观察

- 当前是否有 unstaged、staged、untracked 内容？
- `HEAD` 指向哪次提交？
- 当前分支和远程/标签/引用的关系是什么？

## 恢复

- 不确定前先创建备份分支：`git branch backup/lab-model-index-01`。
- 对未共享历史优先在本地修正；对已共享历史优先使用新增提交修复。
- 如实验仓库混乱，回到 `labs` 目录重新运行 `./bin/git-lab.sh LAB-MODEL-INDEX-01 --force`。

## 清理

```bash
rm -rf git-tutorial/labs/workspaces/model_index_01
```
