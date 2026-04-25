# LAB-GOV-DISASTER-01: 分支策略和灾难恢复卡片

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-GOV-DISASTER-01 --force
cd workspaces/gov-disaster/incident-lab
```

## 执行

- 打开 `POLICY.md` 填写分支策略。
- 打开 `INCIDENT-CARDS.md` 为三类事故填写观察和恢复步骤。

## 观察

```bash
git status -sb
git log --oneline --graph --decorate --all --max-count=12
```

```bash
git show-ref --heads --tags
git log --oneline --graph --decorate --all --max-count=20
```

## 恢复

- 任何模拟恢复前先创建 `rescue/<case>` 分支保存证据。

## 清理

```bash
cd git-tutorial/labs
rm -rf workspaces/gov-disaster
```

## 预期结果

你能输出一页小团队 Git 协作最低规则和三张事故卡片。
