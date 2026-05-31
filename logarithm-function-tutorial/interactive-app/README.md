# Log Quest · 对数函数多邻国式互动学习

把《从零到高阶的对数函数教程》全部 20 章变成**多邻国（Duolingo）式闯关**：
关卡树 → 课程 → 多题型答题 → 即时对错反馈与讲解 → XP / 连胜 / 心数 / 完课撒花。
进度用 `localStorage` 保存，刷新不丢。

> 当前为**纵切片 demo**：全 20 单元（60 课）。
> 9 课、约 70 题，覆盖 4 种题型。后续可按同样数据结构扩到全 24 章。

## 运行

```bash
cd trigonometry-tutorial/interactive-app
npm install
npm run dev          # 本地开发，打开终端提示的 http://localhost:5173
# 或构建静态产物：
npm run build        # 输出到 dist/，base 为相对路径，可托管 GitHub Pages
npm run preview      # 预览 dist/
```

> 注意：构建产物用了 ES Module，**不能直接双击 dist/index.html（file:// 会被 CORS 拦）**，
> 请用 `npm run preview` 或任意静态服务器（`npx serve dist`）打开。

## 功能

- **4 种题型**：选择 / 填空（容错匹配，`√3/2`、`sqrt3/2`、`0.866`、`π/6` 等写法均可）/ 判断 / 配对（角↔值）。
- **游戏化**：每题对得 XP（连击有加成）、心数 5 颗（答错 −1，归零重试）、连胜天数 streak、完课 confetti、关卡树顺序解锁。
- **公式**：题干与讲解用 `$...$`（KaTeX 渲染）。
- **深链接**：`#lesson=u1-l1` 可直接进入某课（刷新/分享稳定）。

## 结构

```
src/
  content/units.ts        # 全部题库（单元→课程→题目，强类型）
  types.ts                # 内容与进度的类型
  progress.tsx            # localStorage 存档 + XP/streak 逻辑（Context）
  lib/check.ts            # 输入题容错匹配 + 日期/streak 工具
  components/
    SkillTree.tsx         # 关卡树主页
    Lesson.tsx            # 单课答题流程（核心状态机）
    TopBar.tsx FeedbackBar.tsx LessonComplete.tsx Math.tsx
    questions/            # 4 种题型组件
```

## 扩到全 24 章

往 `src/content/units.ts` 的 `UNITS` 数组继续追加单元/课程/题目即可——
组件与游戏化逻辑全部数据驱动，无需改动。
