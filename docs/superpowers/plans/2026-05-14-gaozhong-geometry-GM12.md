# 高中几何教程 GM12 里程碑实施计划（补全 figures）

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 在现有 82 张 SVG 基础上补 ~15 张精选图，主要覆盖各 Part 的"04 应用"章节空白 + Part 10 综合不足，目标总 ~100 张。

**前序里程碑:** GM0-GM11

---

## 补图清单（精选 15 张）

### 应用类章节（之前 0 图，共 7 张）

| 图编号 | 章节 | 内容 |
|---|---|---|
| geo-p2-04-1 | Part 2/04 向量应用 | 物理力合成（平行四边形法则）+ 速度合成 |
| geo-p3-05-1 | Part 3/05 直线综合 | 点关于直线对称（含轴 / 中点连接） |
| geo-p4-04-1 | Part 4/04 圆综合 | 圆上点到直线最值（圆心距 ± 半径） |
| geo-p5-04-1 | Part 5/04 椭圆应用 | 椭圆光学反射性质（焦点入射 → 焦点反射） |
| geo-p6-04-1 | Part 6/04 双曲线应用 | 声呐定位双曲线（双站距离差恒定） |
| geo-p7-04-1 | Part 7/04 抛物线应用 | 抛物面反射（平行光 → 焦点） |
| geo-p7-04-2 | Part 7/04 抛物线应用 | 弹道抛物线（斜抛运动） |

### 综合补图（共 5 张）

| 图编号 | 章节 | 内容 |
|---|---|---|
| geo-p9-02-2 | Part 9/02 空间数量积 | 空间垂直判定 |
| geo-p9-02-3 | Part 9/02 空间数量积 | 投影几何意义 |
| geo-p10-02-2 | Part 10/02 含参圆锥 | $\dfrac{x^2}{4}+\dfrac{y^2}{m}=1$ 分类（椭圆 / 圆 / 双曲线） |
| geo-p10-04-2 | Part 10/04 定点定值 | 定点问题套路（参数分离法示意） |
| geo-p10-06-2 | Part 10/06 向量综合 | 向量 + 圆锥曲线 + 函数交叉 |

### 立体补图（共 2 张）

| 图编号 | 章节 | 内容 |
|---|---|---|
| geo-p4-04-2 | Part 4/04 圆综合 | 阿波罗尼斯圆 $\|PA\|=k\|PB\|$ 轨迹 |
| geo-p8-06-2 | Part 8/06 立体综合 | 正方体 8 种截面完整图（三角 / 四边 / 五边 / 六边各 2） |

**合计 14 张**（目标范围内）。

---

## 调度

**2 个 sonnet subagent 并行：**
- A: 应用 + 阿波罗尼斯（8 张）
- B: 综合 + 立体补（6 张）

每图源文件 + SVG 一起 commit。

## Task GM12 收尾

```bash
cd /Users/yangyang/ai_projs/math
ls gaozhong_math/geometry/figures/svg/ | wc -l  # 应为 ~96
printf '\n---\n**GM12 完成于：2026-05-14**\n' >> docs/superpowers/plans/2026-05-14-gaozhong-geometry-GM12.md
git add docs/superpowers/plans/2026-05-14-gaozhong-geometry-GM12.md
git commit -m "docs(gaozhong/geometry): mark GM12 milestone complete (figures补全)"
git push origin master
```
