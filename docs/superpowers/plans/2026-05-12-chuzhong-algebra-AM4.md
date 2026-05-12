# 中考代数教程 AM4 里程碑实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** 完成 `chuzhong_algebra/` 的 part6（一元二次方程, 6 章）。

**Architecture:** 4 章模板 C + **2 章模板 D**（配方法、韦达定理）。每章独立提交。

**Spec 参考:** `docs/superpowers/specs/2026-05-12-chuzhong-algebra-design.md`
**前序里程碑:** AM0–AM3（已完成）

---

## 文件结构

```
chuzhong_algebra/part6-quadratic-eq/                # 一元二次方程 [初三上]
├── 01-quadratic-basics.md                          # 概念与一般形式
├── 02-direct-and-factoring.md                      # 直接开平方 / 因式分解法
├── 03-completing-square.md                         # ⭐ 配方法
├── 04-quadratic-formula.md                         # 求根公式与判别式
├── 05-vieta-theorem.md                             # ⭐ 韦达定理
└── 06-quadratic-applications.md                    # 一元二次方程应用
```

---

## Task 1: part6/01 一元二次方程基础

**File:** `chuzhong_algebra/part6-quadratic-eq/01-quadratic-basics.md`（模板 C，1800-2500 字）

Title: `# 一元二次方程的概念与一般形式`

- **概念**：含 1 个未知数、最高次为 2、整式方程
- **一般形式**：$ax^2 + bx + c = 0$（**$a \neq 0$**）
- **关键三参数**：$a$（二次项系数）、$b$（一次项系数）、$c$（常数项）
- **判断是否是一元二次方程**：化简到一般形式后判断 $a \neq 0$
- **解的概念**：使方程成立的未知数的值
- **典型应用** 3 例：
  - 例 1：判断 $x^2 + 3 = 5x$、$\frac{1}{x^2} - 1 = 0$、$(m-1)x^2 + x = 0$ 中哪些是一元二次方程（最后一个需 $m \neq 1$）
  - 例 2：化为一般形式 $(x-2)^2 = 3x - 5$ 并指出系数
  - 例 3：已知 $x = 2$ 是方程 $x^2 - 3x + k = 0$ 的解，求 $k$
- **易错点**：$a \neq 0$；化为一般形式时移项要变号；分式方程化简后未必是二次
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part6/01 (quadratic equation basics)`

---

## Task 2: part6/02 直接开平方 + 因式分解法

**File:** `chuzhong_algebra/part6-quadratic-eq/02-direct-and-factoring.md`（模板 C，1800-2500 字）

Title: `# 直接开平方法与因式分解法`

- **直接开平方法**：$x^2 = p$（$p \geq 0$）→ $x = \pm\sqrt{p}$
  - 适用：方程能化为 $(\text{未知项})^2 = \text{非负数}$ 形式
  - 推广：$(ax+b)^2 = c$ → $ax + b = \pm\sqrt{c}$
- **因式分解法**：把 $ax^2 + bx + c = 0$ 分解为 $(\cdot)(\cdot) = 0$，由"积 = 0 → 因子 = 0"得解
  - 适用：方程能因式分解（含平方差 / 完全平方 / 十字相乘）
- **典型应用** 3 例每带【思路】:
  - 例 1：直接开平方 $4(x-1)^2 = 9$（$x - 1 = \pm \frac{3}{2}$）
  - 例 2：因式分解 $x^2 - 5x + 6 = 0$（十字相乘 → $(x-2)(x-3) = 0$）
  - 例 3：因式分解 $3x^2 = 6x$（提公因式 → $3x(x-2) = 0$，注意**不能两边除以 $x$ 丢解**）
- **易错点**：丢解（除以可能为零的因式）；忘记 $\pm$；直接开平方需保证右边 $\geq 0$
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part6/02 (direct square root and factoring)`

---

## Task 3: part6/03 配方法 ⭐（模板 D）

**File:** `chuzhong_algebra/part6-quadratic-eq/03-completing-square.md`（**模板 D**，3000-4000 字）

Title: `# 配方法解一元二次方程`

**一例速记**：
> $x^2 + 6x - 7 = 0$ → $x^2 + 6x = 7$ → $x^2 + 6x + 9 = 16$ → $(x+3)^2 = 16$ → $x = 1$ 或 $-7$。**配方三步：移项 → 加"一次项系数一半的平方" → 开方**。

### 一、引入题
> 解 $x^2 + 6x - 7 = 0$。盲算求根公式繁琐；用配方 30 秒。

### 二、思维路径还原（12-18 行 `>` 引用）
> "$x^2 + 6x - 7 = 0$ —— 二次三项式但**不是十字相乘标准型**（试 $-7$ 因数对发现凑不出 6）。  
> 看到 $x^2 + 6x$ → **完全平方提示**：缺一个常数让它变成 $(x + p)^2$。  
> $(x + 3)^2 = x^2 + 6x + 9$ —— **缺的常数是 9**（$6 \div 2 = 3$，再平方）。  
> 把方程改写：$x^2 + 6x - 7 = 0$ → $x^2 + 6x = 7$（移常数到右边）  
> 两边都加 9：$x^2 + 6x + 9 = 7 + 9 = 16$  
> 左边折叠成 $(x+3)^2 = 16$  
> 开方：$x + 3 = \pm 4$ → $x = 1$ 或 $x = -7$  
> **关键反射**：见 $x^2 + bx$ 立即想'加 $(\frac{b}{2})^2$'凑完全平方。"

### 三、抽象成方法
**配方法三步流程**（$x^2$ 系数为 1 时）：
1. **移项**：常数项移到右边
2. **配方**：两边同加"一次项系数一半的平方" $\left(\frac{b}{2}\right)^2$
3. **开方解**：$(x + \frac{b}{2})^2 = \cdots$ → $x = -\frac{b}{2} \pm \sqrt{\cdots}$

**当 $a \neq 1$**：先两边除以 $a$ 化二次项系数为 1，再配方。

### 四、方法变形
- **配方求最值**（二次函数）：$y = x^2 + bx + c = (x + \frac{b}{2})^2 + (c - \frac{b^2}{4})$
- **配方判别正负**：如证 $x^2 - 4x + 5 > 0$ → $(x-2)^2 + 1 > 0$
- **完全平方公式逆用**：见 $a^2 + 2ab + b^2$ 立即想 $(a+b)^2$

### 五、思考路标（≥6 条）
- 见一元二次方程 + 一次项系数偶数 → 优先配方
- 见 $x^2 + bx$ → 加 $(\frac{b}{2})^2$
- 见 $ax^2 + bx + c$（$a \neq 1$）→ 先除以 $a$ 再配
- 求二次函数最值 → 配成顶点式
- 证不等式 $f(x) > 0$ → 配方变 $(\cdot)^2 + (\text{正数})$
- 因式分解不行 + 系数复杂 → 求根公式优先（配方在 $a, b$ 简单时更快）

### 六、应用例题 3 例
- 例 1：$x^2 - 4x - 5 = 0$ 配方解（$(x-2)^2 = 9, x = 5$ 或 $-1$）
- 例 2：$2x^2 - 8x + 3 = 0$（先除以 2 再配方）
- 例 3：求 $y = x^2 - 6x + 11$ 的最值（配成 $(x-3)^2 + 2$，最小值 2，呼应 part10 二次函数）

### 七、自测题 5 题

Commit: `docs(chuzhong_algebra): add part6/03 (completing the square)`

---

## Task 4: part6/04 求根公式与判别式

**File:** `chuzhong_algebra/part6-quadratic-eq/04-quadratic-formula.md`（模板 C，2200-3000 字）

Title: `# 求根公式与判别式`

- **求根公式**：$ax^2 + bx + c = 0$（$a \neq 0$）的解为
  $$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$
- **推导**：由配方法（呼应 part6/03）推出——把通用情形配方
- **判别式 $\Delta = b^2 - 4ac$**：
  - $\Delta > 0$ → 两个**不相等**实数根
  - $\Delta = 0$ → 两个**相等**实数根（即一个根）
  - $\Delta < 0$ → **无**实数根
- **几何意义**（呼应 part10 二次函数）：$\Delta$ 对应抛物线与 x 轴交点个数
- **典型应用** 3 例每带【思路】:
  - 例 1：用求根公式解 $2x^2 - 3x - 1 = 0$
  - 例 2：判别 $x^2 + 2x + 3 = 0$ 的根的情况（$\Delta = 4 - 12 = -8 < 0$，无实根）
  - 例 3：含参 $x^2 + 2x + m = 0$ 有两个不相等实根，求 $m$ 范围（$\Delta > 0$ → $4 - 4m > 0$ → $m < 1$）
- **易错点**：求根公式中 $-b$ 的符号；$\Delta$ 在 $b$ 处也要平方（$b$ 含负号注意）；$\Delta = 0$ 是一个根（重根）不是无根
- **自测题** 5 题。

Commit: `docs(chuzhong_algebra): add part6/04 (quadratic formula and discriminant)`

---

## Task 5: part6/05 韦达定理 ⭐（模板 D）

**File:** `chuzhong_algebra/part6-quadratic-eq/05-vieta-theorem.md`（**模板 D**，3000-4000 字）

Title: `# 韦达定理（根与系数关系）`

**一例速记**：
> $ax^2 + bx + c = 0$ 两根 $x_1, x_2$：$x_1 + x_2 = -\frac{b}{a}$、$x_1 x_2 = \frac{c}{a}$。**不解方程**就能用根的对称表达式。

### 一、引入题
> 已知 $x^2 + 3x - 5 = 0$ 两根 $x_1, x_2$，求 $x_1^2 + x_2^2$。

### 二、思维路径还原（12-18 行 `>` 引用）
> "题目给方程，问 $x_1^2 + x_2^2$。  
> **错误思路**：用求根公式解出 $x_1, x_2$，再代入计算 —— 涉及根式平方很复杂  
> **正确思路**：$x_1^2 + x_2^2$ 是**两根的对称表达式**——用韦达定理！  
> 韦达：$x_1 + x_2 = -3$，$x_1 x_2 = -5$  
> 用代数恒等式：$x_1^2 + x_2^2 = (x_1+x_2)^2 - 2 x_1 x_2$  
> $= (-3)^2 - 2 \times (-5) = 9 + 10 = 19$  
> 答：19  
> **关键反射**：见'两根的对称表达式'（如 $x_1+x_2, x_1 x_2, x_1^2 + x_2^2, \frac{1}{x_1}+\frac{1}{x_2}$）→ 韦达定理，**不解方程**。"

### 三、抽象成方法
**韦达定理**：$ax^2 + bx + c = 0$（$a \neq 0$）两根满足
$$x_1 + x_2 = -\frac{b}{a}, \quad x_1 x_2 = \frac{c}{a}$$

**常用对称表达式变形**（必背）：
- $x_1^2 + x_2^2 = (x_1 + x_2)^2 - 2 x_1 x_2$
- $(x_1 - x_2)^2 = (x_1 + x_2)^2 - 4 x_1 x_2$
- $\frac{1}{x_1} + \frac{1}{x_2} = \frac{x_1 + x_2}{x_1 x_2}$
- $x_1^3 + x_2^3 = (x_1+x_2)^3 - 3 x_1 x_2 (x_1 + x_2)$

### 四、方法变形
- **含参题**：方程有根的条件 = $\Delta \geq 0$；两正根 = $\Delta \geq 0, x_1 + x_2 > 0, x_1 x_2 > 0$
- **构造方程**：已知两数的和与积，构造一元二次方程 $t^2 - st + p = 0$
- **韦达定理与因式分解**：$ax^2 + bx + c = a(x - x_1)(x - x_2)$

### 五、思考路标（≥6 条）
- 见方程 + 问"两根对称表达式" → 韦达定理（不解方程）
- 见"$x_1, x_2$ 是方程的根" → 立刻写韦达两式
- 见 $x_1^2 + x_2^2$ → 用 $(x_1+x_2)^2 - 2 x_1 x_2$
- 见"两根同正" → $\Delta \geq 0, x_1 + x_2 > 0, x_1 x_2 > 0$
- 见"两根异号" → $x_1 x_2 < 0$（自动 $\Delta > 0$）
- 含参题 → 韦达 + 判别式联合用

### 六、应用例题 3 例
- 例 1：$x_1^2 + x_2^2$（同引入题）
- 例 2：方程 $x^2 - 4x + k = 0$ 两正根，求 $k$（$\Delta \geq 0$, $x_1+x_2 = 4 > 0$ ✓, $x_1 x_2 = k > 0$ → $0 < k \leq 4$）
- 例 3：构造方程——以 $2 + \sqrt{3}$ 和 $2 - \sqrt{3}$ 为根（和 4 积 1，方程 $x^2 - 4x + 1 = 0$）

### 七、自测题 5 题

Commit: `docs(chuzhong_algebra): add part6/05 (Vieta's theorem)`

---

## Task 6: part6/06 一元二次方程应用

**File:** `chuzhong_algebra/part6-quadratic-eq/06-quadratic-applications.md`（模板 C，2200-3000 字）

Title: `# 一元二次方程的应用`

- **应用题列方程的步骤**（与 part5 一致）：
  1. 审题：识别**等量关系**和**未知量**
  2. 设元：直接设 / 间接设
  3. 列方程：用已知量与未知量表达等量关系
  4. 解方程：四选一（直接开方 / 因式分解 / 配方 / 求根公式）
  5. **检验实际意义**：负根 / 不合理解要**舍去**
- **5 类典型应用题**：
  1. **面积问题**：长方形 / 周长面积配对
  2. **增长率问题**：$a(1+r)^n = b$（$n=2$ 时是二次方程）
  3. **几何问题**：勾股定理 + 二次方程
  4. **数字问题**：两位数 = $10a + b$
  5. **传播问题**：每轮传 $x$ 人，连续传播 → 几何级数化二次
- **典型应用** 3 例每带【思路】:
  - 例 1：**面积题** —— 长比宽多 3，面积 28，求长宽（设宽 $x$，$x(x+3) = 28$）
  - 例 2：**增长率** —— 某公司 2 年内利润从 100 万增到 144 万，求年均增长率（$100(1+r)^2 = 144$ → $r = 0.2$）
  - 例 3：**传播题** —— 一种新型病毒传播，第一轮一人传染 $x$ 人，第二轮这 $x+1$ 人各传 $x$ 人，共 $169$ 人。求 $x$（$(1+x)^2 = 169$ → $x = 12$）
- **易错点**：负根要舍；增长率 $r$ 为小数；面积题别忘 $\geq 0$；要看清"增长 $r$" vs "变为原来 $r$ 倍"
- **自测题** 5 题。

Commit: `docs(chuzhong_algebra): add part6/06 (quadratic equation applications)`

---

## Task 7: AM4 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' chuzhong_algebra/part6-quadratic-eq
ls chuzhong_algebra/part6-quadratic-eq
printf '\n---\n**AM4 完成于：YYYY-MM-DD**\n' >> docs/superpowers/plans/2026-05-12-chuzhong-algebra-AM4.md
git add docs/superpowers/plans/2026-05-12-chuzhong-algebra-AM4.md
git commit -m "docs(chuzhong_algebra): mark AM4 milestone complete"
git push origin master
```

---
**AM4 完成于：2026-05-12**
