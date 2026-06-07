import type { Lesson } from '../../src/types'

export const LESSONS: Lesson[] = [
  // ─────────────────────────────────────────────────────────────────────────
  // 关 c14-a  原函数与不定积分：动机 + 定义 + 基本公式
  // ─────────────────────────────────────────────────────────────────────────
  {
    id: 'c14-a',
    title: '原函数与不定积分',
    subtitle: '求导的逆运算·定义·基本积分表',
    intro: [
      // 卡 1：动机——求导的逆运算
      {
        title: '为什么要"反着求导"？',
        body:
          '你已经会对 $F(x)=x^3$ 求导，得到 $F\'(x)=3x^2$。\n\n' +
          '现在反过来问：已知导数 $f(x)=3x^2$，能把 $F(x)$ 找回来吗？\n\n' +
          '这就是**不定积分**要解决的问题。物理中"已知速度 $v(t)$ 求位移 $s(t)$"、' +
          '工程中"已知边际成本求总成本"，都是同一类反问题——已知变化率，恢复原量。',
        tip: '求导是"给原函数，求导数"；不定积分是"给导数，找原函数"——方向正好相反。',
      },
      // 卡 2：原函数的定义
      {
        title: '原函数是什么？',
        body:
          '若函数 $F(x)$ 满足 $F\'(x)=f(x)$，就称 $F(x)$ 是 $f(x)$ 的一个**原函数**。\n\n' +
          '**具体例子**：$f(x)=3x^2$ 的原函数是 $F(x)=x^3$，因为 $(x^3)\'=3x^2$。\n\n' +
          '但 $x^3+1$、$x^3-5$、$x^3+100$ 求导后全都等于 $3x^2$——' +
          '所以原函数**不唯一**，两个原函数之间只差一个常数。',
        formula: 'F\'(x) = f(x)',
        tip: '验证方法超简单：把你求出的 $F(x)$ 对 $x$ 求导，结果等于 $f(x)$ 就对了。这是万能检验。',
      },
      // 卡 3：不定积分的定义与记号
      {
        title: '不定积分的定义与记号',
        body:
          '$f(x)$ 的**全体原函数**称为它的不定积分，记作\n\n' +
          '$\\int f(x)\\,dx = F(x)+C$\n\n' +
          '记号说明：$\\int$ 是积分号，$f(x)$ 是被积函数，$dx$ 表示积分变量是 $x$，$C$ 是任意常数（**必须写！**）。\n\n' +
          '**具体例子**：$\\int 3x^2\\,dx = x^3+C$，因为 $(x^3+C)\'=3x^2$。',
        formula: '\\int f(x)\\,dx = F(x)+C',
        tip: '积分结果是一族曲线，不是一条——$C$ 取不同值，得到 $y$ 方向平移的同族曲线。漏写 $+C$ 是最常见失分点。',
      },
      // 卡 4：互逆关系
      {
        title: '积分与求导是互逆运算',
        body:
          '积分与求导互为逆运算，有两个精确的表述：\n\n' +
          '**先积分再求导**：$\\left(\\int f(x)\\,dx\\right)\'=f(x)$（积分后求导，还原被积函数）。\n\n' +
          '**先求导再积分**：$\\int F\'(x)\\,dx=F(x)+C$（求导后积分，还原原函数，差一个 $+C$）。\n\n' +
          '注意：先积分再求导，$C$ 的导数为 0，恰好消掉；先求导再积分，$C$ 无法恢复，所以结果中带 $+C$。',
        steps: [
          '先积分再求导：$\\left(\\int 2x\\,dx\\right)\'=\\left(x^2+C\\right)\'=2x$ ✓',
          '先求导再积分：$\\int (x^2)\'\\,dx=\\int 2x\\,dx=x^2+C$ ✓（多了 $+C$）',
        ],
      },
      // 卡 5：基本积分表（核心 6 条）
      {
        title: '基本积分表——核心 6 条',
        body:
          '每条积分公式都来自对应的导数公式"反着读"：\n\n' +
          '**幂函数**：$\\int x^n\\,dx=\\dfrac{x^{n+1}}{n+1}+C$（$n\\neq-1$）。\n\n' +
          '**倒数**：$\\int\\dfrac{1}{x}\\,dx=\\ln|x|+C$（绝对值不可漏，$x$ 取负数时也成立）。\n\n' +
          '**指数**：$\\int e^x\\,dx=e^x+C$（唯一"求导不变"的基本函数）。\n\n' +
          '**正弦 / 余弦**：$\\int\\cos x\\,dx=\\sin x+C$；$\\int\\sin x\\,dx=-\\cos x+C$（注意负号）。',
        formula:
          '\\int x^n\\,dx = \\frac{x^{n+1}}{n+1}+C \\quad (n\\neq -1)',
        tip: '$\\int\\sin x\\,dx=-\\cos x+C$ 中的负号经常被遗忘——记忆诀窍：对 $\\cos x$ 求导得 $-\\sin x$，加个负号就是 $\\sin x$，反过来读就行。',
      },
      // 卡 6：线性性质——拆开分别积
      {
        title: '不定积分的线性性质',
        body:
          '不定积分满足**线性性**：常数因子可以提到积分号外，两函数之和的积分等于积分之和。\n\n' +
          '用公式写：$\\int[\\alpha f(x)+\\beta g(x)]\\,dx=\\alpha\\int f(x)\\,dx+\\beta\\int g(x)\\,dx$。\n\n' +
          '**例**：$\\int(3x^2-2\\sin x+\\frac{1}{x})\\,dx=3\\cdot\\frac{x^3}{3}-2\\cdot(-\\cos x)+\\ln|x|+C=x^3+2\\cos x+\\ln|x|+C$。\n\n' +
          '**陷阱**：只有**常数**可以提出，**变量** $x$ 不能提出！$\\int x\\cdot f(x)\\,dx\\neq x\\int f(x)\\,dx$。',
        tip: '三项分别套公式，最后合并为一个 $+C$ 就够了，不用写三个 $+C_1,+C_2,+C_3$。',
      },
      // 卡 7：例题精讲 + 易错点 + reveal
      {
        title: '例题精讲与易错点',
        body:
          '**例**：求 $\\int\\left(x+\\dfrac{1}{x^2}\\right)dx$。\n\n' +
          '拆开：$\\int x\\,dx+\\int x^{-2}\\,dx$。\n\n' +
          '分别套幂函数公式（$n=1$ 和 $n=-2$）：\n\n' +
          '$\\dfrac{x^2}{2}+\\dfrac{x^{-1}}{-1}+C=\\dfrac{x^2}{2}-\\dfrac{1}{x}+C$。\n\n' +
          '**验证**：$\\left(\\dfrac{x^2}{2}-\\dfrac{1}{x}\\right)\'=x+x^{-2}=x+\\dfrac{1}{x^2}$ ✓',
        steps: [
          '识别：两项之和，套线性性质拆开。',
          '对 $x=x^1$ 用公式：$\\int x^1\\,dx=\\dfrac{x^2}{2}+C_1$。',
          '对 $\\frac{1}{x^2}=x^{-2}$ 用公式：$\\int x^{-2}\\,dx=\\dfrac{x^{-1}}{-1}+C_2=-\\dfrac{1}{x}+C_2$。',
          '合并：$\\dfrac{x^2}{2}-\\dfrac{1}{x}+C$，最后验证求导 ✓。',
        ],
        tip: '两大易错点：① $n=-2$ 时套公式得 $\\frac{x^{-1}}{-1}$，分母是 $-1$ 不是 $1$；② 合并后**务必写 $+C$**。',
        reveal: {
          q: '求 $\\int(2e^x-\\cos x)\\,dx$，结果是什么？',
          a: '$2e^x-\\sin x+C$。分别套公式：$\\int 2e^x\\,dx=2e^x+C_1$，$\\int\\cos x\\,dx=\\sin x+C_2$，合并得 $2e^x-\\sin x+C$。',
        },
      },
      // 卡 8：小结——思路图
      {
        title: '小结：求不定积分的思路',
        body:
          '遇到 $\\int f(x)\\,dx$，按以下顺序思考：\n\n' +
          '**第一步**：能否直接查基本公式表？（幂函数、指数、三角等）\n\n' +
          '**第二步**：能否先用线性性质拆开，再分别套公式？\n\n' +
          '**第三步**：如果都不行，等下一关学换元法和分部积分。\n\n' +
          '**万能检验**：求出结果 $F(x)+C$ 后，对 $x$ 求导，等于 $f(x)$ 即正确。',
        formula: '\\left(\\int f(x)\\,dx\\right)^{\\prime} = f(x)',
      },
    ],
    questions: [
      // 题 1：judge — 原函数定义
      {
        id: 'c14-a-q1',
        type: 'judge',
        prompt:
          '若 $F\'(x)=f(x)$，则 $F(x)$ 是 $f(x)$ 的一个原函数。',
        answer: true,
        explain:
          '原函数的定义就是：$F\'(x)=f(x)$ 成立，则 $F$ 是 $f$ 的原函数。这是基本定义，直接成立。',
      },
      // 题 2：choice — 原函数非唯一性
      {
        id: 'c14-a-q2',
        type: 'choice',
        prompt:
          '$f(x)=3x^2$ 的原函数中，下列哪一个**不是**它的原函数？',
        options: [
          '$x^3$',
          '$x^3+1$',
          '$x^3-2024$',
          '$3x^3$',
        ],
        answer: 3,
        explain:
          '$(3x^3)\'=9x^2\\neq 3x^2$，所以 $3x^3$ 不是 $3x^2$ 的原函数。前三项求导都等于 $3x^2$，均为原函数。',
      },
      // 题 3：input — 幂函数积分计算
      {
        id: 'c14-a-q3',
        type: 'input',
        prompt:
          '求 $\\int x^4\\,dx=\\dfrac{x^n}{n}+C$，其中指数 $n$ 等于多少？',
        accept: ['5'],
        placeholder: '输入整数',
        explain:
          '套公式 $\\int x^4\\,dx=\\dfrac{x^{4+1}}{4+1}+C=\\dfrac{x^5}{5}+C$，所以分子指数 $n=5$。',
      },
      // 题 4：judge — 常数提出与变量提出的区别
      {
        id: 'c14-a-q4',
        type: 'judge',
        prompt:
          '$\\int x\\sin x\\,dx = x\\int\\sin x\\,dx$，等式成立。',
        answer: false,
        explain:
          '错误！只有**常数**才能提到积分号外。$x$ 是变量，不能提出。正确做法是分部积分（后续关卡），等式右边 $x\\int\\sin x\\,dx=x(-\\cos x+C)$ 并不等于 $\\int x\\sin x\\,dx$。',
      },
      // 题 5：choice — 指数函数积分
      {
        id: 'c14-a-q5',
        type: 'choice',
        prompt: '$\\int e^x\\,dx$ 的结果是？',
        options: [
          '$xe^{x-1}+C$',
          '$e^x+C$',
          '$\\dfrac{e^x}{x}+C$',
          '$e^{x+1}+C$',
        ],
        answer: 1,
        explain:
          '$e^x$ 的导数还是 $e^x$，因此 $e^x$ 就是它自己的原函数。$\\int e^x\\,dx=e^x+C$，不需要加指数或除以 $x$。',
      },
      // 题 6：input — 线性性质计算
      {
        id: 'c14-a-q6',
        type: 'input',
        prompt:
          '$\\int(2x-\\cos x)\\,dx = x^2 - \\sin x + C$ 中，$x^2$ 前的系数是多少？（提示：先拆开再套公式）',
        accept: ['1', '1.0'],
        placeholder: '输入整数',
        explain:
          '$\\int(2x-\\cos x)\\,dx=2\\cdot\\dfrac{x^2}{2}-\\sin x+C=x^2-\\sin x+C$，$x^2$ 的系数是 $1$。',
      },
      // 题 7：match — 基本积分公式配对
      {
        id: 'c14-a-q7',
        type: 'match',
        prompt: '将被积函数与对应的不定积分结果配对（不含 $+C$）。',
        left: [
          '$\\cos x$',
          '$\\sin x$',
          '$\\dfrac{1}{x}$',
          '$e^x$',
        ],
        right: [
          '$\\sin x$',
          '$-\\cos x$',
          '$\\ln|x|$',
          '$e^x$',
        ],
      },
      // 题 8：judge — 绝对值
      {
        id: 'c14-a-q8',
        type: 'judge',
        prompt:
          '$\\int\\dfrac{1}{x}\\,dx = \\ln x + C$（不加绝对值）在 $x<0$ 时仍然正确。',
        answer: false,
        explain:
          '当 $x<0$ 时，$\\ln x$ 无意义。正确写法是 $\\ln|x|+C$，加绝对值后对 $x>0$ 和 $x<0$ 都成立：$(\\ln|x|)\'=\\dfrac{1}{x}$。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────────────
  // 关 c14-b  第一换元法（凑微分法）
  // ─────────────────────────────────────────────────────────────────────────
  {
    id: 'c14-b',
    title: '凑微分法',
    subtitle: '第一换元法·链式法则的逆运算·常见凑微分技巧',
    intro: [
      // 卡 1：动机——表里有复合结构
      {
        title: '为什么基本公式不够用？',
        body:
          '求 $\\int\\cos 2x\\,dx$，公式表里只有 $\\int\\cos x\\,dx=\\sin x+C$。但这里是 $\\cos 2x$，不是 $\\cos x$——直接套会得到错误答案 $\\sin 2x+C$（验证：$(\\sin 2x)\'=2\\cos 2x\\neq\\cos 2x$）。\n\n' +
          '问题出在：$\\cos 2x$ 是**复合函数**（外层 $\\cos$，内层 $2x$），求导时链式法则会多出一个 $2$，积分就要补偿这个因子。\n\n' +
          '**凑微分法**正是链式法则的逆运算，专门处理这类复合结构。',
        tip: '判断信号：被积函数像 $f(g(x))\\cdot g\'(x)$ 的形式——外函数套着内函数，内函数的导数同时出现。',
      },
      // 卡 2：定理与核心思想
      {
        title: '第一换元法（凑微分）的定理',
        body:
          '**定理**：若 $\\int f(u)\\,du=F(u)+C$，$u=\\varphi(x)$ 连续可导，则\n\n' +
          '$\\int f(\\varphi(x))\\cdot\\varphi\'(x)\\,dx=F(\\varphi(x))+C$\n\n' +
          '**核心技巧**：把 $\\varphi\'(x)\\,dx$ 改写成 $d(\\varphi(x))$，然后整体换元 $u=\\varphi(x)$，对 $u$ 积分，最后回代。\n\n' +
          '**理论本质**：链式法则说"对 $F(\\varphi(x))$ 求导等于 $f(\\varphi(x))\\cdot\\varphi\'(x)$"；反过来读，就是第一换元法。',
        formula:
          '\\int f(\\varphi(x))\\cdot\\varphi\'(x)\\,dx = F(\\varphi(x))+C',
      },
      // 卡 3：第一个例子——cos 2x
      {
        title: '例 1：$\\int\\cos 2x\\,dx$（找配比系数）',
        body:
          '**思路**：被积函数 $\\cos 2x$ 的内层是 $2x$，其导数是 $2$。被积式里没有 $2$，需要手动"凑"出来。\n\n' +
          '技巧：$dx=\\dfrac{1}{2}\\cdot 2\\,dx=\\dfrac{1}{2}d(2x)$——在积分号外补 $\\dfrac{1}{2}$，积分号内补 $2$。',
        steps: [
          '改写：$\\int\\cos 2x\\,dx=\\dfrac{1}{2}\\int\\cos 2x\\cdot 2\\,dx$',
          '识别 $d(2x)=2\\,dx$，换元写法：$=\\dfrac{1}{2}\\int\\cos 2x\\,d(2x)$',
          '令 $u=2x$，套公式：$=\\dfrac{1}{2}\\int\\cos u\\,du=\\dfrac{1}{2}\\sin u+C$',
          '回代 $u=2x$：$=\\dfrac{1}{2}\\sin 2x+C$',
          '验证：$\\left(\\dfrac{1}{2}\\sin 2x\\right)\'=\\dfrac{1}{2}\\cdot 2\\cos 2x=\\cos 2x$ ✓',
        ],
        tip: '"在积分号外乘以系数的倒数，积分号内乘以该系数"——这叫**配系数**，是凑微分的最常用操作。',
      },
      // 卡 4：第二个例子——x·e^(x^2)
      {
        title: '例 2：$\\int xe^{x^2}\\,dx$（内层导数恰好出现）',
        body:
          '**识别结构**：内层函数是 $x^2$，其导数是 $2x$；外面有 $x$，差一个系数 $2$。这是凑微分的完美场景。\n\n' +
          '**为什么 $\\int e^{x^2}\\,dx$ 没有初等原函数，而 $\\int xe^{x^2}\\,dx$ 有？** 因为多了 $x$ 恰好能"配出" $d(x^2)=2x\\,dx$，从而化为 $\\int e^u\\,du$。',
        steps: [
          '注意到 $x\\,dx=\\dfrac{1}{2}\\cdot 2x\\,dx=\\dfrac{1}{2}d(x^2)$',
          '改写：$\\int xe^{x^2}\\,dx=\\dfrac{1}{2}\\int e^{x^2}\\cdot 2x\\,dx=\\dfrac{1}{2}\\int e^{x^2}\\,d(x^2)$',
          '令 $u=x^2$：$=\\dfrac{1}{2}\\int e^u\\,du=\\dfrac{1}{2}e^u+C$',
          '回代：$=\\dfrac{1}{2}e^{x^2}+C$',
          '验证：$\\left(\\dfrac{1}{2}e^{x^2}\\right)\'=\\dfrac{1}{2}\\cdot 2x\\cdot e^{x^2}=xe^{x^2}$ ✓',
        ],
      },
      // 卡 5：第三个例子——tan x（改写分子分母）
      {
        title: '例 3：$\\int\\tan x\\,dx$（变形后凑微分）',
        body:
          '被积函数 $\\tan x=\\dfrac{\\sin x}{\\cos x}$ 不在基本公式表里，需要先变形再识别结构。\n\n' +
          '关键观察：$d(\\cos x)=-\\sin x\\,dx$，即 $\\sin x\\,dx=-d(\\cos x)$——分子恰好是分母 $\\cos x$ 导数的相反数！',
        steps: [
          '改写：$\\int\\tan x\\,dx=\\int\\dfrac{\\sin x}{\\cos x}\\,dx$',
          '凑微分：$=-\\int\\dfrac{1}{\\cos x}\\cdot(-\\sin x)\\,dx=-\\int\\dfrac{1}{\\cos x}\\,d(\\cos x)$',
          '令 $u=\\cos x$：$=-\\int\\dfrac{1}{u}\\,du=-\\ln|u|+C$',
          '回代：$=-\\ln|\\cos x|+C=\\ln|\\sec x|+C$',
          '验证：$(-\\ln|\\cos x|)\'=\\dfrac{\\sin x}{\\cos x}=\\tan x$ ✓',
        ],
        tip: '见到 $\\int\\dfrac{g\'(x)}{g(x)}\\,dx$ 的形式，立刻想到 $\\int\\dfrac{d(g(x))}{g(x)}=\\ln|g(x)|+C$。这是凑微分的高频模式。',
      },
      // 卡 6：常用凑微分识别清单
      {
        title: '凑微分识别清单',
        body:
          '记住以下"看到……立刻凑成……"的映射，可以秒判断：\n\n' +
          '**$x^{n-1}\\,dx\\to\\frac{1}{n}d(x^n)$**（含幂函数的内层）\n\n' +
          '**$\\frac{1}{x}\\,dx\\to d(\\ln|x|)$**（倒数乘 $dx$）\n\n' +
          '**$e^x\\,dx\\to d(e^x)$**（指数函数）\n\n' +
          '**$\\cos x\\,dx\\to d(\\sin x)$**；**$\\sin x\\,dx\\to -d(\\cos x)$**（三角）\n\n' +
          '**$\\frac{dx}{1+x^2}\\to d(\\arctan x)$**；**$\\frac{dx}{\\sqrt{1-x^2}}\\to d(\\arcsin x)$**（反三角）',
        tip: '判断"能否凑微分"的本质：外面有没有内层函数导数的倍数。有就凑，没有就换其它方法。',
        reveal: {
          q: '判断：$\\int x^2 e^{x^3}\\,dx$ 能用凑微分法吗？凑微分后 $u$ 是什么？',
          a: '能！内层是 $x^3$，其导数是 $3x^2$，外面有 $x^2$，差系数 $3$。凑微分：$x^2\\,dx=\\frac{1}{3}d(x^3)$，令 $u=x^3$，得 $\\frac{1}{3}e^{x^3}+C$。',
        },
      },
      // 卡 7：易错点总结
      {
        title: '凑微分的易错点',
        body:
          '**易错 1：系数没配平**。$\\int\\cos 3x\\,dx=\\dfrac{1}{3}\\sin 3x+C$，不是 $\\sin 3x+C$。验证：$(\\frac{1}{3}\\sin 3x)\'=\\cos 3x$ ✓；$(\\sin 3x)\'=3\\cos 3x$ ✗。\n\n' +
          '**易错 2：内层导数凑不出**。$\\int x^2 e^{x^3}\\,dx$ 能凑（外面有 $x^2$，内层 $x^3$ 导数含 $x^2$）；但 $\\int e^{x^2}\\,dx$ 不能凑（外面没有 $x$）——后者没有初等原函数。\n\n' +
          '**易错 3：换元后忘记回代**。用 $u=\\varphi(x)$ 换元后，最终结果必须把 $u$ 换回 $x$ 的表达式。\n\n' +
          '**易错 4：漏 $+C$**。',
        tip: '每做完一道，**求导验证**一遍——这比背错误习惯强多了。',
      },
    ],
    questions: [
      // 题 1：choice — 识别凑微分结构
      {
        id: 'c14-b-q1',
        type: 'choice',
        prompt:
          '对 $\\int 2x\\cdot e^{x^2}\\,dx$，用凑微分令 $u=x^2$，则 $du=$？',
        options: [
          '$dx$',
          '$2x\\,dx$',
          '$x^2\\,dx$',
          '$2\\,dx$',
        ],
        answer: 1,
        explain:
          '$u=x^2$，对 $x$ 求导：$\\dfrac{du}{dx}=2x$，故 $du=2x\\,dx$。正好与被积式中的 $2x\\,dx$ 吻合，完美凑微分。',
      },
      // 题 2：judge — 系数配平
      {
        id: 'c14-b-q2',
        type: 'judge',
        prompt:
          '$\\int\\sin 5x\\,dx = \\cos 5x + C$（等式成立）。',
        answer: false,
        explain:
          '错误。正确结果是 $-\\dfrac{1}{5}\\cos 5x+C$。验证：$\\left(-\\dfrac{1}{5}\\cos 5x\\right)\'=\\dfrac{1}{5}\\cdot 5\\sin 5x=\\sin 5x$ ✓。不配系数直接写 $\\cos 5x+C$ 求导会得 $-5\\sin 5x$，差了因子 $5$。',
      },
      // 题 3：input — tan x 积分系数
      {
        id: 'c14-b-q3',
        type: 'input',
        prompt:
          '$\\int\\tan x\\,dx = -\\ln|\\cos x|+C$。请问若改求 $\\int\\tan 2x\\,dx$，结果中的系数是多少？（即结果为"系数$\\times(-\\ln|\\cos 2x|)+C$"）',
        accept: ['1/2', '0.5'],
        placeholder: '输入分数或小数',
        explain:
          '$\\int\\tan 2x\\,dx$：令 $u=2x$，$du=2\\,dx$，故 $dx=\\dfrac{du}{2}$。$=\\dfrac{1}{2}\\int\\tan u\\,du=\\dfrac{1}{2}\\cdot(-\\ln|\\cos u|)+C=-\\dfrac{1}{2}\\ln|\\cos 2x|+C$，系数为 $\\dfrac{1}{2}$。',
      },
      // 题 4：choice — 哪个能凑微分
      {
        id: 'c14-b-q4',
        type: 'choice',
        prompt:
          '下列四个不定积分中，哪一个最适合直接用凑微分法（第一换元法）？',
        options: [
          '$\\int e^{x^2}\\,dx$',
          '$\\int x\\ln x\\,dx$',
          '$\\int x\\cdot e^{x^2}\\,dx$',
          '$\\int\\dfrac{1}{\\sqrt{1-x^2}}\\,dx$',
        ],
        answer: 2,
        explain:
          '$\\int xe^{x^2}\\,dx$：内层 $x^2$ 导数为 $2x$，外面有 $x$，可凑 $\\frac{1}{2}d(x^2)$，完美适用。\n\n' +
          '$e^{x^2}$ 没有初等原函数；$x\\ln x$ 需要分部积分；$\\frac{1}{\\sqrt{1-x^2}}$ 直接查公式表是 $\\arcsin x+C$。',
      },
      // 题 5：input — 计算具体结果
      {
        id: 'c14-b-q5',
        type: 'input',
        prompt:
          '求 $\\int\\dfrac{x}{1+x^2}\\,dx$。结果形如 $\\dfrac{1}{2}\\ln(1+x^2)+C$，分母中的系数是多少？（即 $\\int\\dfrac{x}{1+x^2}\\,dx=\\frac{1}{?}\\ln(1+x^2)+C$，填写 $?$）',
        accept: ['2'],
        placeholder: '输入整数',
        explain:
          '令 $u=1+x^2$，$du=2x\\,dx$，故 $x\\,dx=\\frac{1}{2}du$。原式 $=\\dfrac{1}{2}\\int\\dfrac{du}{u}=\\dfrac{1}{2}\\ln|u|+C=\\dfrac{1}{2}\\ln(1+x^2)+C$（$1+x^2>0$ 去绝对值）。分母系数是 $2$。',
      },
      // 题 6：judge — 凑微分链式逆运算
      {
        id: 'c14-b-q6',
        type: 'judge',
        prompt:
          '第一换元法（凑微分法）的理论本质是链式求导法则的逆运算。',
        answer: true,
        explain:
          '正确。链式法则：$\\dfrac{d}{dx}F(\\varphi(x))=F\'(\\varphi(x))\\cdot\\varphi\'(x)$。反过来读：若被积函数形如 $f(\\varphi(x))\\cdot\\varphi\'(x)$，其原函数就是 $F(\\varphi(x))$——这正是第一换元法。',
      },
      // 题 7：match — 被积函数与凑出的微分配对
      {
        id: 'c14-b-q7',
        type: 'match',
        prompt: '将左列被积式中可"凑出"的微分，与右列配对。',
        left: [
          '$x^3\\,dx$（内层 $x^4$）',
          '$\\cos x\\,dx$',
          '$e^x\\,dx$',
          '$\\dfrac{1}{x}\\,dx$',
        ],
        right: [
          '$\\dfrac{1}{4}d(x^4)$',
          '$d(\\sin x)$',
          '$d(e^x)$',
          '$d(\\ln|x|)$',
        ],
      },
      // 题 8：choice — 验证结果正确性
      {
        id: 'c14-b-q8',
        type: 'choice',
        prompt:
          '有人算出 $\\int\\cos(3x+1)\\,dx=\\sin(3x+1)+C$，通过求导验证，这个结果是否正确？',
        options: [
          '正确，$(\\sin(3x+1))\'=\\cos(3x+1)$',
          '错误，$(\\sin(3x+1))\'=3\\cos(3x+1)$，应为 $\\dfrac{1}{3}\\sin(3x+1)+C$',
          '错误，应为 $-\\sin(3x+1)+C$',
          '正确，因为 $\\cos x$ 的原函数是 $\\sin x$',
        ],
        answer: 1,
        explain:
          '$(\\sin(3x+1))\'=\\cos(3x+1)\\cdot 3=3\\cos(3x+1)\\neq\\cos(3x+1)$，结果**错误**。内层 $3x+1$ 对 $x$ 求导得 $3$，需配系数：正确结果是 $\\dfrac{1}{3}\\sin(3x+1)+C$。验证：$\\left(\\dfrac{1}{3}\\sin(3x+1)\\right)\'=\\dfrac{1}{3}\\cdot 3\\cos(3x+1)=\\cos(3x+1)$ ✓。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────────────
  // 关 c14-c  分部积分法
  // ─────────────────────────────────────────────────────────────────────────
  {
    id: 'c14-c',
    title: '分部积分法',
    subtitle: '乘积法则的逆运算·LIATE 选 $u$·循环积分',
    intro: [
      // 卡 1：动机——两函数相乘，凑微分失败
      {
        title: '凑微分搞不定的情形',
        body:
          '求 $\\int x e^x\\,dx$。被积函数是 $x$（多项式）与 $e^x$（指数）的乘积。\n\n' +
          '尝试凑微分：内层候选有 $x$，其导数是 $1$，但还剩一个 $e^x$，无法化为 $\\int f(u)\\,du$ 的形式。\n\n' +
          '根本原因：**积分没有乘积法则**（$(uv)\'=u\'v+uv\'$，但积分拆不回去）。' +
          '不过，乘积法则可以被**反向利用**——这就是分部积分法。',
        tip: '见到"多项式 × 指数"、"多项式 × 三角"、"对数 × 幂函数"这类乘积——凑微分失败时，立刻想到分部积分。',
      },
      // 卡 2：公式推导
      {
        title: '分部积分公式推导',
        body:
          '从乘积法则出发：$(uv)\'=u\'v+uv\'$，移项得 $uv\'=(uv)\'-u\'v$，两边积分：\n\n' +
          '$\\int uv\'\\,dx=uv-\\int u\'v\\,dx$\n\n' +
          '用微分形式改写（令 $dv=v\'\\,dx$，$du=u\'\\,dx$）：\n\n' +
          '$\\int u\\,dv=uv-\\int v\\,du$\n\n' +
          '**思想**：把"难积分"的 $\\int u\\,dv$ 转化为"更容易积分"的 $\\int v\\,du$。',
        formula: '\\int u\\,dv = uv - \\int v\\,du',
      },
      // 卡 3：LIATE 法则——选 u 的优先顺序
      {
        title: 'LIATE 法则：怎么选 $u$？',
        body:
          '分部积分的关键是选哪部分作 $u$，哪部分作 $dv$。LIATE 给出优先顺序（越靠前优先作 $u$）：\n\n' +
          '**L**：对数函数（$\\ln x$）\n\n' +
          '**I**：反三角函数（$\\arctan x$，$\\arcsin x$）\n\n' +
          '**A**：代数/多项式（$x^n$）\n\n' +
          '**T**：三角函数（$\\sin x$，$\\cos x$）\n\n' +
          '**E**：指数函数（$e^x$，$a^x$）\n\n' +
          '**原理**：把"求导后变简单的"（如 $\\ln x$）选为 $u$；把"积分后形式不变的"（如 $e^x$）选为 $dv$。',
        tip: '口诀记忆：**"对反幂三指"**——对数、反三角、幂函数、三角、指数，前面的优先作 $u$。',
      },
      // 卡 4：例 1——x·e^x
      {
        title: '例 1：$\\int xe^x\\,dx$（多项式 × 指数）',
        body:
          'LIATE：$x$ 是多项式（A），$e^x$ 是指数（E），A 优先于 E，故取 $u=x$，$dv=e^x\\,dx$。',
        steps: [
          '令 $u=x$，$dv=e^x\\,dx$',
          '求 $du=dx$，$v=e^x$',
          '代入公式：$\\int xe^x\\,dx=xe^x-\\int e^x\\,dx$',
          '$=xe^x-e^x+C=(x-1)e^x+C$',
          '验证：$[(x-1)e^x]\'=e^x+(x-1)e^x=xe^x$ ✓',
        ],
        tip: '求 $v$ 时只需一个原函数，不加 $+C$；最后统一加一个 $+C$。',
      },
      // 卡 5：例 2——ln x（对数独立出现）
      {
        title: '例 2：$\\int\\ln x\\,dx$（对数函数）',
        body:
          '只有 $\\ln x$，看似没有"第二个函数"，实际上可以写成 $\\ln x\\cdot 1$，取 $u=\\ln x$，$dv=dx$（即 $v=x$）。',
        steps: [
          '令 $u=\\ln x$，$dv=dx$',
          '求 $du=\\dfrac{1}{x}dx$，$v=x$',
          '代入公式：$\\int\\ln x\\,dx=x\\ln x-\\int x\\cdot\\dfrac{1}{x}\\,dx$',
          '$=x\\ln x-\\int 1\\,dx=x\\ln x-x+C$',
          '验证：$(x\\ln x-x)\'=\\ln x+1-1=\\ln x$ ✓',
        ],
        formula: '\\int \\ln x\\,dx = x\\ln x - x + C',
      },
      // 卡 6：例 3——e^x cos x（循环积分）
      {
        title: '例 3：$\\int e^x\\cos x\\,dx$（循环积分）',
        body:
          '两次分部积分后，原积分 $I$ 会"循环"出现——这时不要继续分部，而是**解方程**。\n\n' +
          '设 $I=\\int e^x\\cos x\\,dx$，取 $u=\\cos x$，$dv=e^x\\,dx$：',
        steps: [
          '第一次分部：$I=e^x\\cos x+\\int e^x\\sin x\\,dx$',
          '对 $\\int e^x\\sin x\\,dx$，取 $u=\\sin x$，$dv=e^x\\,dx$：',
          '$\\int e^x\\sin x\\,dx=e^x\\sin x-\\int e^x\\cos x\\,dx=e^x\\sin x-I$',
          '代入：$I=e^x\\cos x+e^x\\sin x-I$',
          '解方程：$2I=e^x(\\cos x+\\sin x)$，故 $I=\\dfrac{e^x(\\cos x+\\sin x)}{2}+C$',
        ],
        formula: '\\int e^x\\cos x\\,dx = \\frac{e^x(\\cos x+\\sin x)}{2}+C',
        tip: '识别循环积分的信号：**两次分部后被积函数形式不变**（出现了 $-I$）。此时立刻列方程，不要第三次分部。',
      },
      // 卡 7：总结与选方法决策
      {
        title: '小结：凑微分 vs 分部积分',
        body:
          '拿到 $\\int f(x)\\,dx$，快速判断：\n\n' +
          '**先看**：能否查基本公式表？$\\to$ 直接套。\n\n' +
          '**再看**：有没有 $f(g(x))\\cdot g\'(x)$ 的结构？$\\to$ 凑微分。\n\n' +
          '**最后看**：两类函数相乘（LIATE 类型）？$\\to$ 分部积分。\n\n' +
          '**循环出现**：两次分部后 $I$ 再现 $\\to$ 解方程。\n\n' +
          '**万能检验**：结果求导等于被积函数才放心。',
        reveal: {
          q: '$\\int x^2 e^x\\,dx$ 需要几次分部积分？最终结果是什么？',
          a: '两次。第一次：$u=x^2$，$dv=e^x\\,dx$，得 $x^2e^x-2\\int xe^x\\,dx$；第二次：$u=x$，$dv=e^x\\,dx$，得 $(x-1)e^x$。合并：$\\int x^2e^x\\,dx=(x^2-2x+2)e^x+C$。',
        },
      },
    ],
    questions: [
      // 题 1：judge — 分部积分来源
      {
        id: 'c14-c-q1',
        type: 'judge',
        prompt:
          '分部积分公式 $\\int u\\,dv=uv-\\int v\\,du$ 来自**乘积求导法则**的逆运算。',
        answer: true,
        explain:
          '正确。乘积法则 $(uv)\'=u\'v+uv\'$，对 $uv\'=(uv)\'-u\'v$ 两边积分，即得 $\\int uv\'\\,dx=uv-\\int u\'v\\,dx$，改写为微分形式就是 $\\int u\\,dv=uv-\\int v\\,du$。',
      },
      // 题 2：choice — LIATE 选 u
      {
        id: 'c14-c-q2',
        type: 'choice',
        prompt:
          '用分部积分计算 $\\int x\\cos x\\,dx$，按 LIATE 应选哪个作 $u$？',
        options: [
          '$u=\\cos x$',
          '$u=x$',
          '$u=x\\cos x$',
          '$u=1$',
        ],
        answer: 1,
        explain:
          'LIATE：$x$ 是多项式（A），$\\cos x$ 是三角（T）。A 优先于 T，故取 $u=x$，$dv=\\cos x\\,dx$。代入：$\\int x\\cos x\\,dx=x\\sin x-\\int\\sin x\\,dx=x\\sin x+\\cos x+C$。',
      },
      // 题 3：input — 计算 ∫x·e^x dx
      {
        id: 'c14-c-q3',
        type: 'input',
        prompt:
          '$\\int xe^x\\,dx=(x-1)e^x+C$。验证：对 $(x-1)e^x$ 求导，结果等于 $xe^x$，其中 $x$ 前的系数是多少？',
        accept: ['1'],
        placeholder: '输入整数',
        explain:
          '$[(x-1)e^x]\'=e^x+(x-1)e^x=[1+(x-1)]e^x=xe^x$，系数为 $1$，验证正确。',
      },
      // 题 4：choice — ∫ln x dx
      {
        id: 'c14-c-q4',
        type: 'choice',
        prompt: '$\\int\\ln x\\,dx$ 的结果是？',
        options: [
          '$\\dfrac{\\ln^2 x}{2}+C$',
          '$\\dfrac{1}{x}+C$',
          '$x\\ln x - x + C$',
          '$x\\ln x + C$',
        ],
        answer: 2,
        explain:
          '取 $u=\\ln x$，$dv=dx$，则 $du=\\dfrac{dx}{x}$，$v=x$。$\\int\\ln x\\,dx=x\\ln x-\\int x\\cdot\\dfrac{1}{x}\\,dx=x\\ln x-x+C$。少写 $-x$ 是常见错误。',
      },
      // 题 5：judge — 循环积分解方程
      {
        id: 'c14-c-q5',
        type: 'judge',
        prompt:
          '若两次分部积分后出现 $I=\\cdots-I$，则应再做一次分部积分来消去 $I$。',
        answer: false,
        explain:
          '错误。出现 $I=\\cdots-I$ 说明积分循环，此时应**直接解方程**：$2I=\\cdots$，$I=\\cdots/2+C$。再次分部只会产生更复杂的结果，不会消去 $I$。',
      },
      // 题 6：input — 循环积分系数
      {
        id: 'c14-c-q6',
        type: 'input',
        prompt:
          '$\\int e^x\\sin x\\,dx = \\dfrac{e^x(\\sin x-\\cos x)}{2}+C$。若改求 $\\int e^x\\cos x\\,dx$，分母中的数字是多少？',
        accept: ['2'],
        placeholder: '输入整数',
        explain:
          '$\\int e^x\\cos x\\,dx=\\dfrac{e^x(\\cos x+\\sin x)}{2}+C$，分母同样是 $2$。两个循环积分的推导结构对称，分母都是 $2$（因为两次分部后 $I$ 系数为 $1+1=2$）。',
      },
      // 题 7：match — 被积函数与选 u 配对
      {
        id: 'c14-c-q7',
        type: 'match',
        prompt: '按 LIATE 法则，将被积函数与应选的 $u$ 配对。',
        left: [
          '$x\\ln x$',
          '$x\\sin x$',
          '$e^x\\cos x$',
          '$x^2 e^x$',
        ],
        right: [
          '$u=\\ln x$（L 优先）',
          '$u=x$（A 优先于 T）',
          '$u=\\cos x$（T 优先于 E）',
          '$u=x^2$（A 优先于 E）',
        ],
      },
      // 题 8：choice — 综合判断用哪种方法
      {
        id: 'c14-c-q8',
        type: 'choice',
        prompt:
          '下列四个积分中，哪一个用**分部积分法**最合适？',
        options: [
          '$\\int x^2\\cdot e^{x^3}\\,dx$',
          '$\\int\\dfrac{\\cos x}{\\sin x}\\,dx$',
          '$\\int x\\arctan x\\,dx$',
          '$\\int e^{3x}\\,dx$',
        ],
        answer: 2,
        explain:
          '$\\int x\\arctan x\\,dx$：反三角函数（I）× 多项式（A），LIATE 选 $u=\\arctan x$，$dv=x\\,dx$，适合分部积分。\n\n' +
          '其余：$\\int x^2 e^{x^3}\\,dx$ 凑微分（$d(x^3)=3x^2\\,dx$）；$\\int\\dfrac{\\cos x}{\\sin x}\\,dx$ 凑微分（$d(\\sin x)=\\cos x\\,dx$）；$\\int e^{3x}\\,dx$ 直接凑微分。',
      },
    ],
  },
]
