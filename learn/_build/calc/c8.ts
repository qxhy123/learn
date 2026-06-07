export const LESSONS = [
  // ─────────────────────────────────────────
  // 关 c8-a  函数极限的直观与 ε-δ 定义
  // ─────────────────────────────────────────
  {
    id: 'c8-a',
    title: '函数极限与 ε-δ 定义',
    subtitle: '趋近的精确含义·去心邻域·三步证明法',
    intro: [
      // 卡 1：动机——为什么要精确化"趋近"
      {
        title: '为什么要精确定义"趋近"？',
        body:
          '说"$x$ 趋于 2 时，$f(x)=3x-1$ 趋于 5"，直觉上没有问题——把 $x=1.9,1.99,1.999$ 代进去，函数值确实越来越靠近 5。\n\n' +
          '但"越来越靠近"本身是模糊的。"靠近到什么程度？"如果精度要求是 $0.001$，能保证做到吗？如果要求是 $0.000001$ 呢？\n\n' +
          '**ε-δ 定义**就是把这个模糊的"趋近"变成精确的数学语言：无论精度要求 $\\varepsilon$ 多么苛刻，只要 $x$ 离 $x_0$ 足够近（近到某个 $\\delta$ 以内），函数值就能满足精度要求。\n\n' +
          '这是分析学的基石，也是理解"极限"真正含义的关键一步。',
        tip: '极限只关心 $x$ 趋近于 $x_0$ 的过程，$x_0$ 处函数值有没有定义无所谓。',
      },
      // 卡 2：ε-δ 定义的精确表述
      {
        title: 'ε-δ 定义：什么叫"极限等于 L"',
        body:
          '设函数 $f(x)$ 在 $x_0$ 的某个**去心邻域**内有定义。\n\n' +
          '**定义**：若对任意给定的 $\\varepsilon > 0$，都能找到 $\\delta > 0$，使得只要 $0 < |x - x_0| < \\delta$，就有 $|f(x) - L| < \\varepsilon$，则称 $L$ 是 $f(x)$ 当 $x \\to x_0$ 时的极限。\n\n' +
          '**两个关键细节**：\n\n' +
          '① 条件 $0 < |x - x_0|$ 表明 $x \\neq x_0$，极限只管"趋近"，不管"到达"。\n\n' +
          '② $\\varepsilon$ 是任意给定的（挑战方设定），$\\delta$ 是随后找的（应答方响应）。$\\delta$ 通常依赖于 $\\varepsilon$——$\\varepsilon$ 越小，$\\delta$ 一般也越小。',
        formula:
          '\\lim_{x\\to x_0}f(x)=L\\iff\\forall\\varepsilon>0,\\,\\exists\\delta>0,\\,0<|x-x_0|<\\delta\\Rightarrow|f(x)-L|<\\varepsilon',
        tip: '几何理解：无论纵向精度带 $(L-\\varepsilon, L+\\varepsilon)$ 多窄，总能找到横向区间 $(x_0-\\delta,x_0+\\delta)$，使函数值都落进精度带。',
      },
      // 卡 3：找 δ 的核心技巧——反向估计
      {
        title: '找 δ 的技巧：从结论反向解出',
        body:
          '很多同学卡在"$\\delta$ 怎么找"这一步——其实不是靠猜，而是**把目标不等式倒着推**。\n\n' +
          '目标是让 $|f(x) - L| < \\varepsilon$ 成立，把 $f(x)$ 代入、化简，直到写成 $C \\cdot |x - x_0| < \\varepsilon$ 的形式，则 $|x - x_0| < \\varepsilon / C$，于是 $\\delta = \\varepsilon / C$。\n\n' +
          '**具体例子**（$f(x) = 3x - 1$，$x_0 = 2$，$L = 5$）：\n\n' +
          '反向推：$|(3x-1)-5| = |3x-6| = 3|x-2| < \\varepsilon$，解得 $|x-2| < \\varepsilon/3$。\n\n' +
          '于是取 $\\delta = \\varepsilon/3$，验证时直接代回即可。\n\n' +
          '**口诀**：设 $\\varepsilon$ → 反推算 $\\delta$ → 代入验证，三步固定模板。',
        formula:
          '|f(x)-L|<\\varepsilon\\xRightarrow{\\text{反向推}}|x-x_0|<\\frac{\\varepsilon}{C}\\implies\\delta=\\frac{\\varepsilon}{C}',
      },
      // 卡 4：线性极限完整证明示范
      {
        title: '完整证明：$\\lim_{x\\to 2}(3x-1)=5$',
        body:
          '**第 1 步：设 $\\varepsilon > 0$ 任意给定。**\n\n' +
          '**第 2 步：反向找 $\\delta$。**\n\n' +
          '计算 $|(3x-1)-5| = |3x-6| = 3|x-2|$。\n\n' +
          '要使 $3|x-2| < \\varepsilon$，只需 $|x-2| < \\varepsilon/3$。取 $\\delta = \\varepsilon/3$。\n\n' +
          '**第 3 步：验证。**\n\n' +
          '若 $0 < |x-2| < \\delta = \\varepsilon/3$，则\n\n' +
          '$|(3x-1)-5| = 3|x-2| < 3 \\cdot \\varepsilon/3 = \\varepsilon$。\n\n' +
          '证毕。\n\n' +
          '**推广**：对 $f(x) = ax + b$，同理得 $\\delta = \\varepsilon / |a|$（$a \\neq 0$）。',
        steps: [
          '设任意 $\\varepsilon > 0$',
          '计算 $|f(x) - L|$，化简为 $C|x-x_0|$ 的形式',
          '令 $C|x-x_0| < \\varepsilon$，解出 $\\delta = \\varepsilon/C$',
          '假设 $0 < |x-x_0| < \\delta$，代回验证 $|f(x)-L| < \\varepsilon$',
        ],
      },
      // 卡 5：非线性情形——先限制再找 δ
      {
        title: '非线性极限：先限制 |x-x₀|，再处理',
        body:
          '当 $f(x)$ 含有 $x^2$ 等非线性项时，$|f(x) - L|$ 中会出现"可变因子"，不能直接除。\n\n' +
          '**例题**：证明 $\\lim_{x \\to 1} x^2 = 1$。\n\n' +
          '$|x^2 - 1| = |x-1| \\cdot |x+1|$。$|x+1|$ 随 $x$ 变化，不是常数。\n\n' +
          '**解决方案**：先限定 $|x-1| < 1$，得 $0 < x < 2$，从而 $|x+1| < 3$。\n\n' +
          '此时 $|x^2 - 1| < 3|x-1|$，解 $3|x-1| < \\varepsilon$ 得 $|x-1| < \\varepsilon/3$。\n\n' +
          '最终取 $\\delta = \\min\\{1,\\, \\varepsilon/3\\}$，两个条件同时满足。\n\n' +
          '**大原则**：含非线性项时，**先令 $|x-x_0| < 1$ 限制范围**，再处理剩余不等式。',
        formula:
          '\\delta = \\min\\!\\left\\{1,\\,\\frac{\\varepsilon}{3}\\right\\}',
        tip: '取 min 是"两个条件都要满足"的标准写法，缺一不可。',
      },
      // 卡 6：左极限、右极限与极限存在的充要条件
      {
        title: '左极限与右极限：极限存在的充要条件',
        body:
          '有时 $x$ 从左边趋近和从右边趋近的情况不同，需要分别讨论。\n\n' +
          '**左极限**（$x < x_0$，从左趋近）：$\\lim_{x \\to x_0^-} f(x)$，记作 $f(x_0^-)$。\n\n' +
          '**右极限**（$x > x_0$，从右趋近）：$\\lim_{x \\to x_0^+} f(x)$，记作 $f(x_0^+)$。\n\n' +
          '**充要条件**：$\\lim_{x \\to x_0} f(x) = L$ 当且仅当左极限和右极限都存在且都等于 $L$。\n\n' +
          '**反例**：符号函数 $\\text{sgn}(x)$ 在 $x=0$ 处，左极限为 $-1$，右极限为 $1$，不相等，故极限不存在。\n\n' +
          '**应用场景**：分段函数在分界点处的极限，必须用左右极限来判断。',
        formula:
          '\\lim_{x\\to x_0}f(x)=L\\iff\\lim_{x\\to x_0^-}f(x)=\\lim_{x\\to x_0^+}f(x)=L',
      },
      // 卡 7：易错点汇总与 reveal
      {
        title: '易错点汇总与自测',
        body:
          '**易错点一**：把极限值 $L$ 与函数值 $f(x_0)$ 混淆。极限只关心 $x$ 趋近 $x_0$ 时的行为，$f(x_0)$ 可以完全无关，甚至没有定义。\n\n' +
          '**易错点二**：$\\delta$ 中忘记写 $\\min\\{1, \\cdots\\}$。非线性情形中，两个限制条件缺一不可。\n\n' +
          '**易错点三**：左右极限不相等就说"极限等于某个值"。极限存在要求左右极限都存在且相等，两者缺一不可。\n\n' +
          '**易错点四**：证明顺序写反——先猜 $\\delta$（草稿），再按"设 $\\varepsilon$ → 取 $\\delta$ → 验证"的顺序写正式证明。',
        tip: 'ε-δ 证明的正式顺序：① 设 $\\varepsilon > 0$ → ② 给出 $\\delta$ → ③ 验证结论。草稿中的反向推导不出现在最终证明里。',
        reveal: {
          q: '用 ε-δ 定义证明 $\\lim_{x\\to 3}(2x+1)=7$，应取 $\\delta$ 等于多少？',
          a: '$|(2x+1)-7|=|2x-6|=2|x-3|<\\varepsilon$ 解得 $|x-3|<\\varepsilon/2$，取 $\\delta=\\varepsilon/2$。验证：$|(2x+1)-7|=2|x-3|<2\\cdot(\\varepsilon/2)=\\varepsilon$。',
        },
      },
      // 卡 8：典型例题串讲
      {
        title: '典型例题：$\\lim_{x\\to1}\\dfrac{x^2-1}{x-1}$',
        body:
          '这道题在 $x=1$ 处分母为 $0$，函数没有定义——但极限是否存在？\n\n' +
          '**关键观察**：极限只管 $x \\neq 1$ 时的行为，所以可以先约分：\n\n' +
          '$\\dfrac{x^2-1}{x-1} = \\dfrac{(x-1)(x+1)}{x-1} = x+1$（$x \\neq 1$）。\n\n' +
          '当 $x \\to 1$ 时，$x+1 \\to 2$，所以 $\\lim_{x\\to1}\\dfrac{x^2-1}{x-1}=2$。\n\n' +
          '**重要结论**：分子分母有公因子 $(x-a)$，且求的是 $x \\to a$ 的极限，**先约公因子再求极限**，是最常用的化简技巧。\n\n' +
          '这道题的函数值 $f(1)$ 没有定义，但极限值 $2$ 存在——完美体现了"极限不管函数值"。',
      },
    ],
    questions: [
      // 题 1：choice
      {
        id: 'c8-a-q1',
        type: 'choice',
        prompt:
          '函数 $f(x)=\\dfrac{x^2-4}{x-2}$ 在 $x=2$ 处没有定义，那么 $\\lim_{x\\to 2}f(x)$ 的值是？',
        options: ['不存在', '$2$', '$4$', '$0$'],
        answer: 2,
        explain:
          '$\\dfrac{x^2-4}{x-2}=\\dfrac{(x-2)(x+2)}{x-2}=x+2$（$x\\neq2$）。当 $x\\to2$ 时，$x+2\\to4$。极限与函数在该点有无定义无关。',
      },
      // 题 2：judge
      {
        id: 'c8-a-q2',
        type: 'judge',
        prompt: '$\\lim_{x\\to x_0}f(x)=L$ 要求 $f(x_0)$ 必须有定义且等于 $L$。',
        answer: false,
        explain:
          '极限只关心 $x$ 趋近 $x_0$（即 $0<|x-x_0|<\\delta$）时的行为，$f$ 在 $x_0$ 处可以没有定义，两者完全独立。',
      },
      // 题 3：input
      {
        id: 'c8-a-q3',
        type: 'input',
        prompt:
          '用 ε-δ 定义证明 $\\lim_{x\\to3}(2x+1)=7$ 时，应取 $\\delta=$（用 $\\varepsilon$ 表示，填分数形式分子，分母为 2）。分子是多少？',
        accept: ['1'],
        explain:
          '$|(2x+1)-7|=2|x-3|<\\varepsilon$，解得 $|x-3|<\\varepsilon/2$，故取 $\\delta=\\varepsilon/2$，分子为 $1$。',
      },
      // 题 4：choice
      {
        id: 'c8-a-q4',
        type: 'choice',
        prompt:
          '证明 $\\lim_{x\\to1}x^2=1$ 时，为控制非线性项 $|x+1|$，第一步通常先限制 $|x-1|<$？',
        options: ['$\\varepsilon$', '$\\delta$', '$1$', '$3$'],
        answer: 2,
        explain:
          '先令 $|x-1|<1$，得 $0<x<2$，从而 $|x+1|<3$，使得 $|x^2-1|=|x-1||x+1|<3|x-1|$。最终取 $\\delta=\\min\\{1,\\varepsilon/3\\}$。',
      },
      // 题 5：judge
      {
        id: 'c8-a-q5',
        type: 'judge',
        prompt:
          '若 $\\lim_{x\\to x_0^-}f(x)=3$ 且 $\\lim_{x\\to x_0^+}f(x)=3$，则 $\\lim_{x\\to x_0}f(x)=3$。',
        answer: true,
        explain:
          '极限存在的充要条件：左极限 $=$ 右极限 $=L$。此时两侧极限均为 $3$，故双侧极限存在且等于 $3$。',
      },
      // 题 6：match
      {
        id: 'c8-a-q6',
        type: 'match',
        prompt: '将 ε-δ 证明的三个步骤与其含义对应。',
        left: ['第 1 步：设 $\\varepsilon>0$ 任意给定', '第 2 步：反向估计找 $\\delta$', '第 3 步：验证'],
        right: [
          '表明精度要求是对手给定的，不能我们自己选',
          '从 $|f(x)-L|<\\varepsilon$ 倒推出 $\\delta$ 的大小',
          '假设 $0<|x-x_0|<\\delta$，代回证明 $|f(x)-L|<\\varepsilon$',
        ],
      },
      // 题 7：choice
      {
        id: 'c8-a-q7',
        type: 'choice',
        prompt:
          '已知分段函数 $f(x)=\\begin{cases}x+1,&x<0\\\\2x-1,&x\\geq0\\end{cases}$，则 $\\lim_{x\\to0}f(x)$？',
        options: ['$1$', '$-1$', '不存在', '$0$'],
        answer: 2,
        explain:
          '左极限：$\\lim_{x\\to0^-}(x+1)=1$；右极限：$\\lim_{x\\to0^+}(2x-1)=-1$。左右极限不等，故极限不存在。',
      },
      // 题 8：input
      {
        id: 'c8-a-q8',
        type: 'input',
        prompt:
          '计算 $\\lim_{x\\to2}\\dfrac{x^2-3x+2}{x-2}$（约分后直接代入，填整数）。',
        accept: ['-1', '-1.0'],
        explain:
          '$x^2-3x+2=(x-2)(x-1)$，约分得 $x-1$（$x\\neq2$）。令 $x\\to2$，结果为 $2-1=-1$。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c8-b  两个重要极限
  // ─────────────────────────────────────────
  {
    id: 'c8-b',
    title: '两个重要极限',
    subtitle: '$\\lim_{x\\to0}\\frac{\\sin x}{x}=1$ · $\\lim_{x\\to\\infty}(1+\\frac{1}{x})^x=e$ · 变形与应用',
    intro: [
      // 卡 1：第一重要极限的动机
      {
        title: '为什么 $\\sin x / x$ 在 $x\\to0$ 时趋于 1？',
        body:
          '直接代入 $x=0$：分子 $\\sin 0=0$，分母也是 $0$，得到 $0/0$ 型不定式，无法直接算。\n\n' +
          '但用计算器试试：$\\sin(0.1)/0.1 \\approx 0.9983$，$\\sin(0.01)/0.01 \\approx 0.99998$……越来越接近 1。\n\n' +
          '**直观解释**：当弧度 $x$ 很小时，圆弧长 $x$ 与对应的弦（$\\sin x$）几乎一样长，所以比值趋于 1。\n\n' +
          '这个极限是三角函数导数、一系列等价无穷小的出发点，也是"弧度制"之所以自然的数学根源。\n\n' +
          '**注意**：$x$ 必须是**弧度**，不是角度！用度数会得到完全不同的结果。',
        tip: '记忆口诀：小弧度下，弦长 $\\approx$ 弧长，所以 $\\sin x \\approx x$，比值趋于 1。',
      },
      // 卡 2：夹逼定理证明第一重要极限
      {
        title: '用夹逼定理证明 $\\lim_{x\\to0}\\dfrac{\\sin x}{x}=1$',
        body:
          '设 $0 < x < \\pi/2$，在单位圆中比较三个面积：\n\n' +
          '内接三角形面积 $S_1 = \\tfrac{1}{2}\\sin x$；扇形面积 $S_2 = \\tfrac{1}{2}x$；外切三角形面积 $S_3 = \\tfrac{1}{2}\\tan x$。\n\n' +
          '由几何关系 $S_1 < S_2 < S_3$：$\\sin x < x < \\tan x$。\n\n' +
          '各项除以 $\\sin x > 0$，再取倒数（不等号翻转）：\n\n' +
          '$\\cos x < \\dfrac{\\sin x}{x} < 1$。\n\n' +
          '当 $x \\to 0^+$ 时，$\\cos x \\to 1$，由**夹逼定理**得 $\\lim_{x\\to0^+}\\dfrac{\\sin x}{x}=1$。\n\n' +
          '又因为 $\\dfrac{\\sin x}{x}$ 是偶函数，左极限也等于 $1$，故整体极限为 $1$。',
        formula:
          '\\cos x < \\frac{\\sin x}{x} < 1\\xrightarrow{x\\to0}\\lim_{x\\to0}\\frac{\\sin x}{x}=1',
        steps: [
          '单位圆内比较面积：内接三角形 $<$ 扇形 $<$ 外切三角形',
          '化简为 $\\cos x < \\sin x / x < 1$',
          '$x\\to0^+$ 时左端 $\\cos x\\to1$，夹逼得极限为 $1$',
          '偶函数性质推出 $x\\to0^-$ 时同样等于 $1$',
        ],
      },
      // 卡 3：第一重要极限的变形与推论
      {
        title: '第一重要极限的变形与推论',
        body:
          '核心公式 $\\lim_{x\\to0}\\dfrac{\\sin x}{x}=1$ 有多种常见变形，必须熟练掌握。\n\n' +
          '**变形 1（换元法）**：$\\lim_{x\\to0}\\dfrac{\\sin 3x}{x}$——分子分母同乘 $3$：\n\n' +
          '$\\dfrac{\\sin 3x}{x} = 3 \\cdot \\dfrac{\\sin 3x}{3x}$，令 $t = 3x$，$t\\to0$，极限 $= 3 \\times 1 = 3$。\n\n' +
          '**变形 2（比值相乘）**：$\\lim_{x\\to0}\\dfrac{\\sin 5x}{\\sin 3x} = \\dfrac{\\sin 5x}{5x} \\cdot \\dfrac{3x}{\\sin 3x} \\cdot \\dfrac{5}{3} = 1 \\cdot 1 \\cdot \\dfrac{5}{3} = \\dfrac{5}{3}$。\n\n' +
          '**重要推论**：$x\\to0$ 时，\n\n' +
          '$\\tan x \\sim x$，$\\arcsin x \\sim x$，$\\arctan x \\sim x$，$1 - \\cos x \\sim \\dfrac{x^2}{2}$。\n\n' +
          '这些等价无穷小直接由第一重要极限导出，是求极限的快捷工具。',
        formula:
          '\\lim_{x\\to0}\\frac{\\sin\\alpha x}{\\alpha x}=1\\;(\\alpha\\neq0);\\quad\\lim_{x\\to0}\\frac{\\tan x}{x}=1;\\quad\\lim_{x\\to0}\\frac{1-\\cos x}{x^2}=\\frac{1}{2}',
        tip: '口诀：$\\sin$(某式) 与 "某式" 在趋于 0 时等价——关键是括号里整体趋于 0，而不仅仅是 $x$。',
      },
      // 卡 4：第二重要极限的动机
      {
        title: '第二重要极限：数 e 的极限来源',
        body:
          '数 $e \\approx 2.71828$ 是自然指数函数 $e^x$ 的底数，但它是怎么定义的？\n\n' +
          '**银行复利的故事**：年利率 $r = 100\\%$，本金 1 元。\n\n' +
          '每年结一次利：最终得 $(1+1)^1 = 2$ 元。\n\n' +
          '每半年结一次：$(1+\\tfrac{1}{2})^2 = 2.25$ 元。\n\n' +
          '每季度结一次：$(1+\\tfrac{1}{4})^4 \\approx 2.441$ 元。\n\n' +
          '结利次数 $\\to \\infty$（连续复利）：$(1+\\tfrac{1}{x})^x \\to ?$\n\n' +
          '极限正是 $e$！这就是第二个重要极限，也是数 $e$ 的一种定义方式。',
        formula:
          '\\lim_{x\\to\\infty}\\left(1+\\frac{1}{x}\\right)^x=e',
        tip: '等价形式：令 $t=1/x$，$x\\to\\infty$ 对应 $t\\to0$，得 $\\lim_{t\\to0}(1+t)^{1/t}=e$。',
      },
      // 卡 5：识别 1^∞ 型与凑 e 极限
      {
        title: '识别 $1^{\\infty}$ 型：凑标准形式求 $e$',
        body:
          '"底数趋于 1，指数趋于无穷"的结构叫 **$1^{\\infty}$ 待定型**——不能直接写成 $1^{\\infty}=1$，那是错误的！\n\n' +
          '**识别信号**：底 $\\to 1$，指数 $\\to \\infty$（或底 $\\to 1^+$，指数 $\\to +\\infty$）。\n\n' +
          '**凑公式法**：将底写成 $1 + \\alpha$，则 $(1+\\alpha)^{\\beta}$，当 $\\alpha\\to0,\\beta\\to\\infty$ 时：\n\n' +
          '极限 $= e^{\\lim \\alpha\\beta}$（"底减 1 乘以指数，取极限作指数"）。\n\n' +
          '**例题**：$\\lim_{x\\to0}(1+2x)^{1/x}$。\n\n' +
          '识别：底 $1+2x\\to1$，指数 $1/x\\to\\infty$，是 $1^{\\infty}$ 型。\n\n' +
          '套公式：$\\alpha = 2x$，$\\beta = 1/x$，$\\alpha\\beta = 2$，极限 $= e^2$。',
        formula:
          '\\lim_{x\\to x_0}(1+\\alpha)^{\\beta}=e^{\\lim\\alpha\\beta}\\quad(\\alpha\\to0,\\,\\beta\\to\\infty)',
        tip: '口诀："底减 1 乘指数，极限值作指数，$e$ 为底"。三步到答案。',
      },
      // 卡 6：典型例题串讲
      {
        title: '例题串讲：$\\lim_{x\\to\\infty}\\left(1+\\dfrac{2}{x}\\right)^x$ 与 $\\lim_{x\\to0}(1+3x)^{2/x}$',
        body:
          '**例 1**：$\\lim_{x\\to\\infty}\\left(1+\\dfrac{2}{x}\\right)^x$。\n\n' +
          '底减 1 得 $\\alpha = 2/x$，指数 $\\beta = x$，$\\alpha\\beta = 2$，故极限 $= e^2$。\n\n' +
          '验证（换元法）：令 $t = x/2$，原式 $= \\left[(1+1/t)^t\\right]^2 \\to e^2$ ✓。\n\n' +
          '**例 2**：$\\lim_{x\\to0}(1+3x)^{2/x}$。\n\n' +
          '$\\alpha = 3x$，$\\beta = 2/x$，$\\alpha\\beta = 6$，极限 $= e^6$。\n\n' +
          '验证（换元法）：令 $t = 3x$，$(1+t)^{6/t} = [(1+t)^{1/t}]^6 \\to e^6$ ✓。\n\n' +
          '**常见错误**：看到 $1^\\infty$ 直接写 $= 1$，这是错的！$e^6 \\approx 403$，离 1 差得很远。',
        reveal: {
          q: '$\\lim_{x\\to+\\infty}\\left(1-\\dfrac{2}{x}\\right)^x$ 等于多少？',
          a: '$\\alpha = -2/x$，$\\beta = x$，$\\alpha\\beta = -2$，极限 $= e^{-2}$。也可令 $t = x/2$，写成 $[(1-1/t)^{-t}]^{-2}=([(1+(-1/t))^{-t}])^{-2}$。由于 $(-1/t)\\to0$，$(-t)\\to-\\infty$，但乘积 $(-1/t)\\cdot(-t)=1$，结果为 $e^{-2}$。',
        },
      },
      // 卡 7：两个重要极限对比与易错点
      {
        title: '两个重要极限对比与易错点',
        body:
          '**第一重要极限**：$\\lim_{x\\to0}\\dfrac{\\sin x}{x}=1$（$x$ 趋于 0，$0/0$ 型）。\n\n' +
          '**第二重要极限**：$\\lim_{x\\to\\infty}(1+1/x)^x=e$（$x$ 趋于 $\\infty$，$1^\\infty$ 型）。\n\n' +
          '**易错点一**：$\\lim_{x\\to0}\\dfrac{\\sin 2x}{\\sin 3x}$，不能直接写 $\\sin 2x \\approx 2x$ 然后一步给出答案，要写清楚等价过程，结果为 $2/3$。\n\n' +
          '**易错点二**：$\\lim_{x\\to0}(1+x^2)^{1/x}$——注意这里 $\\alpha = x^2$，$\\beta = 1/x$，$\\alpha\\beta = x \\to 0$，故极限 $= e^0 = 1$，不是 $e$！\n\n' +
          '**易错点三**：$1^{\\infty}$ 不是 $1$。$e \\approx 2.718$，$e^2 \\approx 7.39$，都不是 1。\n\n' +
          '**核对步骤**：使用公式后，检查 $\\alpha\\to0$ 和 $\\beta\\to\\infty$ 是否同时成立。',
        tip: '两个极限触发场景：看到 $\\sin(\\cdot)/(\\cdot)$（$\\to0$）→ 第一重要极限；看到 $(1+\\cdot)^{1/\\cdot}$（$1^\\infty$）→ 第二重要极限。',
      },
      // 卡 8：综合应用——$1-\cos x$ 的极限
      {
        title: '推论应用：$\\lim_{x\\to0}\\dfrac{1-\\cos x}{x^2}=\\dfrac{1}{2}$',
        body:
          '这个推论由第一重要极限导出，是求含 $\\cos$ 极限的重要工具。\n\n' +
          '**推导**：利用半角公式 $1 - \\cos x = 2\\sin^2(x/2)$：\n\n' +
          '$\\dfrac{1-\\cos x}{x^2} = \\dfrac{2\\sin^2(x/2)}{x^2} = \\dfrac{1}{2}\\cdot\\left(\\dfrac{\\sin(x/2)}{x/2}\\right)^2$。\n\n' +
          '令 $t = x/2$，$t \\to 0$ 时，$(\\sin t / t)^2 \\to 1$，所以整体极限为 $\\dfrac{1}{2}$。\n\n' +
          '**等价写法**：$x\\to0$ 时，$1 - \\cos x \\sim \\dfrac{x^2}{2}$。\n\n' +
          '**应用**：$\\lim_{x\\to0}\\dfrac{1-\\cos x}{x\\sin x} = \\lim_{x\\to0}\\dfrac{x^2/2}{x\\cdot x} = \\dfrac{1}{2}$（分子用 $1-\\cos x\\sim x^2/2$，分母用 $\\sin x \\sim x$）。',
        formula:
          '\\lim_{x\\to0}\\frac{1-\\cos x}{x^2}=\\frac{1}{2},\\quad x\\to0\\text{ 时 }1-\\cos x\\sim\\frac{x^2}{2}',
      },
    ],
    questions: [
      // 题 1：choice
      {
        id: 'c8-b-q1',
        type: 'choice',
        prompt: '$\\lim_{x\\to0}\\dfrac{\\sin 5x}{x}$ 等于多少？',
        options: ['$1$', '$5$', '$1/5$', '$0$'],
        answer: 1,
        explain:
          '$\\dfrac{\\sin 5x}{x}=5\\cdot\\dfrac{\\sin 5x}{5x}$。令 $t=5x$，$t\\to0$ 时 $\\dfrac{\\sin t}{t}\\to1$，故极限 $=5\\times1=5$。',
      },
      // 题 2：judge
      {
        id: 'c8-b-q2',
        type: 'judge',
        prompt: '$\\lim_{x\\to\\infty}\\left(1+\\dfrac{1}{x}\\right)^x=1$，因为底数趋于 1，任何数的 $\\infty$ 次方都是 1。',
        answer: false,
        explain:
          '$1^\\infty$ 是待定型，不能直接计算。$\\lim_{x\\to\\infty}(1+1/x)^x=e\\approx2.718$，不是 $1$。"底趋于 1、指数趋于无穷"必须用极限公式，不能直接代入。',
      },
      // 题 3：input
      {
        id: 'c8-b-q3',
        type: 'input',
        prompt:
          '$\\lim_{x\\to0}\\dfrac{\\sin 3x}{\\sin 5x}$ 等于多少？（填分数，用小数填写如 0.6）',
        accept: ['3/5', '0.6'],
        explain:
          '$\\dfrac{\\sin 3x}{\\sin 5x}=\\dfrac{\\sin 3x}{3x}\\cdot\\dfrac{5x}{\\sin 5x}\\cdot\\dfrac{3}{5}$。两个因子各自趋于 $1$，故极限 $=3/5=0.6$。',
      },
      // 题 4：choice
      {
        id: 'c8-b-q4',
        type: 'choice',
        prompt: '$\\lim_{x\\to0}(1+2x)^{3/x}$ 等于多少？',
        options: ['$e^2$', '$e^3$', '$e^6$', '$e$'],
        answer: 2,
        explain:
          '$\\alpha=2x\\to0$，$\\beta=3/x\\to\\infty$，$\\alpha\\beta=6$，极限 $=e^6$。也可令 $t=2x$，$(1+t)^{6/t}=[(1+t)^{1/t}]^6\\to e^6$。',
      },
      // 题 5：judge
      {
        id: 'c8-b-q5',
        type: 'judge',
        prompt: '$x\\to0$ 时，$1-\\cos x$ 与 $x^2/2$ 等价（即 $\\lim_{x\\to0}\\dfrac{1-\\cos x}{x^2/2}=1$）。',
        answer: true,
        explain:
          '由 $\\lim_{x\\to0}\\dfrac{1-\\cos x}{x^2}=\\dfrac{1}{2}$ 可知 $\\lim_{x\\to0}\\dfrac{1-\\cos x}{x^2/2}=1$，即 $1-\\cos x\\sim x^2/2$（$x\\to0$）。',
      },
      // 题 6：match
      {
        id: 'c8-b-q6',
        type: 'match',
        prompt: '将极限与其值对应（$x\\to0$）。',
        left: [
          '$\\lim_{x\\to0}\\dfrac{\\sin x}{x}$',
          '$\\lim_{x\\to0}\\dfrac{\\tan x}{x}$',
          '$\\lim_{x\\to0}\\dfrac{1-\\cos x}{x^2}$',
        ],
        right: ['$1$（第一重要极限本身）', '$1$（$\\sin x\\sim x$，$\\cos x\\to1$）', '$1/2$（半角公式推导）'],
      },
      // 题 7：input
      {
        id: 'c8-b-q7',
        type: 'input',
        prompt:
          '$\\lim_{x\\to+\\infty}\\left(1-\\dfrac{1}{x}\\right)^x$ 等于 $e$ 的几次方？（填整数，可为负）',
        accept: ['-1'],
        explain:
          '$\\alpha=-1/x$，$\\beta=x$，$\\alpha\\beta=-1$，故极限 $=e^{-1}$，即 $e$ 的 $-1$ 次方。也可令 $t=-x$，写成 $[(1+1/t)^t]^{-1}\\to e^{-1}$。',
      },
      // 题 8：choice
      {
        id: 'c8-b-q8',
        type: 'choice',
        prompt: '$\\lim_{x\\to0}\\dfrac{e^{2x}-1}{\\sin 3x}$ 等于多少？（用等价无穷小）',
        options: ['$2/3$', '$3/2$', '$1$', '$6$'],
        answer: 0,
        explain:
          '$x\\to0$ 时，$e^{2x}-1\\sim2x$（由 $e^u-1\\sim u$，$u=2x$），$\\sin 3x\\sim 3x$。故极限 $=\\dfrac{2x}{3x}=\\dfrac{2}{3}$。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c8-c  等价无穷小与极限运算综合
  // ─────────────────────────────────────────
  {
    id: 'c8-c',
    title: '等价无穷小与极限运算',
    subtitle: '无穷小比较·替换定理·加减陷阱·四则运算',
    intro: [
      // 卡 1：无穷小的定义
      {
        title: '无穷小：趋于 0 的变量，不是很小的数',
        body:
          '**定义**：若 $\\lim_{x\\to x_0}f(x)=0$，则称 $f(x)$ 为当 $x\\to x_0$ 时的**无穷小量**。\n\n' +
          '**关键点**：无穷小是一个**变量**（随 $x$ 变化），不是某个固定的很小的数。只有数字 $0$ 既是常数又是无穷小。\n\n' +
          '**常见无穷小**（$x\\to0$ 时）：$x$、$\\sin x$、$x^2$、$e^x-1$、$\\ln(1+x)$，它们各自趋于 $0$。\n\n' +
          '**无穷小与极限的关系**：$\\lim f(x)=L \\iff f(x)=L+\\alpha(x)$，其中 $\\alpha(x)$ 是无穷小。\n\n' +
          '也就是说，"极限等于 $L$"等价于"$f(x)$ 与 $L$ 的差是无穷小"——这把"趋近"变成了加减运算。',
        tip: '无穷小 $\\neq$ 无穷小量——后者才是专业术语。注意与"无穷大"区分：无穷大是 $|f(x)|\\to\\infty$，两者互为倒数关系。',
      },
      // 卡 2：无穷小的比较——高阶、同阶、等价
      {
        title: '无穷小的比较：谁"小"得更快？',
        body:
          '设 $\\alpha,\\beta$ 都是无穷小（同一极限过程），且 $\\beta\\neq0$，比较 $\\lim\\dfrac{\\alpha}{\\beta}$：\n\n' +
          '若极限为 $0$ → $\\alpha$ 比 $\\beta$ 趋于 $0$ **更快**，称 $\\alpha$ 是 $\\beta$ 的**高阶无穷小**，记 $\\alpha=o(\\beta)$。\n\n' +
          '若极限为非零常数 $c$ → 两者**同阶**，记 $\\alpha=O(\\beta)$（同阶无穷小）。\n\n' +
          '若极限恰好等于 $1$ → 两者**等价**，记 $\\alpha\\sim\\beta$。\n\n' +
          '**具体例子**（$x\\to0$）：$\\lim\\dfrac{x^2}{x}=\\lim x=0$，故 $x^2=o(x)$（$x^2$ 是 $x$ 的高阶无穷小）；\n\n' +
          '$\\lim\\dfrac{\\sin x}{x}=1$，故 $\\sin x\\sim x$（等价无穷小）。',
        formula:
          '\\alpha\\sim\\beta\\iff\\lim\\frac{\\alpha}{\\beta}=1',
        tip: '高阶无穷小趋于 0 的速度更快——就像 $x^2$ 比 $x$ 更快趋于 0，因为 $|x^2|<|x|$ 当 $|x|<1$。',
      },
      // 卡 3：常用等价无穷小表（$x\\to0$）
      {
        title: '常用等价无穷小（$x\\to0$ 时）',
        body:
          '下面是求极限时最常用的等价无穷小，必须熟记：\n\n' +
          '$\\sin x \\sim x$（来自第一重要极限）\n\n' +
          '$\\tan x \\sim x$\n\n' +
          '$\\arcsin x \\sim x$\n\n' +
          '$\\arctan x \\sim x$\n\n' +
          '$1 - \\cos x \\sim \\dfrac{x^2}{2}$\n\n' +
          '$e^x - 1 \\sim x$\n\n' +
          '$\\ln(1+x) \\sim x$\n\n' +
          '$(1+x)^\\alpha - 1 \\sim \\alpha x$（$\\alpha\\neq0$）\n\n' +
          '**特别注意**：以上都要求括号内整体趋于 $0$。比如 $e^{2x}-1\\sim2x$，$\\ln(1+3x)\\sim3x$，把 $x$ 换成了 $2x$、$3x$。',
        formula:
          'x\\to0\\text{ 时}:\\;\\sin x\\sim x,\\;e^x-1\\sim x,\\;\\ln(1+x)\\sim x,\\;1-\\cos x\\sim\\tfrac{x^2}{2}',
      },
      // 卡 4：等价无穷小替换定理
      {
        title: '等价无穷小替换：只能用在乘除！',
        body:
          '**定理**：设 $\\alpha\\sim\\alpha\'$，$\\beta\\sim\\beta\'$，且 $\\lim\\dfrac{\\alpha\'}{\\beta\'}$ 存在，则 $\\lim\\dfrac{\\alpha}{\\beta}=\\lim\\dfrac{\\alpha\'}{\\beta\'}$。\n\n' +
          '**用途**：在乘除关系中，把复杂的无穷小替换成等价的简单形式（通常是 $x$ 的幂次）。\n\n' +
          '**例子**：$\\lim_{x\\to0}\\dfrac{e^{3x}-1}{\\sin 2x}$。替换：$e^{3x}-1\\sim3x$，$\\sin 2x\\sim2x$，极限 $=\\dfrac{3x}{2x}=\\dfrac{3}{2}$。\n\n' +
          '**绝对不能在加减中替换！** 下一卡专门讨论这个陷阱。\n\n' +
          '**验证步骤**：① 确认趋于同一极限过程；② 确认是乘除关系；③ 替换；④ 化简。',
        formula:
          '\\lim\\frac{\\alpha}{\\beta}=\\lim\\frac{\\alpha\'}{\\beta\'}\\quad(\\alpha\\sim\\alpha\',\\,\\beta\\sim\\beta\')',
        steps: [
          '识别分子分母中的无穷小因子',
          '用等价无穷小（如 $\\sin u\\sim u$）替换',
          '化简后直接计算',
          '检查替换是否用在了乘除关系中',
        ],
      },
      // 卡 5：加减陷阱——等价替换失效的反例
      {
        title: '最大陷阱：加减中不能替换等价无穷小',
        body:
          '**错误示范**：求 $\\lim_{x\\to0}\\dfrac{\\sin x - \\tan x}{x^3}$。\n\n' +
          '若把 $\\sin x$ 和 $\\tan x$ 都替换为 $x$，得 $\\dfrac{x-x}{x^3}=\\dfrac{0}{x^3}=0$，**这是错的**！\n\n' +
          '**正确做法（精确计算）**：\n\n' +
          '$\\sin x - \\tan x = \\sin x\\left(1 - \\dfrac{1}{\\cos x}\\right) = \\dfrac{\\sin x(\\cos x - 1)}{\\cos x}$。\n\n' +
          '用等价无穷小（在**乘除**中）：$\\sin x\\sim x$，$1-\\cos x\\sim x^2/2$，$\\cos x\\to1$。\n\n' +
          '故分子 $\\sim x\\cdot(-x^2/2)/1 = -x^3/2$，极限 $= -\\dfrac{1}{2}$，不是 $0$！\n\n' +
          '**原因**：加减相消时，主项（一阶无穷小）相互抵消，高阶项成为主导——替换主项会丢失这个信息。',
        tip: '口诀：等价无穷小替换，**乘除放心用，加减别乱动**。加减时用变形后的乘除结构，或者 Taylor 展开。',
      },
      // 卡 6：极限的四则运算法则
      {
        title: '极限四则运算法则',
        body:
          '设 $\\lim_{x\\to x_0}f(x)=A$，$\\lim_{x\\to x_0}g(x)=B$，则：\n\n' +
          '$\\lim[f(x)\\pm g(x)]=A\\pm B$\n\n' +
          '$\\lim[f(x)\\cdot g(x)]=A\\cdot B$\n\n' +
          '$\\lim\\dfrac{f(x)}{g(x)}=\\dfrac{A}{B}$（要求 $B\\neq0$）\n\n' +
          '**注意**：法则要求分子分母极限都存在且分母不为 0。若出现 $0/0$、$\\infty/\\infty$、$1^\\infty$ 等待定型，必须先化简再用法则。\n\n' +
          '**典型应用**：$\\lim_{x\\to2}\\dfrac{x^2-4}{x-2}$，直接代入得 $0/0$ 型，**不能用商法则**，需先因式分解约分：$\\dfrac{(x-2)(x+2)}{x-2}=x+2\\to4$。',
        formula:
          '\\lim[f\\pm g]=\\lim f\\pm\\lim g;\\quad\\lim(fg)=(\\lim f)(\\lim g);\\quad\\lim\\frac{f}{g}=\\frac{\\lim f}{\\lim g}\\;(\\lim g\\neq0)',
      },
      // 卡 7：综合方法流程与 reveal
      {
        title: '求函数极限的综合流程',
        body:
          '**第 1 步：代入检验。** 直接代 $x_0$，看是否有定义且函数连续。若是，直接得答案。\n\n' +
          '**第 2 步：识别待定型。**\n\n' +
          '$0/0$ 型 → 因式分解约公因子 / 有理化 / 等价无穷小（乘除）\n\n' +
          '$\\infty/\\infty$ 型 → 分子分母同除最高次项\n\n' +
          '$1^\\infty$ 型 → 凑 $e$ 极限公式\n\n' +
          '**第 3 步：化简，选合适工具。** 优先等价无穷小（快）；若有加减，改写为乘除结构再替换，或用 Taylor 展开。\n\n' +
          '**第 4 步：验算。** 检查分母非零、替换仅在乘除中使用。',
        tip: '看到 $0/0$：先想因式分解和约分，再想等价替换；看到 $1^\\infty$：立刻用"底减 1 乘指数"公式。',
        reveal: {
          q: '求 $\\lim_{x\\to0}\\dfrac{\\ln(1+2x)}{\\arctan 3x}$。',
          a: '$\\ln(1+2x)\\sim2x$，$\\arctan 3x\\sim3x$（$x\\to0$），故极限 $=\\dfrac{2x}{3x}=\\dfrac{2}{3}$。',
        },
      },
      // 卡 8：例题——无穷大与无穷小的关系
      {
        title: '无穷大与无穷小互为倒数',
        body:
          '**定理**：若 $\\lim f(x)=\\infty$（无穷大），则 $\\lim 1/f(x)=0$（无穷小）；反之亦然（$f(x)\\neq0$）。\n\n' +
          '**例子**：$\\lim_{x\\to0^+}\\ln x=-\\infty$（无穷大），故 $\\lim_{x\\to0^+}\\dfrac{1}{\\ln x}=0$（无穷小）。\n\n' +
          '**应用技巧**：遇到 $\\infty/\\infty$ 型，分子分母同除以分母（最高次项），使分子分母都化为有限极限。\n\n' +
          '**例题**：$\\lim_{x\\to\\infty}\\dfrac{3x^2+2}{x^2-1}$，分子分母同除 $x^2$：$\\dfrac{3+2/x^2}{1-1/x^2}\\to\\dfrac{3+0}{1-0}=3$。',
        formula:
          '\\lim f(x)=\\infty\\implies\\lim\\frac{1}{f(x)}=0;\\quad\\lim\\frac{a_nx^n+\\cdots}{b_mx^m+\\cdots}=\\begin{cases}a_n/b_n,&n=m\\\\0,&n<m\\\\\\infty,&n>m\\end{cases}',
      },
    ],
    questions: [
      // 题 1：choice
      {
        id: 'c8-c-q1',
        type: 'choice',
        prompt:
          '$x\\to0$ 时，下列哪个是 $x$ 的高阶无穷小（即比 $x$ 趋于 $0$ 更快）？',
        options: ['$\\sin x$', '$\\arctan x$', '$x^2$', '$2x$'],
        answer: 2,
        explain:
          '$\\lim_{x\\to0}\\dfrac{x^2}{x}=\\lim_{x\\to0}x=0$，故 $x^2=o(x)$，是 $x$ 的高阶无穷小。$\\sin x\\sim x$，$\\arctan x\\sim x$，$2x$ 与 $x$ 同阶。',
      },
      // 题 2：judge
      {
        id: 'c8-c-q2',
        type: 'judge',
        prompt:
          '求 $\\lim_{x\\to0}\\dfrac{\\sin x - x}{x^3}$ 时，可以用等价无穷小 $\\sin x\\sim x$ 直接替换，得极限为 $0$。',
        answer: false,
        explain:
          '等价无穷小替换**不能用于加减**。分子 $\\sin x - x$ 的主项相消，高阶项成为主导。正确做法用 Taylor 展开：$\\sin x = x - x^3/6 + o(x^3)$，故 $\\sin x - x \\approx -x^3/6$，极限为 $-1/6$，不是 $0$。',
      },
      // 题 3：input
      {
        id: 'c8-c-q3',
        type: 'input',
        prompt:
          '$\\lim_{x\\to0}\\dfrac{e^{3x}-1}{\\ln(1+2x)}$ 等于多少？（填分数，如 1.5）',
        accept: ['3/2', '1.5'],
        explain:
          '$e^{3x}-1\\sim3x$，$\\ln(1+2x)\\sim2x$（均在 $x\\to0$ 时）。等价替换（乘除中合法）：极限 $=\\dfrac{3x}{2x}=\\dfrac{3}{2}=1.5$。',
      },
      // 题 4：match
      {
        id: 'c8-c-q4',
        type: 'match',
        prompt: '将 $x\\to0$ 时的无穷小与其等价无穷小对应。',
        left: ['$e^x-1$', '$\\ln(1+x)$', '$1-\\cos x$'],
        right: ['$x$', '$x$（对数的等价）', '$x^2/2$'],
      },
      // 题 5：choice
      {
        id: 'c8-c-q5',
        type: 'choice',
        prompt:
          '$\\lim_{x\\to\\infty}\\dfrac{2x^3+x}{3x^3-1}$ 等于多少？',
        options: ['$0$', '$2/3$', '$\\infty$', '$1$'],
        answer: 1,
        explain:
          '分子分母同除 $x^3$：$\\dfrac{2+1/x^2}{3-1/x^3}\\to\\dfrac{2+0}{3-0}=\\dfrac{2}{3}$。分子分母最高次相同（均为 $x^3$），极限为最高次系数之比。',
      },
      // 题 6：judge
      {
        id: 'c8-c-q6',
        type: 'judge',
        prompt:
          '若 $\\lim_{x\\to x_0}f(x)=A\\neq0$，$\\lim_{x\\to x_0}g(x)=B\\neq0$，则 $\\lim_{x\\to x_0}\\dfrac{f(x)}{g(x)}=\\dfrac{A}{B}$。',
        answer: true,
        explain:
          '极限的商法则：只要两个极限都存在，且分母极限不为 $0$，商的极限等于极限的商。$A\\neq0$，$B\\neq0$ 满足条件，故结论成立。',
      },
      // 题 7：input
      {
        id: 'c8-c-q7',
        type: 'input',
        prompt:
          '$\\lim_{x\\to0}\\dfrac{(1+x)^{1/2}-1}{x}$ 等于多少？（利用 $(1+x)^\\alpha-1\\sim\\alpha x$，填分数如 0.5）',
        accept: ['1/2', '0.5'],
        explain:
          '$(1+x)^{1/2}-1\\sim\\dfrac{1}{2}x$（等价无穷小，$\\alpha=1/2$），故极限 $=\\dfrac{(1/2)x}{x}=\\dfrac{1}{2}$。也可有理化验证。',
      },
      // 题 8：choice
      {
        id: 'c8-c-q8',
        type: 'choice',
        prompt:
          '下列极限计算，哪个步骤有误？',
        options: [
          '$\\lim_{x\\to0}\\dfrac{\\tan x}{\\sin x}=\\lim_{x\\to0}\\dfrac{x}{x}=1$（$\\tan x\\sim x$，$\\sin x\\sim x$）',
          '$\\lim_{x\\to0}\\dfrac{\\tan x-\\sin x}{x^3}=\\lim_{x\\to0}\\dfrac{x-x}{x^3}=0$（等价替换）',
          '$\\lim_{x\\to2}\\dfrac{x^2-4}{x-2}=\\lim_{x\\to2}(x+2)=4$（约分）',
          '$\\lim_{x\\to\\infty}\\dfrac{x^2+1}{2x^2}=\\dfrac{1}{2}$（同除 $x^2$）',
        ],
        answer: 1,
        explain:
          '第 2 项错误：$\\tan x - \\sin x$ 是加减关系，不能把 $\\tan x$ 和 $\\sin x$ 分别替换为 $x$，那会导致主项相消而丢失信息。正确结果为 $1/2$（不是 $0$）。其余三项均正确。',
      },
    ],
  },
];
