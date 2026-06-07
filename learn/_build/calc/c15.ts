import type { Lesson } from '../../src/types'

export const LESSONS: Lesson[] = [
  // =========================================================================
  // c15-a  从面积到定积分——黎曼和与定义
  // =========================================================================
  {
    id: 'c15-a',
    title: '从面积到定积分',
    subtitle: '黎曼和 → 极限 → 精确面积',
    intro: [
      {
        title: '为什么需要定积分？',
        body: '初中我们会算矩形、三角形的面积，但曲线下方的面积怎么算？\n\n比如 $y = x^2$ 从 $x=0$ 到 $x=1$ 围成的图形——它不是任何规则图形，但我们能用一个绝妙的想法：**用无数个矩形去逼近它**。\n\n核心思路：把区间切成 $n$ 段，每段用一个小矩形近似，然后令 $n \\to \\infty$，误差趋于 0，得到精确值。这就是**黎曼和取极限**的思想。',
        tip: '矩形近似 → 黎曼和 → 取极限 → 定积分。这四步是整章的灵魂，后面所有内容都建立在这个想法上。',
      },
      {
        title: '黎曼和：把区间切碎',
        body: '把区间 $[a, b]$ 分成 $n$ 小段：\n$a = x_0 < x_1 < x_2 < \\cdots < x_n = b$\n\n每段长度 $\\Delta x_i = x_i - x_{i-1}$，在第 $i$ 段内任取一点 $\\xi_i$，用小矩形高度 $f(\\xi_i)$、宽度 $\\Delta x_i$ 近似该段面积。\n\n所有小矩形面积之和叫做**黎曼和**：',
        formula: 'S_n = \\sum_{i=1}^{n} f(\\xi_i)\\,\\Delta x_i',
        tip: '$\\xi_i$ 可以取该段的左端点、右端点或中点——取哪个都可以，极限结果相同（只要函数连续）。',
      },
      {
        title: '定积分的严格定义',
        body: '记最大分段长度 $\\lambda = \\max\\{\\Delta x_i\\}$。若不管怎么分割、$\\xi_i$ 怎么取，只要 $\\lambda \\to 0$，黎曼和都趋向同一个数 $I$，则称 $f$ 在 $[a,b]$ 上**可积**，$I$ 就是定积分：',
        formula: '\\int_a^b f(x)\\,dx = \\lim_{\\lambda \\to 0} \\sum_{i=1}^{n} f(\\xi_i)\\,\\Delta x_i',
        steps: [
          '记号解读：$\\int_a^b$ 是积分符号（拉长的 S，代表求和 Sum），$a$ 是下限，$b$ 是上限。',
          '$f(x)\\,dx$ 中 $x$ 叫**哑变量**（积分变量），换成 $t$、$u$ 结果完全相同：$\\int_a^b f(x)\\,dx = \\int_a^b f(t)\\,dt$。',
          '结果是一个**数**（不是函数），代表有向面积（$f(x) < 0$ 时贡献负值）。',
        ],
      },
      {
        title: '具体算算：$\\int_0^1 x^2\\,dx$ 用定义',
        body: '把 $[0,1]$ 等分 $n$ 份，取右端点 $\\xi_i = \\dfrac{i}{n}$，$\\Delta x_i = \\dfrac{1}{n}$。',
        steps: [
          '黎曼和：$S_n = \\displaystyle\\sum_{i=1}^{n} \\left(\\frac{i}{n}\\right)^2 \\cdot \\frac{1}{n} = \\frac{1}{n^3}\\sum_{i=1}^{n} i^2$',
          '用公式 $\\displaystyle\\sum_{i=1}^n i^2 = \\frac{n(n+1)(2n+1)}{6}$：$S_n = \\frac{1}{n^3} \\cdot \\frac{n(n+1)(2n+1)}{6} = \\frac{(n+1)(2n+1)}{6n^2}$',
          '取极限：$\\displaystyle\\lim_{n\\to\\infty} S_n = \\lim_{n\\to\\infty} \\frac{(n+1)(2n+1)}{6n^2} = \\frac{2}{6} = \\frac{1}{3}$',
          '结论：$\\displaystyle\\int_0^1 x^2\\,dx = \\frac{1}{3}$。（后面用牛顿-莱布尼茨公式 5 秒就能算出来！）',
        ],
        tip: '用定义直接算定积分很繁琐，需要 $\\sum i^2$ 等公式。幸运的是，微积分基本定理给了我们简便方法——但理解定义是基础。',
      },
      {
        title: '定积分的几何意义',
        body: '当 $f(x) \\geq 0$ 时，$\\displaystyle\\int_a^b f(x)\\,dx$ 就是曲线 $y = f(x)$、$x$ 轴与直线 $x=a$、$x=b$ 围成的面积。\n\n当 $f(x)$ 有正有负时，定积分是**有向面积**：$x$ 轴上方面积记正，下方记负。\n\n**面积 $\\neq$ 定积分**：若要求总面积，应计算 $\\displaystyle\\int_a^b |f(x)|\\,dx$（需先找零点）。',
        formula: 'S_{\\text{area}} = \\int_a^b |f(x)|\\,dx',
      },
      {
        title: '常用基本性质',
        body: '**线性性**：$\\displaystyle\\int_a^b [k_1 f + k_2 g]\\,dx = k_1\\int_a^b f\\,dx + k_2\\int_a^b g\\,dx$\n\n**区间可加**：$\\displaystyle\\int_a^b f\\,dx = \\int_a^c f\\,dx + \\int_c^b f\\,dx$（$c$ 在 $a,b$ 之间或之外均成立）\n\n**特殊约定**：$\\displaystyle\\int_a^a f\\,dx = 0$；$\\displaystyle\\int_a^b f\\,dx = -\\int_b^a f\\,dx$（上下限互换变号）\n\n**保号性**：若 $f(x) \\geq 0$，则 $\\displaystyle\\int_a^b f\\,dx \\geq 0$（$a < b$）',
        tip: '区间可加性是拆分、合并积分区间的利器，在分段函数和绝对值函数的定积分中频繁使用。',
      },
      {
        title: '易错点与 reveal',
        body: '**易错 1**：定积分与不定积分有本质区别——定积分是一个**数**，不定积分是一族**函数**。\n\n**易错 2**：$\\displaystyle\\int_a^b f(x)\\,dx$ 中的 $x$ 是哑变量，结果和 $x$ 无关。把 $\\int_0^1 x^2\\,dx$ 说成"和 $x$ 有关"是典型错误。\n\n**易错 3**：黎曼和的极限值与分割方式和 $\\xi_i$ 的选取无关（只要函数连续），这一点需要记住。',
        reveal: {
          q: '$\\sin x$ 在 $[0, 2\\pi]$ 上的定积分 $\\displaystyle\\int_0^{2\\pi} \\sin x\\,dx$ 等于多少？为什么？',
          a: '等于 $0$。$\\sin x$ 在 $[0,\\pi]$ 为正（上方面积 $= 2$），在 $[\\pi, 2\\pi]$ 为负（下方面积 $= -2$），两部分抵消。这说明定积分是**有向面积**，正负可以相消。若问总面积则为 $4$，需算 $\\int_0^{2\\pi}|\\sin x|\\,dx$。',
        },
      },
    ],
    questions: [
      {
        id: 'c15-a-q1',
        type: 'choice',
        prompt: '把 $[0,2]$ 等分 $n$ 份，取右端点，求 $\\displaystyle\\sum_{i=1}^n \\frac{i}{n} \\cdot \\frac{2}{n}$ 当 $n\\to\\infty$ 的极限，等于哪个定积分？',
        options: [
          '$\\displaystyle\\int_0^2 x\\,dx$',
          '$\\displaystyle\\int_0^1 x\\,dx$',
          '$\\displaystyle\\int_0^2 2x\\,dx$',
          '$\\displaystyle\\int_0^2 x^2\\,dx$',
        ],
        answer: 0,
        explain: '右端点 $x_i = \\frac{2i}{n}$，$\\Delta x = \\frac{2}{n}$，黎曼和为 $\\sum f(x_i)\\Delta x = \\sum \\frac{2i}{n}\\cdot\\frac{2}{n}$。但注意 $f(x) = x$ 且 $\\xi_i = \\frac{i}{n}\\cdot 2 = \\frac{2i}{n}$，故对应的定积分是 $\\int_0^2 x\\,dx$（区间 $[0,2]$，被积函数 $x$）。验证：$\\int_0^2 x\\,dx = \\frac{x^2}{2}\\big|_0^2 = 2$，极限也应为 $2$。',
      },
      {
        id: 'c15-a-q2',
        type: 'judge',
        prompt: '$\\displaystyle\\int_a^b f(x)\\,dx = \\int_a^b f(t)\\,dt$，两者结果相同，因为积分变量只是哑变量。',
        answer: true,
        explain: '积分变量（$x$ 或 $t$）是哑变量，在定积分中只是临时占位符，最终结果是数值，与变量字母无关。',
      },
      {
        id: 'c15-a-q3',
        type: 'choice',
        prompt: '$y = \\sin x$ 在 $[0, 2\\pi]$ 上与 $x$ 轴围成的**总面积**是多少？',
        options: ['$0$', '$2$', '$4$', '$\\pi$'],
        answer: 2,
        explain: '$\\sin x$ 在 $[0,\\pi]$ 上方面积 $= \\int_0^\\pi \\sin x\\,dx = [-\\cos x]_0^\\pi = 2$；在 $[\\pi,2\\pi]$ 下方面积（绝对值）$= \\int_\\pi^{2\\pi}(-\\sin x)\\,dx = 2$。总面积 $= 2+2 = 4$。注意 $\\int_0^{2\\pi}\\sin x\\,dx = 0$ 是有向面积，不是总面积。',
      },
      {
        id: 'c15-a-q4',
        type: 'input',
        prompt: '利用定积分的区间可加性，已知 $\\displaystyle\\int_0^3 f(x)\\,dx = 5$ 且 $\\displaystyle\\int_0^1 f(x)\\,dx = 2$，则 $\\displaystyle\\int_1^3 f(x)\\,dx =$ ？（填数字）',
        accept: ['3'],
        placeholder: '3',
        explain: '由区间可加性：$\\int_0^3 f\\,dx = \\int_0^1 f\\,dx + \\int_1^3 f\\,dx$，故 $\\int_1^3 f\\,dx = 5 - 2 = 3$。',
      },
      {
        id: 'c15-a-q5',
        type: 'judge',
        prompt: '若 $f(x) \\geq 0$ 在 $[a,b]$ 上成立，则 $\\displaystyle\\int_a^b f(x)\\,dx$ 一定大于 $0$。',
        answer: false,
        explain: '若 $f(x) \\geq 0$，则 $\\int_a^b f\\,dx \\geq 0$，但不一定**严格**大于 0。例如 $f(x) \\equiv 0$，则积分为 $0$。（若 $f$ 连续且不恒为零则严格大于 0，但题目没有这个条件。）',
      },
      {
        id: 'c15-a-q6',
        type: 'match',
        prompt: '将各描述与对应的概念配对：',
        left: [
          '$\\displaystyle\\sum_{i=1}^n f(\\xi_i)\\,\\Delta x_i$',
          '$\\displaystyle\\lim_{\\lambda\\to 0}\\sum_{i=1}^n f(\\xi_i)\\,\\Delta x_i$',
          '定积分的结果',
          '积分中的 $x$（在 $\\int_a^b f(x)\\,dx$ 内）',
        ],
        right: [
          '黎曼和（矩形面积之和）',
          '定积分（精确值）',
          '一个数（不是函数）',
          '哑变量，可用任意字母替换',
        ],
        explain: '黎曼和是有限矩形之和，取极限后得到定积分。定积分的结果是数，积分变量是哑变量。',
      },
      {
        id: 'c15-a-q7',
        type: 'choice',
        prompt: '估计 $\\displaystyle\\int_0^1 e^{-x^2}\\,dx$ 的范围。在 $[0,1]$ 上，$e^{-1} \\leq e^{-x^2} \\leq 1$，由估值定理得：',
        options: [
          '$0 \\leq \\displaystyle\\int_0^1 e^{-x^2}\\,dx \\leq 1$',
          '$e^{-1} \\leq \\displaystyle\\int_0^1 e^{-x^2}\\,dx \\leq 1$',
          '$e^{-1} \\leq \\displaystyle\\int_0^1 e^{-x^2}\\,dx \\leq e$',
          '$\\frac{1}{2} \\leq \\displaystyle\\int_0^1 e^{-x^2}\\,dx \\leq 1$',
        ],
        answer: 1,
        explain: '估值定理：若 $m \\leq f(x) \\leq M$ 在 $[a,b]$ 上成立，则 $m(b-a) \\leq \\int_a^b f\\,dx \\leq M(b-a)$。这里 $m = e^{-1}$，$M = 1$，$b-a = 1$，故 $\\frac{1}{e} \\leq \\int_0^1 e^{-x^2}\\,dx \\leq 1$（约 $0.368$ 到 $1$）。',
      },
      {
        id: 'c15-a-q8',
        type: 'choice',
        prompt: '黎曼和 $S_n = \\displaystyle\\sum_{i=1}^n f(\\xi_i)\\Delta x_i$ 中，$\\xi_i$ 可以是每个小区间的：',
        options: [
          '只能是左端点',
          '只能是右端点',
          '只能是中点',
          '区间 $[x_{i-1}, x_i]$ 内任意一点（极限结果相同）',
        ],
        answer: 3,
        explain: '黎曼和的定义允许 $\\xi_i$ 在每个子区间 $[x_{i-1}, x_i]$ 内任意选取。只要函数连续（或满足可积条件），不论如何选取 $\\xi_i$，当 $\\lambda\\to 0$ 时黎曼和的极限都相同，等于定积分。',
      },
    ],
  },

  // =========================================================================
  // c15-b  微积分基本定理与牛顿-莱布尼茨公式
  // =========================================================================
  {
    id: 'c15-b',
    title: '微积分基本定理',
    subtitle: '积分与求导互为逆运算——牛顿-莱布尼茨公式',
    intro: [
      {
        title: '引入：积分能被求导"撤销"吗？',
        body: '上一关我们看到，用定义算 $\\int_0^1 x^2\\,dx$ 需要算级数求极限，很麻烦。\n\n有没有更快的方法？答案是：**找原函数，代入上下限做差**。\n\n$x^2$ 的原函数是 $\\dfrac{x^3}{3}$，代入得 $\\dfrac{1^3}{3} - \\dfrac{0^3}{3} = \\dfrac{1}{3}$。不到 5 秒！\n\n但为什么可以这样算？这是微积分最深刻的定理——**微积分基本定理**的答案。',
        tip: '微积分基本定理说：求积分与求导是互逆的操作，就像加法与减法、乘法与除法一样。',
      },
      {
        title: '变上限积分函数',
        body: '给定连续函数 $f$ 和固定下限 $a$，定义**变上限积分函数**：\n$\\Phi(x) = \\displaystyle\\int_a^x f(t)\\,dt$，$x \\in [a, b]$\n\n$\\Phi(x)$ 是关于上限 $x$ 的函数——把 $x$ 理解为"扫过面积的右边界"，$x$ 增大时 $\\Phi(x)$ 增大（若 $f > 0$）。\n\n注意：积分变量是 $t$（哑变量），$\\Phi$ 是关于 $x$ 的函数，两者不混。',
        formula: '\\Phi(x) = \\int_a^x f(t)\\,dt',
        tip: '很多同学把 $\\int_a^x f(t)\\,dt$ 里的 $t$ 和 $x$ 混淆——记住，$t$ 是积分内部的哑变量，$x$ 才是外层函数 $\\Phi$ 的自变量。',
      },
      {
        title: '微积分第一基本定理（为什么成立？）',
        body: '**定理**：若 $f$ 在 $[a,b]$ 连续，则 $\\Phi(x) = \\displaystyle\\int_a^x f(t)\\,dt$ 可导，且：',
        formula: "\\Phi'(x) = \\frac{d}{dx}\\int_a^x f(t)\\,dt = f(x)",
        steps: [
          '用导数定义：$\\Phi(x+\\Delta x) - \\Phi(x) = \\int_a^{x+\\Delta x}f(t)\\,dt - \\int_a^x f(t)\\,dt = \\int_x^{x+\\Delta x}f(t)\\,dt$（区间可加性）',
          '由积分中值定理：$\\int_x^{x+\\Delta x}f(t)\\,dt = f(\\xi)\\cdot\\Delta x$，其中 $\\xi$ 在 $x$ 与 $x+\\Delta x$ 之间。',
          '因此 $\\dfrac{\\Phi(x+\\Delta x)-\\Phi(x)}{\\Delta x} = f(\\xi)$。',
          '令 $\\Delta x \\to 0$，则 $\\xi \\to x$，由 $f$ 连续得 $f(\\xi) \\to f(x)$，故 $\\Phi\'(x) = f(x)$。',
        ],
      },
      {
        title: '第一定理的重要推论：链式法则',
        body: '若上限是 $x$ 的函数 $g(x)$，需用链式法则：\n$\\dfrac{d}{dx}\\displaystyle\\int_a^{g(x)} f(t)\\,dt = f(g(x)) \\cdot g\'(x)$\n\n**例**：$\\dfrac{d}{dx}\\displaystyle\\int_0^{x^2} \\cos t\\,dt = \\cos(x^2) \\cdot (x^2)\' = 2x\\cos(x^2)$\n\n**错误做法**：直接写 $\\cos(x^2)$，忘记乘 $g\'(x) = 2x$——这是最常见错误！',
        formula: '\\frac{d}{dx}\\int_a^{g(x)} f(t)\\,dt = f(g(x))\\cdot g\'(x)',
        tip: '上限是 $x$ 直接代；上限是 $g(x)$ 必须再乘 $g\'(x)$。记住：**上限复合，乘链式导数**。',
      },
      {
        title: '微积分第二基本定理（牛顿-莱布尼茨公式）',
        body: '**定理**：若 $f$ 在 $[a,b]$ 连续，$F$ 是 $f$ 的任意一个原函数（即 $F\'=f$），则：',
        formula: '\\int_a^b f(x)\\,dx = F(b) - F(a) \\triangleq F(x)\\Big|_a^b',
        steps: [
          '**为什么成立？** 由第一定理，$\\Phi(x)=\\int_a^x f(t)\\,dt$ 也是 $f$ 的原函数，故 $F(x) = \\Phi(x) + C$（原函数差常数）。',
          '计算 $F(b) - F(a) = [\\Phi(b)+C] - [\\Phi(a)+C] = \\Phi(b) - \\Phi(a)$。',
          '$\\Phi(b) = \\int_a^b f(t)\\,dt$，$\\Phi(a) = \\int_a^a f(t)\\,dt = 0$。',
          '故 $F(b) - F(a) = \\int_a^b f(x)\\,dx$。常数 $C$ 自动消去，取哪个原函数都行！',
        ],
      },
      {
        title: '牛顿-莱布尼茨公式：典型例题',
        body: '**例 1**：$\\displaystyle\\int_0^{\\pi/2} \\cos x\\,dx$\n\n$\\cos x$ 的原函数是 $\\sin x$，故结果为 $\\sin\\dfrac{\\pi}{2} - \\sin 0 = 1 - 0 = 1$。\n\n**例 2**：$\\displaystyle\\int_1^e \\dfrac{1}{x}\\,dx$\n\n$\\dfrac{1}{x}$ 的原函数是 $\\ln x$（$x>0$），故结果为 $\\ln e - \\ln 1 = 1 - 0 = 1$。\n\n**使用前必须验证**：被积函数在 $[a,b]$ 上**连续**（或至少可积且原函数存在）。$\\displaystyle\\int_{-1}^1 \\dfrac{1}{x^2}\\,dx$ 直接套公式得 $-2$ 是**错误**的，因为 $\\dfrac{1}{x^2}$ 在 $x=0$ 无界！',
        tip: '**使用牛顿-莱布尼茨公式的第一步**：验证被积函数在积分区间上连续（无间断点、无无界点）。否则先查是否为广义积分，不能直接套公式。',
      },
      {
        title: '易错点汇总与 reveal',
        body: '**易错 1**：变上限积分求导后，结果是 $f(x)$，不是 $f(t)$——$t$ 是哑变量，已被消掉。\n\n**易错 2**：上限含复合函数必须乘链式导数，例如 $\\dfrac{d}{dx}\\int_0^{x^2} f(t)\\,dt = f(x^2) \\cdot 2x$，不是 $f(x^2)$。\n\n**易错 3**：上限换元法后**不需要换回原变量**——直接代入新积分限计算，这是定积分换元与不定积分换元的最大区别。\n\n**易错 4**：$[a,b]$ 上有间断点时，不能直接用 N-L 公式——需先用区间可加性分段处理。',
        reveal: {
          q: '用极限 $\\displaystyle\\lim_{x\\to 0}\\dfrac{\\int_0^x te^{t^2}\\,dt}{x^2}$，直接代 $x=0$ 得 $\\dfrac{0}{0}$，应如何计算？',
          a: '用洛必达法则：分子对 $x$ 求导用微积分第一基本定理得 $xe^{x^2}$，分母求导得 $2x$，故极限为 $\\lim_{x\\to 0}\\dfrac{xe^{x^2}}{2x} = \\lim_{x\\to 0}\\dfrac{e^{x^2}}{2} = \\dfrac{1}{2}$。关键：变上限积分求导直接得被积函数在上限处的值。',
        },
      },
    ],
    questions: [
      {
        id: 'c15-b-q1',
        type: 'choice',
        prompt: '已知 $F(x) = \\displaystyle\\int_1^x \\sqrt{1+t^3}\\,dt$，则 $F\'(x) = $？',
        options: [
          '$\\sqrt{1+x^3}$',
          '$\\sqrt{1+t^3}$',
          '$\\dfrac{1}{2\\sqrt{1+x^3}}$',
          '$3x^2 \\cdot \\sqrt{1+x^3}$',
        ],
        answer: 0,
        explain: '由微积分第一基本定理，$F\'(x) = \\sqrt{1+x^3}$（用 $x$ 替换被积函数中的哑变量 $t$，直接代入上限即可）。',
      },
      {
        id: 'c15-b-q2',
        type: 'choice',
        prompt: '$\\dfrac{d}{dx}\\displaystyle\\int_0^{x^3} e^{t^2}\\,dt = $？',
        options: [
          '$e^{x^6}$',
          '$3x^2 e^{x^6}$',
          '$e^{x^2}$',
          '$3x^2 e^{x^2}$',
        ],
        answer: 1,
        explain: '上限为 $g(x) = x^3$，用链式法则：$f(g(x))\\cdot g\'(x) = e^{(x^3)^2}\\cdot 3x^2 = 3x^2 e^{x^6}$。注意 $g(x)=x^3$ 代入被积函数得 $e^{(x^3)^2}=e^{x^6}$，再乘 $g\'(x)=3x^2$。',
      },
      {
        id: 'c15-b-q3',
        type: 'input',
        prompt: '计算 $\\displaystyle\\int_0^{\\pi} \\sin x\\,dx$，结果是多少？（填数字）',
        accept: ['2'],
        placeholder: '2',
        explain: '$\\sin x$ 的原函数是 $-\\cos x$，由牛顿-莱布尼茨公式：$-\\cos x\\big|_0^\\pi = -\\cos\\pi - (-\\cos 0) = -(-1) - (-1) = 1 + 1 = 2$。',
      },
      {
        id: 'c15-b-q4',
        type: 'judge',
        prompt: '$\\displaystyle\\int_{-1}^1 \\dfrac{1}{x^2}\\,dx = \\left(-\\dfrac{1}{x}\\right)\\bigg|_{-1}^1 = -1 - 1 = -2$，此计算正确。',
        answer: false,
        explain: '错误！$\\dfrac{1}{x^2}$ 在 $x=0$ 处无界，$[−1,1]$ 上不连续，不能直接用牛顿-莱布尼茨公式。这实际上是一个广义积分（且是发散的）。使用 N-L 公式前必须验证连续性。',
      },
      {
        id: 'c15-b-q5',
        type: 'choice',
        prompt: '计算极限 $\\displaystyle\\lim_{x\\to 0}\\dfrac{\\int_0^x \\sin t^2\\,dt}{x^3}$：',
        options: [
          '$0$',
          '$\\dfrac{1}{3}$',
          '$1$',
          '$\\dfrac{1}{2}$',
        ],
        answer: 1,
        explain: '$\\dfrac{0}{0}$ 型，用洛必达：分子求导得 $\\sin x^2$，分母求导得 $3x^2$，极限变为 $\\lim_{x\\to 0}\\dfrac{\\sin x^2}{3x^2}$。令 $u=x^2\\to 0$，用等价无穷小 $\\sin u \\sim u$，得 $\\dfrac{u}{3u} = \\dfrac{1}{3}$。',
      },
      {
        id: 'c15-b-q6',
        type: 'match',
        prompt: '将各表达式与其导数（对 $x$ 求导）配对：',
        left: [
          '$\\displaystyle\\int_0^x \\cos t\\,dt$',
          '$\\displaystyle\\int_0^{x^2} \\cos t\\,dt$',
          '$\\displaystyle\\int_x^0 \\cos t\\,dt$',
          '$\\displaystyle\\int_1^3 \\cos t\\,dt$',
        ],
        right: [
          '$\\cos x$',
          '$2x\\cos(x^2)$',
          '$-\\cos x$',
          '$0$（常数）',
        ],
        explain: '第一：直接代入上限。第二：链式法则，乘 $(x^2)\'=2x$。第三：交换上下限变号，得 $-\\cos x$。第四：定积分是常数，导数为 $0$。',
      },
      {
        id: 'c15-b-q7',
        type: 'input',
        prompt: '计算 $\\displaystyle\\int_1^e \\dfrac{\\ln x}{x}\\,dx$。（提示：令 $u = \\ln x$，答案化简后是分数，填分子即可，分母为 2）',
        accept: ['1/2', '0.5'],
        placeholder: '1/2',
        explain: '令 $u = \\ln x$，$du = \\frac{1}{x}dx$，$x=1\\Rightarrow u=0$，$x=e\\Rightarrow u=1$。积分变为 $\\int_0^1 u\\,du = \\frac{u^2}{2}\\big|_0^1 = \\frac{1}{2}$。注意换元后直接代入新上下限 $0$ 和 $1$，不必换回 $x$。',
      },
      {
        id: 'c15-b-q8',
        type: 'judge',
        prompt: '微积分第一基本定理说明：对连续函数 $f$，变上限积分 $\\Phi(x)=\\int_a^x f(t)\\,dt$ 一定是 $f$ 的一个原函数。',
        answer: true,
        explain: '正确。第一基本定理的核心结论就是 $\\Phi\'(x)=f(x)$，即 $\\Phi$ 是 $f$ 的原函数。这是 N-L 公式成立的根基：任意原函数 $F$ 与 $\\Phi$ 只差常数，代入上下限后常数消去，所以取哪个原函数都能正确计算定积分。',
      },
    ],
  },

  // =========================================================================
  // c15-c  定积分计算技巧——对称性与换元
  // =========================================================================
  {
    id: 'c15-c',
    title: '定积分计算技巧',
    subtitle: '对称性简化 + 凑微分换元',
    intro: [
      {
        title: '为什么要学计算技巧？',
        body: '牛顿-莱布尼茨公式让我们能快速算定积分，但有些积分用"暴力"找原函数还是很繁琐。\n\n本关介绍两类强力工具：\n\n1. **对称性**：利用奇偶函数在对称区间上的特殊性质，秒杀一大类积分。\n\n2. **换元法**：把复杂的积分变量换成简单变量，同时积分限也跟着换——关键是**换元后不必换回来**，直接用新积分限算。',
        tip: '对称性是最优先检查的技巧——碰到 $\\int_{-a}^a$ 型区间，先看被积函数是奇函数还是偶函数，往往一步解决问题。',
      },
      {
        title: '奇偶函数的对称性',
        body: '设 $f$ 在 $[-a, a]$ 上连续，则：\n\n- 若 $f$ 是**奇函数**（$f(-x) = -f(x)$）：$\\displaystyle\\int_{-a}^a f(x)\\,dx = 0$\n- 若 $f$ 是**偶函数**（$f(-x) = f(x)$）：$\\displaystyle\\int_{-a}^a f(x)\\,dx = 2\\displaystyle\\int_0^a f(x)\\,dx$\n\n**推导**：令 $x=-t$ 代入 $\\int_{-a}^0 f(x)\\,dx$，得 $\\int_0^a f(-t)\\,dt$；再用奇/偶性代入，加上 $\\int_0^a f(t)\\,dt$，奇函数抵消得 0，偶函数翻倍得 $2\\int_0^a$。',
        formula: '\\int_{-a}^a f(x)\\,dx = \\begin{cases} 0 & (f \\text{ odd}) \\\\ 2\\displaystyle\\int_0^a f(x)\\,dx & (f \\text{ even}) \\end{cases}',
        tip: '检查奇偶性只需看 $f(-x)$ 是否等于 $\\pm f(x)$。对多项式，**奇次幂项**是奇函数，**偶次幂项**是偶函数——可以拆开分别处理！',
      },
      {
        title: '对称性例题：拆开处理混合函数',
        body: '**例**：计算 $\\displaystyle\\int_{-1}^1 (x^3 + x^4)\\,dx$\n\n区间 $[-1,1]$ 关于原点对称，先拆：',
        steps: [
          '$x^3$ 是奇函数：$\\displaystyle\\int_{-1}^1 x^3\\,dx = 0$',
          '$x^4$ 是偶函数：$\\displaystyle\\int_{-1}^1 x^4\\,dx = 2\\int_0^1 x^4\\,dx = 2\\cdot\\dfrac{x^5}{5}\\Big|_0^1 = \\dfrac{2}{5}$',
          '合并：$\\displaystyle\\int_{-1}^1 (x^3+x^4)\\,dx = 0 + \\dfrac{2}{5} = \\dfrac{2}{5}$',
          '对比不用对称性：$\\dfrac{x^4}{4}\\Big|_{-1}^1 + \\dfrac{x^5}{5}\\Big|_{-1}^1 = (\\frac{1}{4}-\\frac{1}{4}) + (\\frac{1}{5}+\\frac{1}{5}) = \\frac{2}{5}$，结果相同但更繁。',
        ],
      },
      {
        title: '第一类换元（凑微分）：定积分版本',
        body: '不定积分换元：凑出 $u=\\varphi(x)$，换回 $x$ 表示。\n\n定积分换元的关键优势：**换元后直接用新积分限，不必换回**。\n\n**公式**：$\\displaystyle\\int_a^b f(\\varphi(x))\\varphi\'(x)\\,dx = \\int_{\\varphi(a)}^{\\varphi(b)} f(u)\\,du$\n\n**例**：$\\displaystyle\\int_0^{\\pi/2} \\sin^3 x\\cos x\\,dx$\n\n令 $u=\\sin x$，$du=\\cos x\\,dx$；$x=0\\Rightarrow u=0$，$x=\\pi/2\\Rightarrow u=1$：\n\n$= \\displaystyle\\int_0^1 u^3\\,du = \\dfrac{u^4}{4}\\Big|_0^1 = \\dfrac{1}{4}$',
        formula: '\\int_a^b f(\\varphi(x))\\varphi\'(x)\\,dx = \\int_{\\varphi(a)}^{\\varphi(b)} f(u)\\,du',
        tip: '换元后**立刻把上下限换成新变量的值**，写在积分符号旁——这样不可能忘记，也不会犯"换元后还要换回"的错误。',
      },
      {
        title: '换元法经典：三角替换消根号',
        body: '含 $\\sqrt{a^2 - x^2}$ 的积分，令 $x = a\\sin t$（$t\\in[-\\pi/2,\\pi/2]$），则：\n$\\sqrt{a^2 - x^2} = a\\cos t$，$dx = a\\cos t\\,dt$\n\n**例**：$\\displaystyle\\int_0^1 \\sqrt{1-x^2}\\,dx$（单位圆第一象限面积）\n\n令 $x=\\sin t$，$dx=\\cos t\\,dt$；$x=0\\Rightarrow t=0$，$x=1\\Rightarrow t=\\pi/2$：\n\n$= \\displaystyle\\int_0^{\\pi/2}\\cos t \\cdot \\cos t\\,dt = \\int_0^{\\pi/2}\\cos^2 t\\,dt = \\int_0^{\\pi/2}\\dfrac{1+\\cos 2t}{2}\\,dt = \\dfrac{\\pi}{4}$\n\n**几何验证**：正好是单位圆四分之一面积 $\\dfrac{\\pi\\cdot 1^2}{4} = \\dfrac{\\pi}{4}$。',
        tip: '三角替换后积分限的变换：把 $x$ 的值代入 $x=a\\sin t$ 反解 $t$，即 $t = \\arcsin(x/a)$。记住换元后直接用 $t$ 的积分限，不必换回。',
      },
      {
        title: '分部积分法（定积分版）',
        body: '定积分分部积分公式：\n$\\displaystyle\\int_a^b u\\,dv = uv\\Big|_a^b - \\int_a^b v\\,du$\n\n选择 $u$ 的口诀：**反对幂指三**（反三角、对数、幂次、指数、三角——前面的优先选 $u$）。\n\n**例**：$\\displaystyle\\int_0^1 xe^x\\,dx$\n\n取 $u=x$，$dv=e^x\\,dx$，则 $du=dx$，$v=e^x$：\n\n$= xe^x\\Big|_0^1 - \\displaystyle\\int_0^1 e^x\\,dx = (e-0) - (e^x\\big|_0^1) = e - (e-1) = 1$',
        formula: '\\int_a^b u\\,dv = uv\\Big|_a^b - \\int_a^b v\\,du',
        tip: '"反对幂指三"——从左到右优先选 $u$：**反**三角最优先，**三**角函数最末。两个函数相乘时，前面的当 $u$，后面的当 $dv$。',
      },
      {
        title: '易错点与总结',
        body: '**易错 1**：换元后忘记换积分限。换元 $u=\\varphi(x)$ 后，积分限 $a,b$ 必须相应变成 $\\varphi(a), \\varphi(b)$，否则计算的是错误的区间。\n\n**易错 2**：面积 vs 有向积分。$\\int_0^{2\\pi}\\sin x\\,dx = 0$，但 $\\sin x$ 与 $x$ 轴围成总面积为 $4$（需加绝对值）。\n\n**计算定积分的决策树**：\n① 先看对称性（$[-a,a]$ 区间，奇偶函数）\n② 再看能否凑微分\n③ 含根号考虑三角替换\n④ 含 ln 或反三角考虑分部积分',
        reveal: {
          q: '计算 $\\displaystyle\\int_{-\\pi/2}^{\\pi/2} \\dfrac{\\cos x}{1+e^x}\\,dx$，看起来很难，有没有技巧？',
          a: '用区间再现技巧：令 $I = \\int_{-\\pi/2}^{\\pi/2}\\dfrac{\\cos x}{1+e^x}\\,dx$，再令 $x\\to -x$（区间对称），得 $I = \\int_{-\\pi/2}^{\\pi/2}\\dfrac{\\cos x}{1+e^{-x}}\\,dx = \\int_{-\\pi/2}^{\\pi/2}\\dfrac{e^x\\cos x}{1+e^x}\\,dx$。两式相加：$2I = \\int_{-\\pi/2}^{\\pi/2}\\dfrac{(1+e^x)\\cos x}{1+e^x}\\,dx = \\int_{-\\pi/2}^{\\pi/2}\\cos x\\,dx = 2$，故 $I=1$。',
        },
      },
    ],
    questions: [
      {
        id: 'c15-c-q1',
        type: 'choice',
        prompt: '$\\displaystyle\\int_{-2}^2 (x^5 + 3x^3 + x^2 + 1)\\,dx$ 等于多少？（先用对称性化简）',
        options: [
          '$\\dfrac{28}{3}$',
          '$0$',
          '$\\dfrac{28}{3} + 0 = \\dfrac{28}{3}$',
          '$\\dfrac{56}{3}$',
        ],
        answer: 0,
        explain: '$x^5 + 3x^3$ 是奇函数，在 $[-2,2]$ 上积分为 $0$；$x^2+1$ 是偶函数，积分为 $2\\int_0^2(x^2+1)\\,dx = 2\\left[\\frac{x^3}{3}+x\\right]_0^2 = 2(\\frac{8}{3}+2) = 2\\cdot\\frac{14}{3} = \\frac{28}{3}$。',
      },
      {
        id: 'c15-c-q2',
        type: 'judge',
        prompt: '定积分换元 $u = \\varphi(x)$ 之后，积分限必须同步更新为 $\\varphi(a)$ 到 $\\varphi(b)$，无需换回原变量即可直接计算。',
        answer: true,
        explain: '这是定积分换元与不定积分换元的核心区别：定积分换元后，积分限变成新变量的值，直接代入计算，无需最后换回 $x$。',
      },
      {
        id: 'c15-c-q3',
        type: 'choice',
        prompt: '计算 $\\displaystyle\\int_0^4 x\\sqrt{x^2+9}\\,dx$，令 $u = x^2+9$，$du = 2x\\,dx$，新积分限为：',
        options: [
          '$u$：从 $0$ 到 $4$',
          '$u$：从 $9$ 到 $25$',
          '$u$：从 $3$ 到 $5$',
          '$u$：从 $9$ 到 $16$',
        ],
        answer: 1,
        explain: '令 $u=x^2+9$：$x=0$ 时 $u=9$，$x=4$ 时 $u=16+9=25$。故新积分限为 $9$ 到 $25$。积分变为 $\\frac{1}{2}\\int_9^{25}\\sqrt{u}\\,du = \\frac{1}{2}\\cdot\\frac{2}{3}u^{3/2}\\big|_9^{25} = \\frac{1}{3}(125-27) = \\frac{98}{3}$。',
      },
      {
        id: 'c15-c-q4',
        type: 'input',
        prompt: '计算 $\\displaystyle\\int_0^{\\pi/2} \\sin^3 x\\cos x\\,dx$（令 $u=\\sin x$），结果是（填分数如 1/4）：',
        accept: ['1/4', '0.25'],
        placeholder: '1/4',
        explain: '令 $u=\\sin x$，$du=\\cos x\\,dx$。$x=0\\Rightarrow u=0$，$x=\\pi/2\\Rightarrow u=1$。积分变为 $\\int_0^1 u^3\\,du = \\frac{u^4}{4}\\big|_0^1 = \\frac{1}{4}$。',
      },
      {
        id: 'c15-c-q5',
        type: 'choice',
        prompt: '计算 $\\displaystyle\\int_1^e \\ln x\\,dx$（分部积分，取 $u=\\ln x$，$dv=dx$），结果是：',
        options: ['$1$', '$e-1$', '$e$', '$e+1$'],
        answer: 0,
        explain: '$u=\\ln x$，$dv=dx$；$du=\\frac{1}{x}dx$，$v=x$。$\\int_1^e\\ln x\\,dx = x\\ln x\\big|_1^e - \\int_1^e x\\cdot\\frac{1}{x}\\,dx = (e\\cdot 1 - 1\\cdot 0) - \\int_1^e 1\\,dx = e - (e-1) = 1$。',
      },
      {
        id: 'c15-c-q6',
        type: 'judge',
        prompt: '$\\displaystyle\\int_{-3}^3 x^3\\sin x\\,dx = 0$，因为 $x^3\\sin x$ 是奇函数。',
        answer: false,
        explain: '$x^3$ 是奇函数，$\\sin x$ 也是奇函数，两个奇函数相乘得**偶函数**（奇×奇=偶）。$x^3\\sin x$ 是偶函数，所以 $\\int_{-3}^3 x^3\\sin x\\,dx = 2\\int_0^3 x^3\\sin x\\,dx \\neq 0$。',
      },
      {
        id: 'c15-c-q7',
        type: 'match',
        prompt: '将各积分与最适合的计算技巧配对：',
        left: [
          '$\\displaystyle\\int_{-1}^1 (x^3 + x^4)\\,dx$',
          '$\\displaystyle\\int_0^1 x e^{x^2}\\,dx$',
          '$\\displaystyle\\int_0^1 x^2 e^x\\,dx$',
          '$\\displaystyle\\int_0^1 \\sqrt{1-x^2}\\,dx$',
        ],
        right: [
          '对称性（奇偶分离）',
          '凑微分（令 $u=x^2$）',
          '分部积分（$u=x^2$，$dv=e^x dx$）',
          '三角换元（$x=\\sin t$）',
        ],
        explain: '第一：$[-1,1]$ 对称区间，奇偶分离。第二：出现 $xe^{x^2}$，凑微分 $d(x^2)=2x\\,dx$。第三：幂次 × 指数，选幂次当 $u$ 做分部。第四：含 $\\sqrt{1-x^2}$，三角替换消根号。',
      },
      {
        id: 'c15-c-q8',
        type: 'choice',
        prompt: '设 $\\displaystyle I = \\int_0^{\\pi/2}\\dfrac{\\sin x}{\\sin x + \\cos x}\\,dx$。令 $x = \\pi/2 - t$，则 $I$ 等于 $\\displaystyle\\int_0^{\\pi/2}\\dfrac{\\cos x}{\\sin x+\\cos x}\\,dx$。由此得 $2I = $ ？',
        options: [
          '$0$',
          '$\\pi$',
          '$\\dfrac{\\pi}{2}$',
          '$1$',
        ],
        answer: 2,
        explain: '两式相加：$2I = \\int_0^{\\pi/2}\\dfrac{\\sin x + \\cos x}{\\sin x+\\cos x}\\,dx = \\int_0^{\\pi/2}1\\,dx = \\dfrac{\\pi}{2}$。故 $I = \\dfrac{\\pi}{4}$。这是区间再现技巧的经典应用。',
      },
    ],
  },
]
