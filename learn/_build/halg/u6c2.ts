export const LESSONS = [
  {
    id: 'u6c2',
    title: '复数的四则运算',
    subtitle: '加减：实虚分开；乘法：展开用 $\\mathrm{i}^2=-1$；除法：分母乘共轭；$|z_1 z_2|=|z_1||z_2|$',
    intro: [
      {
        title: '引入：运算规则从哪里来？',
        body: '复数的四则运算遵循两条设计原则：\n1. **保留实数的运算律**——加法、乘法的交换律、结合律、分配律全部成立\n2. **与实数兼容**——虚部为零时退化为普通实数运算\n\n只需把 $\\mathrm{i}$ 当作普通的代数符号，按多项式规则运算，最后用 $\\mathrm{i}^2 = -1$ 化简，所有规则**自然导出**。\n\n本章目标：掌握加减、乘法、除法的完整步骤，理解模的乘性和共轭的运算性质。',
        tip: '复数运算的核心就是"把 $\\mathrm{i}$ 当字母算，算完把 $\\mathrm{i}^2$ 换成 $-1$"。',
      },
      {
        title: '加减法：实部加实部，虚部加虚部',
        body: '设 $z_1 = a + b\\mathrm{i}$，$z_2 = c + d\\mathrm{i}$（$a,b,c,d \\in \\mathbb{R}$），定义：\n$$z_1 \\pm z_2 = (a \\pm c) + (b \\pm d)\\mathrm{i}$$\n\n**几何意义**：复数加减完全对应**向量加减**：\n- $z_1 + z_2$ 对应向量 $\\vec{OZ_1} + \\vec{OZ_2}$（平行四边形法则）\n- $z_1 - z_2$ 对应向量 $\\vec{Z_2 Z_1}$（从 $Z_2$ 指向 $Z_1$）\n\n**重要推论**：$|z_1 - z_2|$ 表示复平面内两点 $Z_1$、$Z_2$ 之间的**距离**。',
        formula: 'z_1 \\pm z_2 = (a \\pm c) + (b \\pm d)\\mathrm{i}',
      },
      {
        title: '乘法：展开后用 $\\mathrm{i}^2 = -1$ 化简',
        body: '按多项式乘法展开，最后用 $\\mathrm{i}^2 = -1$：\n$$(a + b\\mathrm{i})(c + d\\mathrm{i}) = ac + ad\\mathrm{i} + bc\\mathrm{i} + bd\\mathrm{i}^2 = (ac - bd) + (ad + bc)\\mathrm{i}$$\n\n**两个高频小结论（必背）**：\n- $(1 + \\mathrm{i})^2 = 1 + 2\\mathrm{i} + \\mathrm{i}^2 = 2\\mathrm{i}$\n- $(1 - \\mathrm{i})^2 = 1 - 2\\mathrm{i} + \\mathrm{i}^2 = -2\\mathrm{i}$\n\n由此可快速得到：\n- $\\dfrac{1+\\mathrm{i}}{1-\\mathrm{i}} = \\dfrac{(1+\\mathrm{i})^2}{(1+\\mathrm{i})(1-\\mathrm{i})} = \\dfrac{2\\mathrm{i}}{2} = \\mathrm{i}$\n- $(1+\\mathrm{i})^4 = (2\\mathrm{i})^2 = -4$',
        formula: '(a + b\\mathrm{i})(c + d\\mathrm{i}) = (ac - bd) + (ad + bc)\\mathrm{i}',
        steps: [
          '按多项式展开：$ac + ad\\mathrm{i} + bc\\mathrm{i} + bd\\mathrm{i}^2$',
          '将 $\\mathrm{i}^2 = -1$ 代入：$bd\\mathrm{i}^2 \\to -bd$',
          '合并实部 $(ac - bd)$ 和虚部 $(ad + bc)$',
        ],
      },
      {
        title: '除法：分子分母同乘分母的共轭',
        body: '**核心步骤**：分子分母同乘以分母的共轭，把分母化为实数：\n$$\\frac{a + b\\mathrm{i}}{c + d\\mathrm{i}} = \\frac{(a + b\\mathrm{i})(c - d\\mathrm{i})}{(c + d\\mathrm{i})(c - d\\mathrm{i})} = \\frac{(ac + bd) + (bc - ad)\\mathrm{i}}{c^2 + d^2}$$\n\n**倒数公式**：\n$$\\frac{1}{z} = \\frac{\\overline{z}}{|z|^2}$$\n\n**操作要点**：\n1. 看到形如 $\\dfrac{\\cdots}{c + d\\mathrm{i}}$ 的式子，立刻想到"乘 $c - d\\mathrm{i}$"\n2. 分母变成 $c^2 + d^2$（正实数）\n3. 分子用乘法展开，分离实部和虚部',
        formula: '\\frac{a + b\\mathrm{i}}{c + d\\mathrm{i}} = \\frac{ac + bd}{c^2 + d^2} + \\frac{bc - ad}{c^2 + d^2}\\,\\mathrm{i}',
        tip: '分子和分母都要乘！只乘分母会改变分数的值。这是除法最常见的失误。',
      },
      {
        title: '模的乘性与共轭的运算性质',
        body: '**模的乘性**（恒等式）：\n$$|z_1 z_2| = |z_1| \\cdot |z_2|,\\quad \\left|\\frac{z_1}{z_2}\\right| = \\frac{|z_1|}{|z_2|}$$\n\n**应用**：$\\left|\\dfrac{2 + a\\mathrm{i}}{1 - \\mathrm{i}}\\right| = \\dfrac{\\sqrt{4 + a^2}}{\\sqrt{2}}$，直接由模的商性得出，**不需要化简再开方**。\n\n**共轭的运算性质**：\n$$\\overline{z_1 \\pm z_2} = \\overline{z_1} \\pm \\overline{z_2},\\quad \\overline{z_1 z_2} = \\overline{z_1} \\cdot \\overline{z_2},\\quad \\overline{\\left(\\frac{z_1}{z_2}\\right)} = \\frac{\\overline{z_1}}{\\overline{z_2}}$$\n\n注意：$|z_1 + z_2| \\leq |z_1| + |z_2|$（三角不等式），加法**没有**乘性！',
        formula: '|z_1 z_2| = |z_1||z_2|,\\quad \\overline{z_1 z_2} = \\overline{z_1}\\cdot\\overline{z_2}',
      },
      {
        title: '实系数方程的虚根成对定理',
        body: '**定理**：若实系数多项式方程 $f(x) = 0$ 有虚根 $z = p + q\\mathrm{i}$（$q \\neq 0$），则 $\\overline{z} = p - q\\mathrm{i}$ 也是该方程的根。\n\n**证明思路**：把 $z$ 代入 $f(z) = 0$，利用共轭对运算可分配（$\\overline{a_i} = a_i$ 因为系数是实数），得 $f(\\overline{z}) = \\overline{f(z)} = 0$。\n\n**实系数二次方程**（$\\Delta < 0$ 时）的虚根：\n$$x = \\frac{-b \\pm \\sqrt{4ac - b^2}\\,\\mathrm{i}}{2a}$$\n\n**韦达定理依然成立**：$x_1 + x_2 = -\\dfrac{b}{a}$，$x_1 x_2 = \\dfrac{c}{a}$（即便根是复数）。',
        tip: '前提是"实系数"！复系数方程的虚根不一定成对。例如 $x^2 + \\mathrm{i} = 0$ 的两根不互为共轭。',
      },
      {
        title: '例题精讲与易错点',
        body: '**例 1（含参数的实部/虚部）**：$z = \\dfrac{2 + a\\mathrm{i}}{1 - \\mathrm{i}}$，求 $z$ 是实数时 $a$ 的值。\n\n分母乘共轭：$z = \\dfrac{(2+a\\mathrm{i})(1+\\mathrm{i})}{2} = \\dfrac{(2-a)+(2+a)\\mathrm{i}}{2}$。\n$z$ 是实数 $\\Leftrightarrow$ 虚部 $= 0$：$2 + a = 0$，$a = -2$。\n\n**例 2（高次幂化简）**：$\\dfrac{(1+\\mathrm{i})^{10}}{(1-\\mathrm{i})^8}$。\n$(1+\\mathrm{i})^{10} = [(1+\\mathrm{i})^2]^5 = (2\\mathrm{i})^5 = 32\\mathrm{i}$；$(1-\\mathrm{i})^8 = [(-2\\mathrm{i})]^4 = 16$。答：$2\\mathrm{i}$。\n\n**三大易错点**：① 除法忘了分子也要乘；② 乘法展开漏掉 $bd\\mathrm{i}^2 \\to -bd$；③ 误以为 $|z_1+z_2| = |z_1|+|z_2|$（只有乘法才有等号）。',
        reveal: {
          q: '已知 $1 - 2\\mathrm{i}$ 是实系数方程 $x^2 + px + q = 0$ 的一个根，求 $p$、$q$。',
          a: '实系数方程虚根成对，另一根为 $1 + 2\\mathrm{i}$。由韦达定理：$p = -(x_1 + x_2) = -(2) = -2$；$q = x_1 x_2 = (1-2\\mathrm{i})(1+2\\mathrm{i}) = 1 + 4 = 5$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '计算 $(2 + 3\\mathrm{i}) - (1 - \\mathrm{i})$ 的结果是（  ）。',
        options: [
          '$1 + 2\\mathrm{i}$',
          '$1 + 4\\mathrm{i}$',
          '$3 + 2\\mathrm{i}$',
          '$3 + 4\\mathrm{i}$',
        ],
        answer: 1,
        explain: '$(2 + 3\\mathrm{i}) - (1 - \\mathrm{i}) = (2 - 1) + (3 - (-1))\\mathrm{i} = 1 + 4\\mathrm{i}$。',
      },
      {
        type: 'choice',
        prompt: '计算 $(1 + 2\\mathrm{i})(3 - \\mathrm{i})$ 的结果是（  ）。',
        options: [
          '$5 + 5\\mathrm{i}$',
          '$3 - 2\\mathrm{i}$',
          '$1 + 5\\mathrm{i}$',
          '$5 - \\mathrm{i}$',
        ],
        answer: 0,
        explain: '$(1+2\\mathrm{i})(3-\\mathrm{i}) = 3 - \\mathrm{i} + 6\\mathrm{i} - 2\\mathrm{i}^2 = 3 + 5\\mathrm{i} - 2(-1) = 5 + 5\\mathrm{i}$。',
      },
      {
        type: 'input',
        prompt: '计算 $\\dfrac{3 + \\mathrm{i}}{1 - \\mathrm{i}}$，将结果写成 $a + b\\mathrm{i}$ 的形式（填写如 $1+2i$）。',
        accept: ['1+2i', '1 + 2i', '1+2\\mathrm{i}'],
        explain: '分子分母同乘 $1 + \\mathrm{i}$：$\\dfrac{(3+\\mathrm{i})(1+\\mathrm{i})}{(1-\\mathrm{i})(1+\\mathrm{i})} = \\dfrac{3 + 3\\mathrm{i} + \\mathrm{i} + \\mathrm{i}^2}{1+1} = \\dfrac{(3-1) + 4\\mathrm{i}}{2} = \\dfrac{2 + 4\\mathrm{i}}{2} = 1 + 2\\mathrm{i}$。',
      },
      {
        type: 'judge',
        prompt: '对任意复数 $z_1$、$z_2$，都有 $|z_1 + z_2| = |z_1| + |z_2|$。',
        answer: false,
        explain: '这是错误的。正确结论是三角不等式 $|z_1 + z_2| \\leq |z_1| + |z_2|$，等号成立当且仅当 $z_1$、$z_2$ "同向"（$z_2 = \\lambda z_1$，$\\lambda \\geq 0$）。**乘法**才有 $|z_1 z_2| = |z_1||z_2|$（恒等式）。',
      },
      {
        type: 'choice',
        prompt: '设 $z = \\dfrac{1 + a\\mathrm{i}}{2 - \\mathrm{i}}$（$a \\in \\mathbb{R}$）是纯虚数，则 $a$ 等于（  ）。',
        options: [
          '$a = -2$',
          '$a = 2$',
          '$a = \\dfrac{1}{2}$',
          '$a = -\\dfrac{1}{2}$',
        ],
        answer: 1,
        explain: '化简：$z = \\dfrac{(1+a\\mathrm{i})(2+\\mathrm{i})}{(2-\\mathrm{i})(2+\\mathrm{i})} = \\dfrac{(2-a)+(1+2a)\\mathrm{i}}{5}$。纯虚数要求实部 $= 0$ 且虚部 $\\neq 0$：$2 - a = 0 \\Rightarrow a = 2$；验证虚部 $1 + 2(2) = 5 \\neq 0$，符合。故 $a = 2$。',
      },
      {
        type: 'match',
        prompt: '将下列运算结果与正确答案匹配。',
        left: [
          '$(1+\\mathrm{i})^2$',
          '$(1-\\mathrm{i})^4$',
          '$\\dfrac{1+\\mathrm{i}}{1-\\mathrm{i}}$',
        ],
        right: [
          '$\\mathrm{i}$',
          '$2\\mathrm{i}$',
          '$-4$',
        ],
        explain: '$(1+\\mathrm{i})^2 = 1+2\\mathrm{i}-1 = 2\\mathrm{i}$；$(1-\\mathrm{i})^4 = [(1-\\mathrm{i})^2]^2 = (-2\\mathrm{i})^2 = -4$；$\\dfrac{1+\\mathrm{i}}{1-\\mathrm{i}} = \\dfrac{(1+\\mathrm{i})^2}{2} = \\dfrac{2\\mathrm{i}}{2} = \\mathrm{i}$。',
      },
      {
        type: 'choice',
        prompt: '已知 $2 + 3\\mathrm{i}$ 是实系数方程 $x^2 + px + q = 0$ 的一个根，则 $p + q$ 等于（  ）。',
        options: [
          '$9$',
          '$-9$',
          '$13$',
          '$17$',
        ],
        answer: 0,
        explain: '实系数方程的虚根成对，另一根为 $2 - 3\\mathrm{i}$。由韦达定理：$x_1 + x_2 = 4 = -p$，故 $p = -4$；$x_1 x_2 = (2+3\\mathrm{i})(2-3\\mathrm{i}) = 4+9 = 13 = q$。因此 $p + q = -4 + 13 = 9$。',
      },
      {
        type: 'input',
        prompt: '计算 $\\dfrac{(1+\\mathrm{i})^{10}}{(1-\\mathrm{i})^8}$，结果写成 $a + b\\mathrm{i}$ 的形式（如 $2i$）。',
        accept: ['2i', '2\\mathrm{i}', '0+2i'],
        explain: '$(1+\\mathrm{i})^2 = 2\\mathrm{i}$，故 $(1+\\mathrm{i})^{10} = (2\\mathrm{i})^5 = 32\\mathrm{i}^5 = 32\\mathrm{i}$（因为 $\\mathrm{i}^5 = \\mathrm{i}$）。$(1-\\mathrm{i})^2 = -2\\mathrm{i}$，故 $(1-\\mathrm{i})^8 = (-2\\mathrm{i})^4 = 16\\mathrm{i}^4 = 16$。结果为 $\\dfrac{32\\mathrm{i}}{16} = 2\\mathrm{i}$。',
      },
    ],
  },
]
