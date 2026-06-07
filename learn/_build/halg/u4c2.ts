export const LESSONS = [
  {
    id: 'u4c2',
    title: '对数运算与对数函数',
    subtitle: '$\\log_a b = N \\Leftrightarrow a^N = b$，三大法则，换底约分，与指数互为反函数',
    intro: [
      {
        title: '引入：链式乘积的巧解',
        body: '计算 $\\log_2 3 \\cdot \\log_3 4 \\cdot \\log_4 5 \\cdot \\log_5 8$。\n\n直接用小数逼近很繁琐。仔细观察：第一项的真数 $3$ 恰好是第二项的底数；第二项的真数 $4$ 是第三项的底数……这是一条**链**。\n\n将每项用换底公式统一成自然对数：各分子分母首尾相消，只剩 $\\dfrac{\\ln 8}{\\ln 2} = \\log_2 8 = 3$。\n\n**一般化**：$\\log_{a_1} a_2 \\cdot \\log_{a_2} a_3 \\cdots \\log_{a_{n-1}} a_n = \\log_{a_1} a_n$（对数链式法则）。',
        steps: [
          '换底：$\\log_2 3 = \\dfrac{\\ln 3}{\\ln 2}$，$\\log_3 4 = \\dfrac{\\ln 4}{\\ln 3}$，$\\log_4 5 = \\dfrac{\\ln 5}{\\ln 4}$，$\\log_5 8 = \\dfrac{\\ln 8}{\\ln 5}$',
          '连乘：$\\dfrac{\\ln 3}{\\ln 2} \\cdot \\dfrac{\\ln 4}{\\ln 3} \\cdot \\dfrac{\\ln 5}{\\ln 4} \\cdot \\dfrac{\\ln 8}{\\ln 5} = \\dfrac{\\ln 8}{\\ln 2}$（中间全约掉）',
          '$\\dfrac{\\ln 8}{\\ln 2} = \\dfrac{3\\ln 2}{\\ln 2} = 3$',
        ],
      },
      {
        title: '对数的定义与基本恒等式',
        body: '**定义**：若 $a^N = b$（$a > 0$，$a \\ne 1$，$b > 0$），则称 $N$ 为以 $a$ 为底 $b$ 的对数，记作 $N = \\log_a b$。\n\n- 底数 $a > 0$ 且 $a \\ne 1$（与指数函数相同的限制）\n- 真数 $b > 0$（对数的真数必须为正数！）\n- 对数值 $N$ 可以是任意实数\n\n**特殊记法**：$\\log_{10} b$ 简记为 $\\lg b$；$\\log_e b$ 简记为 $\\ln b$（$e \\approx 2.718$）。\n\n**四个基本恒等式**：$a^{\\log_a b} = b$，$\\log_a a^n = n$，$\\log_a 1 = 0$，$\\log_a a = 1$。',
        formula: '\\log_a b = N \\Longleftrightarrow a^N = b \\quad (a>0,\\ a\\ne 1,\\ b>0)',
        tip: '最常犯的错误：令真数为负数或零。如 $\\log_2(-3)$ 无意义，因为 $2^x > 0$ 恒成立，不可能等于 $-3$。',
      },
      {
        title: '三大运算法则',
        body: '设 $a > 0$，$a \\ne 1$，$M > 0$，$N > 0$，$n \\in \\mathbb{R}$：\n\n**乘积法则**：$\\log_a(MN) = \\log_a M + \\log_a N$（真数相乘 $\\to$ 对数相加）\n\n**商的法则**：$\\log_a \\dfrac{M}{N} = \\log_a M - \\log_a N$（真数相除 $\\to$ 对数相减）\n\n**幂的法则**：$\\log_a M^n = n\\log_a M$（真数的指数 $\\to$ 对数的系数）\n\n**化简示例**：$2\\lg 2 + \\lg 25 = \\lg 4 + \\lg 25 = \\lg 100 = 2$。',
        formula: '\\log_a(MN) = \\log_a M + \\log_a N,\\quad \\log_a M^n = n\\log_a M',
        tip: '反用法则：加法合并成乘法，减法合并成除法，系数压进指数。逆方向同样重要。',
      },
      {
        title: '换底公式与互倒关系',
        body: '**换底公式**：$\\log_a b = \\dfrac{\\log_c b}{\\log_c a}$（$c > 0$，$c \\ne 1$）。常换成 $\\lg$ 或 $\\ln$：$\\log_a b = \\dfrac{\\lg b}{\\lg a} = \\dfrac{\\ln b}{\\ln a}$。\n\n**互倒关系**：$\\log_a b \\cdot \\log_b a = 1$，即 $\\log_a b = \\dfrac{1}{\\log_b a}$。\n\n**应用示例**：计算 $\\log_4 9 \\cdot \\log_3 8 = \\dfrac{\\lg 9}{\\lg 4} \\cdot \\dfrac{\\lg 8}{\\lg 3} = \\dfrac{2\\lg 3}{2\\lg 2} \\cdot \\dfrac{3\\lg 2}{\\lg 3} = 3$。',
        formula: '\\log_a b = \\frac{\\ln b}{\\ln a},\\quad \\log_a b \\cdot \\log_b a = 1',
        tip: '换底公式的核心价值：把不同底的对数变成同底的分式，创造约分机会。',
      },
      {
        title: '对数函数的图象与性质',
        body: '**对数函数** $y = \\log_a x$（$a > 0$，$a \\ne 1$，$x > 0$）：\n\n- 定义域 $(0, +\\infty)$，值域 $\\mathbb{R}$，恒过点 $(1, 0)$\n- $a > 1$：严格递增；$x \\to 0^+$ 时 $y \\to -\\infty$，$x \\to +\\infty$ 时 $y \\to +\\infty$\n- $0 < a < 1$：严格递减；$x \\to 0^+$ 时 $y \\to +\\infty$，$x \\to +\\infty$ 时 $y \\to -\\infty$\n\n**与指数函数的关系**：$y = \\log_a x$ 与 $y = a^x$ 互为**反函数**，图象关于直线 $y = x$ 对称。\n\n口诀：指数函数值域正，对数函数定义域正；两者图象关于 $y = x$ 互镜。',
        tip: '对数函数定义域只含正数，$x \\leq 0$ 无意义。复合对数 $y = \\log_a f(x)$ 要先解 $f(x) > 0$。',
      },
      {
        title: '例题：复合对数函数的定义域与单调区间',
        body: '求 $y = \\log_{1/3}(x^2 - x - 2)$ 的定义域与单调递减区间。\n\n**定义域**：真数 $x^2 - x - 2 > 0$，即 $(x-2)(x+1) > 0$，解得 $x < -1$ 或 $x > 2$。\n\n**单调性**（复合函数，同增异减）：\n- 内层 $g(x) = x^2 - x - 2$，顶点 $x = \\dfrac{1}{2}$，在 $(-\\infty, \\frac{1}{2})$ 递减，在 $(\\frac{1}{2}, +\\infty)$ 递增\n- 外层 $\\log_{1/3}$：底数 $< 1$，递减\n- 内减外减（同号）$\\to$ 整体递增；内增外减（异号）$\\to$ 整体递减\n\n与定义域取交集：递减区间为 $(2, +\\infty)$。',
        steps: [
          '解真数 $> 0$：$(x-2)(x+1) > 0$，定义域 $(-\\infty,-1)\\cup(2,+\\infty)$',
          '内层顶点 $x=1/2$；定义域内 $(-\\infty,-1)$ 段内层递减，$(2,+\\infty)$ 段内层递增',
          '外层 $\\log_{1/3}$ 递减：内减外减 $\\to$ 递增；内增外减 $\\to$ 递减',
          '单调递减区间：$(2,+\\infty)$',
        ],
      },
      {
        title: '例题：对数方程',
        body: '解方程 $\\log_x 4 + \\log_4 x = \\dfrac{5}{2}$（$x > 0$，$x \\ne 1$）。\n\n设 $t = \\log_4 x$，由互倒关系 $\\log_x 4 = \\dfrac{1}{t}$。\n\n方程变为 $\\dfrac{1}{t} + t = \\dfrac{5}{2}$，两边乘 $t$（$t \\ne 0$）：$2t^2 - 5t + 2 = 0$。\n\n$(2t-1)(t-2) = 0$，解得 $t = \\dfrac{1}{2}$ 或 $t = 2$。\n\n- $t = \\dfrac{1}{2}$：$\\log_4 x = \\dfrac{1}{2}$，$x = 4^{1/2} = 2$\n- $t = 2$：$\\log_4 x = 2$，$x = 16$\n\n验证均满足条件。答：$x = 2$ 或 $x = 16$。',
        steps: [
          '令 $t = \\log_4 x$，则 $\\log_x 4 = 1/t$',
          '代入得 $1/t + t = 5/2$，整理为 $2t^2 - 5t + 2 = 0$',
          '$(2t-1)(t-2) = 0$，$t = 1/2$ 或 $t = 2$',
          '还原：$x = 2$ 或 $x = 16$，均满足 $x > 0$，$x \\ne 1$',
        ],
      },
      {
        title: '易错点与小结',
        body: '**易错 1**：真数必须为正，忘记验根。对数方程求解后要检验是否使真数大于零。\n\n**易错 2**：换底公式分子分母搞反。$\\log_a b = \\dfrac{\\lg b}{\\lg a}$，分子是目标真数，分母是原底数。\n\n**易错 3**：复合对数函数单调性判断时"同增异减"的主语——是内层与外层的方向比较，方向相同则整体增，方向相反则整体减。\n\n**易错 4**：$\\log_a(-x)$ 定义域解法：$-x > 0$，即 $x < 0$，不要漏掉负号。',
        reveal: {
          q: '已知 $\\log_2 3 = m$，用 $m$ 表示 $\\log_4 6$。',
          a: '$\\log_4 6 = \\dfrac{\\lg 6}{\\lg 4} = \\dfrac{\\lg 2 + \\lg 3}{2\\lg 2} = \\dfrac{1 + \\log_2 3}{2} = \\dfrac{1+m}{2}$。',
        },
      },
    ],
    questions: [
      {
        type: 'judge',
        prompt: '$\\log_3(-9)$ 的值是 $-2$。',
        answer: false,
        explain: '对数的真数必须为正数。$-9 < 0$，$\\log_3(-9)$ 无意义，不存在实数值。',
      },
      {
        type: 'choice',
        prompt: '化简 $2\\lg 5 + \\lg 4$，结果等于（  ）。',
        options: ['$1$', '$2$', '$\\lg 54$', '$\\lg 100$'],
        answer: 1,
        explain: '$2\\lg 5 + \\lg 4 = \\lg 25 + \\lg 4 = \\lg(25 \\times 4) = \\lg 100 = 2$。',
      },
      {
        type: 'input',
        prompt: '计算 $\\log_2 3 \\cdot \\log_3 4 \\cdot \\log_4 8$，结果为？',
        accept: ['3'],
        explain: '链式法则：$\\log_2 3 \\cdot \\log_3 4 \\cdot \\log_4 8 = \\log_2 8 = \\log_2 2^3 = 3$。',
      },
      {
        type: 'choice',
        prompt: '函数 $y = \\log_{0.5}(3-x)$ 的单调递增区间是（  ）。',
        options: [
          '$(3, +\\infty)$',
          '$(-\\infty, 3)$',
          '$(-\\infty, 3]$',
          '无单调递增区间',
        ],
        answer: 1,
        explain: '$y = \\log_{0.5}(3-x)$：定义域 $3-x > 0$，即 $x < 3$。底数 $0.5 \\in (0,1)$，外层递减；内层 $3-x$ 在整个定义域上递减。内减（$-1$斜率）外减 $\\to$ 整体递增。单调递增区间为 $(-\\infty, 3)$（开区间，$x=3$ 不在定义域内）。',
      },
      {
        type: 'input',
        prompt: '已知 $\\lg 2 = a$，$\\lg 3 = b$，则 $\\lg 24 = $？（用含 $a$，$b$ 的式子表示，如 $3a+b$）',
        accept: ['3a+b'],
        explain: '$\\lg 24 = \\lg(8 \\times 3) = \\lg 2^3 + \\lg 3 = 3\\lg 2 + \\lg 3 = 3a + b$。',
      },
      {
        type: 'judge',
        prompt: '$y = \\log_a x$ 与 $y = a^x$ 的图象关于直线 $y = x$ 对称。',
        answer: true,
        explain: '$y = \\log_a x$ 是 $y = a^x$ 的反函数，互为反函数的图象关于直线 $y = x$ 对称。将 $y = a^x$ 中 $x$ 与 $y$ 互换即得 $x = a^y$，即 $y = \\log_a x$。',
      },
      {
        type: 'match',
        prompt: '将下列表达式与化简结果配对。',
        left: [
          '$\\log_3 27$',
          '$\\log_4 2$',
          '$\\log_5 1$',
        ],
        right: ['$0$', '$3$', '$\\dfrac{1}{2}$'],
        explain: '$\\log_3 27 = \\log_3 3^3 = 3$；$\\log_4 2 = \\log_4 4^{1/2} = \\dfrac{1}{2}$；$\\log_5 1 = 0$（任何合法底数的 $0$ 次方等于 $1$）。',
      },
      {
        type: 'choice',
        prompt: '求函数 $y = \\log_2(3 - 2x - x^2)$ 的单调递增区间（  ）。',
        options: [
          '$(-3, -1)$',
          '$(-1, 1)$',
          '$(-3, 1)$',
          '$(-1, 3)$',
        ],
        answer: 0,
        explain: '定义域：$3-2x-x^2 > 0$，即 $x^2+2x-3 < 0$，$(x+3)(x-1) < 0$，$-3 < x < 1$。内层 $g(x) = -(x+1)^2+4$，顶点 $x=-1$，在 $(-3,-1)$ 上递增，$(-1,1)$ 上递减。外层 $\\log_2$ 递增。内增外增 $\\to$ 整体递增，区间为 $(-3,-1)$。',
      },
    ],
  },
]
