export const LESSONS = [
  {
    id: 'u5c4',
    title: '同角三角恒等式与诱导公式',
    subtitle: '$\\sin^2\\alpha + \\cos^2\\alpha = 1$；诱导口诀：奇变偶不变，符号看象限',
    intro: [
      {
        title: '引入：已知 $\\sin\\alpha$ 求 $\\cos\\alpha$ 的完整思路',
        body: '**题目**：已知 $\\sin\\alpha = \\dfrac{3}{5}$，且 $\\alpha$ 在第二象限，求 $\\cos\\alpha$ 和 $\\tan\\alpha$。\n\n很多同学直接写 $\\cos\\alpha = \\dfrac{4}{5}$——这是错的，忽略了"第二象限"这个关键条件！\n\n**正确三步法**：\n1. 用勾股恒等式求 $|\\cos\\alpha|$：$\\cos^2\\alpha = 1 - \\left(\\dfrac{3}{5}\\right)^2 = \\dfrac{16}{25}$，$|\\cos\\alpha| = \\dfrac{4}{5}$\n2. 用象限定符号：第二象限 $\\cos < 0$，故 $\\cos\\alpha = -\\dfrac{4}{5}$\n3. 用商式求正切：$\\tan\\alpha = \\dfrac{3/5}{-4/5} = -\\dfrac{3}{4}$\n\n**验证**：第二象限"$\\sin > 0$，$\\cos < 0$，$\\tan < 0$"——完全自洽。',
        tip: '三步缺一不可：绝对值（勾股）→ 符号（象限）→ 正切（商式）。',
      },
      {
        title: '三大同角恒等式',
        body: '**勾股型**（最重要）：由单位圆上 $(\\cos\\alpha, \\sin\\alpha)$ 满足 $x^2 + y^2 = 1$ 直接得到：\n$$\\sin^2\\alpha + \\cos^2\\alpha = 1$$\n\n**常用变形**：\n- $\\sin^2\\alpha = 1 - \\cos^2\\alpha$，$\\cos^2\\alpha = 1 - \\sin^2\\alpha$\n- $(\\sin\\alpha + \\cos\\alpha)^2 = 1 + 2\\sin\\alpha\\cos\\alpha$\n- $(\\sin\\alpha - \\cos\\alpha)^2 = 1 - 2\\sin\\alpha\\cos\\alpha$\n\n**商式恒等式**：由 $\\tan\\alpha = \\dfrac{y}{x} = \\dfrac{\\sin\\alpha}{\\cos\\alpha}$ 直接得到（$\\cos\\alpha \\ne 0$）：\n$$\\tan\\alpha = \\frac{\\sin\\alpha}{\\cos\\alpha}$$\n\n**平方和型**（两边除以 $\\cos^2\\alpha$）：\n$$1 + \\tan^2\\alpha = \\frac{1}{\\cos^2\\alpha}$$',
        formula: '\\sin^2\\alpha + \\cos^2\\alpha = 1,\\quad \\tan\\alpha = \\frac{\\sin\\alpha}{\\cos\\alpha}',
        image: 'gz-p5-04-1.svg',
        imageCaption: '单位圆上 $\\sin^2\\alpha + \\cos^2\\alpha = 1$ 的几何意义',
      },
      {
        title: '诱导公式：什么是"奇变偶不变，符号看象限"？',
        body: '**诱导公式**的目标：把 $\\dfrac{k\\pi}{2} \\pm \\alpha$ 的三角函数，化为 $\\alpha$（锐角）的三角函数。\n\n**口诀拆解**：\n- "奇偶"指 $k$ 是奇数还是偶数（$\\dfrac{k\\pi}{2}$ 中的 $k$）\n- "变"指正弦与余弦互换；"不变"指函数名不变\n- "符号看象限"：把 $\\alpha$ 当锐角，判断 $\\dfrac{k\\pi}{2} \\pm \\alpha$ 落在哪个象限，原函数（变换前的）在该象限的符号即为结果的符号\n\n**举例**：$\\sin\\!\\left(\\pi + \\alpha\\right)$：$k = 2$（偶数）$\\to$ 不变，还是 $\\sin$；$\\pi + \\alpha$ 在第三象限，$\\sin < 0$，故 $\\sin(\\pi + \\alpha) = -\\sin\\alpha$。',
        tip: '口诀的关键：先数 $k$ 奇偶决定"变不变"，再判断整体角所在象限决定正负号。不要背结果，用口诀现推。',
      },
      {
        title: '六组常用诱导公式',
        body: '| 变换 | $\\sin$ 的结果 | $\\cos$ 的结果 | $\\tan$ 的结果 |\n|------|------|------|------|\n| $-\\alpha$（偶不变） | $-\\sin\\alpha$ | $\\cos\\alpha$ | $-\\tan\\alpha$ |\n| $\\pi - \\alpha$（偶不变） | $\\sin\\alpha$ | $-\\cos\\alpha$ | $-\\tan\\alpha$ |\n| $\\pi + \\alpha$（偶不变） | $-\\sin\\alpha$ | $-\\cos\\alpha$ | $\\tan\\alpha$ |\n| $2\\pi - \\alpha$（偶不变） | $-\\sin\\alpha$ | $\\cos\\alpha$ | $-\\tan\\alpha$ |\n| $\\dfrac{\\pi}{2} - \\alpha$（奇变） | $\\cos\\alpha$ | $\\sin\\alpha$ | $\\dfrac{1}{\\tan\\alpha}$ |\n| $\\dfrac{\\pi}{2} + \\alpha$（奇变） | $\\cos\\alpha$ | $-\\sin\\alpha$ | $-\\dfrac{1}{\\tan\\alpha}$ |\n\n**重点记忆**：$\\sin(\\pi - \\alpha) = \\sin\\alpha$（终边关于 $y$ 轴对称，纵坐标不变）；$\\sin(\\pi + \\alpha) = -\\sin\\alpha$（关于原点对称，纵坐标取反）。',
        steps: [
          '确定 $k$ 值（$\\dfrac{k\\pi}{2}$ 的 $k$），判断奇偶',
          '奇数：函数名 $\\sin \\leftrightarrow \\cos$ 互换；偶数：函数名不变',
          '把 $\\alpha$ 当锐角，判断整体角所在象限',
          '该象限中原函数（变换后的那个函数）为正还是负，即为结果的符号',
        ],
      },
      {
        title: '同角恒等式的进阶变形',
        body: '**已知 $\\sin\\alpha + \\cos\\alpha = k$，求 $\\sin\\alpha\\cos\\alpha$**：\n\n两边平方：$(\\sin\\alpha + \\cos\\alpha)^2 = 1 + 2\\sin\\alpha\\cos\\alpha = k^2$\n$$\\sin\\alpha\\cos\\alpha = \\frac{k^2 - 1}{2}$$\n\n**已知 $\\tan\\alpha = t$，求 $\\dfrac{\\sin\\alpha - 2\\cos\\alpha}{\\sin\\alpha + \\cos\\alpha}$**：\n\n分子分母同除以 $\\cos\\alpha$（$\\cos\\alpha \\ne 0$）：\n$$= \\frac{\\tan\\alpha - 2}{\\tan\\alpha + 1} = \\frac{t - 2}{t + 1}$$\n\n**综合化简例**：化简 $\\dfrac{\\sin(\\pi+\\alpha)\\cos(2\\pi-\\alpha)}{\\tan(-\\alpha-\\pi)\\sin(\\pi-\\alpha)}$\n\n$\\sin(\\pi+\\alpha) = -\\sin\\alpha$；$\\cos(2\\pi-\\alpha) = \\cos\\alpha$；$\\tan(-\\alpha-\\pi) = -\\tan\\alpha$；$\\sin(\\pi-\\alpha) = \\sin\\alpha$，代入化简得 $\\dfrac{\\cos^2\\alpha}{\\sin\\alpha}$。',
        tip: '见到分式含 $\\tan$ 时，优先把分子分母同除以 $\\cos\\alpha$，将分式转化成 $\\tan\\alpha$ 的代数表达式，化简往往最简洁。',
      },
      {
        title: '典型例题：从已知求三角值',
        body: '**例**：已知 $\\cos\\theta = -\\dfrac{\\sqrt{5}}{5}$，$\\theta$ 在第二象限，求 $\\sin\\theta$ 和 $\\tan\\theta$。\n\n**第一步（勾股求绝对值）**：\n$$\\sin^2\\theta = 1 - \\cos^2\\theta = 1 - \\frac{1}{5} = \\frac{4}{5}$$\n\n**第二步（象限定符号）**：第二象限 $\\sin\\theta > 0$，故 $\\sin\\theta = \\dfrac{2}{\\sqrt{5}} = \\dfrac{2\\sqrt{5}}{5}$\n\n**第三步（商式求正切）**：\n$$\\tan\\theta = \\frac{\\sin\\theta}{\\cos\\theta} = \\frac{2\\sqrt{5}/5}{-\\sqrt{5}/5} = -2$$\n\n**验证**：$\\sin^2\\theta + \\cos^2\\theta = \\dfrac{4}{5} + \\dfrac{1}{5} = 1$ ✓；$\\tan\\theta = \\dfrac{\\sin}{\\cos} = -2$ ✓。',
      },
      {
        title: '易错点总结',
        body: '**易错 1**：由 $\\sin^2\\alpha + \\cos^2\\alpha = 1$ 求到 $\\cos^2\\alpha = \\dfrac{16}{25}$ 后，直接写 $\\cos\\alpha = \\dfrac{4}{5}$，忘记用象限符号——最常见失分！\n\n**易错 2**：$\\tan\\alpha = \\dfrac{\\sin\\alpha}{\\cos\\alpha}$ 在 $\\cos\\alpha = 0$ 时不可用，需先检查分母。\n\n**易错 3**：诱导公式用"函数名"变换时，仅 $\\sin \\leftrightarrow \\cos$，正切有其自身变换规律（半周期时 $\\tan$ 不变；四分之一周期时 $\\tan \\to \\cot$）。\n\n**易错 4**：化简诱导公式后忘记验证符号，"看象限"这一步不可省略。',
        reveal: {
          q: '已知 $\\sin\\alpha + \\cos\\alpha = \\dfrac{\\sqrt{2}}{2}$，求 $\\tan\\alpha + \\dfrac{1}{\\tan\\alpha}$ 的值。',
          a: '两边平方：$1 + 2\\sin\\alpha\\cos\\alpha = \\dfrac{1}{2}$，故 $\\sin\\alpha\\cos\\alpha = -\\dfrac{1}{4}$。目标式 $= \\dfrac{\\sin^2\\alpha + \\cos^2\\alpha}{\\sin\\alpha\\cos\\alpha} = \\dfrac{1}{-1/4} = -4$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '已知 $\\sin\\alpha = \\dfrac{5}{13}$，$\\alpha$ 在第一象限，则 $\\tan\\alpha = $（  ）。',
        options: [
          '$\\dfrac{5}{12}$',
          '$-\\dfrac{5}{12}$',
          '$\\dfrac{12}{13}$',
          '$\\dfrac{5}{13}$',
        ],
        answer: 0,
        explain: '$\\cos^2\\alpha = 1 - \\dfrac{25}{169} = \\dfrac{144}{169}$，第一象限 $\\cos\\alpha > 0$，故 $\\cos\\alpha = \\dfrac{12}{13}$。$\\tan\\alpha = \\dfrac{5/13}{12/13} = \\dfrac{5}{12}$。',
      },
      {
        type: 'judge',
        prompt: '$\\sin(\\pi - \\alpha) = -\\sin\\alpha$ 对所有 $\\alpha$ 成立。',
        answer: false,
        explain: '$\\pi - \\alpha$ 中，$k = 2$（$\\pi = \\dfrac{2\\pi}{2}$，$k = 2$，偶数），函数名不变，仍为 $\\sin$；$\\pi - \\alpha$ 在第二象限（$\\alpha$ 为锐角时），$\\sin > 0$，故 $\\sin(\\pi - \\alpha) = +\\sin\\alpha$，符号为正，不是负号。',
      },
      {
        type: 'choice',
        prompt: '化简 $\\sin\\!\\left(\\dfrac{3\\pi}{2} - \\alpha\\right) + \\cos(\\pi + \\alpha)$，结果为（  ）。',
        options: [
          '$-2\\cos\\alpha$',
          '$2\\cos\\alpha$',
          '$-2\\sin\\alpha$',
          '$0$',
        ],
        answer: 0,
        explain: '$\\sin\\!\\left(\\dfrac{3\\pi}{2} - \\alpha\\right)$：$k = 3$（奇），$\\sin \\to \\cos$；第三象限 $\\cos < 0$，故 $= -\\cos\\alpha$。$\\cos(\\pi + \\alpha)$：$k = 2$（偶），不变；第三象限 $\\cos < 0$，故 $= -\\cos\\alpha$。合并：$-\\cos\\alpha + (-\\cos\\alpha) = -2\\cos\\alpha$。',
      },
      {
        type: 'input',
        prompt: '已知 $\\tan\\alpha = 2$，求 $\\dfrac{\\sin\\alpha - 2\\cos\\alpha}{\\sin\\alpha + \\cos\\alpha}$ 的值（填整数或分数）。',
        accept: ['0'],
        explain: '分子分母同除以 $\\cos\\alpha$：$\\dfrac{\\tan\\alpha - 2}{\\tan\\alpha + 1} = \\dfrac{2 - 2}{2 + 1} = 0$。',
      },
      {
        type: 'choice',
        prompt: '已知 $\\cos\\alpha < 0$ 且 $\\tan\\alpha > 0$，则 $\\alpha$ 的终边在（  ）。',
        options: [
          '第一象限',
          '第二象限',
          '第三象限',
          '第四象限',
        ],
        answer: 2,
        explain: '$\\cos\\alpha < 0$ 说明终边在第二或三象限；$\\tan\\alpha > 0$ 说明终边在第一或三象限。取交集，终边在第三象限。',
      },
      {
        type: 'match',
        prompt: '将各诱导公式与结果对应（$\\alpha$ 为任意角）。',
        left: [
          '$\\cos(2\\pi - \\alpha)$',
          '$\\sin(\\pi + \\alpha)$',
          '$\\tan(-\\alpha)$',
        ],
        right: [
          '$-\\tan\\alpha$',
          '$\\cos\\alpha$',
          '$-\\sin\\alpha$',
        ],
        explain: '$\\cos(2\\pi - \\alpha) = \\cos(-\\alpha) = \\cos\\alpha$（偶函数）；$\\sin(\\pi + \\alpha) = -\\sin\\alpha$（第三象限，$\\sin < 0$）；$\\tan(-\\alpha) = -\\tan\\alpha$（奇函数）。',
      },
      {
        type: 'judge',
        prompt: '已知 $\\sin\\alpha = \\dfrac{3}{5}$，则 $\\cos\\alpha = \\dfrac{4}{5}$。',
        answer: false,
        explain: '$\\sin^2\\alpha + \\cos^2\\alpha = 1$，$\\cos^2\\alpha = \\dfrac{16}{25}$，故 $\\cos\\alpha = \\pm\\dfrac{4}{5}$。符号取决于 $\\alpha$ 所在象限，不给象限信息无法唯一确定，直接写 $\\dfrac{4}{5}$ 是错误的。',
      },
      {
        type: 'input',
        prompt: '已知 $\\sin\\alpha + \\cos\\alpha = \\dfrac{1}{2}$，求 $\\sin\\alpha \\cdot \\cos\\alpha$ 的值（用分数表示）。',
        accept: ['-3/8', '-\\frac{3}{8}'],
        explain: '两边平方：$(\\sin\\alpha + \\cos\\alpha)^2 = \\dfrac{1}{4}$，即 $1 + 2\\sin\\alpha\\cos\\alpha = \\dfrac{1}{4}$，故 $\\sin\\alpha\\cos\\alpha = \\dfrac{1/4 - 1}{2} = \\dfrac{-3/4}{2} = -\\dfrac{3}{8}$。',
      },
    ],
  },
]
