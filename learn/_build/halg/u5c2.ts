export const LESSONS = [
  {
    id: 'u5c2',
    title: '三角函数的定义（单位圆）',
    subtitle: '终边交单位圆于 $P(x,y)$：$\\cos\\alpha = x$，$\\sin\\alpha = y$，$\\tan\\alpha = y/x$',
    intro: [
      {
        title: '引入：为什么要推广三角函数的定义？',
        body: '初中用直角三角形定义正弦、余弦、正切——只对 $0°$ 到 $90°$ 的锐角有意义。\n\n但现实中的振动、旋转等现象涉及钝角乃至负角，我们需要**对任意角都成立**的三角函数定义。\n\n**直观想法**：把角放在坐标系中，让终边与圆相交，用交点坐标定义三角值。为让坐标直接就是函数值（不需除以半径），选用**半径为 1 的圆**，即单位圆。',
        tip: '单位圆的方程是 $x^2 + y^2 = 1$，圆上任意一点的横纵坐标都满足 $-1 \\leq x \\leq 1$，$-1 \\leq y \\leq 1$。',
      },
      {
        title: '单位圆定义三角函数',
        body: '在直角坐标系中，以原点为圆心、1 为半径的圆叫**单位圆**，方程为 $x^2 + y^2 = 1$。\n\n设 $\\alpha$ 是任意角，顶点在原点，初始边与 $x$ 轴正半轴重合，终边与单位圆的交点为 $P(x, y)$，则：\n\n$$\\sin\\alpha = y,\\quad \\cos\\alpha = x,\\quad \\tan\\alpha = \\frac{y}{x}\\ (x \\ne 0)$$\n\n**几何意义**：\n- $\\cos\\alpha$ = 终边与单位圆交点的**横坐标**\n- $\\sin\\alpha$ = 交点的**纵坐标**\n- $\\tan\\alpha$ = 终边的**斜率**（$y/x$）\n\n这对所有角（正角、负角、零角、超过 $2\\pi$ 的角）都成立。',
        formula: '\\sin\\alpha = y,\\quad \\cos\\alpha = x,\\quad \\tan\\alpha = \\frac{y}{x}\\ (x \\ne 0)',
        image: 'gz-p5-02-1.svg',
        imageCaption: '单位圆上的点 $P(\\cos\\alpha, \\sin\\alpha)$：横坐标是余弦，纵坐标是正弦',
      },
      {
        title: '定义域与特殊情况',
        body: '$\\sin\\alpha$ 和 $\\cos\\alpha$ 对所有 $\\alpha \\in \\mathbb{R}$ 都有意义（单位圆上的点总存在）。\n\n$\\tan\\alpha = \\dfrac{y}{x}$ 要求 $x \\ne 0$，即终边不能落在 $y$ 轴上。$x = 0$ 当且仅当 $\\alpha = \\dfrac{\\pi}{2} + k\\pi$（$k \\in \\mathbb{Z}$）。\n\n$$\\text{正切函数定义域：}\\alpha \\ne \\frac{\\pi}{2} + k\\pi,\\ k \\in \\mathbb{Z}$$\n\n**值域**：由 $x^2 + y^2 = 1$，得 $-1 \\leq \\cos\\alpha \\leq 1$，$-1 \\leq \\sin\\alpha \\leq 1$；$\\tan\\alpha$ 的值域为 $\\mathbb{R}$。',
        formula: '-1 \\leq \\sin\\alpha \\leq 1,\\quad -1 \\leq \\cos\\alpha \\leq 1',
        tip: '正弦和余弦有界（值在 $[-1,1]$ 内），正切无界（值域为整个实数集）。',
      },
      {
        title: '各象限三角函数的符号',
        body: '单位圆上的交点 $(x, y)$ 在不同象限中符号不同：\n\n| 象限 | $\\cos\\alpha$（$x$）| $\\sin\\alpha$（$y$）| $\\tan\\alpha$（$y/x$）|\n|------|--------|--------|--------|\n| 第一象限 | $+$ | $+$ | $+$ |\n| 第二象限 | $-$ | $+$ | $-$ |\n| 第三象限 | $-$ | $-$ | $+$ |\n| 第四象限 | $+$ | $-$ | $-$ |\n\n**口诀**：一全二正弦三切四余弦\n- 第一象限：**全**部为正\n- 第二象限：仅**正弦**为正\n- 第三象限：仅**正切**为正\n- 第四象限：仅**余弦**为正',
        tip: '验证口诀：第二象限 $x < 0$，$y > 0$，故 $\\sin > 0$（$y$），$\\cos < 0$（$x$），$\\tan < 0$（$y/x < 0$）——完全吻合。',
      },
      {
        title: '特殊角的三角函数值',
        body: '利用几何关系，精确求出以下特殊角的函数值：\n\n| $\\alpha$ | $0$ | $\\dfrac{\\pi}{6}$ | $\\dfrac{\\pi}{4}$ | $\\dfrac{\\pi}{3}$ | $\\dfrac{\\pi}{2}$ | $\\pi$ | $\\dfrac{3\\pi}{2}$ |\n|---|---|---|---|---|---|---|---|\n| $\\sin\\alpha$ | $0$ | $\\dfrac{1}{2}$ | $\\dfrac{\\sqrt{2}}{2}$ | $\\dfrac{\\sqrt{3}}{2}$ | $1$ | $0$ | $-1$ |\n| $\\cos\\alpha$ | $1$ | $\\dfrac{\\sqrt{3}}{2}$ | $\\dfrac{\\sqrt{2}}{2}$ | $\\dfrac{1}{2}$ | $0$ | $-1$ | $0$ |\n| $\\tan\\alpha$ | $0$ | $\\dfrac{\\sqrt{3}}{3}$ | $1$ | $\\sqrt{3}$ | 不存在 | $0$ | 不存在 |\n\n**记忆技巧**：$\\sin$ 在 $0$ 到 $\\dfrac{\\pi}{2}$ 依次为 $\\dfrac{\\sqrt{0}}{2}, \\dfrac{\\sqrt{1}}{2}, \\dfrac{\\sqrt{2}}{2}, \\dfrac{\\sqrt{3}}{2}, \\dfrac{\\sqrt{4}}{2}$（根号下依次 0、1、2、3、4）；$\\cos$ 顺序恰好相反。',
      },
      {
        title: '奇偶性与周期性',
        body: '**奇偶性**（由单位圆对称性得到）：\n- $\\sin(-\\alpha) = -\\sin\\alpha$：$\\sin$ 是**奇函数**，图象关于原点对称\n- $\\cos(-\\alpha) = \\cos\\alpha$：$\\cos$ 是**偶函数**，图象关于 $y$ 轴对称\n- $\\tan(-\\alpha) = -\\tan\\alpha$：$\\tan$ 是**奇函数**\n\n**周期性**：\n- $\\sin(\\alpha + 2\\pi) = \\sin\\alpha$，$\\cos(\\alpha + 2\\pi) = \\cos\\alpha$：以 $2\\pi$ 为周期\n- $\\tan(\\alpha + \\pi) = \\tan\\alpha$：以 $\\pi$ 为周期（终边方向不变，斜率不变）',
        formula: '\\sin(-\\alpha) = -\\sin\\alpha,\\quad \\cos(-\\alpha) = \\cos\\alpha',
        tip: '"正弦奇，余弦偶"——记住这两个字就够，正切同为奇函数。',
      },
      {
        title: '例题：已知坐标求三角值、利用周期化简',
        body: '**例 1（坐标法）**：角 $\\alpha$ 终边上有点 $P(-3, 4)$，求三角函数值。\n\n$|OP| = \\sqrt{9 + 16} = 5$，单位化后交点坐标为 $\\left(-\\dfrac{3}{5}, \\dfrac{4}{5}\\right)$，故：\n$$\\sin\\alpha = \\frac{4}{5},\\quad \\cos\\alpha = -\\frac{3}{5},\\quad \\tan\\alpha = \\frac{4}{-3} = -\\frac{4}{3}$$\n\n**例 2（奇函数+周期）**：求 $\\sin\\!\\left(-\\dfrac{13\\pi}{6}\\right)$。\n\n先用奇函数：$= -\\sin\\dfrac{13\\pi}{6}$；再用周期：$\\dfrac{13\\pi}{6} = 2\\pi + \\dfrac{\\pi}{6}$，故 $\\sin\\dfrac{13\\pi}{6} = \\sin\\dfrac{\\pi}{6} = \\dfrac{1}{2}$；最终结果为 $-\\dfrac{1}{2}$。',
        steps: [
          '终边上一点 $P(x_0, y_0)$（不在原点）：先求 $r = \\sqrt{x_0^2 + y_0^2}$',
          '然后 $\\sin\\alpha = y_0/r$，$\\cos\\alpha = x_0/r$，$\\tan\\alpha = y_0/x_0$',
          '用奇偶性先去掉负号，再用周期性化简到 $[0, 2\\pi)$ 内查特殊值',
        ],
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '设角 $\\alpha$ 的终边上（除原点外）有一点 $P(1, -\\sqrt{3})$，则 $\\sin\\alpha = $（  ）。',
        options: [
          '$\\dfrac{\\sqrt{3}}{2}$',
          '$-\\dfrac{\\sqrt{3}}{2}$',
          '$\\dfrac{1}{2}$',
          '$-\\dfrac{1}{2}$',
        ],
        answer: 1,
        explain: '$|OP| = \\sqrt{1 + 3} = 2$。$\\sin\\alpha = \\dfrac{y}{r} = \\dfrac{-\\sqrt{3}}{2} = -\\dfrac{\\sqrt{3}}{2}$。',
      },
      {
        type: 'judge',
        prompt: '正弦函数和余弦函数的值域都是 $[-1, 1]$，正切函数的值域也是 $[-1, 1]$。',
        answer: false,
        explain: '正弦和余弦的值域是 $[-1, 1]$（正确）；但正切函数的值域是 $\\mathbb{R}$（全体实数），无界，不局限于 $[-1, 1]$。',
      },
      {
        type: 'choice',
        prompt: '下列各值中，正确的是（  ）。',
        options: [
          '$\\sin\\dfrac{2\\pi}{3} = -\\dfrac{\\sqrt{3}}{2}$',
          '$\\cos\\dfrac{3\\pi}{4} = \\dfrac{\\sqrt{2}}{2}$',
          '$\\tan\\dfrac{5\\pi}{6} = -\\dfrac{\\sqrt{3}}{3}$',
          '$\\sin\\dfrac{5\\pi}{4} = \\dfrac{\\sqrt{2}}{2}$',
        ],
        answer: 2,
        explain: '$\\dfrac{5\\pi}{6} = \\pi - \\dfrac{\\pi}{6}$，在第二象限，$\\tan$ 为负，$\\tan\\dfrac{5\\pi}{6} = -\\tan\\dfrac{\\pi}{6} = -\\dfrac{\\sqrt{3}}{3}$。A 中 $\\sin\\dfrac{2\\pi}{3} = \\sin\\left(\\pi - \\dfrac{\\pi}{3}\\right) = \\sin\\dfrac{\\pi}{3} = \\dfrac{\\sqrt{3}}{2}$（正），故 A 错；B 中 $\\cos\\dfrac{3\\pi}{4} = -\\dfrac{\\sqrt{2}}{2}$（第二象限余弦为负），故 B 错；D 中 $\\dfrac{5\\pi}{4}$ 在第三象限，$\\sin < 0$，故 D 错。',
      },
      {
        type: 'input',
        prompt: '已知 $\\sin\\alpha = -\\dfrac{5}{13}$，且 $\\alpha$ 在第三象限，则 $\\tan\\alpha = $？（填分数，如 5/12）',
        accept: ['5/12'],
        explain: '第三象限 $\\cos\\alpha < 0$。$\\cos^2\\alpha = 1 - \\left(-\\dfrac{5}{13}\\right)^2 = 1 - \\dfrac{25}{169} = \\dfrac{144}{169}$，故 $\\cos\\alpha = -\\dfrac{12}{13}$。$\\tan\\alpha = \\dfrac{\\sin\\alpha}{\\cos\\alpha} = \\dfrac{-5/13}{-12/13} = \\dfrac{5}{12}$。',
      },
      {
        type: 'choice',
        prompt: '$\\sin\\left(-\\dfrac{11\\pi}{6}\\right)$ 的值是（  ）。',
        options: [
          '$-\\dfrac{1}{2}$',
          '$\\dfrac{1}{2}$',
          '$-\\dfrac{\\sqrt{3}}{2}$',
          '$\\dfrac{\\sqrt{3}}{2}$',
        ],
        answer: 1,
        explain: '$\\sin\\left(-\\dfrac{11\\pi}{6}\\right) = -\\sin\\dfrac{11\\pi}{6}$（奇函数）。$\\dfrac{11\\pi}{6} = 2\\pi - \\dfrac{\\pi}{6}$，故 $\\sin\\dfrac{11\\pi}{6} = \\sin\\left(-\\dfrac{\\pi}{6}\\right) = -\\sin\\dfrac{\\pi}{6} = -\\dfrac{1}{2}$。所以原式 $= -\\left(-\\dfrac{1}{2}\\right) = \\dfrac{1}{2}$。',
      },
      {
        type: 'match',
        prompt: '将下列三角函数与其值对应。',
        left: [
          '$\\cos\\dfrac{2\\pi}{3}$',
          '$\\sin\\dfrac{7\\pi}{6}$',
          '$\\tan\\dfrac{3\\pi}{4}$',
        ],
        right: [
          '$-1$',
          '$-\\dfrac{1}{2}$',
          '$-\\dfrac{1}{2}$',
        ],
        explain: '$\\cos\\dfrac{2\\pi}{3} = -\\cos\\dfrac{\\pi}{3} = -\\dfrac{1}{2}$；$\\sin\\dfrac{7\\pi}{6} = -\\sin\\dfrac{\\pi}{6} = -\\dfrac{1}{2}$；$\\tan\\dfrac{3\\pi}{4} = -\\tan\\dfrac{\\pi}{4} = -1$。',
      },
      {
        type: 'judge',
        prompt: '$\\sin\\alpha > 0$ 能唯一确定 $\\alpha$ 在第一象限。',
        answer: false,
        explain: '$\\sin\\alpha > 0$ 只说明终边在 $x$ 轴上方（纵坐标 $y > 0$），终边可能在第一象限，也可能在第二象限，或恰好在 $y$ 轴正半轴上。不能唯一确定象限。',
      },
      {
        type: 'input',
        prompt: '化简 $\\cos\\left(\\dfrac{11\\pi}{4}\\right)$，结果用最简根式分数表示（如 -sqrt(2)/2）。',
        accept: ['-\\sqrt{2}/2', '-sqrt(2)/2', '-\\frac{\\sqrt{2}}{2}'],
        explain: '$\\dfrac{11\\pi}{4} = 2\\pi + \\dfrac{3\\pi}{4}$，故 $\\cos\\dfrac{11\\pi}{4} = \\cos\\dfrac{3\\pi}{4}$。$\\dfrac{3\\pi}{4}$ 在第二象限，$\\cos\\dfrac{3\\pi}{4} = -\\cos\\dfrac{\\pi}{4} = -\\dfrac{\\sqrt{2}}{2}$。',
      },
    ],
  },
]
