export const LESSONS = [
  {
    id: 'u5c6',
    title: '简单三角方程',
    subtitle: '$\\sin x = a$：$x = (-1)^k\\arcsin a + k\\pi$；$\\cos x = a$：$x = \\pm\\arccos a + 2k\\pi$；$\\tan x = a$：$x = \\arctan a + k\\pi$',
    intro: [
      {
        title: '引入：基本三角方程的通解',
        body: '与一元二次方程有求根公式类似，三类基本三角方程也有固定的通解格式：\n\n**方程 $\\sin x = a$**（$|a| \\leq 1$）：\n$$x = (-1)^k\\arcsin a + k\\pi,\\quad k \\in \\mathbb{Z}$$\n**方程 $\\cos x = a$**（$|a| \\leq 1$）：\n$$x = \\pm\\arccos a + 2k\\pi,\\quad k \\in \\mathbb{Z}$$\n**方程 $\\tan x = a$**：\n$$x = \\arctan a + k\\pi,\\quad k \\in \\mathbb{Z}$$\n\n高中常遇到 $a$ 是特殊值，直接写特殊角而非 $\\arcsin/\\arccos/\\arctan$。\n\n**为什么 $\\sin x = a$ 用 $(-1)^k$，$\\cos x = a$ 用 $\\pm$？**\n$\\sin x$ 在 $[0,\\pi]$ 上的两根关于 $x = \\dfrac{\\pi}{2}$ 对称（若一根为 $x_0$，另一根为 $\\pi - x_0 = \\pi + (-1)^1\\cdot x_0$）；$\\cos x$ 的两根关于 $y$ 轴对称（$x_0$ 和 $-x_0$）。',
        tip: '记住：正弦用"奇偶符号 $(-1)^k$"，余弦用"$\\pm$"，正切最简洁只用"$k\\pi$"。不要混用。',
      },
      {
        title: '特殊值对应的标准通解',
        body: '当 $a$ 是高中常见特殊值时，通解直接用特殊角表示（无需 $\\arcsin$）：\n\n**$\\sin x = a$ 的常用解**：\n- $\\sin x = 0$：$x = k\\pi$\n- $\\sin x = \\dfrac{1}{2}$：$x = \\dfrac{\\pi}{6} + 2k\\pi$ 或 $x = \\dfrac{5\\pi}{6} + 2k\\pi$\n- $\\sin x = \\dfrac{\\sqrt{2}}{2}$：$x = \\dfrac{\\pi}{4} + 2k\\pi$ 或 $x = \\dfrac{3\\pi}{4} + 2k\\pi$\n- $\\sin x = \\dfrac{\\sqrt{3}}{2}$：$x = \\dfrac{\\pi}{3} + 2k\\pi$ 或 $x = \\dfrac{2\\pi}{3} + 2k\\pi$\n\n**$\\cos x = a$ 的常用解**：\n- $\\cos x = \\dfrac{1}{2}$：$x = \\pm\\dfrac{\\pi}{3} + 2k\\pi$\n- $\\cos x = 0$：$x = \\dfrac{\\pi}{2} + k\\pi$\n\n**$\\tan x = a$ 的常用解**：\n- $\\tan x = 1$：$x = \\dfrac{\\pi}{4} + k\\pi$\n- $\\tan x = \\sqrt{3}$：$x = \\dfrac{\\pi}{3} + k\\pi$',
        formula: '\\sin x = \\sin\\theta \\Rightarrow x = \\theta + 2k\\pi\\ \\text{或}\\ x = \\pi - \\theta + 2k\\pi,\\quad k \\in \\mathbb{Z}',
      },
      {
        title: '换元化归：含复合角的方程',
        body: '方程中含 $\\omega x + \\varphi$ 这样的复合角时，令 $u = \\omega x + \\varphi$ 先求 $u$，再解 $x$。\n\n**步骤**：\n1. 令 $u = \\omega x + \\varphi$，方程化为 $\\sin u = a$（或 $\\cos/\\tan$）\n2. 确定 $x$ 的范围对应 $u$ 的范围\n3. 用通解求 $u$，筛选在范围内的值\n4. 回代 $u = \\omega x + \\varphi$ 解出 $x$\n\n**例**：解 $\\cos\\!\\left(2x + \\dfrac{\\pi}{3}\\right) = -\\dfrac{1}{2}$（$x \\in [0, \\pi]$）\n\n令 $u = 2x + \\dfrac{\\pi}{3}$，$x \\in [0, \\pi]$ 时 $u \\in \\left[\\dfrac{\\pi}{3}, \\dfrac{7\\pi}{3}\\right]$。\n\n$\\cos u = -\\dfrac{1}{2}$，通解 $u = \\pm\\dfrac{2\\pi}{3} + 2k\\pi$，在范围内筛选：$u = \\dfrac{2\\pi}{3}$ 或 $u = \\dfrac{4\\pi}{3}$。\n\n回代：$x = \\dfrac{\\pi}{6}$ 或 $x = \\dfrac{\\pi}{2}$。',
        steps: [
          '令 $u = \\omega x + \\varphi$，写出 $u$ 的范围（由 $x$ 的范围推出）',
          '对 $u$ 写基本方程的通解',
          '在 $u$ 的范围内筛选有效解',
          '回代 $u = \\omega x + \\varphi$ 求 $x$',
        ],
      },
      {
        title: '代换法：化为一元代数方程',
        body: '当方程含不同三角函数时，用同角恒等式统一变量，令 $t = \\sin x$（或 $\\cos x$），化为关于 $t$ 的代数方程。\n\n**关键约束**：$|t| \\leq 1$（若代数方程解出 $t = 2$ 之类的值，必须舍去）。\n\n**例 1**：解 $2\\sin^2 x - 3\\sin x + 1 = 0$（$x \\in [0, 2\\pi)$）\n\n令 $t = \\sin x$，$2t^2 - 3t + 1 = 0$，$(2t-1)(t-1) = 0$，$t = \\dfrac{1}{2}$ 或 $t = 1$。\n\n$\\sin x = \\dfrac{1}{2}$：$x = \\dfrac{\\pi}{6}$ 或 $x = \\dfrac{5\\pi}{6}$；$\\sin x = 1$：$x = \\dfrac{\\pi}{2}$。\n\n**例 2（含 $\\cos 2x$）**：解 $\\cos 2x + 3\\sin x = 2$（$x \\in [0, 2\\pi)$）\n\n用 $\\cos 2x = 1 - 2\\sin^2 x$ 代入：$1 - 2\\sin^2 x + 3\\sin x = 2$，即 $2\\sin^2 x - 3\\sin x + 1 = 0$，与例 1 相同。',
        tip: '见到 $\\cos 2x$ 与 $\\sin x$ 同时出现，优先用 $\\cos 2x = 1 - 2\\sin^2 x$ 统一为 $\\sin x$；与 $\\cos x$ 同时出现则用 $\\cos 2x = 2\\cos^2 x - 1$。',
      },
      {
        title: '和差化积解方程',
        body: '当方程含 $\\sin mx \\pm \\sin nx$（$m \\ne n$）时，用和差化积公式因式分解。\n\n**和差化积公式**：\n$$\\sin A + \\sin B = 2\\sin\\frac{A+B}{2}\\cos\\frac{A-B}{2}$$\n$$\\cos A - \\cos B = -2\\sin\\frac{A+B}{2}\\sin\\frac{A-B}{2}$$\n\n**例**：解 $\\sin 3x + \\sin x = 0$（$x \\in [0, 2\\pi)$）\n\n$$2\\sin\\frac{3x+x}{2}\\cos\\frac{3x-x}{2} = 2\\sin 2x\\cos x = 0$$\n\n$\\sin 2x = 0$：$x = 0, \\dfrac{\\pi}{2}, \\pi, \\dfrac{3\\pi}{2}$；$\\cos x = 0$：$x = \\dfrac{\\pi}{2}, \\dfrac{3\\pi}{2}$（已包含）。\n\n答：$x = 0, \\dfrac{\\pi}{2}, \\pi, \\dfrac{3\\pi}{2}$。',
        formula: '\\sin A + \\sin B = 2\\sin\\frac{A+B}{2}\\cos\\frac{A-B}{2}',
      },
      {
        title: '方程解的个数分析',
        body: '利用三角函数的图象分析方程 $f(x) = k$ 的解的个数：\n\n**$\\cos x = m$（$x \\in [0, 2\\pi]$）解的个数**：\n- $m = 1$：$x = 0$ 和 $x = 2\\pi$，**2 个解**\n- $-1 < m < 1$：图象下降段和上升段各一个交点，**2 个解**\n- $m = -1$：仅 $x = \\pi$，**1 个解**\n- $|m| > 1$：**无解**\n\n结论：$\\cos x = m$ 在 $[0, 2\\pi]$ 上有两个不同实数解当且仅当 $-1 < m \\leq 1$。\n\n**$\\sin x = m$（$x \\in [0, 2\\pi)$）类似分析**：\n- $m = 1$：仅 $x = \\dfrac{\\pi}{2}$，**1 个解**\n- $0 < m < 1$：第一和第二象限各一个，**2 个解**\n- $m = 0$：$x = 0$ 和 $x = \\pi$，**2 个解**\n- $-1 \\leq m < 0$：**2 个解**（$m = -1$ 时为 1 个）',
        tip: '解题时先画函数图象和水平线 $y = k$ 的草图，从图形直觉判断解的个数，再精确求值。',
      },
      {
        title: '易错点总结',
        body: '**易错 1**：正弦通解写成 $\\pm\\arcsin a + 2k\\pi$（那是余弦的形式！）。正弦用 $(-1)^k$，余弦用 $\\pm$。\n\n**易错 2**：换元后忘记确定 $u$ 的范围，直接写全部通解而不筛选——这是最常见失分原因。\n\n**易错 3**：令 $t = \\sin x$ 后解出 $t > 1$ 或 $t < -1$ 的值，忘记舍去（三角函数值域限制）。\n\n**易错 4**：含 $\\cos 2x$ 时选错降幂公式，应根据方程中其余项含 $\\sin x$ 还是 $\\cos x$ 来选对应形式。\n\n**易错 5**：$\\sin x = \\dfrac{1}{2}$ 在 $[0, 2\\pi)$ 中有两个解（$\\dfrac{\\pi}{6}$ 和 $\\dfrac{5\\pi}{6}$），常漏写一个。',
        reveal: {
          q: '求方程 $\\sin\\!\\left(2x - \\dfrac{\\pi}{3}\\right) = \\dfrac{1}{2}$ 的通解。',
          a: '令 $u = 2x - \\dfrac{\\pi}{3}$，$\\sin u = \\dfrac{1}{2}$。通解：$u = \\dfrac{\\pi}{6} + 2k\\pi$ 或 $u = \\dfrac{5\\pi}{6} + 2k\\pi$。回代：情形 1：$2x = \\dfrac{\\pi}{6} + \\dfrac{\\pi}{3} + 2k\\pi = \\dfrac{\\pi}{2} + 2k\\pi$，$x = \\dfrac{\\pi}{4} + k\\pi$；情形 2：$2x = \\dfrac{5\\pi}{6} + \\dfrac{\\pi}{3} + 2k\\pi = \\dfrac{7\\pi}{6} + 2k\\pi$，$x = \\dfrac{7\\pi}{12} + k\\pi$（$k \\in \\mathbb{Z}$）。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '方程 $\\cos x = -\\dfrac{\\sqrt{3}}{2}$ 的通解是（  ）。',
        options: [
          '$x = \\pm\\dfrac{5\\pi}{6} + 2k\\pi$，$k \\in \\mathbb{Z}$',
          '$x = \\dfrac{5\\pi}{6} + k\\pi$，$k \\in \\mathbb{Z}$',
          '$x = \\pm\\dfrac{\\pi}{6} + 2k\\pi$，$k \\in \\mathbb{Z}$',
          '$x = (-1)^k\\dfrac{5\\pi}{6} + k\\pi$，$k \\in \\mathbb{Z}$',
        ],
        answer: 0,
        explain: '$\\cos x = -\\dfrac{\\sqrt{3}}{2}$，$\\arccos\\!\\left(-\\dfrac{\\sqrt{3}}{2}\\right) = \\dfrac{5\\pi}{6}$，余弦方程通解为 $x = \\pm\\dfrac{5\\pi}{6} + 2k\\pi$（$k \\in \\mathbb{Z}$）。',
      },
      {
        type: 'judge',
        prompt: '$\\sin x = a$（$|a| \\leq 1$）的通解可以写成 $x = \\pm\\arcsin a + 2k\\pi$（$k \\in \\mathbb{Z}$）。',
        answer: false,
        explain: '$x = \\pm\\arcsin a + 2k\\pi$ 是**余弦**方程的通解形式。正弦方程的正确通解是 $x = (-1)^k\\arcsin a + k\\pi$（$k \\in \\mathbb{Z}$），两者不可混用。',
      },
      {
        type: 'input',
        prompt: '方程 $\\tan\\!\\left(x + \\dfrac{\\pi}{4}\\right) = 1$ 的通解，$x = $？（填 $k\\pi$ 形式，$k \\in \\mathbb{Z}$）',
        accept: ['k\\pi', 'kπ'],
        explain: '令 $u = x + \\dfrac{\\pi}{4}$，$\\tan u = 1$，通解 $u = \\dfrac{\\pi}{4} + k\\pi$。回代：$x + \\dfrac{\\pi}{4} = \\dfrac{\\pi}{4} + k\\pi$，故 $x = k\\pi$（$k \\in \\mathbb{Z}$）。',
      },
      {
        type: 'choice',
        prompt: '解方程 $2\\cos^2 x + \\cos x - 1 = 0$，$x \\in [0, 2\\pi)$ 的解集是（  ）。',
        options: [
          '$\\left\\{\\dfrac{\\pi}{3}, \\pi, \\dfrac{5\\pi}{3}\\right\\}$',
          '$\\left\\{\\dfrac{\\pi}{3}, \\dfrac{5\\pi}{3}\\right\\}$',
          '$\\left\\{\\dfrac{2\\pi}{3}, \\pi, \\dfrac{4\\pi}{3}\\right\\}$',
          '$\\left\\{\\pi\\right\\}$',
        ],
        answer: 0,
        explain: '令 $t = \\cos x$，$2t^2 + t - 1 = (2t-1)(t+1) = 0$，$t = \\dfrac{1}{2}$ 或 $t = -1$。$\\cos x = \\dfrac{1}{2}$：$x = \\dfrac{\\pi}{3}$ 或 $x = \\dfrac{5\\pi}{3}$；$\\cos x = -1$：$x = \\pi$。解集为 $\\left\\{\\dfrac{\\pi}{3}, \\pi, \\dfrac{5\\pi}{3}\\right\\}$。',
      },
      {
        type: 'choice',
        prompt: '方程 $\\cos x = m$ 在 $[0, 2\\pi]$ 上恰好有两个不同实数解，则 $m$ 的范围是（  ）。',
        options: [
          '$-1 \\leq m \\leq 1$',
          '$-1 < m \\leq 1$',
          '$-1 \\leq m < 1$',
          '$-1 < m < 1$',
        ],
        answer: 1,
        explain: '$\\cos x$ 在 $[0,2\\pi]$ 上先从 1 递减到 $-1$ 再递增到 1。$m = 1$：交点 $x = 0$ 和 $x = 2\\pi$（2 个解，包含）；$-1 < m < 1$：2 个解；$m = -1$：仅 $x = \\pi$（1 个解，不含）。故 $-1 < m \\leq 1$。',
      },
      {
        type: 'match',
        prompt: '将各方程与其在 $[0, 2\\pi)$ 内的解对应。',
        left: [
          '$\\sin x = 0$',
          '$\\cos x = 1$',
          '$\\tan x = -1$',
        ],
        right: [
          '$x = \\dfrac{3\\pi}{4}$ 或 $x = \\dfrac{7\\pi}{4}$',
          '$x = 0$',
          '$x = 0$ 或 $x = \\pi$',
        ],
        explain: '$\\sin x = 0$：$x = 0$ 或 $x = \\pi$；$\\cos x = 1$：$x = 0$；$\\tan x = -1$：$x = \\dfrac{3\\pi}{4}$ 或 $x = \\dfrac{7\\pi}{4}$（在 $[0, 2\\pi)$ 内）。',
      },
      {
        type: 'judge',
        prompt: '解方程 $\\sin 2x = \\cos x$（$x \\in [0, 2\\pi)$）时，将左边用倍角公式展开后可以得到 4 个解。',
        answer: true,
        explain: '$\\sin 2x = 2\\sin x\\cos x$，方程变为 $\\cos x(2\\sin x - 1) = 0$。$\\cos x = 0$：$x = \\dfrac{\\pi}{2}, \\dfrac{3\\pi}{2}$（2 个）；$\\sin x = \\dfrac{1}{2}$：$x = \\dfrac{\\pi}{6}, \\dfrac{5\\pi}{6}$（2 个）。共 4 个解。',
      },
      {
        type: 'input',
        prompt: '解方程 $\\cos 2x - \\cos x = 0$ 的通解，将所有解写成统一形式，最简的通解 $x = $ （填最简分数 $\\pi$ 表达式，如 $2k\\pi/3$）。',
        accept: ['2k\\pi/3', '\\frac{2k\\pi}{3}', '2kπ/3'],
        explain: '和差化积：$\\cos 2x - \\cos x = -2\\sin\\dfrac{3x}{2}\\sin\\dfrac{x}{2} = 0$。$\\sin\\dfrac{3x}{2} = 0$：$\\dfrac{3x}{2} = k\\pi$，$x = \\dfrac{2k\\pi}{3}$；$\\sin\\dfrac{x}{2} = 0$：$x = 2k\\pi$（已包含在 $k$ 为 3 的倍数时）。通解：$x = \\dfrac{2k\\pi}{3}$（$k \\in \\mathbb{Z}$）。',
      },
    ],
  },
]
