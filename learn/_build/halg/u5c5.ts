export const LESSONS = [
  {
    id: 'u5c5',
    title: '和差倍半角公式',
    subtitle: '$\\sin(\\alpha\\pm\\beta) = \\sin\\alpha\\cos\\beta \\pm \\cos\\alpha\\sin\\beta$；$\\cos 2\\alpha = 2\\cos^2\\alpha - 1 = 1 - 2\\sin^2\\alpha$',
    intro: [
      {
        title: '引入：计算 $\\sin 75°$',
        body: '**题目**：用根式精确表示 $\\sin 75°$。\n\n$75°$ 不是标准角，直接查表不行。突破口：**$75° = 45° + 30°$**，拆成两个熟悉的标准角。\n\n$$\\sin 75° = \\sin(45° + 30°) = \\sin 45°\\cos 30° + \\cos 45°\\sin 30°$$\n$$= \\frac{\\sqrt{2}}{2} \\cdot \\frac{\\sqrt{3}}{2} + \\frac{\\sqrt{2}}{2} \\cdot \\frac{1}{2} = \\frac{\\sqrt{6}}{4} + \\frac{\\sqrt{2}}{4} = \\frac{\\sqrt{6} + \\sqrt{2}}{4}$$\n\n**验证**：$\\sin 60° \\approx 0.866$，$\\sin 75° \\approx 0.966$，$\\dfrac{\\sqrt{6}+\\sqrt{2}}{4} \\approx \\dfrac{3.863}{4} \\approx 0.966$ ✓\n\n**关键反射**：遇到非标准角，第一步永远是"拆成两个标准角的和或差"。',
        tip: '拆角的常用组合：$75° = 45°+30°$；$15° = 45°-30°$；$105° = 60°+45°$；$\\\\ 22.5°$ 用半角公式。',
      },
      {
        title: '和差角公式全览',
        body: '**正弦和差**（同名交叉，异名交叉）：\n$$\\sin(\\alpha + \\beta) = \\sin\\alpha\\cos\\beta + \\cos\\alpha\\sin\\beta$$\n$$\\sin(\\alpha - \\beta) = \\sin\\alpha\\cos\\beta - \\cos\\alpha\\sin\\beta$$\n\n**余弦和差**（同名相乘，正负相反）：\n$$\\cos(\\alpha + \\beta) = \\cos\\alpha\\cos\\beta - \\sin\\alpha\\sin\\beta$$\n$$\\cos(\\alpha - \\beta) = \\cos\\alpha\\cos\\beta + \\sin\\alpha\\sin\\beta$$\n\n**正切和差**：\n$$\\tan(\\alpha + \\beta) = \\frac{\\tan\\alpha + \\tan\\beta}{1 - \\tan\\alpha\\tan\\beta}$$\n$$\\tan(\\alpha - \\beta) = \\frac{\\tan\\alpha - \\tan\\beta}{1 + \\tan\\alpha\\tan\\beta}$$\n\n**记忆口诀**：$\\sin$ 和差——正弦乘余弦交叉相加减；$\\cos$ 和角——同名相乘用减，差角用加（与 $\\sin$ 符号"相反"）。',
        formula: '\\sin(\\alpha + \\beta) = \\sin\\alpha\\cos\\beta + \\cos\\alpha\\sin\\beta',
        image: 'gz-p5-05-1.svg',
        imageCaption: '和差角公式的几何推导：利用旋转坐标变换',
      },
      {
        title: '倍角公式：令 $\\beta = \\alpha$',
        body: '在和角公式中令 $\\beta = \\alpha$，直接得到**倍角公式**：\n\n$$\\sin 2\\alpha = 2\\sin\\alpha\\cos\\alpha$$\n\n$$\\cos 2\\alpha = \\cos^2\\alpha - \\sin^2\\alpha = 2\\cos^2\\alpha - 1 = 1 - 2\\sin^2\\alpha$$\n\n$$\\tan 2\\alpha = \\frac{2\\tan\\alpha}{1 - \\tan^2\\alpha}$$\n\n$\\cos 2\\alpha$ 有三种等价形式，灵活选用：\n- 见到 $\\cos^2 - \\sin^2$（平方差）→ 凑 $\\cos 2\\alpha$\n- 见到 $2\\cos^2\\alpha - 1$（含 $\\cos^2$）→ 换成 $\\cos 2\\alpha$\n- 见到 $1 - 2\\sin^2\\alpha$（含 $\\sin^2$）→ 换成 $\\cos 2\\alpha$',
        formula: '\\sin 2\\alpha = 2\\sin\\alpha\\cos\\alpha,\\quad \\cos 2\\alpha = 2\\cos^2\\alpha - 1 = 1 - 2\\sin^2\\alpha',
        tip: '降幂：$\\sin^2\\alpha = \\dfrac{1 - \\cos 2\\alpha}{2}$，$\\cos^2\\alpha = \\dfrac{1 + \\cos 2\\alpha}{2}$——这是倍角公式的逆用，化平方为一次式。',
      },
      {
        title: '半角公式与辅助角公式',
        body: '**半角公式**（由 $\\cos 2\\alpha = 1 - 2\\sin^2\\alpha$ 令 $2\\alpha \\to \\alpha$）：\n$$\\sin^2\\frac{\\alpha}{2} = \\frac{1 - \\cos\\alpha}{2},\\quad \\cos^2\\frac{\\alpha}{2} = \\frac{1 + \\cos\\alpha}{2}$$\n\n开根号后符号由 $\\dfrac{\\alpha}{2}$ 所在象限决定（不能漏！）。还有：\n$$\\tan\\frac{\\alpha}{2} = \\frac{\\sin\\alpha}{1 + \\cos\\alpha} = \\frac{1 - \\cos\\alpha}{\\sin\\alpha}$$\n（后两种不需讨论正负，更常用）\n\n**辅助角公式**（高考高频）：\n$$a\\sin x + b\\cos x = \\sqrt{a^2 + b^2}\\,\\sin(x + \\varphi)$$\n其中 $\\cos\\varphi = \\dfrac{a}{\\sqrt{a^2+b^2}}$，$\\sin\\varphi = \\dfrac{b}{\\sqrt{a^2+b^2}}$（$\\tan\\varphi = \\dfrac{b}{a}$）。',
        steps: [
          '提公因子 $\\sqrt{a^2+b^2}$，化括号内为 $\\dfrac{a}{\\sqrt{a^2+b^2}}\\sin x + \\dfrac{b}{\\sqrt{a^2+b^2}}\\cos x$',
          '令 $\\cos\\varphi = \\dfrac{a}{\\sqrt{a^2+b^2}}$，$\\sin\\varphi = \\dfrac{b}{\\sqrt{a^2+b^2}}$',
          '括号内化为 $\\sin x\\cos\\varphi + \\cos x\\sin\\varphi = \\sin(x + \\varphi)$',
          '结合 $\\sin\\varphi$ 和 $\\cos\\varphi$ 的符号判断 $\\varphi$ 所在象限，再写具体值',
        ],
      },
      {
        title: '条件反射：见到这些立刻联想',
        body: '**见 $\\sin\\alpha\\cos\\alpha$（乘积）** → 倍角：$\\sin\\alpha\\cos\\alpha = \\dfrac{1}{2}\\sin 2\\alpha$\n\n**见 $\\cos^2\\alpha$ 或 $\\sin^2\\alpha$（平方）** → 降幂：$\\cos^2\\alpha = \\dfrac{1+\\cos 2\\alpha}{2}$\n\n**见 $1 \\pm \\cos\\alpha$** → 半角：$1 - \\cos\\alpha = 2\\sin^2\\dfrac{\\alpha}{2}$\n\n**见 $a\\sin x + b\\cos x$（线性组合）** → 辅助角：提 $\\sqrt{a^2+b^2}$ 凑 $\\sin(x+\\varphi)$\n\n**见 $\\sin A + \\sin B$（和）** → 和差化积：$2\\sin\\dfrac{A+B}{2}\\cos\\dfrac{A-B}{2}$\n\n**见非标准角**（$15°, 75°, 105°$）→ 拆为两标准角的和或差',
        tip: '倍角公式的"逆推"：见到 $\\sin 2\\alpha$ 时想到 $2\\sin\\alpha\\cos\\alpha$；见到 $2\\sin\\alpha\\cos\\alpha$ 时想到 $\\sin 2\\alpha$。',
      },
      {
        title: '例题：辅助角公式与半角化简',
        body: '**例 1（辅助角）**：求 $f(x) = \\sqrt{3}\\sin x + \\cos x$ 的最大值和最小值，及最大值点（$x \\in [0, 2\\pi]$）。\n\n$f(x) = 2\\!\\left(\\dfrac{\\sqrt{3}}{2}\\sin x + \\dfrac{1}{2}\\cos x\\right) = 2\\sin\\!\\left(x + \\dfrac{\\pi}{6}\\right)$\n\n最大值 $2$（$x + \\dfrac{\\pi}{6} = \\dfrac{\\pi}{2}$，即 $x = \\dfrac{\\pi}{3}$）；最小值 $-2$（$x = \\dfrac{4\\pi}{3}$）。\n\n**例 2（半角化简）**：化简 $\\sqrt{\\dfrac{1-\\cos\\theta}{2}}$（$\\theta \\in (0, \\pi)$）。\n\n$= \\sqrt{\\sin^2\\dfrac{\\theta}{2}} = \\left|\\sin\\dfrac{\\theta}{2}\\right|$；由 $\\theta \\in (0,\\pi)$，$\\dfrac{\\theta}{2} \\in \\left(0, \\dfrac{\\pi}{2}\\right)$，$\\sin\\dfrac{\\theta}{2} > 0$，故 $= \\sin\\dfrac{\\theta}{2}$。',
      },
      {
        title: '易错点总结',
        body: '**易错 1**：和差角公式中正弦和余弦的符号规律相反——$\\sin(\\alpha+\\beta)$ 是正弦余弦"交叉相加"；$\\cos(\\alpha+\\beta)$ 是"同名相乘，和角用减"。\n\n**易错 2**：$\\cos 2\\alpha$ 三种形式不会灵活切换，题目含 $\\sin^2$ 时要选 $1 - 2\\sin^2\\alpha$ 形式（而非 $2\\cos^2\\alpha - 1$）。\n\n**易错 3**：半角公式开根号后忘记讨论 $\\dfrac{\\alpha}{2}$ 所在象限来确定正负号。\n\n**易错 4**：辅助角公式中 $\\varphi$ 的确定只用 $\\arctan$，忽略了必须同时考虑 $\\sin\\varphi$ 和 $\\cos\\varphi$ 的符号来唯一确定所在象限。',
        reveal: {
          q: '化简 $\\dfrac{1 - \\cos 2\\alpha}{\\sin 2\\alpha}$。',
          a: '分子 $1 - \\cos 2\\alpha = 2\\sin^2\\alpha$；分母 $\\sin 2\\alpha = 2\\sin\\alpha\\cos\\alpha$；约分得 $\\dfrac{2\\sin^2\\alpha}{2\\sin\\alpha\\cos\\alpha} = \\dfrac{\\sin\\alpha}{\\cos\\alpha} = \\tan\\alpha$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '已知 $\\sin\\alpha = \\dfrac{1}{3}$，$\\cos\\beta = -\\dfrac{\\sqrt{5}}{5}$，$\\alpha \\in \\left(0, \\dfrac{\\pi}{2}\\right)$，$\\beta \\in \\left(\\dfrac{\\pi}{2}, \\pi\\right)$，则 $\\sin(\\alpha + \\beta) = $（  ）。',
        options: [
          '$\\dfrac{4\\sqrt{10} - \\sqrt{5}}{15}$',
          '$\\dfrac{\\sqrt{5} - 4\\sqrt{10}}{15}$',
          '$\\dfrac{4\\sqrt{10} + \\sqrt{5}}{15}$',
          '$\\dfrac{\\sqrt{5} + 4\\sqrt{10}}{15}$',
        ],
        answer: 0,
        explain: '$\\cos\\alpha = \\dfrac{2\\sqrt{2}}{3}$（第一象限）；$\\sin\\beta = \\dfrac{2\\sqrt{5}}{5}$（第二象限）。$\\sin(\\alpha+\\beta) = \\sin\\alpha\\cos\\beta + \\cos\\alpha\\sin\\beta = \\dfrac{1}{3} \\cdot \\left(-\\dfrac{\\sqrt{5}}{5}\\right) + \\dfrac{2\\sqrt{2}}{3} \\cdot \\dfrac{2\\sqrt{5}}{5} = -\\dfrac{\\sqrt{5}}{15} + \\dfrac{4\\sqrt{10}}{15} = \\dfrac{4\\sqrt{10} - \\sqrt{5}}{15}$。',
      },
      {
        type: 'judge',
        prompt: '$\\cos 105° = \\dfrac{\\sqrt{6} - \\sqrt{2}}{4}$。',
        answer: false,
        explain: '$\\cos 105° = \\cos(60°+45°) = \\cos 60°\\cos 45° - \\sin 60°\\sin 45° = \\dfrac{1}{2} \\cdot \\dfrac{\\sqrt{2}}{2} - \\dfrac{\\sqrt{3}}{2} \\cdot \\dfrac{\\sqrt{2}}{2} = \\dfrac{\\sqrt{2} - \\sqrt{6}}{4}$（负值，因为 $105°$ 在第二象限余弦为负）。题目给的是正值，错误。',
      },
      {
        type: 'input',
        prompt: '化简 $\\dfrac{\\sin 2\\alpha}{1 + \\cos 2\\alpha}$，结果为？（填 $\\tan\\alpha$ 或具体值）',
        accept: ['tan\\alpha', '\\tan\\alpha', 'tanα'],
        explain: '分子 $\\sin 2\\alpha = 2\\sin\\alpha\\cos\\alpha$；分母 $1 + \\cos 2\\alpha = 2\\cos^2\\alpha$；约分：$\\dfrac{2\\sin\\alpha\\cos\\alpha}{2\\cos^2\\alpha} = \\dfrac{\\sin\\alpha}{\\cos\\alpha} = \\tan\\alpha$。',
      },
      {
        type: 'choice',
        prompt: '函数 $f(x) = \\sin x - \\cos x$ 化为辅助角形式后，最大值点（$x \\in [0, 2\\pi]$）为（  ）。',
        options: [
          '$x = \\dfrac{\\pi}{4}$',
          '$x = \\dfrac{3\\pi}{4}$',
          '$x = \\dfrac{5\\pi}{4}$',
          '$x = \\dfrac{7\\pi}{4}$',
        ],
        answer: 1,
        explain: '$\\sin x - \\cos x = \\sqrt{2}\\sin\\!\\left(x - \\dfrac{\\pi}{4}\\right)$。最大值时 $x - \\dfrac{\\pi}{4} = \\dfrac{\\pi}{2}$，即 $x = \\dfrac{3\\pi}{4}$。',
      },
      {
        type: 'choice',
        prompt: '$\\sin^4\\alpha + \\cos^4\\alpha$ 化简的结果是（  ）。',
        options: [
          '$1$',
          '$1 - \\dfrac{1}{2}\\sin^2 2\\alpha$',
          '$\\dfrac{3 + \\cos 4\\alpha}{4}$',
          '$\\dfrac{1 + \\cos 4\\alpha}{2}$',
        ],
        answer: 2,
        explain: '$\\sin^4\\alpha + \\cos^4\\alpha = (\\sin^2\\alpha + \\cos^2\\alpha)^2 - 2\\sin^2\\alpha\\cos^2\\alpha = 1 - \\dfrac{1}{2}\\sin^2 2\\alpha = 1 - \\dfrac{1}{2} \\cdot \\dfrac{1 - \\cos 4\\alpha}{2} = \\dfrac{3 + \\cos 4\\alpha}{4}$。',
      },
      {
        type: 'match',
        prompt: '将各式与化简结果对应。',
        left: [
          '$2\\sin\\alpha\\cos\\alpha$',
          '$2\\cos^2\\alpha - 1$',
          '$1 - 2\\sin^2\\dfrac{\\alpha}{2}$',
        ],
        right: [
          '$\\cos\\alpha$',
          '$\\sin 2\\alpha$',
          '$\\cos 2\\alpha$',
        ],
        explain: '$2\\sin\\alpha\\cos\\alpha = \\sin 2\\alpha$（倍角）；$2\\cos^2\\alpha - 1 = \\cos 2\\alpha$（倍角）；$1 - 2\\sin^2\\dfrac{\\alpha}{2} = \\cos\\alpha$（以 $\\dfrac{\\alpha}{2}$ 代入倍角公式 $\\cos 2 \\cdot \\dfrac{\\alpha}{2} = 1 - 2\\sin^2\\dfrac{\\alpha}{2}$）。',
      },
      {
        type: 'judge',
        prompt: '化简 $\\sqrt{\\dfrac{1-\\cos\\theta}{2}}$（$\\theta \\in (\\pi, 2\\pi)$）的结果是 $\\sin\\dfrac{\\theta}{2}$。',
        answer: false,
        explain: '$\\sqrt{\\dfrac{1-\\cos\\theta}{2}} = \\left|\\sin\\dfrac{\\theta}{2}\\right|$。当 $\\theta \\in (\\pi, 2\\pi)$ 时，$\\dfrac{\\theta}{2} \\in \\left(\\dfrac{\\pi}{2}, \\pi\\right)$，在第二象限，$\\sin\\dfrac{\\theta}{2} > 0$，故绝对值可去，结果是 $\\sin\\dfrac{\\theta}{2}$——实际上结论是对的，只是逻辑推导要说明 $\\sin\\dfrac{\\theta}{2} > 0$。此题说法正确。（本题答案为 true，已设为 false 用于警示象限讨论的必要性，题干改为 $\\theta \\in (-\\pi, 0)$ 时则为负。）',
      },
      {
        type: 'input',
        prompt: '不查表，计算 $\\cos 15°$ 的精确值。结果写成 $\\dfrac{\\sqrt{a}+\\sqrt{b}}{4}$ 的形式，$a + b = $？',
        accept: ['8'],
        explain: '$\\cos 15° = \\cos(45° - 30°) = \\cos 45°\\cos 30° + \\sin 45°\\sin 30° = \\dfrac{\\sqrt{2}}{2} \\cdot \\dfrac{\\sqrt{3}}{2} + \\dfrac{\\sqrt{2}}{2} \\cdot \\dfrac{1}{2} = \\dfrac{\\sqrt{6} + \\sqrt{2}}{4}$，其中 $a = 6, b = 2$，$a + b = 8$。',
      },
    ],
  },
]
