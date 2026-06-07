export const LESSONS = [
  {
    id: 'u2c1',
    title: '平面向量的数量积',
    subtitle: '夹角、垂直与投影的统一语言',
    intro: [
      {
        title: '为什么需要数量积？',
        body: '两个向量可以相加、可以数乘，但能不能"相乘"出一个有意义的量？物理里力做的功等于力的大小乘以位移大小再乘以夹角余弦——这正是数量积的原型。数量积把两个向量"压缩"成一个数，同时保留了方向信息（夹角），是计算夹角、判断垂直、求投影的核心工具。',
      },
      {
        title: '数量积的定义',
        body: '设向量 $\\vec{a}$ 与 $\\vec{b}$ 的夹角为 $\\theta$（$0 \\leq \\theta \\leq \\pi$），则数量积（点积）定义为：',
        formula: '\\vec{a} \\cdot \\vec{b} = |\\vec{a}||\\vec{b}|\\cos\\theta',
        tip: '结果是一个数（标量），不是向量。夹角范围限定在 $[0, \\pi]$，零向量与任何向量的数量积为 $0$。',
        image: 'geo-p2-01-1.svg',
        imageCaption: '数量积的几何意义：$|\\vec{b}|\\cos\\theta$ 是 $\\vec{b}$ 在 $\\vec{a}$ 方向上的投影',
      },
      {
        title: '坐标计算公式',
        body: '若 $\\vec{a} = (x_1, y_1)$，$\\vec{b} = (x_2, y_2)$，则数量积直接用坐标计算：',
        formula: '\\vec{a} \\cdot \\vec{b} = x_1 x_2 + y_1 y_2',
        steps: [
          '对应分量相乘：$x_1 x_2$ 和 $y_1 y_2$',
          '两项相加，得到一个数',
          '不需要求模长，不需要求夹角，直接出结果',
        ],
        tip: '坐标公式是定义式的等价形式。两种形式互相转换是本章的核心技能。',
      },
      {
        title: '引入题逐步还原',
        body: '已知 $\\vec{a} = (3, 4)$，$\\vec{b} = (1, -2)$，求数量积、模长与夹角余弦。三步走，每步独立完成再代入：',
        steps: [
          '第一步：数量积 $\\vec{a} \\cdot \\vec{b} = 3 \\times 1 + 4 \\times (-2) = 3 - 8 = -5$（负数 → 夹角为钝角）',
          '第二步：模长 $|\\vec{a}| = \\sqrt{9+16} = 5$，$|\\vec{b}| = \\sqrt{1+4} = \\sqrt{5}$',
          '第三步：夹角余弦 $\\cos\\theta = \\dfrac{-5}{5\\sqrt{5}} = -\\dfrac{\\sqrt{5}}{5} \\approx -0.447$，对应 $\\theta \\approx 116.6°$，是钝角，与第一步吻合',
        ],
        tip: '关键节奏：先算点积 → 再算模长 → 最后商。不要混在一起算。',
      },
      {
        title: '三大应用',
        body: '数量积的三个最常用方向：',
        steps: [
          '求夹角：$\\cos\\theta = \\dfrac{\\vec{a} \\cdot \\vec{b}}{|\\vec{a}||\\vec{b}|}$，先算点积和模长再相除',
          '判垂直：$\\vec{a} \\perp \\vec{b} \\Leftrightarrow \\vec{a} \\cdot \\vec{b} = 0 \\Leftrightarrow x_1 x_2 + y_1 y_2 = 0$',
          '求投影：$\\vec{b}$ 在 $\\vec{a}$ 方向上的投影 $= \\dfrac{\\vec{a} \\cdot \\vec{b}}{|\\vec{a}|}$（分母是基准向量的模）',
        ],
        image: 'geo-p2-01-2.svg',
        imageCaption: '垂直时数量积为零：$\\vec{a} \\cdot \\vec{b} = |\\vec{a}||\\vec{b}|\\cos 90° = 0$',
      },
      {
        title: '模长展开与极化恒等式',
        body: '利用 $|\\vec{a}|^2 = \\vec{a} \\cdot \\vec{a}$，可以展开向量和差的模长：',
        formula: '|\\vec{a} + \\vec{b}|^2 = |\\vec{a}|^2 + 2\\vec{a}\\cdot\\vec{b} + |\\vec{b}|^2',
        steps: [
          '同理 $|\\vec{a} - \\vec{b}|^2 = |\\vec{a}|^2 - 2\\vec{a}\\cdot\\vec{b} + |\\vec{b}|^2$',
          '两式相减除以 4，得极化恒等式：$\\vec{a}\\cdot\\vec{b} = \\dfrac{1}{4}(|\\vec{a}+\\vec{b}|^2 - |\\vec{a}-\\vec{b}|^2)$',
          '题目给出对角线长（即 $|\\vec{a}+\\vec{b}|$ 和 $|\\vec{a}-\\vec{b}|$）时，直接用极化恒等式求点积',
        ],
        tip: '看到"已知两对角线长"→ 立刻想极化恒等式。',
      },
      {
        title: '运算性质与陷阱',
        body: '数量积满足交换律和分配律，但不满足结合律：',
        steps: [
          '交换律：$\\vec{a}\\cdot\\vec{b} = \\vec{b}\\cdot\\vec{a}$（可以）',
          '分配律：$\\vec{a}\\cdot(\\vec{b}+\\vec{c}) = \\vec{a}\\cdot\\vec{b} + \\vec{a}\\cdot\\vec{c}$（可以）',
          '结合律：$(\\vec{a}\\cdot\\vec{b})\\vec{c} \\neq \\vec{a}(\\vec{b}\\cdot\\vec{c})$（不可以！）',
        ],
        tip: '陷阱：$(\\vec{a}\\cdot\\vec{b})$ 已经是一个数，乘以 $\\vec{c}$ 得到向量；$\\vec{a}(\\vec{b}\\cdot\\vec{c})$ 方向是 $\\vec{a}$ 的方向，两者方向不同，不相等。',
      },
      {
        title: '易错辨析',
        body: '本章最常见的三类错误：',
        steps: [
          '错误1：把 $\\vec{a}\\cdot\\vec{b}$ 当成向量——实际上点积结果是数（标量），不能再做点积',
          '错误2：投影公式分母搞反——$\\vec{b}$ 在 $\\vec{a}$ 方向上的投影，分母是 $|\\vec{a}|$，不是 $|\\vec{b}|$',
          '错误3：忽视夹角范围——向量夹角 $\\theta \\in [0, \\pi]$，不是 $(-\\pi, \\pi)$；判断锐/直/钝角要先确认是否为零向量',
        ],
        reveal: {
          q: '若 $|\\vec{a}| = 2$，$|\\vec{b}| = 3$，夹角 $60°$，求 $\\vec{b}$ 在 $\\vec{a}$ 方向上的投影。',
          a: '先算点积：$\\vec{a}\\cdot\\vec{b} = 2 \\times 3 \\times \\cos 60° = 3$。投影 $= \\dfrac{\\vec{a}\\cdot\\vec{b}}{|\\vec{a}|} = \\dfrac{3}{2}$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '已知 $\\vec{a} = (3, 4)$，$\\vec{b} = (1, -2)$，则 $\\vec{a} \\cdot \\vec{b}$ 等于：',
        options: ['$-5$', '$5$', '$-1$', '$11$'],
        answer: 0,
        explain: '$\\vec{a}\\cdot\\vec{b} = 3\\times1 + 4\\times(-2) = 3 - 8 = -5$。',
      },
      {
        type: 'choice',
        prompt: '已知 $|\\vec{a}| = 2$，$|\\vec{b}| = 3$，$\\vec{a}$ 与 $\\vec{b}$ 的夹角为 $60°$，则 $|\\vec{a}+\\vec{b}|$ 等于：',
        options: ['$\\sqrt{19}$', '$\\sqrt{7}$', '$5$', '$\\sqrt{13}$'],
        answer: 0,
        explain: '$\\vec{a}\\cdot\\vec{b} = 2\\times3\\times\\frac{1}{2} = 3$。$|\\vec{a}+\\vec{b}|^2 = 4 + 2\\times3 + 9 = 19$，故 $|\\vec{a}+\\vec{b}| = \\sqrt{19}$。',
      },
      {
        type: 'choice',
        prompt: '向量 $\\vec{a} = (\\lambda, 2)$ 与 $\\vec{b} = (1, \\lambda - 1)$ 垂直，则 $\\lambda$ 等于：',
        options: ['$\\dfrac{2}{3}$', '$-\\dfrac{2}{3}$', '$2$', '$-2$'],
        answer: 0,
        explain: '$\\vec{a}\\cdot\\vec{b} = \\lambda\\times1 + 2\\times(\\lambda-1) = 3\\lambda - 2 = 0$，解得 $\\lambda = \\dfrac{2}{3}$。验证：$\\vec{a}\\cdot\\vec{b} = \\dfrac{2}{3} + 2\\times(-\\dfrac{1}{3}) = 0$ ✓。',
      },
      {
        type: 'input',
        prompt: '已知 $\\vec{a} = (-3, 4)$，$\\vec{b} = (2, 1)$，求 $\\vec{a}\\cdot\\vec{b}$（填数字）。',
        accept: ['-2'],
        explain: '$\\vec{a}\\cdot\\vec{b} = (-3)\\times2 + 4\\times1 = -6 + 4 = -2$。',
      },
      {
        type: 'judge',
        prompt: '数量积满足结合律，即 $(\\vec{a}\\cdot\\vec{b})\\cdot\\vec{c} = \\vec{a}\\cdot(\\vec{b}\\cdot\\vec{c})$ 成立。',
        answer: false,
        explain: '数量积不满足结合律。$(\\vec{a}\\cdot\\vec{b})$ 是标量，$(\\vec{a}\\cdot\\vec{b})\\vec{c}$ 是向量（$\\vec{c}$ 方向）；$\\vec{b}\\cdot\\vec{c}$ 也是标量，$\\vec{a}(\\vec{b}\\cdot\\vec{c})$ 是向量（$\\vec{a}$ 方向）。两者方向不同，一般不相等。',
      },
      {
        type: 'choice',
        prompt: '已知 $\\vec{a} = (1, \\sqrt{3})$，则 $\\vec{a}$ 与 $x$ 轴正方向（即 $\\vec{e}_1 = (1,0)$）的夹角为：',
        options: ['$30°$', '$45°$', '$60°$', '$120°$'],
        answer: 2,
        explain: '$\\vec{a}\\cdot\\vec{e}_1 = 1$，$|\\vec{a}| = \\sqrt{1+3} = 2$，$|\\vec{e}_1| = 1$。$\\cos\\theta = \\dfrac{1}{2}$，$\\theta = 60°$。',
      },
      {
        type: 'match',
        prompt: '将下列条件与对应结论配对：',
        left: [
          '$\\vec{a}\\cdot\\vec{b} > 0$',
          '$\\vec{a}\\cdot\\vec{b} = 0$',
          '$\\vec{a}\\cdot\\vec{b} < 0$',
        ],
        right: [
          '夹角为锐角（两向量均非零）',
          '两向量垂直',
          '夹角为钝角',
        ],
        explain: '由定义 $\\vec{a}\\cdot\\vec{b} = |\\vec{a}||\\vec{b}|\\cos\\theta$：$\\cos\\theta > 0$ 对应 $\\theta \\in (0°, 90°)$（锐角）；$\\cos\\theta = 0$ 对应 $\\theta = 90°$（垂直）；$\\cos\\theta < 0$ 对应 $\\theta \\in (90°, 180°]$（钝角）。',
      },
      {
        type: 'judge',
        prompt: '若 $\\vec{a} = (3, 4)$，则 $\\vec{a}$ 在 $\\vec{e}_1 = (1, 0)$ 方向上的投影等于 $3$，与 $\\vec{a}$ 的 $x$ 分量相同。',
        answer: true,
        explain: '投影 $= \\dfrac{\\vec{a}\\cdot\\vec{e}_1}{|\\vec{e}_1|} = \\dfrac{3\\times1+4\\times0}{1} = 3$，恰为 $x$ 分量。这说明坐标分量就是向量在坐标轴方向上的投影。',
      },
      {
        type: 'input',
        prompt: '已知 $|\\vec{a}| = 1$，$|\\vec{b}| = 2$，$|\\vec{a}+\\vec{b}| = \\sqrt{7}$，则 $\\vec{a}$ 与 $\\vec{b}$ 的夹角为多少度（填整数）？',
        accept: ['60'],
        explain: '$|\\vec{a}+\\vec{b}|^2 = 1 + 2\\vec{a}\\cdot\\vec{b} + 4 = 7$，故 $\\vec{a}\\cdot\\vec{b} = 1$。$\\cos\\theta = \\dfrac{1}{1\\times2} = \\dfrac{1}{2}$，$\\theta = 60°$。',
      },
    ],
  },
]
