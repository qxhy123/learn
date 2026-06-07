export const LESSONS = [
  {
    id: 'u3c1',
    title: '直线的斜率与倾斜角',
    subtitle: '用一个数刻画直线的"倾斜程度"',
    intro: [
      {
        title: '为什么需要倾斜程度的概念？',
        body: '生活中斜坡有陡有缓，坐标平面里的直线也有各种倾斜方式。如何用一个数字统一刻画这些差异，让它可以参与代数运算？数学给出两个工具：倾斜角（角度视角）和斜率（数值视角）。',
      },
      {
        title: '倾斜角的定义',
        body: '从 $x$ 轴正方向出发，逆时针旋转到直线向上方向所经过的最小非负角，称为该直线的倾斜角，记作 $\\alpha$。规定范围：',
        formula: '\\alpha \\in [0,\\, \\pi)',
        tip: '水平线 $\\alpha = 0$，垂直线 $\\alpha = \\dfrac{\\pi}{2}$，范围右端开区间（不含 $\\pi$），因为 $\\alpha = \\pi$ 与 $\\alpha = 0$ 代表同一方向。',
      },
      {
        title: '为什么范围是 $[0,\\pi)$ 而不是 $[0,2\\pi)$？',
        body: '直线没有方向之分——从两端看都是同一条直线。逆时针转过 $\\pi$ 后，方向与原来完全相同。因此半个圆（$[0,\\pi)$）足以覆盖所有直线的倾斜情形，不需要整圈。',
        image: 'geo-p3-01-1.svg',
        imageCaption: '倾斜角示意：从 $x$ 轴正方向逆时针量到直线向上方向',
      },
      {
        title: '斜率的定义与两点公式',
        body: '由倾斜角定义斜率 $k = \\tan\\alpha$（当 $\\alpha \\neq \\dfrac{\\pi}{2}$ 时）。若直线过两点 $(x_1, y_1)$ 和 $(x_2, y_2)$，且 $x_1 \\neq x_2$，则：',
        formula: 'k = \\frac{y_2 - y_1}{x_2 - x_1}',
        steps: [
          '分子 $y_2 - y_1$ 是纵向变化量（rise）',
          '分母 $x_2 - x_1$ 是横向变化量（run）',
          '斜率 = 每向右走一格，纵坐标变化了多少',
          '两点顺序互换，分子分母同时变号，商不变',
        ],
      },
      {
        title: '倾斜角与斜率的对应关系',
        body: '四种典型情形的对应关系，帮助建立直觉：',
        steps: [
          '$\\alpha = 0$：$k = 0$，水平线，形如 $y = c$',
          '$0 < \\alpha < \\dfrac{\\pi}{2}$：$k > 0$，右上倾斜（锐角）',
          '$\\alpha = \\dfrac{\\pi}{2}$：$k$ 不存在，垂直线，形如 $x = c$',
          '$\\dfrac{\\pi}{2} < \\alpha < \\pi$：$k < 0$，右下倾斜（钝角）',
        ],
        image: 'geo-p3-01-2.svg',
        imageCaption: '四种倾斜角情形：水平、锐角、垂直、钝角',
      },
      {
        title: '例题：已知两点求斜率和倾斜角',
        body: '直线 $l$ 过点 $A(1,2)$ 和 $B(4,-1)$，求斜率和倾斜角。',
        steps: [
          '代入两点公式：$k = \\dfrac{-1-2}{4-1} = \\dfrac{-3}{3} = -1$',
          '$k < 0$，故 $\\alpha \\in \\left(\\dfrac{\\pi}{2}, \\pi\\right)$',
          '$\\tan\\alpha = -1 = \\tan\\left(\\pi - \\dfrac{\\pi}{4}\\right)$',
          '故 $\\alpha = \\dfrac{3\\pi}{4}$（即 $135°$）',
        ],
      },
      {
        title: '易错点：四个高频混淆',
        body: '以下四个易错点是历年高考的常见考点，务必分清：',
        steps: [
          '垂直线斜率不存在，不是 $k = 0$ 也不是 $k = \\infty$',
          '$k = 0$ 是水平线，$k$ 不存在是垂直线——两者截然不同',
          '倾斜角范围是 $[0,\\pi)$，不含 $\\pi$，含 $0$',
          '由 $\\tan\\alpha = k$ 反解 $\\alpha$ 时，必须先确定 $\\alpha$ 所在区间',
        ],
        tip: '判断"直线 $x = 3$ 的斜率为 $0$"——错误！斜率不存在。',
      },
      {
        title: '拓展思考',
        body: '"斜率越大，直线越陡"这句话对吗？',
        reveal: {
          q: '斜率 $k=-3$ 的直线，比斜率 $k=2$ 的直线倾斜角大还是小？',
          a: '大。$\\tan\\alpha=-3$ 对应 $\\alpha\\approx108°$，$\\tan\\alpha=2$ 对应 $\\alpha\\approx63°$。"斜率越大越陡"是错的；应用倾斜角与 $\\dfrac{\\pi}{2}$ 的距离来衡量陡度。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '直线的倾斜角 $\\alpha$ 的范围是哪个区间？',
        options: [
          '$[0, 2\\pi)$',
          '$[0, \\pi]$',
          '$[0, \\pi)$',
          '$\\left(0, \\dfrac{\\pi}{2}\\right)$',
        ],
        answer: 2,
        explain: '倾斜角规定在 $[0,\\pi)$：含左端点 $0$（水平线），不含右端点 $\\pi$（与 $0$ 代表同一方向）。',
      },
      {
        type: 'judge',
        prompt: '直线 $x = 5$ 的斜率为 $0$。',
        answer: false,
        explain: '$x = 5$ 是垂直于 $x$ 轴的直线，倾斜角为 $\\dfrac{\\pi}{2}$，斜率不存在（undefined），而非 $0$。斜率为 $0$ 是水平线 $y = c$ 的性质。',
      },
      {
        type: 'input',
        prompt: '直线过点 $A(1,2)$ 和 $B(4,-1)$，求斜率 $k$（化简为分数或整数）。',
        accept: ['-1', '−1'],
        explain: '$k = \\dfrac{-1-2}{4-1} = \\dfrac{-3}{3} = -1$。',
      },
      {
        type: 'choice',
        prompt: '若直线的倾斜角 $\\alpha = \\dfrac{3\\pi}{4}$，则斜率 $k$ 等于？',
        options: [
          '$\\sqrt{3}$',
          '$-1$',
          '$1$',
          '$-\\sqrt{3}$',
        ],
        answer: 1,
        explain: '$k = \\tan\\dfrac{3\\pi}{4} = \\tan\\left(\\pi - \\dfrac{\\pi}{4}\\right) = -\\tan\\dfrac{\\pi}{4} = -1$。',
      },
      {
        type: 'choice',
        prompt: '已知直线斜率 $k \\in (0, \\sqrt{3}]$，则倾斜角 $\\alpha$ 的范围是？',
        options: [
          '$\\left(0, \\dfrac{\\pi}{3}\\right]$',
          '$\\left[0, \\dfrac{\\pi}{3}\\right]$',
          '$\\left(0, \\dfrac{\\pi}{6}\\right]$',
          '$\\left(0, \\dfrac{\\pi}{2}\\right)$',
        ],
        answer: 0,
        explain: '$k > 0$ 说明 $\\alpha \\in \\left(0, \\dfrac{\\pi}{2}\\right)$。在此区间 $\\tan$ 单调递增，$\\tan\\alpha \\leq \\sqrt{3} = \\tan\\dfrac{\\pi}{3}$，故 $\\alpha \\leq \\dfrac{\\pi}{3}$；又 $\\tan\\alpha > 0$，故 $\\alpha > 0$，综合得 $\\alpha \\in \\left(0, \\dfrac{\\pi}{3}\\right]$。',
      },
      {
        type: 'match',
        prompt: '将倾斜角与对应的直线类型配对。',
        left: [
          '$\\alpha = 0$',
          '$\\alpha = \\dfrac{\\pi}{2}$',
          '$0 < \\alpha < \\dfrac{\\pi}{2}$',
          '$\\dfrac{\\pi}{2} < \\alpha < \\pi$',
        ],
        right: [
          '水平线，$k = 0$',
          '垂直线，$k$ 不存在',
          '右上倾斜，$k > 0$',
          '右下倾斜，$k < 0$',
        ],
        explain: '倾斜角与斜率的对应：水平 $\\leftrightarrow k=0$，垂直 $\\leftrightarrow k$ 不存在，锐角 $\\leftrightarrow k>0$，钝角 $\\leftrightarrow k<0$。',
      },
      {
        type: 'input',
        prompt: '已知 $A(a, 3)$，$B(2, -1)$，直线 $AB$ 的倾斜角为 $\\dfrac{\\pi}{4}$，求 $a$ 的值。',
        accept: ['6'],
        explain: '$k = \\tan\\dfrac{\\pi}{4} = 1$，代入两点公式：$\\dfrac{-1-3}{2-a} = 1$，解得 $-4 = 2-a$，即 $a = 6$。验证：$k = \\dfrac{-4}{2-6} = 1$ ✓。',
      },
      {
        type: 'judge',
        prompt: '斜率越大的直线，倾斜角也越大。',
        answer: false,
        explain: '错误。斜率 $k=-3$ 对应倾斜角约 $108°$，斜率 $k=2$ 对应约 $63°$。负斜率的直线倾斜角可能比正斜率的更大。"陡度"应用倾斜角与 $\\dfrac{\\pi}{2}$ 的距离来衡量，而非斜率大小。',
      },
    ],
  },
]
