export const LESSONS = [
  {
    id: 'u9c2',
    title: '空间向量的数量积',
    subtitle: '点积公式、模长、垂直与夹角',
    intro: [
      {
        title: '数量积的动机',
        body: '平面向量中，数量积 $\\vec{a}\\cdot\\vec{b}=|\\vec{a}||\\vec{b}|\\cos\\theta$ 用于计算投影和夹角。空间中，两个向量同样可以确定一个夹角，因此数量积的定义完全相同，只需把坐标从 $(x,y)$ 扩展到 $(x,y,z)$。最终结果是一个实数（数量），不是向量，这正是"数量积"名称的由来。',
      },
      {
        title: '数量积的定义与特殊值',
        body: '设空间向量 $\\vec{a}$ 与 $\\vec{b}$ 的夹角为 $\\theta$（$0\\leq\\theta\\leq\\pi$），定义：$\\vec{a}\\cdot\\vec{b}=|\\vec{a}||\\vec{b}|\\cos\\theta$。\n\n三种特殊情形：$\\theta=0$（同向）时 $\\vec{a}\\cdot\\vec{b}=|\\vec{a}||\\vec{b}|$；$\\theta=\\dfrac{\\pi}{2}$（垂直）时 $\\vec{a}\\cdot\\vec{b}=0$；$\\theta=\\pi$（反向）时 $\\vec{a}\\cdot\\vec{b}=-|\\vec{a}||\\vec{b}|$。',
        formula: '\\vec{a}\\cdot\\vec{b} = |\\vec{a}||\\vec{b}|\\cos\\theta',
        image: 'geo-p9-02-1.svg',
        imageCaption: '空间两向量夹角 $\\theta\\in[0,\\pi]$，平移到共起点后取较小夹角',
      },
      {
        title: '坐标公式的推导',
        body: '在空间直角坐标系中，标准基 $\\vec{i},\\vec{j},\\vec{k}$ 两两垂直且为单位向量：$\\vec{i}\\cdot\\vec{i}=\\vec{j}\\cdot\\vec{j}=\\vec{k}\\cdot\\vec{k}=1$，$\\vec{i}\\cdot\\vec{j}=\\vec{j}\\cdot\\vec{k}=\\vec{k}\\cdot\\vec{i}=0$。\n\n将 $\\vec{a}=x_1\\vec{i}+y_1\\vec{j}+z_1\\vec{k}$，$\\vec{b}=x_2\\vec{i}+y_2\\vec{j}+z_2\\vec{k}$ 展开，利用分配律展开 9 项，利用正交关系只剩下三项，得到坐标公式。对比平面：空间多了 $z_1z_2$ 一项。',
        formula: '\\vec{a}\\cdot\\vec{b} = x_1x_2 + y_1y_2 + z_1z_2',
      },
      {
        title: '模长公式与两点距离',
        body: '令 $\\vec{a}=\\vec{b}$，$\\theta=0$，$\\cos0=1$，则 $\\vec{a}\\cdot\\vec{a}=|\\vec{a}|^2$。用坐标：$|\\vec{a}|^2=x^2+y^2+z^2$，所以 $|\\vec{a}|=\\sqrt{x^2+y^2+z^2}$。\n\n几何意义：这是点 $(x,y,z)$ 到原点的欧氏距离，来源于三维勾股定理（两次应用平面勾股定理）。两点 $A(x_1,y_1,z_1)$，$B(x_2,y_2,z_2)$ 之间距离：$|AB|=\\sqrt{(x_2-x_1)^2+(y_2-y_1)^2+(z_2-z_1)^2}$。',
        formula: '|\\vec{a}| = \\sqrt{x^2+y^2+z^2}',
      },
      {
        title: '垂直判定与夹角公式',
        body: '垂直充要条件：$\\vec{a}\\perp\\vec{b}\\Leftrightarrow\\vec{a}\\cdot\\vec{b}=0\\Leftrightarrow x_1x_2+y_1y_2+z_1z_2=0$（两向量均非零时）。\n\n夹角公式（两向量，$\\theta\\in[0,\\pi]$）：$\\cos\\theta=\\dfrac{\\vec{a}\\cdot\\vec{b}}{|\\vec{a}||\\vec{b}|}$，余弦可正可负。\n\n注意：若求两直线所成角或两平面所成角，需对余弦取绝对值（因为线和面方向不区分正负），结果在 $[0,\\frac{\\pi}{2}]$。',
        formula: '\\cos\\theta = \\frac{x_1x_2+y_1y_2+z_1z_2}{\\sqrt{x_1^2+y_1^2+z_1^2}\\cdot\\sqrt{x_2^2+y_2^2+z_2^2}}',
      },
      {
        title: '投影',
        body: '向量 $\\vec{b}$ 在 $\\vec{a}$ 方向上的投影（标量）：$\\text{proj}_{\\vec{a}}\\vec{b}=|\\vec{b}|\\cos\\theta=\\dfrac{\\vec{a}\\cdot\\vec{b}}{|\\vec{a}|}$。投影可正可负：$\\theta<\\dfrac{\\pi}{2}$ 时为正，$\\theta>\\dfrac{\\pi}{2}$ 时为负，$\\theta=\\dfrac{\\pi}{2}$ 时为 $0$。\n\n投影向量（沿 $\\vec{a}$ 方向的分量）：$\\dfrac{\\vec{a}\\cdot\\vec{b}}{|\\vec{a}|^2}\\vec{a}$，用于分解向量和求点到直线的距离。',
        formula: '\\mathrm{proj}_{\\vec{a}}\\vec{b} = \\frac{\\vec{a}\\cdot\\vec{b}}{|\\vec{a}|}',
      },
      {
        title: '运算性质与易错',
        body: '数量积满足：交换律 $\\vec{a}\\cdot\\vec{b}=\\vec{b}\\cdot\\vec{a}$，分配律 $\\vec{a}\\cdot(\\vec{b}+\\vec{c})=\\vec{a}\\cdot\\vec{b}+\\vec{a}\\cdot\\vec{c}$，数乘结合律 $(\\lambda\\vec{a})\\cdot\\vec{b}=\\lambda(\\vec{a}\\cdot\\vec{b})$，自积 $\\vec{a}\\cdot\\vec{a}=|\\vec{a}|^2$。\n\n易错：①结果是数，不是向量，不能写 $(\\vec{a}\\cdot\\vec{b})\\cdot\\vec{c}$（点积）；②数量积没有结合律；③没有消去律（$\\vec{a}\\cdot\\vec{b}=\\vec{a}\\cdot\\vec{c}$ 且 $\\vec{a}\\neq\\vec{0}$ 不能推 $\\vec{b}=\\vec{c}$）；④模长是 $\\sqrt{x^2+y^2+z^2}$，不是 $|x|+|y|+|z|$。',
        reveal: {
          q: '已知 $\\vec{a}=(1,-1,2)$，$\\vec{b}=(2,1,1)$，求夹角 $\\theta$。',
          a: '$\\vec{a}\\cdot\\vec{b}=2-1+2=3$，$|\\vec{a}|=\\sqrt{6}$，$|\\vec{b}|=\\sqrt{6}$，$\\cos\\theta=\\frac{3}{6}=\\frac{1}{2}$，$\\theta=60^\\circ$。',
        },
      },
    ],
    questions: [
      {
        type: 'input',
        prompt: '已知 $\\vec{a}=(2,-1,2)$，$\\vec{b}=(1,3,-2)$，计算 $\\vec{a}\\cdot\\vec{b}$',
        accept: ['-5'],
        explain: '$\\vec{a}\\cdot\\vec{b}=2\\times1+(-1)\\times3+2\\times(-2)=2-3-4=-5$。',
      },
      {
        type: 'input',
        prompt: '已知 $\\vec{a}=(3,-4,0)$，求 $|\\vec{a}|$',
        accept: ['5'],
        explain: '$|\\vec{a}|=\\sqrt{3^2+(-4)^2+0^2}=\\sqrt{9+16}=\\sqrt{25}=5$。',
      },
      {
        type: 'choice',
        prompt: '已知 $\\vec{a}=(2,-1,k)$，$\\vec{b}=(1,3,1)$，若 $\\vec{a}\\perp\\vec{b}$，则 $k=$',
        options: ['$-1$', '$0$', '$1$', '$2$'],
        answer: 2,
        explain: '$\\vec{a}\\cdot\\vec{b}=2\\times1+(-1)\\times3+k\\times1=2-3+k=k-1=0$，故 $k=1$。验证：$\\vec{a}=(2,-1,1)$，$\\vec{b}=(1,3,1)$，$\\vec{a}\\cdot\\vec{b}=2-3+1=0$。正确。',
      },
      {
        type: 'choice',
        prompt: '已知 $\\vec{a}=(1,2,-2)$，$\\vec{b}=(3,-4,0)$，则 $\\vec{a}$ 与 $\\vec{b}$ 的夹角 $\\theta$ 满足 $\\cos\\theta=$',
        options: ['$\\dfrac{1}{3}$', '$-\\dfrac{1}{3}$', '$\\dfrac{1}{5}$', '$-\\dfrac{1}{5}$'],
        answer: 1,
        explain: '$\\vec{a}\\cdot\\vec{b}=1\\times3+2\\times(-4)+(-2)\\times0=3-8+0=-5$。$|\\vec{a}|=\\sqrt{1+4+4}=3$，$|\\vec{b}|=\\sqrt{9+16}=5$。$\\cos\\theta=\\frac{-5}{3\\times5}=-\\frac{1}{3}$。',
      },
      {
        type: 'judge',
        prompt: '若 $\\vec{a}\\cdot\\vec{b}=\\vec{a}\\cdot\\vec{c}$ 且 $\\vec{a}\\neq\\vec{0}$，则一定有 $\\vec{b}=\\vec{c}$。',
        answer: false,
        explain: '数量积没有消去律。反例：$\\vec{a}=(1,0,0)$，$\\vec{b}=(1,1,0)$，$\\vec{c}=(1,0,1)$，则 $\\vec{a}\\cdot\\vec{b}=\\vec{a}\\cdot\\vec{c}=1$，但 $\\vec{b}\\neq\\vec{c}$。',
      },
      {
        type: 'input',
        prompt: '已知空间两点 $P(1,0,2)$，$Q(4,-3,5)$，求 $|PQ|$（填精确值，如 $3\\sqrt{3}$）',
        accept: ['3√3', '$3\\sqrt{3}$'],
        explain: '$\\vec{PQ}=(4-1,-3-0,5-2)=(3,-3,3)$，$|PQ|=\\sqrt{9+9+9}=\\sqrt{27}=3\\sqrt{3}$。',
      },
      {
        type: 'choice',
        prompt: '已知 $|\\vec{a}|=2$，$|\\vec{b}|=\\sqrt{3}$，$\\vec{a}$ 与 $\\vec{b}$ 夹角为 $\\dfrac{\\pi}{6}$，则 $\\vec{b}$ 在 $\\vec{a}$ 方向上的投影为',
        options: ['$\\dfrac{\\sqrt{3}}{2}$', '$\\dfrac{3}{2}$', '$3$', '$\\dfrac{3}{4}$'],
        answer: 1,
        explain: '$\\vec{a}\\cdot\\vec{b}=2\\times\\sqrt{3}\\times\\cos\\frac{\\pi}{6}=2\\sqrt{3}\\times\\frac{\\sqrt{3}}{2}=3$。投影 $=\\frac{\\vec{a}\\cdot\\vec{b}}{|\\vec{a}|}=\\frac{3}{2}$。',
      },
      {
        type: 'match',
        prompt: '将各情形与对应的数量积值或结论配对',
        left: [
          '$\\vec{a}$ 与 $\\vec{b}$ 夹角为 $\\dfrac{\\pi}{2}$',
          '$|\\vec{a}|=1$ 时 $\\vec{a}\\cdot\\vec{a}$',
          '$\\vec{a}=(1,0,0)$，$\\vec{b}=(0,1,0)$',
          '夹角 $\\theta=\\pi$，$|\\vec{a}|=2$，$|\\vec{b}|=3$',
        ],
        right: [
          '$\\vec{a}\\cdot\\vec{b}=0$（垂直）',
          '$-6$',
          '$1$',
          '$\\vec{a}\\cdot\\vec{b}=0$（坐标算）',
        ],
        explain: '①夹角 $\\frac{\\pi}{2}$ 时 $\\cos\\theta=0$，点积为 $0$；②单位向量自积 $=1$；③$(1,0,0)\\cdot(0,1,0)=0$；④$\\theta=\\pi$ 时 $\\cos\\pi=-1$，点积 $=2\\times3\\times(-1)=-6$。',
      },
    ],
  },
]
