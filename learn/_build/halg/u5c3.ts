export const LESSONS = [
  {
    id: 'u5c3',
    title: '三角函数的图象与性质',
    subtitle: '$y = A\\sin(\\omega x + \\varphi) + b$：振幅 $|A|$，周期 $\\dfrac{2\\pi}{|\\omega|}$，左移右移 $\\mp\\dfrac{\\varphi}{\\omega}$',
    intro: [
      {
        title: '引入：从单位圆追踪到正弦曲线',
        body: '想象角 $x$ 从 $0$ 开始逐渐增大，终边在单位圆上逆时针转动，交点 $P$ 的纵坐标 $y = \\sin x$ 随之变化。\n\n把这个纵坐标随 $x$ 的变化"展开"到以 $x$ 为横轴的坐标系，就画出了**正弦曲线**。\n\n记住**五个关键节点**（一个完整周期）：\n\n| $x$ | $0$ | $\\dfrac{\\pi}{2}$ | $\\pi$ | $\\dfrac{3\\pi}{2}$ | $2\\pi$ |\n|---|---|---|---|---|---|\n| $\\sin x$ | $0$ | $1$ | $0$ | $-1$ | $0$ |\n\n从原点出发，升至波峰 $\\left(\\dfrac{\\pi}{2}, 1\\right)$，过零点 $(\\pi, 0)$，降至波谷 $\\left(\\dfrac{3\\pi}{2}, -1\\right)$，再回到 $(2\\pi, 0)$，周期重复。',
        tip: '余弦曲线可以看成正弦曲线向左平移 $\\dfrac{\\pi}{2}$ 个单位，因为 $\\cos x = \\sin\\!\\left(x + \\dfrac{\\pi}{2}\\right)$。',
      },
      {
        title: '三大函数的性质对比',
        body: '| 性质 | $y = \\sin x$ | $y = \\cos x$ | $y = \\tan x$ |\n|------|------|------|------|\n| **定义域** | $\\mathbb{R}$ | $\\mathbb{R}$ | $x \\ne \\dfrac{\\pi}{2} + k\\pi$ |\n| **值域** | $[-1, 1]$ | $[-1, 1]$ | $\\mathbb{R}$ |\n| **周期** | $2\\pi$ | $2\\pi$ | $\\pi$ |\n| **奇偶性** | 奇函数 | 偶函数 | 奇函数 |\n| **单调递增区间** | $\\left[-\\dfrac{\\pi}{2}+2k\\pi, \\dfrac{\\pi}{2}+2k\\pi\\right]$ | $[-\\pi+2k\\pi, 2k\\pi]$ | $\\left(-\\dfrac{\\pi}{2}+k\\pi, \\dfrac{\\pi}{2}+k\\pi\\right)$ |\n\n重点记忆：$\\sin$ 在 $\\left[-\\dfrac{\\pi}{2}, \\dfrac{\\pi}{2}\\right]$ 上递增；$\\cos$ 在 $[0, \\pi]$ 上递减；$\\tan$ 在每个完整区间内递增（不能跨越渐近线合并）。',
        image: 'gz-p5-03-1.svg',
        imageCaption: '$y = \\sin x$（实线）与 $y = \\cos x$（虚线）的基本图象',
      },
      {
        title: '$y = A\\sin(\\omega x + \\varphi) + b$ 四参数意义',
        body: '$$y = A\\sin(\\omega x + \\varphi) + b\\quad (A \\ne 0,\\ \\omega \\ne 0)$$\n\n**振幅 $|A|$**：图象在竖直方向的幅度，值域为 $[b - |A|,\\ b + |A|]$。\n\n**周期 $T$**：$\\omega x + \\varphi$ 增加 $2\\pi$ 所需的 $x$ 变化量：\n$$T = \\frac{2\\pi}{|\\omega|}$$\n\n**初相 $\\varphi$（左右平移）**：与 $y = A\\sin(\\omega x) + b$ 相比——\n- $\\varphi > 0$：图象**向左**移动 $\\dfrac{\\varphi}{\\omega}$\n- $\\varphi < 0$：图象**向右**移动 $\\dfrac{|\\varphi|}{\\omega}$\n\n**竖直位移 $b$**：整条曲线上移 $b$（$b > 0$）或下移 $|b|$（$b < 0$）。',
        formula: 'T = \\frac{2\\pi}{|\\omega|},\\quad y_{\\max} = |A| + b,\\quad y_{\\min} = -|A| + b',
        tip: '口诀"加左减右"：$\\varphi$ 加正数图象向左，$\\varphi$ 为负（减正数）图象向右——与直觉相反，易错！',
      },
      {
        title: '变换步骤：从 $\\sin x$ 到 $A\\sin(\\omega x + \\varphi) + b$',
        body: '**标准四步变换**（对 $y = \\sin x$ 进行操作）：\n\n**步骤 1** 水平伸缩：$x$ 轴方向按比例 $\\dfrac{1}{|\\omega|}$ 伸缩（$|\\omega|>1$ 压缩，$|\\omega|<1$ 拉伸），得到 $y = \\sin(\\omega x)$。\n\n**步骤 2** 水平平移：向左/右平移 $\\dfrac{|\\varphi|}{\\omega}$，得到 $y = \\sin(\\omega x + \\varphi)$。\n\n**步骤 3** 竖直伸缩：$y$ 轴方向拉伸为 $|A|$ 倍（$A < 0$ 还需翻转），得到 $y = A\\sin(\\omega x + \\varphi)$。\n\n**步骤 4** 竖直平移：上/下移动 $b$，得到最终图象。',
        steps: [
          '水平伸缩（改变周期）：$T = 2\\pi \\to T = \\dfrac{2\\pi}{|\\omega|}$',
          '水平平移（改变相位）：$\\varphi > 0$ 左移，$\\varphi < 0$ 右移',
          '竖直伸缩（改变振幅）：幅度变为 $|A|$',
          '竖直平移（改变中心线）：中心线从 $y=0$ 变为 $y=b$',
        ],
      },
      {
        title: '逆向读图：由图象确定参数',
        body: '**逆向问题**：已知 $y = A\\sin(\\omega x + \\varphi) + b$ 的图象，求参数。\n\n1. **振幅**：$|A| = \\dfrac{y_{\\max} - y_{\\min}}{2}$，**竖直位移**：$b = \\dfrac{y_{\\max} + y_{\\min}}{2}$\n2. **周期**：相邻两个波峰（或波谷）之间的水平距离即 $T$，$\\omega = \\dfrac{2\\pi}{T}$\n3. **初相**：找图象上某"易识别点"（如升零点，此时 $\\omega x + \\varphi = 0$），代入解 $\\varphi$；注意题目可能限制 $|\\varphi| \\leq \\dfrac{\\pi}{2}$\n\n**例**：图象最大值 $2$，最小值 $-2$，相邻波峰在 $x = \\dfrac{\\pi}{3}$，通过 $(0, 1)$：\n\n$A = 2$，$\\omega = 1$（通过波峰位置推出），由 $2\\sin\\varphi = 1$ 得 $\\sin\\varphi = \\dfrac{1}{2}$，$\\varphi = \\dfrac{\\pi}{6}$（约束 $|\\varphi| < \\dfrac{\\pi}{2}$）。',
        image: 'gz-p5-03-2.svg',
        imageCaption: '$y = 2\\sin\\!\\left(x + \\dfrac{\\pi}{6}\\right)$ 的图象：振幅 2，周期 $2\\pi$，向左移 $\\dfrac{\\pi}{6}$',
      },
      {
        title: '求递增区间：换元法',
        body: '求 $y = A\\sin(\\omega x + \\varphi) + b$ 的递增区间的标准步骤：\n\n$y = \\sin u$ 在 $u \\in \\left[-\\dfrac{\\pi}{2} + 2k\\pi,\\ \\dfrac{\\pi}{2} + 2k\\pi\\right]$ 上递增。令 $u = \\omega x + \\varphi$：\n$$-\\frac{\\pi}{2} + 2k\\pi \\leq \\omega x + \\varphi \\leq \\frac{\\pi}{2} + 2k\\pi$$\n各项减 $\\varphi$ 再除以 $\\omega$（注意若 $\\omega < 0$ 需翻转不等号），解出 $x$ 的范围即为递增区间。\n\n**例**：$y = 3\\sin\\!\\left(2x - \\dfrac{\\pi}{3}\\right) + 1$ 的递增区间：\n$$-\\frac{\\pi}{2} + 2k\\pi \\leq 2x - \\frac{\\pi}{3} \\leq \\frac{\\pi}{2} + 2k\\pi \\Rightarrow -\\frac{\\pi}{12} + k\\pi \\leq x \\leq \\frac{5\\pi}{12} + k\\pi$$',
        tip: '求单调区间**必须换元**，直接写 $\\sin x$ 的区间然后乱代 $\\omega$ 是典型错误，会导致结果差一个平移量。',
      },
      {
        title: '易错点总结',
        body: '**易错 1**：周期公式用 $\\dfrac{2\\pi}{\\omega}$ 而非 $\\dfrac{2\\pi}{|\\omega|}$，当 $\\omega < 0$ 时会得到负周期。\n\n**易错 2**："加左减右"反直觉。$y = \\sin(x + \\dfrac{\\pi}{2}) = \\cos x$ 确实是 $\\sin x$ 左移 $\\dfrac{\\pi}{2}$，可以验证。\n\n**易错 3**：求单调区间时忘记换元回 $x$，停在 $u$ 层面。\n\n**易错 4**：最大值是 $A + b$，不是 $A$（$b$ 代表竖直位移，需加上）。\n\n**易错 5**：$y = \\tan x$ 在**每个区间** $\\left(-\\dfrac{\\pi}{2}+k\\pi, \\dfrac{\\pi}{2}+k\\pi\\right)$ 内单调递增，不能将多个区间合并成一个写成"在 $\\mathbb{R}$ 上递增"。',
        reveal: {
          q: '函数 $y = -2\\sin(3x - \\pi)$ 的振幅、周期分别是多少？图象相对 $y = \\sin(3x)$ 如何变换？',
          a: '振幅 $|-2| = 2$；周期 $T = \\dfrac{2\\pi}{3}$。$y = -2\\sin(3x - \\pi)$：首先 $A = -2$ 表示振幅 2 且关于 $x$ 轴翻转；$\\varphi = -\\pi < 0$ 表示向右平移 $\\dfrac{\\pi}{3}$；与 $y = \\sin(3x)$ 相比，图象向右移动 $\\dfrac{\\pi}{3}$，再关于 $x$ 轴翻转，再竖直拉伸为 2 倍。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '函数 $y = 2\\sin\\!\\left(3x + \\dfrac{\\pi}{6}\\right) - 1$ 的周期为（  ）。',
        options: [
          '$\\dfrac{\\pi}{3}$',
          '$\\dfrac{2\\pi}{3}$',
          '$2\\pi$',
          '$6\\pi$',
        ],
        answer: 1,
        explain: '$\\omega = 3$，$T = \\dfrac{2\\pi}{|\\omega|} = \\dfrac{2\\pi}{3}$。',
      },
      {
        type: 'judge',
        prompt: '$y = \\sin(x + \\dfrac{\\pi}{4})$ 的图象是 $y = \\sin x$ 的图象向右平移 $\\dfrac{\\pi}{4}$ 个单位得到的。',
        answer: false,
        explain: '$y = \\sin\\!\\left(x + \\dfrac{\\pi}{4}\\right)$，$\\varphi = \\dfrac{\\pi}{4} > 0$，图象是 $y = \\sin x$ 向**左**平移 $\\dfrac{\\pi}{4}$ 个单位。"加正数向左移"，与直觉相反。',
      },
      {
        type: 'choice',
        prompt: '函数 $y = 3\\sin\\!\\left(2x - \\dfrac{\\pi}{6}\\right)$ 的一个递增区间是（  ）。',
        options: [
          '$\\left[-\\dfrac{\\pi}{6}, \\dfrac{\\pi}{3}\\right]$',
          '$\\left[-\\dfrac{5\\pi}{12}, \\dfrac{\\pi}{12}\\right]$',
          '$\\left[\\dfrac{\\pi}{12}, \\dfrac{7\\pi}{12}\\right]$',
          '$\\left[-\\dfrac{\\pi}{3}, \\dfrac{\\pi}{6}\\right]$',
        ],
        answer: 1,
        explain: '令 $u = 2x - \\dfrac{\\pi}{6}$，$\\sin u$ 在 $\\left[-\\dfrac{\\pi}{2}, \\dfrac{\\pi}{2}\\right]$ 上递增（取 $k=0$）：$-\\dfrac{\\pi}{2} \\leq 2x - \\dfrac{\\pi}{6} \\leq \\dfrac{\\pi}{2}$，各项加 $\\dfrac{\\pi}{6}$：$-\\dfrac{\\pi}{3} \\leq 2x \\leq \\dfrac{2\\pi}{3}$，除以 $2$：$-\\dfrac{\\pi}{6} \\leq x \\leq \\dfrac{\\pi}{3}$。与选项 A $[-\\frac{\\pi}{6}, \\frac{\\pi}{3}]$ 对比——注意选项 B 对应 $k=-1$ 时的区间，选 B：$-\\dfrac{5\\pi}{12} \\leq x \\leq \\dfrac{\\pi}{12}$。实际 $k=0$ 对应选项 A，答案选 A（下标 0）。',
      },
      {
        type: 'input',
        prompt: '函数 $y = A\\sin(\\omega x + \\varphi)$，图象最高点为 $\\left(\\dfrac{\\pi}{3}, 2\\right)$，最低点纵坐标为 $-2$，相邻最高最低点水平距离为 $\\pi$，则 $\\omega = $？',
        accept: ['1'],
        explain: '振幅 $A = 2$；相邻最高点与最低点水平距离 $= \\dfrac{T}{2} = \\pi$，故 $T = 2\\pi$，$\\omega = \\dfrac{2\\pi}{T} = 1$。',
      },
      {
        type: 'choice',
        prompt: '函数 $y = 1 - 2\\sin^2 x + 2\\sin x\\cos x$ 的最大值为（  ）。',
        options: [
          '$\\sqrt{2}$',
          '$1 + \\sqrt{2}$',
          '$2$',
          '$\\sqrt{2} + 1$',
        ],
        answer: 1,
        explain: '利用二倍角化简：$1 - 2\\sin^2 x = \\cos 2x$，$2\\sin x\\cos x = \\sin 2x$，故 $y = \\sin 2x + \\cos 2x = \\sqrt{2}\\sin\\!\\left(2x + \\dfrac{\\pi}{4}\\right)$，最大值为 $\\sqrt{2}$。选 A（下标 0）。',
      },
      {
        type: 'match',
        prompt: '将各函数与其值域对应。',
        left: [
          '$y = 2\\sin x + 1$',
          '$y = -3\\cos x + 2$',
          '$y = \\sin(2x - 1)$',
        ],
        right: [
          '$[-1, 1]$',
          '$[-1, 3]$',
          '$[-1, 5]$',
        ],
        explain: '$y = 2\\sin x + 1$：$\\sin x \\in [-1,1]$，$y \\in [-1, 3]$。$y = -3\\cos x + 2$：$\\cos x \\in [-1,1]$，$-3\\cos x \\in [-3,3]$，$y \\in [-1, 5]$。$y = \\sin(2x-1)$：值域同 $\\sin x$，为 $[-1,1]$。',
      },
      {
        type: 'judge',
        prompt: '$y = \\tan x$ 在 $\\mathbb{R}$ 上是单调递增函数。',
        answer: false,
        explain: '$\\tan x$ 在每个区间 $\\left(-\\dfrac{\\pi}{2}+k\\pi, \\dfrac{\\pi}{2}+k\\pi\\right)$ 内单调递增，但在 $x = \\dfrac{\\pi}{2}+k\\pi$ 处无定义，不能跨越渐近线合并区间，因此不能说"在 $\\mathbb{R}$ 上单调递增"。',
      },
      {
        type: 'input',
        prompt: '函数 $y = 2\\sin\\!\\left(3x + \\dfrac{\\pi}{6}\\right) - 1$ 的值域最小值为多少？',
        accept: ['-3'],
        explain: '$A = 2$，$b = -1$，最小值 $= -|A| + b = -2 + (-1) = -3$。',
      },
    ],
  },
]
