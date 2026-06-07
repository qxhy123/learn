export const LESSONS = [
  {
    id: 'u10c2',
    title: '求导法则与基本公式',
    subtitle: '九大公式 + 四则法则 + 链式法则',
    intro: [
      {
        title: '为什么不每次都用定义求导？',
        body: '用极限定义求导虽然严谨，但繁琐。数学家总结出一套高效法则：记住若干基本函数的导数公式，再用四则运算法则和链式法则拆解复杂函数。掌握这套法则，绝大多数高中函数可以直接求导。',
      },
      {
        title: '九大基本导数公式',
        body: '常数：(C)′ = 0。幂函数：(xⁿ)′ = nxⁿ⁻¹。三角：(sin x)′ = cos x，(cos x)′ = −sin x。自然指数：(eˣ)′ = eˣ（自身的导数还是自身，e 由此而来）。一般指数：(aˣ)′ = aˣ ln a。自然对数：(ln x)′ = 1/x。一般对数：(logₐx)′ = 1/(x ln a)。',
        formula: '(x^n)^{\\prime} = nx^{n-1},\\quad (e^x)^{\\prime} = e^x,\\quad (\\ln x)^{\\prime} = \\frac{1}{x}',
        tip: '(cos x)′ = −sin x，负号是关键，容易漏。eˣ 的任意阶导数都是 eˣ，这是 e 被特别选出的理由。',
      },
      {
        title: '和差法则与乘法法则',
        body: '和差法则：(u ± v)′ = u′ ± v′，逐项求导。\n乘法法则（莱布尼茨法则）：(uv)′ = u′v + uv′，口诀"前导后不导，加上前不导后导"。\n常数因子可提：(Cu)′ = C · u′。',
        formula: '(uv)^{\\prime} = u^{\\prime}v + uv^{\\prime}',
        steps: [
          '识别哪部分是 u，哪部分是 v',
          '分别求 u′ 和 v′',
          '代入公式 u′v + uv′',
        ],
      },
      {
        title: '商法则',
        body: '(u/v)′ = (u′v − uv′)/v²，分子是 u′v 在前减 uv′ 在后，中间是减号，整体除以 v²。特别容易错的是分子顺序和符号：若写成 uv′ − u′v 则差一个负号，若写成加号则完全错误。',
        formula: '\\left(\\frac{u}{v}\\right)^{\\prime} = \\frac{u^{\\prime}v - uv^{\\prime}}{v^2}',
        tip: '记忆口诀：子导母不导，减去子不导母导，整体除以母平方。注意分子是"减号"。',
      },
      {
        title: '链式法则：外导乘内导',
        body: '复合函数 y = f(g(x)) 的导数：先对外层函数求导（把内层当整体），再乘以内层函数的导数。例如 (sin 2x)′ = cos(2x) · 2 = 2cos 2x，内层 2x 的导数 2 不能漏掉。多层复合像"剥洋葱"，从外到内逐层求导，全部相乘。',
        formula: '[f(g(x))]^{\\prime} = f^{\\prime}(g(x)) \\cdot g^{\\prime}(x)',
        steps: [
          '辨认外层函数和内层函数',
          '对外层求导（内层当整体 u，结果含 u）',
          '乘以内层对 x 的导数',
          '还原 u 为内层表达式',
        ],
      },
      {
        title: '例题：乘积 + 链式',
        body: '求 f(x) = x² ln(3x+1) 的导数。\n识别乘积结构：u = x²，v = ln(3x+1)。\nu′ = 2x；v 是复合函数，外层 ln，内层 3x+1：v′ = 1/(3x+1) · 3 = 3/(3x+1)。\n代入乘法法则：f′(x) = 2x ln(3x+1) + x² · 3/(3x+1)。',
        formula: 'f^{\\prime}(x) = 2x\\ln(3x+1) + \\frac{3x^2}{3x+1}',
        tip: '易错：链式法则时漏掉内层导数。每写一个复合导数，问自己"内层导数乘了吗？"',
      },
      {
        title: '高阶导数',
        body: '对 f′(x) 再求一次导，得到二阶导数 f″(x)。物理含义：位移 → 一阶导 = 速度 → 二阶导 = 加速度。重要公式：(eˣ)″ = eˣ（任意阶导数都是 eˣ）；(sin x)″ = −sin x；(ln x)″ = −1/x²。',
        formula: 'f^{\\prime\\prime}(x) = [f^{\\prime}(x)]^{\\prime}',
        reveal: {
          q: '求 f(x) = e^{sin(2x)} 的导数（写出链式法则的步骤）。',
          a: '三层复合：外层 eᵘ 导数 eᵘ，中层 sin v 导数 cos v，内层 2x 导数 2。结果：f′(x) = e^{sin(2x)} · cos(2x) · 2 = 2cos(2x) · e^{sin(2x)}。',
        },
      },
    ],
    questions: [
      {
        type: 'input' as const,
        stem: '求 f(x) = 3x⁴ − 2x³ + x − 7 的导数 f′(x)。（格式：12x³−6x²+1）',
        answer: '12x³−6x²+1',
        explanation: '逐项：(3x⁴)′=12x³，(2x³)′=6x²，(x)′=1，常数 7 的导数 0。合并得 12x³−6x²+1。',
      },
      {
        type: 'choice' as const,
        stem: '(cos x)′ 等于',
        options: ['A. sin x', 'B. −sin x', 'C. cos x', 'D. −cos x'],
        answer: 'B',
        explanation: '基本公式：(cos x)′ = −sin x，注意负号，选 B。',
      },
      {
        type: 'input' as const,
        stem: '用商法则求 f(x) = (x²+1)/(x−1) 的导数分子（化简后）。（格式：x²−2x−1）',
        answer: 'x²−2x−1',
        explanation: 'u=x²+1，u′=2x；v=x−1，v′=1。分子：u′v − uv′ = 2x(x−1)−(x²+1)·1 = 2x²−2x−x²−1 = x²−2x−1。',
      },
      {
        type: 'choice' as const,
        stem: '(√x)′ 等于',
        options: [
          'A. 1/(2√x)',
          'B. 2√x',
          'C. −1/(2√x)',
          'D. 1/√x',
        ],
        answer: 'A',
        explanation: '√x = x^(1/2)，用幂函数公式：(1/2)x^(−1/2) = 1/(2√x)，选 A。',
      },
      {
        type: 'input' as const,
        stem: '求 f(x) = sin(x²+1) 的导数 f′(x)（格式：2x·cos(x²+1)）',
        answer: '2x·cos(x²+1)',
        explanation: '链式法则：外层 sin u，导数 cos u；内层 x²+1，导数 2x。f′(x) = cos(x²+1)·2x = 2x cos(x²+1)。',
      },
      {
        type: 'judge' as const,
        stem: '(uv)′ = u′v′（两个因子各自求导再相乘）。',
        answer: '错误',
        explanation: '乘法法则是 (uv)′ = u′v + uv′，不是 u′v′。这是高频易错点。',
      },
      {
        type: 'choice' as const,
        stem: '求 f(x) = x² sin x 的导数，正确结果是',
        options: [
          'A. 2x cos x',
          'B. 2x sin x + x² cos x',
          'C. x² cos x',
          'D. 2x sin x − x² cos x',
        ],
        answer: 'B',
        explanation: '乘法法则：u=x²，v=sin x；u′=2x，v′=cos x。f′=2x sin x + x² cos x，选 B。',
      },
      {
        type: 'match' as const,
        stem: '将函数与其导数正确连线',
        left: ['eˣ', 'ln x', 'aˣ (a>0,a≠1)', 'logₐx (a>0,a≠1)'],
        right: ['aˣ ln a', '1/(x ln a)', '1/x', 'eˣ'],
        answer: ['eˣ', '1/x', 'aˣ ln a', '1/(x ln a)'],
      },
    ],
  },
]
