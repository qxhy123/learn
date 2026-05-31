import type { IntroCard } from '../../types'

// 第四部分（第 11–14 章）每节的“学新知”讲解卡片。
// 12 个 lesson id：u11-l1..l3、u12-l1..l3、u13-l1..l3、u14-l1..l3。
export const INTRO4: Record<string, IntroCard[]> = {
  // ===================== u11 自然对数与 e =====================
  'u11-l1': [
    {
      title: '核心概念',
      body: '**自然对数**是以特殊常数 $e$ 为底的对数。它不是“众多底数中随意的一个”，而是连续变化最自然的坐标。',
      formula: '\\ln x=\\log_e x',
    },
    {
      title: '常数 e',
      body: '$e$ 约等于 $2.71828$，是**无理数**，也是**超越数**。它会在连续增长、极限和微积分中自然出现，不是凑出来的数字。',
      formula: 'e\\approx2.71828',
    },
    {
      title: '两个必记值',
      body: '对数在 $1$ 和底数处的值是地基：\n\n- $e^1=e$，所以 $\\ln e=1$\n- $e^0=1$，所以 $\\ln 1=0$',
      steps: ['$\\ln e=1$（底数处取 $1$）', '$\\ln 1=0$（在 $1$ 处取 $0$）'],
    },
    {
      title: '配套图像',
      body: '数列 $(1+1/n)^n$ 随 $n$ 增大单调上升，收敛到水平渐近线 $y=e\\approx2.718$——这正是 $e$ 自然出现的画面。',
      image: 'log-p4-11-1.svg',
      imageCaption: '$(1+1/n)^n$ 单调上升并收敛到 $e\\approx2.718$',
    },
    {
      title: '试一试',
      body: '巩固一下幂法则与对数的配合。',
      reveal: { q: '$\\ln e^3$ 等于多少？', a: '$\\ln e^3=3\\ln e=3$。' },
    },
  ],
  'u11-l2': [
    {
      title: '经典来源',
      body: '考虑复利模型 $\\left(1+\\frac1n\\right)^n$：当复利次数 $n$ 越来越大，结果趋向一个固定常数，这个极限就是 $e$。',
      formula: 'e=\\lim_{n\\to\\infty}\\left(1+\\frac1n\\right)^n',
    },
    {
      title: '连续复利的直觉',
      body: '以年利率 100% 复利：\n\n- 一年复利 1 次，得到 $2$\n- 一年复利很多次，结果逼近 $e$\n\n所以 $e$ 是“单位增长率连续累积”的自然基准。',
    },
    {
      title: 'e 的等价定义',
      body: '同一个 $e$ 有多种刻画，它们都指向核心性质——$e^x$ 是满足 $f^{\\prime}(x)=f(x)$ 且 $f(0)=1$ 的唯一函数。',
      steps: [
        '复利极限：$e=\\lim_{n\\to\\infty}(1+1/n)^n$',
        '级数求和：$e=\\sum_{n=0}^{\\infty}\\frac1{n!}=1+1+\\frac12+\\frac16+\\cdots$',
        '面积阈值：$\\ln x=\\int_1^x\\frac{dt}{t}$，使 $\\ln e=1$ 的数',
      ],
    },
    {
      title: '一个常见套路',
      body: '把指数里的常数因子提到括号外，就能套用已知极限。例如 $\\left(1+\\frac1n\\right)^{3n}=\\left[\\left(1+\\frac1n\\right)^n\\right]^3$。',
      tip: '内层 $\\to e$，三次幂连续，故整体 $\\to e^3$。常数因子可以“穿过”幂运算。',
    },
    {
      title: '试一试',
      body: '把“小量 $3x$ 与它的倒数”配对是这类极限的通用招。',
      reveal: { q: '$\\lim_{x\\to0}(1+3x)^{1/x}$ 等于？', a: '凑成 $[(1+3x)^{1/(3x)}]^3\\to e^3$。' },
    },
  ],
  'u11-l3': [
    {
      title: '桥梁公式',
      body: '任何指数函数都能写成以 $e$ 为底的形式。于是研究一般底数时，只要研究 $e^x$ 和 $\\ln x$ 即可——自然对数像一个统一接口。',
      formula: 'a^x=e^{x\\ln a}',
    },
    {
      title: '怎么用',
      body: '取 $a=5$ 就得到 $5^x=e^{x\\ln 5}$。一旦写成 $e^{(\\cdot)}$，“指数相等”就直接给出方程，无需再换底。',
      steps: ['$5^x=e^{x\\ln 5}$', '$7^{2x}=e^{2x\\ln 7}$'],
    },
    {
      title: '对数与时间尺度',
      body: '指数增长模型 $y=y_0e^{kt}$ 达到目标 $Y$ 的时间，本质在回答“要经过多少连续增长时间”。',
      formula: 't=\\frac1k\\ln\\frac{Y}{y_0}',
    },
    {
      title: '倍增与半衰',
      body: '两个对偶结论都由 $\\ln 2$ 给出，且与初始量无关（指数模型的“无标度”性质）。',
      steps: ['倍增时间 $T_d=\\frac{\\ln 2}{k}$', '半衰期 $T_{1/2}=\\frac{\\ln 2}{\\lambda}$'],
      tip: '$N_0$ 在约分时被消去，所以达到固定比率所需时间只取决于速率常数。',
    },
    {
      title: '试一试',
      body: '建模题的标准两步：先用条件定 $k$，再代入求时间。',
      reveal: { q: '$y=4e^{0.2t}$ 达到 $20$ 的时间 $t$？', a: '$e^{0.2t}=5\\Rightarrow t=\\frac{\\ln5}{0.2}=5\\ln5$。' },
    },
  ],

  // ===================== u12 极限·增长比较 =====================
  'u12-l1': [
    {
      title: '接近 0 时',
      body: '当 $x\\to0^+$ 时 $\\ln x\\to-\\infty$，对应 $x=0$ 处的竖直渐近线。注意只能写 $x\\to0^+$，因为 $\\ln x$ 只对正数有定义。',
      formula: '\\lim_{x\\to0^+}\\ln x=-\\infty',
    },
    {
      title: '趋向无穷时',
      body: '当 $x\\to+\\infty$ 时 $\\ln x\\to+\\infty$，但增长非常慢。把 $x$ 从 $10$ 增到 $1000$，$\\ln x$ 只从约 $2.3$ 增到约 $6.9$。',
      formula: '\\lim_{x\\to+\\infty}\\ln x=+\\infty',
    },
    {
      title: '双端对照',
      body: '两端放在一起看：\n\n- 靠近 0 时变化非常剧烈\n- 远离 0 后虽继续增长，却越来越慢',
      tip: '“趋于无穷”只表示无上界，不等于“增长很快”——这正是本章要区分的两件事。',
    },
    {
      title: '连续性',
      body: '$\\ln x$ 在定义域 $(0,+\\infty)$ 上**连续**：输入做小变化，输出不会突然跳跃。这使它适合做极限分析、近似与建模。',
      formula: '\\ln x\\ \\text{在}\\ (0,+\\infty)\\ \\text{上连续}',
    },
    {
      title: '试一试',
      body: '小心定义域陷阱。',
      reveal: { q: '$\\ln x$ 在 $x=0$ 处连续吗？', a: '不连续——$x=0$ 不在定义域内，“在 $0$ 处连续”根本不成立。' },
    },
  ],
  'u12-l2': [
    {
      title: '两块基石',
      body: '这两个基本极限是后续导数公式与级数展开的根基，务必记牢。',
      steps: [
        '$\\lim_{x\\to0}\\frac{\\ln(1+x)}{x}=1$',
        '$\\lim_{x\\to+\\infty}\\frac{\\ln x}{x}=0$',
      ],
    },
    {
      title: '极限一的含义',
      body: '当相对变化 $x$ 很小时，$\\ln(1+x)\\approx x$——对数变化近似等于相对变化。',
      formula: '\\lim_{x\\to0}\\frac{\\ln(1+x)}{x}=1',
    },
    {
      title: '极限二的含义',
      body: '对数增长比任何正幂函数都慢。令 $x=e^t$，$\\frac{\\ln x}{x}=\\frac{t}{e^t}\\le\\frac{2}{t}\\to0$。',
      formula: '\\lim_{x\\to+\\infty}\\frac{\\ln x}{x}=0',
    },
    {
      title: '配凑技巧',
      body: '套用 $\\frac{\\ln(1+\\square)}{\\square}\\to1$ 的诀窍是让分母与括号内小量完全一致。例如 $\\frac{\\ln(1+5x)}{x}=5\\cdot\\frac{\\ln(1+5x)}{5x}\\to5$。',
      tip: '“乘 $5$ 除 $5$”把分母补成 $5x$，多出的因子 $5$ 就是答案。',
    },
    {
      title: '试一试',
      body: '换元能把陌生极限变熟悉。',
      reveal: { q: '$\\lim_{x\\to0^+}x\\ln x$ 等于？', a: '令 $x=1/t$，化为 $-\\frac{\\ln t}{t}\\to0$。' },
    },
  ],
  'u12-l3': [
    {
      title: '增长层级',
      body: '把对数、幂、指数排成从慢到快的层级，是分析、算法复杂度和概率尾界中的核心工具。',
      formula: '\\ln x\\ll x^\\alpha\\ll a^x\\quad(\\alpha>0,\\,a>1)',
    },
    {
      title: '对数 ≪ 幂',
      body: '对任意 $\\alpha>0$，幂函数终将彻底压过对数——哪怕 $\\alpha$ 只有 $\\frac12$。',
      formula: '\\lim_{x\\to+\\infty}\\frac{\\ln x}{x^\\alpha}=0',
    },
    {
      title: '幂 ≪ 指数',
      body: '对任意 $a>1$，指数函数终将压过任意幂函数。如 $\\frac{x^2}{e^x}\\to0$（连用两次洛必达，分子降阶、分母不变）。',
      formula: '\\lim_{x\\to+\\infty}\\frac{x^\\alpha}{a^x}=0',
    },
    {
      title: '配套图像',
      body: '同一坐标系里 $\\ln x$、$\\sqrt x$、$2^x$ 三条曲线，直观显示对数最慢、指数最快。',
      image: 'log-p4-12-1.svg',
      imageCaption: '$\\ln x\\ll\\sqrt x\\ll 2^x$ 的增长对比',
      tip: '用洛必达前务必先确认是 $\\frac00$ 或 $\\frac{\\infty}{\\infty}$ 未定式。',
    },
    {
      title: '试一试',
      body: '别被“看起来都慢”迷惑。',
      reveal: { q: '$\\lim_{x\\to+\\infty}\\frac{\\ln x}{\\sqrt x}$ 等于？', a: '$\\sqrt x=x^{1/2}$，$\\alpha=\\frac12>0$，故极限为 $0$。' },
    },
  ],

  // ===================== u13 求导 =====================
  'u13-l1': [
    {
      title: '最核心的公式',
      body: '对数求导最重要的结论，把“相对变化率”直接写进了导数。',
      formula: '(\\ln x)^{\\prime}=\\frac1x\\quad(x>0)',
    },
    {
      title: '一般底数',
      body: '由换底 $\\log_a x=\\frac{\\ln x}{\\ln a}$ 求导，会多出一个常数因子 $\\frac1{\\ln a}$。',
      formula: '(\\log_a x)^{\\prime}=\\frac{1}{x\\ln a}',
      tip: '切忌把 $(\\log_a x)^{\\prime}$ 误写成 $\\frac1x$——别漏掉 $\\frac1{\\ln a}$。',
    },
    {
      title: '为什么是 1/x',
      body: '导数 $\\frac1x$ 表明：$x$ 越大，同样的增量对 $\\ln x$ 的影响越小。这是“尺度压缩”的微积分表达——描述的是相对敏感度。',
    },
    {
      title: '配套图像',
      body: '在 $y=\\ln x$ 上点 $(2,\\ln 2)$ 处作切线，直角三角形 $\\Delta x=2$、$\\Delta y=1$，斜率 $=\\frac1{x_0}=\\frac12$。',
      image: 'log-p4-13-1.svg',
      imageCaption: '$(2,\\ln 2)$ 处切线斜率 $=1/2$',
    },
    {
      title: '试一试',
      body: '注意一般底数的因子。',
      reveal: { q: '$(\\log_5 x)^{\\prime}$ 等于？', a: '$\\frac{1}{x\\ln 5}$。' },
    },
  ],
  'u13-l2': [
    {
      title: '链式法则',
      body: '复合对数求导的口诀是“分母抄原式、分子放内导”。',
      formula: '(\\ln u)^{\\prime}=\\frac{u^{\\prime}}{u}\\quad(u>0)',
    },
    {
      title: '示范',
      body: '分母保留原函数，分子来自内部函数的导数。',
      steps: [
        '$(\\ln(x^2+1))^{\\prime}=\\frac{2x}{x^2+1}$',
        '$(\\ln(3x+1))^{\\prime}=\\frac{3}{3x+1}$',
      ],
    },
    {
      title: '先求定义域',
      body: '导数表达式只在真数为正的区间才有意义。例如 $y=\\ln(x^2-4)$ 需 $x^2-4>0$，即 $|x|>2$。',
      tip: '先求定义域再求导是好习惯，否则结果可能在无意义的区间“看起来成立”。',
    },
    {
      title: '化商为差',
      body: '碰到“对数套着分式”，先用 $\\ln\\frac AB=\\ln A-\\ln B$ 拆开往往更省事——这也是下一节对数求导法的雏形。',
      formula: '\\left(\\ln\\tfrac{x+1}{x-1}\\right)^{\\prime}=\\frac{-2}{x^2-1}',
    },
    {
      title: '试一试',
      body: '内导别忘了。',
      reveal: { q: '$(\\ln(3x+1))^{\\prime}$ 等于？', a: '内层导数为 $3$，故 $\\frac{3}{3x+1}$。' },
    },
  ],
  'u13-l3': [
    {
      title: '对数求导法',
      body: '当函数是乘积、商、幂的复杂组合时，先取对数会很方便：乘积变加法、幂落到前面、结构更易求导。流程是**先取对数 → 再求导 → 再还原**。',
    },
    {
      title: '经典例子 x^x',
      body: '幂指函数 $y=x^x$（底和指数都含变量）无法直接套幂法则或指数法则，取对数后迎刃而解。',
      steps: [
        '取对数：$\\ln y=x\\ln x$',
        '求导：$\\frac{y^{\\prime}}{y}=\\ln x+1$',
        '还原：$y^{\\prime}=x^x(\\ln x+1)$',
      ],
    },
    {
      title: '什么时候特别有用',
      body: '如果函数长这样——多个因子相乘、有分式、指数里也含变量——先取对数通常比硬套乘积法则和商法则清楚得多。',
      formula: 'y=\\frac{(x^2+1)^3\\sqrt{x+2}}{x^5}',
      tip: '取对数后变成 $3\\ln(x^2+1)+\\frac12\\ln(x+2)-5\\ln x$，每项独立求导即可。',
    },
    {
      title: '一个经典结论',
      body: '由 $y^{\\prime}=x^x(\\ln x+1)$，因 $x^x>0$ 恒成立，导数符号由 $\\ln x+1$ 决定。令其为 $0$ 得 $x=\\frac1e$，这是 $x^x$ 的最低点。',
      formula: 'x^x\\ \\text{在}\\ x=\\tfrac1e\\ \\text{取极小值}',
    },
    {
      title: '试一试',
      body: '别忘了最后乘回 $y$。',
      reveal: { q: '$y=x^{\\sin x}$ 的导数？', a: '$\\ln y=\\sin x\\ln x$，得 $y^{\\prime}=x^{\\sin x}(\\cos x\\ln x+\\frac{\\sin x}{x})$。' },
    },
  ],

  // ===================== u14 积分·级数·近似 =====================
  'u14-l1': [
    {
      title: '对数与积分',
      body: '因为 $(\\ln x)^{\\prime}=\\frac1x$，反过来积分就回到对数。绝对值是因为积分区间可能在正半轴或负半轴。',
      formula: '\\int\\frac1x\\,dx=\\ln|x|+C',
      tip: '别忘绝对值！$\\ln x$ 只对 $x>0$ 有定义，$\\ln|x|$ 才能同时覆盖 $x<0$。',
    },
    {
      title: '面积定义',
      body: '自然对数本身可以用面积定义：$\\ln x=\\int_1^x\\frac1t\\,dt$ 就是 $y=1/t$ 曲线下方、从 $t=1$ 到 $t=x$ 之间的面积。',
      formula: '\\ln x=\\int_1^x\\frac{dt}{t}',
    },
    {
      title: 'u′/u 结构',
      body: '识别“分子是分母的导数”是这类积分的关键：$\\int\\frac{f^{\\prime}(x)}{f(x)}\\,dx=\\ln|f(x)|+C$。',
      steps: [
        '$\\int\\frac{2x}{x^2+1}\\,dx=\\ln(x^2+1)+C$（真数恒正可去绝对值）',
        '$\\int\\frac{3}{3x+1}\\,dx=\\ln|3x+1|+C$',
      ],
    },
    {
      title: '配套图像',
      body: '$y=1/t$ 曲线下方从 $1$ 到 $x$（图中取 $x=4$）之间的阴影面积，恰好等于 $\\ln x$。',
      image: 'log-p4-14-1.svg',
      imageCaption: '阴影面积 $=\\ln x$',
    },
    {
      title: '试一试',
      body: '分子差常数倍时先凑因子。',
      reveal: { q: '$\\int\\frac{1}{2x-5}\\,dx$ 等于？', a: '$\\frac12\\int\\frac{2}{2x-5}\\,dx=\\frac12\\ln|2x-5|+C$。' },
    },
  ],
  'u14-l2': [
    {
      title: 'ln x 的积分',
      body: '$\\ln x$ 本身用分部积分（取 $u=\\ln x$、$dv=dx$）即可求出。',
      formula: '\\int\\ln x\\,dx=x\\ln x-x+C',
    },
    {
      title: '一个漂亮的定积分',
      body: '代入上下限并用 $\\ln e=1$、$\\ln 1=0$：',
      steps: [
        '$\\int_1^e\\ln x\\,dx=[x\\ln x-x]_1^e$',
        '$=(e\\cdot1-e)-(0-1)=1$',
      ],
    },
    {
      title: 'ln(1+x) 的级数',
      body: '从几何级数 $\\frac{1}{1+t}=1-t+t^2-\\cdots$ 逐项积分得到。注意收敛域 $-1<x\\le1$：右端 $x=1$ 收敛，左端 $x=-1$ 发散。',
      formula: '\\ln(1+x)=x-\\frac{x^2}{2}+\\frac{x^3}{3}-\\cdots',
    },
    {
      title: '交错和 = ln 2',
      body: '在级数中取 $x=1$（端点收敛），就得到著名的交错调和级数之和。',
      formula: '\\ln 2=1-\\frac12+\\frac13-\\frac14+\\cdots',
      tip: '同一级数不能“去掉负号”去求调和级数——那对应 $x=-1$，落在收敛域之外，会发散。',
    },
    {
      title: '试一试',
      body: '反常积分也能收敛。',
      reveal: { q: '$\\int_0^1\\ln x\\,dx$ 收敛吗？', a: '收敛。配合 $\\lim_{a\\to0^+}a\\ln a=0$，积分值为 $-1$。' },
    },
  ],
  'u14-l3': [
    {
      title: '小量近似',
      body: '当 $x$ 很小时，对数变化与线性变化几乎一致。很多“增长率近似”“百分比近似”都建立在这个公式上。',
      formula: '\\ln(1+x)\\approx x',
    },
    {
      title: '近似的边界',
      body: '这只是近似，不是恒等式。使用前应先判断 $x$ 是否确实足够小。',
      steps: [
        '$x=0.01$：近似通常很好',
        '$x=0.5$：误差已不能忽视（真值约 $0.405$，差约 23%）',
      ],
      tip: '一阶近似误差约 $\\frac{x^2}{2}$，加上二次项 $x-\\frac{x^2}{2}$ 后误差降到约 $\\frac{x^3}{3}$。',
    },
    {
      title: '更精细一点',
      body: '需要更高精度时，多取一项。这也是数值计算里避免精度损失的思路来源。',
      formula: '\\ln(1+x)\\approx x-\\frac{x^2}{2}',
    },
    {
      title: '调和级数与对数',
      body: '调和级数 $H_n=1+\\frac12+\\cdots+\\frac1n$ 发散，但增长极慢，渐近于 $\\ln n$。',
      formula: 'H_n=\\ln n+\\gamma+O(1/n)',
      tip: '$\\gamma\\approx0.5772$ 是欧拉-马歇罗尼常数。快排平均比较次数 $\\approx2n\\ln n$、赠券收集 $\\approx n\\ln n$ 都源于此。',
    },
    {
      title: '试一试',
      body: '别滥用近似。',
      reveal: { q: '$\\ln(1.5)\\approx0.5$ 可靠吗？', a: '不可靠。真值约 $0.405$，$x=0.5$ 不够小，误差约 23%。' },
    },
  ],
}
