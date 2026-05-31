import type { Unit } from '../../types'

// Part 4（第10–14章）：微积分与分析 —— 数 e、极限渐近、求导、积分与微分方程、级数近似。
export const PART4_UNITS: Unit[] = [
  // ===================== u10 数 e 与连续复利 =====================
  {
    id: 'u10',
    title: '数 e 与连续复利',
    color: '#ffc800',
    icon: '🌀',
    blurb: '$e=\\lim(1+1/n)^n$、连续复利 $A=Pe^{rt}$、翻倍时间 $\\ln2/r$',
    lessons: [
      {
        id: 'u10-l1',
        title: 'e 从哪里来',
        subtitle: '离散复利无限加密的极限',
        questions: [
          {
            id: 'u10-l1-q1',
            type: 'choice',
            prompt: '自然常数 $e$ 的极限定义是？',
            options: [
              '$e=\\lim_{n\\to\\infty}\\left(1+\\frac1n\\right)^n$',
              '$e=\\lim_{n\\to\\infty}\\left(1+n\\right)^{1/n}$',
              '$e=\\lim_{n\\to\\infty}\\frac{n}{n+1}$',
              '$e=\\lim_{n\\to\\infty}\\frac1n$',
            ],
            answer: 0,
            explain:
              '$e=\\lim_{n\\to\\infty}\\left(1+\\frac1n\\right)^n$：把增长拆得越来越细后自然冒出的常数。',
          },
          {
            id: 'u10-l1-q2',
            type: 'input',
            prompt: '$e$ 保留三位小数约等于多少？',
            accept: ['2.718', '2,718'],
            placeholder: '2.___',
            explain: '$e\\approx2.71828\\ldots$，常记为 $2.718$。',
          },
          {
            id: 'u10-l1-q3',
            type: 'judge',
            prompt: '序列 $a_n=\\left(1+\\frac1n\\right)^n$ 会随 $n$ 增大无限增大到 $+\\infty$。',
            answer: false,
            explain:
              '不会。二项式展开各项被 $\\frac1{k!}$ 控制，整列 $<\\sum\\frac1{k!}<3$，单调递增且有上界，收敛到 $e$。',
          },
          {
            id: 'u10-l1-q4',
            type: 'choice',
            prompt:
              '为什么 $\\left(1+\\frac1n\\right)^n$ 不会暴涨？关键在于哪两种趋势同时发生？',
            options: [
              '单步增幅变小（只剩 $\\frac1n$），但步数变多（共 $n$ 步）',
              '单步增幅变大，步数也变多',
              '单步增幅不变，步数变多',
              '底数变大，指数变小',
            ],
            answer: 0,
            explain:
              '极限研究的正是“单步变小 + 步数变多”叠加后的总效果，最终稳定到固定值 $e$。',
          },
          {
            id: 'u10-l1-q5',
            type: 'judge',
            prompt: '对任意有限的 $n$，都有 $\\left(1+\\frac1n\\right)^n<e$。',
            answer: true,
            explain:
              '$a_n$ 单调递增地趋于 $e$，故任意有限项都严格偏小，例如 $a_{1000}\\approx2.71692<e$。',
          },
        ],
      },
      {
        id: 'u10-l2',
        title: '从离散到连续复利',
        subtitle: '$A=Pe^{rt}$ 的来源',
        questions: [
          {
            id: 'u10-l2-q1',
            type: 'choice',
            prompt: '本金 $P$、年利率 $r$、每年复利 $m$ 次、$t$ 年后的本利和是？',
            options: [
              '$P\\left(1+\\frac{r}{m}\\right)^{mt}$',
              '$P\\left(1+r\\right)^{mt}$',
              '$P\\left(1+\\frac{r}{t}\\right)^{m}$',
              '$P\\,e^{rm}$',
            ],
            answer: 0,
            explain:
              '每次只加 $\\frac{r}{m}$，一年计息 $m$ 次，共 $t$ 年即 $mt$ 次。',
          },
          {
            id: 'u10-l2-q2',
            type: 'input',
            prompt:
              '当 $m\\to\\infty$ 时，$P\\left(1+\\frac{r}{m}\\right)^{mt}$ 趋于连续复利公式 $A=P\\,e^{\\,\\square}$。$\\square$ 处应填？',
            accept: ['rt', 'r t', 'r*t', 'tr'],
            placeholder: '__',
            explain:
              '$\\left(1+\\frac{r}{m}\\right)^{mt}=\\left[\\left(1+\\frac{r}{m}\\right)^m\\right]^t\\to(e^r)^t=e^{rt}$，故 $A=Pe^{rt}$。',
          },
          {
            id: 'u10-l2-q3',
            type: 'judge',
            prompt:
              '一般极限 $\\left(1+\\frac{x}{n}\\right)^n\\to e^x$ 在 $x=1$ 时就退化成 $e$ 的定义。',
            answer: true,
            explain: '取 $x=1$ 即得 $\\left(1+\\frac1n\\right)^n\\to e^1=e$，正是特例。',
          },
          {
            id: 'u10-l2-q4',
            type: 'choice',
            prompt:
              '指数上的量 $rt$ 必须是“纯数”（无量纲）。若 $r=0.1\\,\\text{年}^{-1}$、$t=2\\,\\text{年}$，则 $rt=$？',
            options: ['$0.2$（无量纲）', '$0.2\\,\\text{年}$', '$0.2\\,\\text{年}^{-1}$', '$5$'],
            answer: 0,
            explain: '$\\text{年}^{-1}\\times\\text{年}$ 单位相消，得纯数 $0.2$，$e^{0.2}$ 才有意义。',
          },
          {
            id: 'u10-l2-q5',
            type: 'match',
            prompt: '把连续复利公式 $A=Pe^{rt}$ 中的符号与含义配对：',
            left: ['$P$', '$r$', '$t$', '$A$'],
            right: ['本金', '名义年利率', '持续时间（年）', '本利和'],
            explain: '$P$ 起点、$r$ 利率、$t$ 时间、$A$ 末值，单位需配套一致。',
          },
          {
            id: 'u10-l2-q6',
            type: 'judge',
            prompt:
              '“连续复利”意味着银行真的在无穷多个瞬间逐次结算利息。',
            answer: false,
            explain:
              '连续是一种极限/建模语言：把计息间隔压到极短的理想化，而非真的无穷次操作。',
          },
        ],
      },
      {
        id: 'u10-l3',
        title: '翻倍时间与频率比较',
        subtitle: '$t=\\ln2/r$ 与 70 法则',
        questions: [
          {
            id: 'u10-l3-q1',
            type: 'choice',
            prompt: '连续复利下本金翻倍所需时间 $t$ 满足 $e^{rt}=2$，解得 $t=$？',
            options: ['$\\frac{\\ln2}{r}$', '$\\frac{r}{\\ln2}$', '$\\frac{2}{r}$', '$r\\ln2$'],
            answer: 0,
            explain: '$e^{rt}=2\\Rightarrow rt=\\ln2\\Rightarrow t=\\frac{\\ln2}{r}$。',
          },
          {
            id: 'u10-l3-q2',
            type: 'input',
            prompt:
              '年化 $r=5\\%$（连续复利），翻倍时间 $t=\\frac{\\ln2}{0.05}\\approx?$ 年（取 $\\ln2\\approx0.693$，保留整数）。',
            accept: ['14', '13.86', '13.9', '约14'],
            placeholder: '__ 年',
            explain: '$t=\\frac{0.693}{0.05}\\approx13.86\\approx14$ 年，这正是金融界的“70 法则”。',
          },
          {
            id: 'u10-l3-q3',
            type: 'judge',
            prompt:
              '同一名义年利率下，按月复利的本利和会大于按年复利，但始终不超过连续复利。',
            answer: true,
            explain:
              '计息越频繁越大，但被上界 $Pe^{rt}$ 卡住；这是 $\\left(1+\\frac{r}{m}\\right)^m\\uparrow e^r$ 的体现。',
          },
          {
            id: 'u10-l3-q4',
            type: 'choice',
            prompt:
              '$\\displaystyle\\lim_{n\\to\\infty}\\left(1+\\frac{3}{n}\\right)^{2n}=?$',
            options: ['$e^{6}$', '$e^{3}$', '$e^{2}$', '$e^{5}$'],
            answer: 0,
            explain:
              '$\\left(1+\\frac3n\\right)^{2n}=\\left[\\left(1+\\frac3n\\right)^n\\right]^2\\to(e^3)^2=e^6$。',
          },
          {
            id: 'u10-l3-q5',
            type: 'choice',
            prompt:
              '题目说“每月固定结算一次利息”，应优先用哪个模型？',
            options: [
              '离散公式 $P\\left(1+\\frac{r}{m}\\right)^{mt}$',
              '连续公式 $Pe^{rt}$',
              '线性公式 $P(1+rt)$',
              '都不行',
            ],
            answer: 0,
            explain: '关键看机制：更新是离散发生的就用离散公式；明确“连续”才用 $Pe^{rt}$。',
          },
          {
            id: 'u10-l3-q6',
            type: 'judge',
            prompt: '“连续复利更高级”，所以任何复利题都该直接写 $A=Pe^{rt}$。',
            answer: false,
            explain: '不对。该用哪个公式取决于题目描述的是离散更新还是连续极限，而非哪个更整齐。',
          },
        ],
      },
    ],
  },

  // ===================== u11 极限·连续·渐近 =====================
  {
    id: 'u11',
    title: '极限·连续·渐近',
    color: '#ff86d0',
    icon: '📉',
    blurb: '$e^x$ 碾压多项式 $\\lim x^n/e^x=0$、$\\lim_{x\\to0}\\frac{e^x-1}{x}=1$、洛必达',
    lessons: [
      {
        id: 'u11-l1',
        title: '连续性与无穷远行为',
        subtitle: '$a^x=e^{x\\ln a}$ 与两类底数',
        questions: [
          {
            id: 'u11-l1-q1',
            type: 'choice',
            prompt:
              '桥梁公式把任意底数指数统一成自然指数，它是？',
            options: [
              '$a^x=e^{x\\ln a}$',
              '$a^x=e^{a\\ln x}$',
              '$a^x=\\ln(x a)$',
              '$a^x=x^{\\ln a}$',
            ],
            answer: 0,
            explain:
              '$a^x=e^{x\\ln a}$：线性内层 $x\\ln a$ 复合 $e^u$，由此推出 $a^x$ 连续。',
          },
          {
            id: 'u11-l1-q2',
            type: 'judge',
            prompt:
              '当 $a>1$ 时，$\\lim_{x\\to-\\infty}a^x=0$，且函数恒为正、永不触及 0。',
            answer: true,
            explain:
              '$a>1$ 递增：右端 $\\to+\\infty$、左端 $\\to0$；水平渐近线 $y=0$ 只逼近不取到。',
          },
          {
            id: 'u11-l1-q3',
            type: 'choice',
            prompt: '$\\displaystyle\\lim_{x\\to+\\infty}3^{-x}=?$',
            options: ['$0$', '$+\\infty$', '$1$', '$3$'],
            answer: 0,
            explain:
              '$3^{-x}=\\left(\\frac13\\right)^x$，底 $\\frac13\\in(0,1)$ 递减，$x\\to+\\infty$ 时 $\\to0$。',
          },
          {
            id: 'u11-l1-q4',
            type: 'match',
            prompt: '把底数情形与 $x\\to+\\infty$ 时 $a^x$ 的极限配对：',
            left: ['$a>1$', '$0<a<1$', '$a=1$'],
            right: ['$+\\infty$', '$0$', '$1$'],
            explain: '$a>1$ 发散、$0<a<1$ 趋 0、$a=1$ 恒为 1。',
          },
          {
            id: 'u11-l1-q5',
            type: 'judge',
            prompt: '“连续”仅指图像画起来一笔不断，与极限性质无关。',
            answer: false,
            explain:
              '分析中的连续是：$x$ 微小变化时 $a^x$ 也做可控的小变化，是极限性质而非单纯画图。',
          },
        ],
      },
      {
        id: 'u11-l2',
        title: '增长层级：对数≪幂≪指数',
        subtitle: '$\\ln x\\ll x^\\alpha\\ll a^x$',
        questions: [
          {
            id: 'u11-l2-q1',
            type: 'choice',
            prompt: '对任意整数 $n\\ge1$ 与 $a>1$，下面哪个极限正确？',
            options: [
              '$\\lim_{x\\to+\\infty}\\frac{x^n}{a^x}=0$',
              '$\\lim_{x\\to+\\infty}\\frac{x^n}{a^x}=+\\infty$',
              '$\\lim_{x\\to+\\infty}\\frac{x^n}{a^x}=1$',
              '$\\lim_{x\\to+\\infty}\\frac{x^n}{a^x}=n!$',
            ],
            answer: 0,
            explain: '这就是“指数胜过幂”的标准版本：指数最终彻底主导。',
          },
          {
            id: 'u11-l2-q2',
            type: 'input',
            prompt:
              '在 $x\\to+\\infty$ 下，把 $\\ln x$、$x^5$、$3^x$ 由慢到快排成 $\\ll$ 链：$\\ln x\\ll\\square\\ll 3^x$。$\\square$ 填？',
            accept: ['x^5', 'x5', 'x⁵', 'x^{5}'],
            placeholder: 'x^?',
            explain: '增长层级：$\\ln x\\ll x^5\\ll 3^x$。',
          },
          {
            id: 'u11-l2-q3',
            type: 'judge',
            prompt:
              '“$3^x$ 增长快于 $x^5$”意味着对所有 $x$ 都有 $3^x>x^5$。',
            answer: false,
            explain:
              '渐近“更快”只说 $x$ 充分大后指数主导；在某些较小区间 $x^5$ 可暂时更大。',
          },
          {
            id: 'u11-l2-q4',
            type: 'choice',
            prompt:
              '对任意 $\\alpha>0$，$\\displaystyle\\lim_{x\\to+\\infty}\\frac{\\ln x}{x^\\alpha}=?$',
            options: ['$0$', '$1$', '$+\\infty$', '$\\alpha$'],
            answer: 0,
            explain: '对数慢于任意正幂，故比值趋 0。',
          },
          {
            id: 'u11-l2-q5',
            type: 'judge',
            prompt:
              '幂函数 $x^n$ 的相对增长率 $\\frac{(x^n)\'}{x^n}=\\frac{n}{x}\\to0$，而指数 $a^x$ 的相对增长率恒为常数 $\\ln a$。',
            answer: true,
            explain:
              '这正是本质差别：幂“不断加高”但相对率衰弱，指数“按固定倍率放大”，故指数终将取胜。',
          },
        ],
      },
      {
        id: 'u11-l3',
        title: '洛必达与关键极限',
        subtitle: '$\\frac{x^n}{e^x}\\to0$ 与 $\\frac{e^x-1}{x}\\to1$',
        questions: [
          {
            id: 'u11-l3-q1',
            type: 'choice',
            prompt:
              '求 $\\displaystyle\\lim_{x\\to+\\infty}\\frac{x^3}{e^x}$，连用洛必达法则三次后分子变为常数，结果是？',
            options: ['$0$', '$6$', '$+\\infty$', '$1$'],
            answer: 0,
            explain:
              '$\\frac{x^3}{e^x}\\to\\frac{3x^2}{e^x}\\to\\frac{6x}{e^x}\\to\\frac{6}{e^x}\\to0$。',
          },
          {
            id: 'u11-l3-q2',
            type: 'input',
            prompt: '$\\displaystyle\\lim_{x\\to0}\\frac{e^x-1}{x}=?$（填一个数）',
            accept: ['1', '１'],
            placeholder: '__',
            explain: '$e^x$ 在 0 点的瞬时增长率恰为 1，这正是 $(e^x)\'=e^x$ 的种子。',
          },
          {
            id: 'u11-l3-q3',
            type: 'judge',
            prompt:
              '用级数不等式 $e^x\\ge\\frac{x^{n+1}}{(n+1)!}$ 也能不借助洛必达，由夹逼证明 $\\frac{x^n}{e^x}\\to0$。',
            answer: true,
            explain:
              '$0\\le\\frac{x^n}{e^x}\\le\\frac{(n+1)!}{x}\\to0$，只用级数截断不等式，更基础。',
          },
          {
            id: 'u11-l3-q4',
            type: 'choice',
            prompt:
              '$\\displaystyle\\lim_{x\\to+\\infty}\\frac{x^{100}}{e^{0.01x}}=?$（提示：换元 $t=0.01x$）',
            options: [
              '$0$',
              '$+\\infty$',
              '$100^{100}$',
              '$1$',
            ],
            answer: 0,
            explain:
              '换元得 $100^{100}\\cdot\\frac{t^{100}}{e^t}\\to100^{100}\\times0=0$；指数系数再小、幂次再高终被压倒。',
          },
          {
            id: 'u11-l3-q5',
            type: 'choice',
            prompt:
              '推论：对一般底数 $a>0$，$\\displaystyle\\lim_{h\\to0}\\frac{a^h-1}{h}=?$',
            options: ['$\\ln a$', '$1$', '$a$', '$0$'],
            answer: 0,
            explain: '由 $a^h=e^{h\\ln a}$ 得极限为 $\\ln a$；$a=e$ 时回到 1。',
          },
        ],
      },
    ],
  },

  // ===================== u12 求导 =====================
  {
    id: 'u12',
    title: '求导',
    color: '#7c4dff',
    icon: '✏️',
    blurb: '$(e^x)\'=e^x$ 切线斜率=高度、$(a^x)\'=a^x\\ln a$、链式、对数求导 $x^x$',
    lessons: [
      {
        id: 'u12-l1',
        title: '$(e^x)\'=e^x$ 的来源',
        subtitle: '差商压缩成关键极限',
        questions: [
          {
            id: 'u12-l1-q1',
            type: 'choice',
            prompt:
              '$\\frac{d}{dx}e^x$ 的差商利用 $e^{x+h}=e^xe^h$ 化简后，剩下哪个关键极限？',
            options: [
              '$e^x\\lim_{h\\to0}\\frac{e^h-1}{h}$',
              '$\\lim_{h\\to0}\\frac{e^x-1}{h}$',
              '$e^x\\lim_{h\\to0}\\frac{h}{e^h}$',
              '$\\lim_{h\\to0}(e^h-1)$',
            ],
            answer: 0,
            explain:
              '$\\frac{e^{x+h}-e^x}{h}=e^x\\cdot\\frac{e^h-1}{h}$，而 $\\lim_{h\\to0}\\frac{e^h-1}{h}=1$。',
          },
          {
            id: 'u12-l1-q2',
            type: 'input',
            prompt: '$\\frac{d}{dx}e^x=?$（用 $x$ 表示，写成 e^x）',
            accept: ['e^x', 'e^{x}', 'ex', 'exp(x)'],
            placeholder: '____',
            explain: '导数仍是自己：$(e^x)\'=e^x$。',
          },
          {
            id: 'u12-l1-q3',
            type: 'judge',
            prompt:
              '在 $y=e^x$ 上任一点 $P(x_0,e^{x_0})$，切线斜率恰等于该点高度 $e^{x_0}$。',
            answer: true,
            explain:
              '“导数=自身”的几何体现：如 $x_0=1$ 时切点 $(1,e)$ 处斜率 $=e=$ 高度。',
          },
          {
            id: 'u12-l1-q4',
            type: 'choice',
            prompt: '$f\'(x)=f(x)$ 这个性质的建模含义是？',
            options: [
              '每个时刻的瞬时变化率恰等于它当前的规模',
              '函数恒为常数',
              '函数图像是一条直线',
              '变化率与时间成正比',
            ],
            answer: 0,
            explain: '“按当前规模成比例变化”的最纯粹微积分表达，正是连续增长的标准模型。',
          },
          {
            id: 'u12-l1-q5',
            type: 'judge',
            prompt: '$e^x$ 的二阶导数 $(e^x)\'\'=e^x>0$，故 $e^x$ 是严格凸函数。',
            answer: true,
            explain: '一阶导恒正（严格递增），二阶导恒正（严格凸），图像始终向上弯曲。',
          },
        ],
      },
      {
        id: 'u12-l2',
        title: '一般底数·链式·乘积',
        subtitle: '$(a^x)\'=a^x\\ln a$ 与组合求导',
        questions: [
          {
            id: 'u12-l2-q1',
            type: 'choice',
            prompt: '$\\frac{d}{dx}a^x=?$（$a>0,a\\neq1$）',
            options: ['$a^x\\ln a$', '$x a^{x-1}$', '$a^x$', '$\\frac{a^x}{\\ln a}$'],
            answer: 0,
            explain:
              '由 $a^x=e^{x\\ln a}$，链式法则带出内层斜率 $\\ln a$：$(a^x)\'=a^x\\ln a$。',
          },
          {
            id: 'u12-l2-q2',
            type: 'input',
            prompt: '$\\frac{d}{dx}e^{x^2}=?$（写成形如 2x e^{x^2}）',
            accept: ['2xe^{x^2}', '2x e^{x^2}', '2xe^x^2', '2x*e^(x^2)', '2xe^{x²}'],
            placeholder: '____',
            explain: '复合指数 $(e^{g})\'=e^{g}g\'$，内层 $g=x^2$、$g\'=2x$，得 $2xe^{x^2}$。',
          },
          {
            id: 'u12-l2-q3',
            type: 'choice',
            prompt: '$\\frac{d}{dx}(xe^x)=?$',
            options: ['$(x+1)e^x$', '$xe^x$', '$e^x$', '$(x-1)e^x$'],
            answer: 0,
            explain: '乘积法则：$e^x+xe^x=(x+1)e^x$。',
          },
          {
            id: 'u12-l2-q4',
            type: 'choice',
            prompt: '$\\frac{d}{dx}2^{3x-1}=?$',
            options: [
              '$3\\ln2\\cdot2^{3x-1}$',
              '$2^{3x-1}\\ln2$',
              '$3\\cdot2^{3x-1}$',
              '$(3x-1)2^{3x-2}$',
            ],
            answer: 0,
            explain:
              '$(a^{g})\'=a^{g}\\ln a\\cdot g\'$，$g\'=3$：得 $2^{3x-1}\\ln2\\cdot3=3\\ln2\\cdot2^{3x-1}$。',
          },
          {
            id: 'u12-l2-q5',
            type: 'match',
            prompt: '把求导题型与第一反应配对：',
            left: ['$e^x,\\ a^x$', '$e^{g(x)},\\ a^{g(x)}$', '$xe^x,\\ (\\ln x)e^x$'],
            right: ['直接套基本公式', '基本公式 + 链式法则', '先认出乘积，再乘积法则'],
            explain: '先判断主结构、选对规则，再动笔计算，可避免漏内层导数或漏因子。',
          },
          {
            id: 'u12-l2-q6',
            type: 'judge',
            prompt: '$a^x$ 与 $x^a$ 的求导公式相同。',
            answer: false,
            explain:
              '$(a^x)\'=a^x\\ln a$（指数函数），$(x^a)\'=ax^{a-1}$（幂函数），二者完全不同，最易混。',
          },
        ],
      },
      {
        id: 'u12-l3',
        title: '凸性·切线不等式·对数求导',
        subtitle: '$e^x\\ge1+x$ 与 $x^x$',
        questions: [
          {
            id: 'u12-l3-q1',
            type: 'choice',
            prompt:
              '由凸性，$e^x$ 图像在 $x=0$ 处的切线 $y=1+x$ 之上，得到哪个不等式？',
            options: ['$e^x\\ge1+x$', '$e^x\\le1+x$', '$e^x\\ge x$', '$e^x\\le x^2$'],
            answer: 0,
            explain: '凸函数图像在切线之上：$e^x\\ge1+x$，当且仅当 $x=0$ 取等。',
          },
          {
            id: 'u12-l3-q2',
            type: 'input',
            prompt: '求 $y=x^x$（$x>0$）的导数，结果形如 $x^x(\\ln x+\\square)$，$\\square$ 填？',
            accept: ['1', '１'],
            placeholder: '__',
            explain:
              '$x^x=e^{x\\ln x}$，内层导数 $(\\ln x+1)$，故 $y\'=x^x(\\ln x+1)$。',
          },
          {
            id: 'u12-l3-q3',
            type: 'judge',
            prompt: '不等式 $e^x\\ge1+x$ 仅对 $x\\ge0$ 成立。',
            answer: false,
            explain:
              '对所有实数 $x$ 都成立（仅 $x=0$ 取等）；构造 $g(x)=e^x-(1+x)$ 求最小值即得。',
          },
          {
            id: 'u12-l3-q4',
            type: 'choice',
            prompt:
              '比较 $e^{2x}$ 与 $3^x$（$x\\to+\\infty$），统一到底数 $e$ 后比哪两个数？',
            options: [
              '$2$ 与 $\\ln3\\approx1.0986$，因 $2>\\ln3$ 故 $e^{2x}$ 更快',
              '$2$ 与 $3$，故 $3^x$ 更快',
              '$e$ 与 $3$',
              '无法比较',
            ],
            answer: 0,
            explain:
              '$3^x=e^{x\\ln3}$，比指数系数 $2$ 与 $\\ln3$；$2>\\ln3$，故 $e^{2x}$ 增长更快。',
          },
          {
            id: 'u12-l3-q5',
            type: 'choice',
            prompt:
              '模型 $f(x)=Ce^{kx}$ 的导数 $f\'(x)=kCe^{kx}=kf(x)$。参数 $k$ 的角色是？',
            options: [
              '相对增长率：$k>0$ 增长、$k<0$ 衰减、$|k|$ 越大越敏感',
              '初始规模',
              '函数的零点',
              '渐近线高度',
            ],
            answer: 0,
            explain: '$k$ 直接决定相对增长率与方向，$C$ 才控制初始规模。',
          },
        ],
      },
    ],
  },

  // ===================== u13 积分·微分方程 =====================
  {
    id: 'u13',
    title: '积分·微分方程',
    color: '#00b894',
    icon: '∫',
    blurb:
      '$\\int e^{kx}dx=\\frac1k e^{kx}+C$、$y\'=ky\\Rightarrow y=Ce^{kt}$、分部、牛顿冷却',
    lessons: [
      {
        id: 'u13-l1',
        title: '指数积分与换元',
        subtitle: '$\\int e^{kx}dx$ 与 $\\int e^{g}g\'dx$',
        questions: [
          {
            id: 'u13-l1-q1',
            type: 'choice',
            prompt: '$\\displaystyle\\int e^{kx}\\,dx=?$（$k\\neq0$）',
            options: [
              '$\\frac1k e^{kx}+C$',
              '$k e^{kx}+C$',
              '$e^{kx}+C$',
              '$\\frac1{k+1}e^{kx}+C$',
            ],
            answer: 0,
            explain:
              '是 $(e^{kx})\'=ke^{kx}$ 的反向操作，最易漏的就是这个 $\\frac1k$。',
          },
          {
            id: 'u13-l1-q2',
            type: 'choice',
            prompt: '$\\displaystyle\\int a^x\\,dx=?$（$a>0,a\\neq1$）',
            options: [
              '$\\frac{a^x}{\\ln a}+C$',
              '$a^x\\ln a+C$',
              '$a^x+C$',
              '$\\frac{a^{x+1}}{x+1}+C$',
            ],
            answer: 0,
            explain: '由 $a^x=e^{x\\ln a}$ 与 $\\int e^{kx}dx=\\frac1ke^{kx}+C$，$k=\\ln a$ 得 $\\frac{a^x}{\\ln a}+C$。',
          },
          {
            id: 'u13-l1-q3',
            type: 'input',
            prompt: '$\\displaystyle\\int 2xe^{x^2}\\,dx=?$（令 $u=x^2$，写成 e^{x^2}+C）',
            accept: ['e^{x^2}+C', 'e^x^2+C', 'e^{x^2}', 'e^(x^2)+C', 'e^{x²}+C'],
            placeholder: '______',
            explain:
              '指数内部 $x^2$ 的导数 $2x$ 恰在外面，$\\int e^u du=e^u+C=e^{x^2}+C$。',
          },
          {
            id: 'u13-l1-q4',
            type: 'judge',
            prompt:
              '$\\displaystyle\\int xe^{x^2}\\,dx=\\frac12 e^{x^2}+C$。',
            answer: true,
            explain:
              '只差常数因子：$\\int xe^{x^2}dx=\\frac12\\int 2xe^{x^2}dx=\\frac12e^{x^2}+C$。',
          },
          {
            id: 'u13-l1-q5',
            type: 'judge',
            prompt: '识别复合指数积分的要点是“长得像指数”就直接套换元。',
            answer: false,
            explain:
              '真正要看的是“指数内部函数的导数是否（差常数地）出现在外面”，模式吻合换元才自然。',
          },
        ],
      },
      {
        id: 'u13-l2',
        title: '$y\'=ky$ 与初值问题',
        subtitle: '解 $y=Ce^{kt}$、累积量、分部积分',
        questions: [
          {
            id: 'u13-l2-q1',
            type: 'choice',
            prompt: '微分方程 $\\frac{dy}{dt}=ky$ 的通解是？',
            options: ['$y=Ce^{kt}$', '$y=Ckt$', '$y=C+kt$', '$y=Ce^{t/k}$'],
            answer: 0,
            explain:
              '“导数等于常数倍自身”正匹配指数结构：$\\frac{d}{dt}e^{kt}=ke^{kt}$，故 $y=Ce^{kt}$。',
          },
          {
            id: 'u13-l2-q2',
            type: 'input',
            prompt:
              '某量满足 $y\'=0.4y$、$y(0)=8$。代入初值得 $C=$？',
            accept: ['8', '８'],
            placeholder: '__',
            explain: '$y(0)=Ce^0=C=8$，故 $y(t)=8e^{0.4t}$，$k=0.4>0$ 为指数增长。',
          },
          {
            id: 'u13-l2-q3',
            type: 'match',
            prompt: '把 $y\'=ky$、$y(0)=y_0$ 的三层结构配对：',
            left: ['状态', '机制', '参数'],
            right: ['$y(t)$', '$y\'=ky$', '$k$'],
            explain: '状态是当前大小、机制是变化率如何依赖状态、参数定强弱与方向。',
          },
          {
            id: 'u13-l2-q4',
            type: 'choice',
            prompt:
              '流入速率 $R(t)=R_0e^{kt}$，从 $0$ 到 $T$ 的总流入量 $\\int_0^T R_0e^{kt}dt=?$（$k\\neq0$）',
            options: [
              '$\\frac{R_0}{k}\\left(e^{kT}-1\\right)$',
              '$R_0 k\\left(e^{kT}-1\\right)$',
              '$R_0\\left(e^{kT}-1\\right)$',
              '$\\frac{R_0}{k}e^{kT}$',
            ],
            answer: 0,
            explain: '积分把瞬时速率累积成总量；注意状态量与累积量回答的是不同问题。',
          },
          {
            id: 'u13-l2-q5',
            type: 'choice',
            prompt: '用分部积分求 $\\displaystyle\\int xe^x\\,dx=?$',
            options: [
              '$(x-1)e^x+C$',
              '$(x+1)e^x+C$',
              '$xe^x+C$',
              '$\\frac{x^2}{2}e^x+C$',
            ],
            answer: 0,
            explain:
              '取 $u=x,\\,dv=e^xdx$：$xe^x-\\int e^xdx=(x-1)e^x+C$。验证求导得 $xe^x$。',
          },
          {
            id: 'u13-l2-q6',
            type: 'judge',
            prompt:
              '$R(3)=5e^{-0.6}$ 表示前 3 小时累计输入量（单位毫克）。',
            answer: false,
            explain:
              '$R(3)$ 是第 3 小时这一瞬间的速率（状态量）；累计量是 $\\int_0^3 5e^{-0.2t}dt$。',
          },
        ],
      },
      {
        id: 'u13-l3',
        title: '积分因子·牛顿冷却',
        subtitle: '$y\'+py=q$ 与平衡项 + 过渡项',
        questions: [
          {
            id: 'u13-l3-q1',
            type: 'choice',
            prompt:
              '解一阶线性方程 $y\'+py=q(x)$，应乘上哪个积分因子，使左边成为乘积导数？',
            options: ['$e^{px}$', '$e^{-px}$', '$e^{qx}$', '$px$'],
            answer: 0,
            explain:
              '乘 $e^{px}$ 后左边 $=\\frac{d}{dx}(e^{px}y)=e^{px}y\'+pe^{px}y$，即可直接积分。',
          },
          {
            id: 'u13-l3-q2',
            type: 'input',
            prompt:
              '解 $y\'+2y=6$、$y(0)=1$，得 $y=3+Ce^{-2x}$。代入初值后 $C=$？',
            accept: ['-2', '−2'],
            placeholder: '__',
            explain: '$1=3+C\\Rightarrow C=-2$，故 $y=3-2e^{-2x}$。',
          },
          {
            id: 'u13-l3-q3',
            type: 'judge',
            prompt:
              '解 $y=3-2e^{-2x}$ 中，$3$ 是平衡水平，$-2e^{-2x}$ 是随时间衰减的过渡项，$x\\to+\\infty$ 时 $y\\to3$。',
            answer: true,
            explain:
              '线性方程的解天然分成“平衡项 + 指数衰减过渡项”，长期行为由平衡项主导。',
          },
          {
            id: 'u13-l3-q4',
            type: 'choice',
            prompt:
              '牛顿冷却 $T\'=-k(T-T_{\\text{env}})$ 的标准手法是？',
            options: [
              '换元温差 $u=T-T_{\\text{env}}$，化成 $u\'=-ku$',
              '直接对 $T$ 积分',
              '令 $T=e^{kt}$',
              '忽略 $T_{\\text{env}}$',
            ],
            answer: 0,
            explain:
              '温差衰减是指数型 $u=u_0e^{-kt}$，平衡值就是环境温度，再用一个数据点定 $k$。',
          },
          {
            id: 'u13-l3-q5',
            type: 'input',
            prompt:
              '$90^\\circ$C 咖啡置于 $20^\\circ$C 房间，温差 $u=T-20$。初始温差 $u(0)=?$',
            accept: ['70', '７０'],
            placeholder: '__',
            explain: '$u(0)=90-20=70$，故 $u(t)=70e^{-kt}$，$T(t)=20+70e^{-kt}$。',
          },
          {
            id: 'u13-l3-q6',
            type: 'judge',
            prompt: '积分因子法只是一个要背的公式，与“制造乘积导数”无关。',
            answer: false,
            explain:
              '其核心正是用指数因子把左边凑成 $\\frac{d}{dx}(e^{px}y)$ 这一乘积导数，从而可直接积分。',
          },
        ],
      },
    ],
  },

  // ===================== u14 级数·近似 =====================
  {
    id: 'u14',
    title: '级数·近似',
    color: '#e84393',
    icon: '➕',
    blurb:
      '$e^x=\\sum x^n/n!$、泰勒部分和逼近、$e^{0.1}$ 近似、误差意识',
    lessons: [
      {
        id: 'u14-l1',
        title: '$e^x$ 的幂级数',
        subtitle: '$\\sum x^n/n!$ 与全域收敛',
        questions: [
          {
            id: 'u14-l1-q1',
            type: 'choice',
            prompt: '$e^x$ 的幂级数展开是？',
            options: [
              '$\\sum_{n=0}^{\\infty}\\frac{x^n}{n!}$',
              '$\\sum_{n=0}^{\\infty}\\frac{x^n}{n}$',
              '$\\sum_{n=0}^{\\infty}n!\\,x^n$',
              '$\\sum_{n=0}^{\\infty}\\frac{n!}{x^n}$',
            ],
            answer: 0,
            explain:
              '$e^x=1+x+\\frac{x^2}{2!}+\\frac{x^3}{3!}+\\cdots$；因 $(e^x)\'=e^x$，各阶导在 0 处都为 1。',
          },
          {
            id: 'u14-l1-q2',
            type: 'input',
            prompt:
              '$e^x$ 级数的一般项是 $\\frac{x^n}{\\square}$。$\\square$ 填（写成 n!）？',
            accept: ['n!', 'n！', '阶乘n', 'factorial(n)'],
            placeholder: '__',
            explain: '一般项 $\\frac{x^n}{n!}$，阶乘分母是全域收敛的关键。',
          },
          {
            id: 'u14-l1-q3',
            type: 'judge',
            prompt:
              '$e^x$ 的幂级数对所有实数 $x$ 都收敛，因为分母阶乘增长最终快于分子幂次增长。',
            answer: true,
            explain:
              '$n!$ 长得比 $x^n$ 快，最终把幂次压住；故 $e^x$ 在整条实轴上都能被同一级数描述。',
          },
          {
            id: 'u14-l1-q4',
            type: 'choice',
            prompt:
              '令 $x=1$，$e=\\sum_{n=0}^\\infty\\frac1{n!}=1+1+\\frac12+\\frac16+\\cdots$，约等于？',
            options: ['$2.71828$', '$3.14159$', '$1.41421$', '$2.30259$'],
            answer: 0,
            explain: '前 10 项即可给出 6 位有效数字 $e\\approx2.71828$。',
          },
          {
            id: 'u14-l1-q5',
            type: 'input',
            prompt:
              '把无穷和 $\\sum_{n=0}^{\\infty}\\frac{2^n}{n!}$ 认成 $e^x$ 在 $x=\\square$ 的值，$\\square$ 填？',
            accept: ['2', '２'],
            placeholder: '__',
            explain: '$\\frac{2^n}{n!}=\\frac{x^n}{n!}\\big|_{x=2}$，故和为 $e^2\\approx7.389$。',
          },
          {
            id: 'u14-l1-q6',
            type: 'match',
            prompt: '把含 $\\frac1{n!}$ 的无穷和与其闭式（$e^x$ 的某个取值）配对：',
            left: [
              '$\\sum_{n=0}^{\\infty}\\frac{1}{n!}$',
              '$\\sum_{n=0}^{\\infty}\\frac{2^n}{n!}$',
              '$\\sum_{n=0}^{\\infty}\\frac{(-1)^n}{n!}$',
            ],
            right: ['$e^1=e$', '$e^2$', '$e^{-1}$'],
            explain:
              '反认 $\\sum\\frac{x^n}{n!}=e^x$：分别取 $x=1,2,-1$，得 $e,\\ e^2,\\ e^{-1}$。',
          },
        ],
      },
      {
        id: 'u14-l2',
        title: '部分和逼近与误差',
        subtitle: '$e^x\\approx1+x$、看下一项',
        questions: [
          {
            id: 'u14-l2-q1',
            type: 'choice',
            prompt: '$x$ 很小时，$e^x$ 的一阶（线性化）近似是？',
            options: ['$1+x$', '$x$', '$1+x^2$', '$1-x$'],
            answer: 0,
            explain:
              '$e^x\\approx1+x$ 是 $e^x$ 在 $x=0$ 处的切线近似，靠近 0 时很接近。',
          },
          {
            id: 'u14-l2-q2',
            type: 'input',
            prompt:
              '用到二次项近似 $e^{0.1}\\approx1+0.1+\\frac{0.1^2}{2}=?$（保留三位小数）',
            accept: ['1.105', '1,105'],
            placeholder: '1.___',
            explain: '$1+0.1+0.005=1.105$，真实值 $1.10517$，误差约 $0.00017$。',
          },
          {
            id: 'u14-l2-q3',
            type: 'judge',
            prompt: '$e^x\\approx1+x$ 是处处成立的恒等式。',
            answer: false,
            explain:
              '它只是 $x$ 接近 0 的局部线性近似，不是恒等式；$x$ 大时误差显著。',
          },
          {
            id: 'u14-l2-q4',
            type: 'choice',
            prompt:
              '“保留几项才够”的判断依据是？',
            options: [
              '看下一项（首个被舍项）的量级是否远小于所需精度',
              '永远保留越多越好',
              '固定保留三项',
              '看 $x$ 的符号',
            ],
            answer: 0,
            explain:
              '只要下一项远小于所需精度即可停；这种“看下一项大小”的误差意识是关键。',
          },
          {
            id: 'u14-l2-q5',
            type: 'input',
            prompt:
              '近似 $e^{0.2}\\approx1+0.2+\\frac{0.2^2}{2}=1.22$，下一项 $\\frac{0.2^3}{3!}\\approx?$（科学计数法量级，填如 $10^{-3}$）',
            accept: ['10^-3', '10^{-3}', '0.00133', '1.33e-3', '千分'],
            placeholder: '10^__',
            explain:
              '$\\frac{0.008}{6}\\approx0.00133$，即 $10^{-3}$ 量级，故截到二次项误差大致千分量级。',
          },
        ],
      },
      {
        id: 'u14-l3',
        title: '递推计算与 Euler 出口',
        subtitle: '把 $x$ 换成 $ix$ 通向 $e^{ix}=\\cos x+i\\sin x$',
        questions: [
          {
            id: 'u14-l3-q1',
            type: 'choice',
            prompt:
              '已知某项为 $\\frac{x^n}{n!}$，下一项 $\\frac{x^{n+1}}{(n+1)!}$ 只需把当前项乘以？',
            options: ['$\\frac{x}{n+1}$', '$\\frac{x}{n}$', '$x(n+1)$', '$\\frac{n+1}{x}$'],
            answer: 0,
            explain:
              '这种递推使每项只在前一项上乘 $\\frac{x}{n+1}$，避免重复算高次幂和阶乘，计算高效。',
          },
          {
            id: 'u14-l3-q2',
            type: 'input',
            prompt:
              '把 $x$ 换成 $ix$，$e^{ix}$ 的实部级数恰是 $\\square x$ 的级数（填一个三角函数名）。',
            accept: ['cos', 'cosx', 'cos x', '余弦', 'cosine'],
            placeholder: '___',
            explain: '实部 $1-\\frac{x^2}{2!}+\\frac{x^4}{4!}-\\cdots=\\cos x$，虚部为 $\\sin x$。',
          },
          {
            id: 'u14-l3-q3',
            type: 'choice',
            prompt: '把 $x$ 换成 $ix$ 整理级数，得到的 Euler 公式是？',
            options: [
              '$e^{ix}=\\cos x+i\\sin x$',
              '$e^{ix}=\\sin x+i\\cos x$',
              '$e^{ix}=\\cos x-i\\sin x$',
              '$e^{ix}=1+ix$',
            ],
            answer: 0,
            explain:
              '$i$ 的四步循环 $1,i,-1,-i$ 把级数劈成偶次（$\\cos x$）与奇次（$\\sin x$）两组。',
          },
          {
            id: 'u14-l3-q4',
            type: 'judge',
            prompt:
              '对正 $x$，$e^x$ 级数各项同号，故“首个被舍项”常可作为截断误差量级的快速估计。',
            answer: true,
            explain:
              '尾和被“首项 + 等比型上界”夹住，故首项常作误差量级估计，如 $e^{0.5}$ 截到三次项误差约 $\\frac{0.5^4}{4!}\\approx0.0026$。',
          },
          {
            id: 'u14-l3-q5',
            type: 'judge',
            prompt:
              'Euler 公式 $e^{ix}=\\cos x+i\\sin x$ 是凭空规定的神奇等式，与级数无关。',
            answer: false,
            explain:
              '它是指数幂级数进入复数世界后的必然结果，正是本章（级数）通向第 15 章的自然出口。',
          },
        ],
      },
    ],
  },
]
