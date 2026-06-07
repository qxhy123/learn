export const LESSONS = [
  {
    id: 'u12c4',
    title: '正态分布',
    subtitle: '钟形曲线——$3\\sigma$ 法则与对称性的联合应用',
    intro: [
      {
        title: '从离散到连续：为什么需要密度函数？',
        body: '身高、寿命、测量误差这类变量的取值充满某个区间，无法用"某个具体值的概率"描述（$P(X=172.000\\ldots\\text{cm})=0$）。要改用**密度函数** $f(x)$：密度曲线下、$x$ 轴上、区间 $[a,b]$ 内的**面积**就是该区间的概率。密度函数满足 $f(x)\\geq0$ 且曲线下总面积为 $1$——与分布列的非负性、归一性对应。',
      },
      {
        title: '正态分布的定义',
        body: '若连续型随机变量 $X$ 的密度函数为钟形曲线，由两个参数完全确定：**均值** $\\mu$（决定中心位置）和**标准差** $\\sigma$（决定曲线胖瘦），则称 $X$ 服从正态分布，记 $X \\sim N(\\mu, \\sigma^2)$。\n正态分布的期望 $E(X)=\\mu$，方差 $D(X)=\\sigma^2$，标准差 $\\sigma$ 越小，曲线越尖（数据越集中）。',
        formula: 'f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}}\\, e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}',
      },
      {
        title: '钟形曲线的五条性质',
        body: '**① 对称性**：曲线关于 $x=\\mu$ 完全对称，故 $P(X>\\mu)=P(X<\\mu)=0.5$，$\\mu$ 既是均值又是中位数。\n**② 最高点在 $\\mu$**：$x=\\mu$ 处密度最大。\n**③ $\\sigma$ 越小越尖**：数据集中于 $\\mu$ 附近；$\\sigma$ 越大曲线越平坦。\n**④ 拐点在 $\\mu\\pm\\sigma$**：曲线在这两点由凸转凹。\n**⑤ 渐近于 $x$ 轴**：$x\\to\\pm\\infty$ 时 $f(x)\\to0$，但永不为 $0$。',
      },
      {
        title: '$3\\sigma$ 法则（必记三个数）',
        body: '正态分布在 $\\mu$ 周围 $1\\sigma$、$2\\sigma$、$3\\sigma$ 范围内的概率是高考核心考点，必须牢记：\n$1\\sigma$ 区间：约 $68.26\\%$ 的数据落在 $[\\mu-\\sigma,\\, \\mu+\\sigma]$ 内。\n$2\\sigma$ 区间：约 $95.44\\%$ 落在 $[\\mu-2\\sigma,\\, \\mu+2\\sigma]$ 内。\n$3\\sigma$ 区间：约 $99.74\\%$ 落在 $[\\mu-3\\sigma,\\, \\mu+3\\sigma]$ 内（"几乎全部"）。',
        formula: 'P(\\mu-k\\sigma < X < \\mu+k\\sigma) \\approx \\begin{cases} 0.6826 & k=1 \\\\ 0.9544 & k=2 \\\\ 0.9974 & k=3 \\end{cases}',
      },
      {
        title: '对称性：拆分复杂区间',
        body: '由对称性可将任意区间概率拆为"左半" + "右半"：\n$P(X>\\mu+a)=P(X<\\mu-a)$（两侧对称）。\n$P(X>\\mu+k\\sigma)=\\dfrac{1-P(\\mu-k\\sigma<X<\\mu+k\\sigma)}{2}$（区间外除以 $2$）。\n例：$P(X>\\mu+2\\sigma)=\\dfrac{1-0.9544}{2}=0.0228$。\n非对称区间（如 $[\\mu-\\sigma, \\mu+2\\sigma]$）：以 $\\mu$ 为界拆成左右两段，分别用 $3\\sigma$ 法则的一半来计算。',
        tip: '口诀：端点先化为 $\\mu\\pm k\\sigma$，再查三个概率值。若区间跨越 $\\mu$，以 $\\mu$ 为界分两块：左块 $=$ 对应 $k\\sigma$ 概率的一半，右块同理。',
      },
      {
        title: '典例：成绩分布',
        body: '某地高三数学成绩 $X\\sim N(85,25)$，即 $\\mu=85, \\sigma=5$。\n(1) $P(80<X<90)$：$80=\\mu-\\sigma, 90=\\mu+\\sigma$，故概率 $\\approx 0.6826$，约 $68\\%$。\n(2) $P(75<X<95)$：$75=\\mu-2\\sigma, 95=\\mu+2\\sigma$，故概率 $\\approx 0.9544$，约 $95\\%$。\n(3) $10000$ 名考生中 $70$ 至 $100$ 分人数：$70=\\mu-3\\sigma, 100=\\mu+3\\sigma$，$P\\approx0.9974$，约 $9974$ 人。',
        steps: [
          '把端点写成 $\\mu\\pm k\\sigma$ 的形式（识别 $k$）',
          '查 $3\\sigma$ 法则读出对应概率',
          '非对称区间以 $\\mu$ 为界拆成两块分别计算',
          '人数 $=$ 总人数 $\\times$ 概率（四舍五入取整）',
        ],
      },
      {
        title: '标准正态与标准化变换',
        body: '$\\mu=0, \\sigma=1$ 的正态分布称为**标准正态分布**，记 $Z\\sim N(0,1)$。任意 $X\\sim N(\\mu,\\sigma^2)$ 都可通过 $Z=\\dfrac{X-\\mu}{\\sigma}$ 变换为标准正态（减均值移中心，除标准差统一单位），从而查标准正态表。高中阶段理解"存在这种变换"即可；$3\\sigma$ 法则直接用 $\\mu,\\sigma$ 计算，不必每次都化为 $Z$。',
        formula: 'Z = \\frac{X - \\mu}{\\sigma} \\sim N(0, 1)',
      },
      {
        title: '易错点与自检',
        body: '**错误 1**：$N(\\mu,\\sigma^2)$ 第二个参数是**方差** $\\sigma^2$，求 $\\sigma$ 需开根号。如 $N(60,100)$ 中 $\\sigma=10$，不是 $100$。\n**错误 2**：三个概率值顺序记混——"$1\\sigma \\to 0.6826$，$2\\sigma \\to 0.9544$，$3\\sigma \\to 0.9974$"，依次递增且接近 $1$。\n**错误 3**：求单侧概率忘除以 $2$——"区间外"由对称性平分给左右两侧。\n**错误 4**：连续型变量 $P(X=a)=0$，端点开闭不影响区间概率。',
        reveal: {
          q: '$X\\sim N(170,36)$（$\\sigma=6$）。求 $P(X>182)$ 和 $P(164<X<176)$。',
          a: '$182=170+2\\times6=\\mu+2\\sigma$；$P(X>\\mu+2\\sigma)=\\dfrac{1-0.9544}{2}=0.0228$。$164=\\mu-\\sigma, 176=\\mu+\\sigma$；$P(164<X<176)\\approx0.6826$。',
        },
      },
    ],
    questions: [
      {
        id: 'u12c4q1',
        type: 'choice',
        question: '$X\\sim N(50,16)$，则 $\\sigma$ 等于（　　），$P(46<X<54)\\approx$（　　）。',
        options: [
          '$\\sigma=16$，$P\\approx0.6826$',
          '$\\sigma=4$，$P\\approx0.6826$',
          '$\\sigma=4$，$P\\approx0.9544$',
          '$\\sigma=16$，$P\\approx0.9544$',
        ],
        answer: '$\\sigma=4$，$P\\approx0.6826$',
        explanation: '$\\sigma=\\sqrt{16}=4$；$46=50-4=\\mu-\\sigma$，$54=50+4=\\mu+\\sigma$，故 $P(46<X<54)\\approx0.6826$。',
      },
      {
        id: 'u12c4q2',
        type: 'choice',
        question: '$X\\sim N(100,25)$（$\\sigma=5$）。下列概率最大的是（　　）。',
        options: [
          '$P(95<X<105)$',
          '$P(90<X<110)$',
          '$P(85<X<115)$',
          '$P(X>100)$',
        ],
        answer: '$P(85<X<115)$',
        explanation: '$P(85<X<115)=P(\\mu-3\\sigma<X<\\mu+3\\sigma)\\approx0.9974$，是三个区间内概率中最大的。$P(X>100)=0.5$；$P(95<X<105)\\approx0.6826$；$P(90<X<110)\\approx0.9544$。',
      },
      {
        id: 'u12c4q3',
        type: 'input',
        question: '$X\\sim N(60,100)$（$\\sigma=10$）。求 $P(X>80)$（保留四位小数）。',
        answer: '0.0228',
        explanation: '$80=60+2\\times10=\\mu+2\\sigma$；$P(X>\\mu+2\\sigma)=\\dfrac{1-0.9544}{2}=\\dfrac{0.0456}{2}=0.0228$。',
      },
      {
        id: 'u12c4q4',
        type: 'judge',
        question: '$X\\sim N(\\mu,\\sigma^2)$，则 $P(X=\\mu)=0.5$，因为 $\\mu$ 是正态分布的中心。此说法正确吗？',
        answer: '错误',
        explanation: '连续型随机变量在任意单点的概率为 $0$，即 $P(X=\\mu)=0$。$\\mu$ 是均值和中位数，$P(X>\\mu)=P(X<\\mu)=0.5$ 是正确的，但 $P(X=\\mu)=0$。',
      },
      {
        id: 'u12c4q5',
        type: 'input',
        question: '$X\\sim N(75,36)$（$\\sigma=6$）。求 $P(X<63)$（保留四位小数）。',
        answer: '0.0228',
        explanation: '$63=75-12=\\mu-2\\sigma$；由对称性 $P(X<\\mu-2\\sigma)=P(X>\\mu+2\\sigma)=\\dfrac{1-0.9544}{2}=0.0228$。',
      },
      {
        id: 'u12c4q6',
        type: 'choice',
        question: '$X\\sim N(60,100)$（$\\sigma=10$）。求 $P(40<X<70)$（　　）。',
        options: ['$0.8185$', '$0.9544$', '$0.6826$', '$0.8413$'],
        answer: '$0.8185$',
        explanation: '$40=\\mu-2\\sigma$，$70=\\mu+\\sigma$。$P(40<X<70)=P(40<X<60)+P(60<X<70)=\\dfrac{0.9544}{2}+\\dfrac{0.6826}{2}=0.4772+0.3413=0.8185$。',
      },
      {
        id: 'u12c4q7',
        type: 'input',
        question: '某品牌灯泡寿命 $X\\sim N(2000,10000)$（$\\sigma=100$）。从 $1000$ 个灯泡中，估计寿命在 $1900$ 至 $2200$ 小时之间的有多少个？（取整）',
        answer: '819',
        explanation: '$1900=\\mu-\\sigma$，$2200=\\mu+2\\sigma$。$P(1900<X<2200)=\\dfrac{0.6826}{2}+\\dfrac{0.9544}{2}=0.3413+0.4772=0.8185$。人数 $\\approx1000\\times0.8185=818.5\\approx819$ 个。',
      },
      {
        id: 'u12c4q8',
        type: 'match',
        question: '将下列区间描述与对应的 $3\\sigma$ 法则概率配对（$X\\sim N(\\mu,\\sigma^2)$）。',
        left: [
          '$P(\\mu-\\sigma<X<\\mu+\\sigma)$',
          '$P(\\mu-2\\sigma<X<\\mu+2\\sigma)$',
          '$P(X>\\mu+\\sigma)$',
          '$P(X>\\mu+3\\sigma)$',
        ],
        right: ['$\\approx0.6826$', '$\\approx0.9544$', '$\\approx0.1587$', '$\\approx0.0013$'],
        answer: ['$\\approx0.6826$', '$\\approx0.9544$', '$\\approx0.1587$', '$\\approx0.0013$'],
        explanation: '$P(X>\\mu+\\sigma)=\\dfrac{1-0.6826}{2}=0.1587$；$P(X>\\mu+3\\sigma)=\\dfrac{1-0.9974}{2}=0.0013$。',
      },
    ],
  },
]
