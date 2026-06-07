// 第30章  概率论中的微积分
// 关 c30-a：概率密度、期望与方差的积分定义（7 卡 7 题）
// 关 c30-b：KL 散度、变量替换与 ELBO（7 卡 7 题）

export const LESSONS = [
  // ─────────────────────────────────────────
  // 关 c30-a  概率密度、期望与方差
  // ─────────────────────────────────────────
  {
    id: 'c30-a',
    title: '概率密度、期望与方差',
    subtitle: '动机·PDF 归一化·期望积分·方差·高斯积分技巧',
    intro: [
      // 卡 1：动机——为什么连续随机变量需要密度函数？
      {
        title: '从频率到密度：为什么单点概率等于零？',
        body:
          '离散情形中，掷一枚骰子，$P(X=3)=1/6$，这完全合理。\n\n' +
          '但如果 $X$ 是一个人的身高（精确到毫米的小数），那 $P(X=170.000\\ldots)$ 是多少？' +
          '直觉告诉我们这个概率应当**趋近于零**——因为实数轴上有无穷多个点，概率不可能分配到每一个点上。\n\n' +
          '解决办法：不问"正好等于某值"的概率，改问**"落在某区间内"**的概率。' +
          '区间的概率等于面积，而描述面积的工具就是**概率密度函数（PDF）**：\n\n' +
          '$P(a \\le X \\le b) = \\int_a^b f(x)\\,dx$\n\n' +
          '密度函数 $f(x)$ 可以大于 $1$（它是密度，不是概率），但面积（积分）永远在 $[0,1]$ 之间。',
        tip: '$f(x)$ 是密度，不是概率。均匀分布 $U[0, 0.1]$ 的密度为 $10$，完全合法——它只是说该区间"很拥挤"。',
      },
      // 卡 2：PDF 的两条合法性条件
      {
        title: 'PDF 的两条合法性条件',
        body:
          '一个函数 $f(x)$ 要成为合法的概率密度函数，必须同时满足两条：\n\n' +
          '**条件一：非负性**——对所有 $x$，$f(x) \\ge 0$（概率密度不能为负）。\n\n' +
          '**条件二：归一化**——$\\int_{-\\infty}^{+\\infty} f(x)\\,dx = 1$（总概率等于 $1$）。\n\n' +
          '**验证例子（指数分布）**：$f(x) = \\lambda e^{-\\lambda x}$，$x \\ge 0$（$\\lambda > 0$）。\n\n' +
          '条件一：$\\lambda > 0$，$e^{-\\lambda x} > 0$，所以 $f(x) > 0$ ✓。\n\n' +
          '条件二：$\\int_0^\\infty \\lambda e^{-\\lambda x}\\,dx = \\left[-e^{-\\lambda x}\\right]_0^\\infty = 0-(-1)=1$ ✓。\n\n' +
          '**累积分布函数（CDF）**由 PDF 积分得到：$F(x) = \\int_{-\\infty}^{x} f(t)\\,dt$，' +
          '由微积分基本定理反过来有 $F\'(x) = f(x)$。',
        formula:
          'f(x) \\ge 0, \\quad \\int_{-\\infty}^{+\\infty} f(x)\\,dx = 1, \\quad F(x) = \\int_{-\\infty}^{x} f(t)\\,dt',
        tip: 'CDF 单调从 $0$ 增到 $1$；PDF 是 CDF 的导数。见到区间概率就想"面积"，见到 CDF 就想"微分还原"。',
      },
      // 卡 3：高斯积分：极坐标换元技巧
      {
        title: '高斯积分：极坐标换元',
        body:
          '正态分布的归一化依赖一个经典积分 $I = \\int_{-\\infty}^{+\\infty} e^{-x^2/2}\\,dx$。' +
          '直接计算没有原函数，但有一个精妙的技巧：**把 $I^2$ 写成二重积分，再转极坐标**。',
        steps: [
          '写 $I^2 = \\int_{-\\infty}^{+\\infty}\\int_{-\\infty}^{+\\infty} e^{-(x^2+y^2)/2}\\,dx\\,dy$。',
          '令 $x = r\\cos\\theta$，$y = r\\sin\\theta$，则 $x^2+y^2 = r^2$，面积元变为 $dx\\,dy = r\\,dr\\,d\\theta$。',
          '积分范围：$r \\in [0,+\\infty)$，$\\theta \\in [0, 2\\pi)$。',
          '$I^2 = \\int_0^{2\\pi}\\int_0^\\infty e^{-r^2/2}\\cdot r\\,dr\\,d\\theta = 2\\pi \\cdot \\int_0^\\infty e^{-r^2/2} r\\,dr$。',
          '令 $u = r^2/2$，则 $du = r\\,dr$，内层积分 $= \\int_0^\\infty e^{-u}\\,du = 1$。',
          '所以 $I^2 = 2\\pi$，故 $I = \\sqrt{2\\pi}$，归一化常数 $\\dfrac{1}{\\sqrt{2\\pi}}$ 得证。',
        ],
        tip: '关键步骤是"$e^{-r^2/2}\\cdot r\\,dr$"里多出来的 $r$——这来自极坐标面积元，正好让积分变成简单的指数积分。',
      },
      // 卡 4：期望——分布的"重心"
      {
        title: '期望：分布的重心',
        body:
          '连续随机变量 $X$ 的**期望**（均值）定义为\n\n' +
          '$\\mathbb{E}[X] = \\int_{-\\infty}^{+\\infty} x\\,f(x)\\,dx$\n\n' +
          '直觉上，期望就是把每个值 $x$ 按其密度 $f(x)$ 加权求和，结果是分布的"重心"。\n\n' +
          '更一般地，若 $g$ 是任意可积函数（LOTUS 法则）：\n\n' +
          '$\\mathbb{E}[g(X)] = \\int_{-\\infty}^{+\\infty} g(x)\\,f(x)\\,dx$\n\n' +
          '**指数分布的期望**（$f(x) = \\lambda e^{-\\lambda x}$，$x \\ge 0$）：\n\n' +
          '$\\mathbb{E}[X] = \\int_0^\\infty x\\lambda e^{-\\lambda x}\\,dx$。' +
          '用分部积分令 $u=x$，$dv=\\lambda e^{-\\lambda x}dx$，得 $\\mathbb{E}[X] = 1/\\lambda$。\n\n' +
          '**正态分布** $\\mathcal{N}(\\mu, \\sigma^2)$：$\\mathbb{E}[X] = \\mu$（对称轴就是均值，积分利用奇函数性质）。',
        formula: '\\mathbb{E}[X] = \\int_{-\\infty}^{+\\infty} x\\,f(x)\\,dx, \\qquad \\mathbb{E}[g(X)] = \\int_{-\\infty}^{+\\infty} g(x)\\,f(x)\\,dx',
        tip: '期望可以是任意实数，包括负数。双峰分布的期望可能恰好落在两峰之间的低谷——它是"重心"，不一定是"最高点"。',
      },
      // 卡 5：方差——分布的"离散程度"
      {
        title: '方差：积分定义与展开公式',
        body:
          '**方差**衡量 $X$ 围绕均值 $\\mu = \\mathbb{E}[X]$ 的散布程度：\n\n' +
          '$\\mathrm{Var}(X) = \\mathbb{E}[(X - \\mu)^2] = \\int_{-\\infty}^{+\\infty} (x-\\mu)^2 f(x)\\,dx$\n\n' +
          '展开后得到更常用的计算公式：\n\n' +
          '$\\mathrm{Var}(X) = \\mathbb{E}[X^2] - (\\mathbb{E}[X])^2$\n\n' +
          '**推导**：$(X-\\mu)^2 = X^2 - 2\\mu X + \\mu^2$，取期望得\n\n' +
          '$\\mathbb{E}[(X-\\mu)^2] = \\mathbb{E}[X^2] - 2\\mu^2 + \\mu^2 = \\mathbb{E}[X^2] - \\mu^2$。\n\n' +
          '**验证**（标准正态 $\\mu=0$）：$\\mathrm{Var}(X) = \\mathbb{E}[X^2] - 0 = \\int x^2 \\frac{e^{-x^2/2}}{\\sqrt{2\\pi}}\\,dx$。' +
          '分部积分（令 $u=x$，$dv=xe^{-x^2/2}dx$）得 $\\mathbb{E}[X^2] = 1$，与 $\\sigma^2=1$ 一致。',
        formula: '\\mathrm{Var}(X) = \\mathbb{E}[X^2] - (\\mathbb{E}[X])^2',
        tip: 'Jensen 不等式保证 $\\mathbb{E}[X^2] \\ge (\\mathbb{E}[X])^2$（$f(x)=x^2$ 是凸函数），因此方差恒 $\\ge 0$。',
      },
      // 卡 6：正态分布的完整参数推导
      {
        title: '正态分布：用换元验证均值与方差',
        body:
          '正态分布 $\\mathcal{N}(\\mu, \\sigma^2)$ 的密度为\n\n' +
          '$f(x) = \\dfrac{1}{\\sqrt{2\\pi\\sigma^2}} e^{-(x-\\mu)^2/(2\\sigma^2)}$\n\n' +
          '**归一化验证**：令 $z = (x-\\mu)/\\sigma$，则 $dx = \\sigma\\,dz$，被积函数变为 $\\dfrac{1}{\\sqrt{2\\pi}} e^{-z^2/2}$，积分为 $1$ ✓。\n\n' +
          '**均值**：令 $z = (x-\\mu)/\\sigma$，则 $x = \\mu + \\sigma z$，\n\n' +
          '$\\mathbb{E}[X] = \\int (\\mu + \\sigma z)\\frac{e^{-z^2/2}}{\\sqrt{2\\pi}}\\,dz = \\mu\\cdot 1 + \\sigma\\cdot 0 = \\mu$\n\n' +
          '（$\\sigma z\\cdot e^{-z^2/2}$ 是奇函数，积分为 $0$）。\n\n' +
          '**方差**：$\\mathbb{E}[(X-\\mu)^2] = \\sigma^2 \\mathbb{E}[Z^2] = \\sigma^2 \\cdot 1 = \\sigma^2$（标准正态二阶矩为 $1$）。',
        steps: [
          '换元 $z = (x-\\mu)/\\sigma$，$dx = \\sigma\\,dz$。',
          '归一化：$\\int_{-\\infty}^{+\\infty}\\frac{1}{\\sqrt{2\\pi}}e^{-z^2/2}\\,dz = 1$，由高斯积分 $I=\\sqrt{2\\pi}$ 得。',
          '均值：$\\int z\\frac{e^{-z^2/2}}{\\sqrt{2\\pi}}\\,dz = 0$（奇函数），所以 $\\mathbb{E}[X] = \\mu$。',
          '方差：$\\int z^2\\frac{e^{-z^2/2}}{\\sqrt{2\\pi}}\\,dz = 1$（分部积分），所以 $\\mathrm{Var}(X) = \\sigma^2$。',
        ],
        tip: '记住换元 $z=(x-\\mu)/\\sigma$ 是处理正态分布积分的万能第一步——它把任意正态化为标准正态。',
      },
      // 卡 7：易错点与 reveal
      {
        title: '易错点与自测',
        body:
          '**易错一**：PDF $f(x)$ 可以大于 $1$，它是**密度**不是概率。只有区间积分 $\\int_a^b f(x)\\,dx$ 才是概率，值域才必须在 $[0,1]$。\n\n' +
          '**易错二**：期望不一定存在——若 $\\int|x|f(x)\\,dx = +\\infty$，期望无定义（如 Cauchy 分布）。遇到重尾分布要检验收敛性。\n\n' +
          '**易错三**：$\\mathrm{Var}(X) = \\mathbb{E}[X^2] - (\\mathbb{E}[X])^2$，注意括号——是 $\\mathbb{E}[X^2]$ 减去 $(\\mathbb{E}[X])^2$，不是 $\\mathbb{E}[X^2 - X^2]$。\n\n' +
          '**易错四**：CDF 单调不减，$F(-\\infty)=0$，$F(+\\infty)=1$；对 CDF 求导才得 PDF，反过来对 PDF 积分才得 CDF。',
        reveal: {
          q: '设 $f(x) = cx^2(1-x)$，$x \\in [0,1]$，其余为 $0$。要使 $f$ 是合法 PDF，$c$ 应等于多少？',
          a: '归一化条件：$c\\int_0^1 x^2(1-x)\\,dx = 1$。展开：$\\int_0^1(x^2-x^3)\\,dx = \\frac{1}{3} - \\frac{1}{4} = \\frac{1}{12}$。故 $c/12 = 1$，$c = 12$。',
        },
      },
    ],
    questions: [
      // 题 1：judge — PDF 可以大于 1
      {
        id: 'c30-a-q1',
        type: 'judge',
        prompt: '概率密度函数 $f(x)$ 的值必须在 $0$ 到 $1$ 之间，否则不是合法的 PDF。',
        answer: false,
        explain:
          '$f(x)$ 是**密度**，不是概率。它只需满足 $f(x) \\ge 0$ 和 $\\int f(x)\\,dx=1$，本身的值可以大于 $1$。' +
          '例如均匀分布 $U[0, 0.1]$ 的密度 $f(x)=10>1$，完全合法。',
      },
      // 题 2：input — 归一化常数
      {
        id: 'c30-a-q2',
        type: 'input',
        prompt:
          '设 $f(x) = cx^2(1-x)$，$x \\in [0,1]$，其余为 $0$。要使 $f$ 是合法 PDF，$c$ 等于多少？',
        accept: ['12'],
        explain:
          '$\\int_0^1 x^2(1-x)\\,dx = \\frac{1}{3}-\\frac{1}{4} = \\frac{1}{12}$，' +
          '故 $c \\cdot \\frac{1}{12} = 1$，得 $c = 12$。',
      },
      // 题 3：choice — 指数分布期望
      {
        id: 'c30-a-q3',
        type: 'choice',
        prompt: '指数分布 $f(x) = \\lambda e^{-\\lambda x}$（$x \\ge 0$）的期望 $\\mathbb{E}[X]$ 等于？',
        options: ['$\\lambda$', '$\\lambda^2$', '$1/\\lambda$', '$1/\\lambda^2$'],
        answer: 2,
        explain:
          '$\\mathbb{E}[X] = \\int_0^\\infty x\\lambda e^{-\\lambda x}\\,dx$。' +
          '分部积分：令 $u=x$，$dv=\\lambda e^{-\\lambda x}dx$，' +
          '得 $\\mathbb{E}[X] = 0 + \\int_0^\\infty e^{-\\lambda x}\\,dx = 1/\\lambda$。',
      },
      // 题 4：choice — 方差展开公式
      {
        id: 'c30-a-q4',
        type: 'choice',
        prompt:
          '若 $\\mathbb{E}[X] = 2$，$\\mathbb{E}[X^2] = 9$，则 $\\mathrm{Var}(X)$ 等于？',
        options: ['$13$', '$7$', '$5$', '$4$'],
        answer: 2,
        explain:
          '$\\mathrm{Var}(X) = \\mathbb{E}[X^2] - (\\mathbb{E}[X])^2 = 9 - 4 = 5$。' +
          '注意是减去 $(\\mathbb{E}[X])^2 = 2^2 = 4$，而不是 $\\mathbb{E}[X] = 2$。',
      },
      // 题 5：judge — 高斯积分结论
      {
        id: 'c30-a-q5',
        type: 'judge',
        prompt: '$\\int_{-\\infty}^{+\\infty} e^{-x^2/2}\\,dx = \\sqrt{2\\pi}$。',
        answer: true,
        explain:
          '令 $I = \\int e^{-x^2/2}\\,dx$。利用极坐标换元，$I^2 = \\int\\!\\int e^{-(x^2+y^2)/2}\\,dx\\,dy = 2\\pi\\int_0^\\infty e^{-r^2/2}r\\,dr = 2\\pi$，' +
          '故 $I = \\sqrt{2\\pi}$。这也说明正态分布的归一化常数是 $1/\\sqrt{2\\pi}$。',
      },
      // 题 6：match — 概念配对
      {
        id: 'c30-a-q6',
        type: 'match',
        prompt: '将概念与其定义/公式配对。',
        left: ['PDF 归一化条件', '期望（LOTUS）', '方差展开式'],
        right: [
          '$\\int_{-\\infty}^{+\\infty} f(x)\\,dx = 1$',
          '$\\int_{-\\infty}^{+\\infty} g(x)f(x)\\,dx$',
          '$\\mathbb{E}[X^2] - (\\mathbb{E}[X])^2$',
        ],
      },
      // 题 7：input — 标准正态二阶矩
      {
        id: 'c30-a-q7',
        type: 'input',
        prompt:
          '标准正态分布 $\\mathcal{N}(0,1)$ 的二阶矩 $\\mathbb{E}[X^2]$ 等于多少？（提示：方差加均值的平方）',
        accept: ['1', '1.0'],
        explain:
          '$\\mathbb{E}[X^2] = \\mathrm{Var}(X) + (\\mathbb{E}[X])^2 = 1 + 0^2 = 1$。' +
          '也可直接计算：$\\int_{-\\infty}^{+\\infty} x^2 \\frac{e^{-x^2/2}}{\\sqrt{2\\pi}}\\,dx = 1$（分部积分）。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c30-b  KL 散度、变量替换与 ELBO
  // ─────────────────────────────────────────
  {
    id: 'c30-b',
    title: 'KL 散度、变量替换与 ELBO',
    subtitle: '动机·KL 积分·换元公式·重参数化·Jensen 推导 ELBO',
    intro: [
      // 卡 1：动机——如何衡量两个分布有多"不像"？
      {
        title: '如何衡量两个分布的差异？',
        body:
          '假设真实世界的降雨量服从分布 $p(x)$，而你用模型拟合出分布 $q(x)$。' +
          '问：你的模型有多"偏"？\n\n' +
          '最朴素的想法是比较两条密度曲线之间的"距离"。但概率论里有一种更深刻的度量——' +
          '**KL 散度（Kullback-Leibler divergence）**，它衡量"用分布 $q$ 近似 $p$ 时，平均损失了多少信息"：\n\n' +
          '$\\mathrm{KL}(p \\| q) = \\int p(x)\\ln\\dfrac{p(x)}{q(x)}\\,dx$\n\n' +
          'KL 散度有两个基本性质：\n\n' +
          '① **非负性**：$\\mathrm{KL}(p\\|q) \\ge 0$，当且仅当 $p=q$ 时等号成立（Jensen 不等式推出）。\n\n' +
          '② **非对称性**：$\\mathrm{KL}(p\\|q) \\ne \\mathrm{KL}(q\\|p)$，这让它区别于通常意义上的"距离"。',
        tip: 'KL 是"用 $q$ 编码 $p$ 时多花的比特数"，方向很重要：$\\mathrm{KL}(p\\|q)$ 读作"$p$ 相对 $q$ 的 KL"。',
      },
      // 卡 2：一维高斯之间的 KL 散度闭式
      {
        title: '两个高斯之间的 KL：闭式推导',
        body:
          '设 $p = \\mathcal{N}(\\mu_1, \\sigma_1^2)$，$q = \\mathcal{N}(\\mu_2, \\sigma_2^2)$，' +
          '它们的对数密度分别为\n\n' +
          '$\\ln p(x) = -\\dfrac{1}{2}\\ln(2\\pi\\sigma_1^2) - \\dfrac{(x-\\mu_1)^2}{2\\sigma_1^2}$\n\n' +
          '$\\ln q(x) = -\\dfrac{1}{2}\\ln(2\\pi\\sigma_2^2) - \\dfrac{(x-\\mu_2)^2}{2\\sigma_2^2}$',
        steps: [
          '写出定义：$\\mathrm{KL}(p\\|q) = \\mathbb{E}_p[\\ln p(x) - \\ln q(x)]$。',
          '代入对数密度之差，按期望线性拆分：得到三类期望——常数项、$\\mathbb{E}_p[X]$、$\\mathbb{E}_p[X^2]$。',
          '用 $\\mathbb{E}_p[X] = \\mu_1$，$\\mathbb{E}_p[(X-\\mu_1)^2] = \\sigma_1^2$，以及 $\\mathbb{E}_p[(X-\\mu_2)^2] = \\sigma_1^2 + (\\mu_1-\\mu_2)^2$。',
          '整理得闭式：$\\mathrm{KL}(p\\|q) = \\ln\\dfrac{\\sigma_2}{\\sigma_1} + \\dfrac{\\sigma_1^2+(\\mu_1-\\mu_2)^2}{2\\sigma_2^2} - \\dfrac{1}{2}$。',
        ],
        formula:
          '\\mathrm{KL}(\\mathcal{N}_1\\|\\mathcal{N}_2) = \\ln\\frac{\\sigma_2}{\\sigma_1}+\\frac{\\sigma_1^2+(\\mu_1-\\mu_2)^2}{2\\sigma_2^2}-\\frac{1}{2}',
        tip: '三项记忆口诀："log 比 + 方差比 + 常数"。当 $\\mu_1=\\mu_2$，$\\sigma_1=\\sigma_2$ 时，三项恰好凑成 $0$，即 $\\mathrm{KL}=0$。',
      },
      // 卡 3：变量替换——分布的 Jacobian
      {
        title: '变量替换：分布变换中的 Jacobian',
        body:
          '若连续随机变量 $Y = g(X)$，$g$ 可逆（单调），则 $Y$ 的密度为\n\n' +
          '$f_Y(y) = f_X(g^{-1}(y)) \\cdot \\left|\\dfrac{d}{dy}g^{-1}(y)\\right|$\n\n' +
          '直觉：$g$ 把 $x$ 轴上的区间"拉伸或压缩"了，Jacobian 因子 $|dg^{-1}/dy|$ 就是"补偿这个形变"——' +
          '保证概率总量（面积）不变。\n\n' +
          '**具体例子**：设 $X \\sim U[0,1]$，令 $Y = -\\ln X$（$X=e^{-Y}$）。' +
          '则 $f_Y(y) = f_X(e^{-y}) \\cdot e^{-y} = 1 \\cdot e^{-y}$，即 $Y \\sim \\mathrm{Exp}(1)$。\n\n' +
          '这说明：对均匀分布取负对数，恰好得到参数为 $1$ 的指数分布——逆变换采样的理论基础。',
        formula:
          'f_Y(y) = f_X(g^{-1}(y))\\cdot\\left|\\frac{d}{dy}g^{-1}(y)\\right|',
        tip: '多元情形中，标量 Jacobian 换成行列式 $|\\det J|$，这是 normalizing flow 的数学核心。',
      },
      // 卡 4：重参数化技巧
      {
        title: '重参数化技巧：让采样变可微',
        body:
          '在变分自编码器（VAE）里，我们需要对期望求梯度：\n\n' +
          '$\\nabla_{\\theta}\\,\\mathbb{E}_{z \\sim q_{\\theta}(z)}[f(z)]$\n\n' +
          '**问题**：梯度无法直接穿透采样步骤——采样是随机操作，不是可微函数。\n\n' +
          '**重参数化**：把 $z \\sim \\mathcal{N}(\\mu, \\sigma^2)$ 改写为\n\n' +
          '$z = \\mu + \\sigma \\varepsilon$，其中 $\\varepsilon \\sim \\mathcal{N}(0,1)$\n\n' +
          '随机性移到与参数无关的 $\\varepsilon$ 上，期望变成对 $\\varepsilon$ 的期望：\n\n' +
          '$\\mathbb{E}_{q_{\\theta}(z)}[f(z)] = \\mathbb{E}_{\\varepsilon \\sim \\mathcal{N}(0,1)}[f(\\mu + \\sigma\\varepsilon)]$\n\n' +
          '现在对 $\\mu$ 和 $\\sigma$ 的梯度可以安全穿透，且方差通常远小于 REINFORCE 方法。',
        formula:
          'z = \\mu + \\sigma\\varepsilon,\\quad \\varepsilon \\sim \\mathcal{N}(0,1)',
        tip: '重参数化的关键：把"随机"移到和参数无关的变量上，让计算图重新变得可微。不可重参数化的分布（如 Bernoulli）需用 Gumbel-Softmax。',
      },
      // 卡 5：Jensen 不等式与 ELBO 推导
      {
        title: 'Jensen 不等式 + 凹函数 = ELBO',
        body:
          '贝叶斯推断的核心困难：边缘似然 $p(x) = \\int p(x,z)\\,dz$ 高维时无法直接计算。\n\n' +
          '**三步推导 ELBO**（对任意辅助分布 $q(z)$）：\n\n' +
          '**第一步：引入辅助分布**\n\n' +
          '$\\ln p(x) = \\ln\\int q(z)\\dfrac{p(x,z)}{q(z)}\\,dz = \\ln\\,\\mathbb{E}_{q}\\!\\left[\\dfrac{p(x,z)}{q(z)}\\right]$\n\n' +
          '**第二步：Jensen 不等式**（$\\ln$ 是凹函数，Jensen 给出 $\\ln\\mathbb{E}[Y] \\ge \\mathbb{E}[\\ln Y]$）\n\n' +
          '$\\ln p(x) \\ge \\mathbb{E}_{q}\\!\\left[\\ln\\dfrac{p(x,z)}{q(z)}\\right] = \\mathbb{E}_q[\\ln p(x,z)] - \\mathbb{E}_q[\\ln q(z)]$\n\n' +
          '**第三步：分解**（代入 $p(x,z) = p(x|z)p(z)$）\n\n' +
          '$\\mathrm{ELBO} = \\mathbb{E}_q[\\ln p(x|z)] - \\mathrm{KL}(q(z)\\|p(z))$',
        formula:
          '\\ln p(x) \\ge \\underbrace{\\mathbb{E}_q[\\ln p(x|z)]}_{\\text{重构项}} - \\underbrace{\\mathrm{KL}(q(z)\\|p(z))}_{\\text{KL 正则项}} = \\mathrm{ELBO}',
        tip: '等号成立的条件：$q(z) = p(z|x)$（真实后验）。最大化 ELBO 等价于最小化 $\\mathrm{KL}(q\\|p(\\cdot|x))$。',
      },
      // 卡 6：前向 KL vs 反向 KL：mode-covering 与 mode-seeking
      {
        title: '前向 KL vs 反向 KL：两种近似策略',
        body:
          '**前向 KL**：$\\mathrm{KL}(p\\|q) = \\int p(x)\\ln\\dfrac{p(x)}{q(x)}\\,dx$\n\n' +
          '当 $p(x) > 0$ 而 $q(x) \\approx 0$ 时，$\\ln(p/q) \\to +\\infty$，贡献无穷大——' +
          'KL 会强烈惩罚"$p$ 有概率而 $q$ 没覆盖"的区域。结果：$q$ 倾向于覆盖 $p$ 的所有 mode（**mode-covering**）。\n\n' +
          '**反向 KL**：$\\mathrm{KL}(q\\|p) = \\int q(x)\\ln\\dfrac{q(x)}{p(x)}\\,dx$\n\n' +
          '当 $q(x) > 0$ 而 $p(x) \\approx 0$ 时，$\\ln(q/p) \\to +\\infty$——' +
          'KL 会强烈惩罚"$q$ 跑到 $p$ 的低概率区域"。结果：$q$ 倾向于集中在 $p$ 的某个 mode 上（**mode-seeking**）。\n\n' +
          'VAE 的 ELBO 中 KL 正则项最小化的是 $\\mathrm{KL}(q\\|p)$（反向 KL），因此编码器学会集中于先验的某个区域。',
        tip: '口诀：前向 KL"零避"（避免 $q=0$ 但 $p>0$）→ 覆盖所有 mode；反向 KL"质避"（避免 $q>0$ 但 $p=0$）→ 集中一个 mode。',
      },
      // 卡 7：易错点与 reveal
      {
        title: '易错点与自测',
        body:
          '**易错一**：$\\mathrm{KL}(p\\|q) \\ne \\mathrm{KL}(q\\|p)$，KL 散度**不对称**，交换方向结果不同，含义也不同。\n\n' +
          '**易错二**：最大化 ELBO $\\ne$ 最大化 $\\ln p(x)$——ELBO 只是下界，但最大化 ELBO 是最小化 $\\mathrm{KL}(q\\|p(\\cdot|x))$ 的等价形式。\n\n' +
          '**易错三**：重参数化技巧中，$\\varepsilon \\sim \\mathcal{N}(0,1)$ 必须**与参数 $\\mu, \\sigma$ 独立**。若写成 $z = \\mu + \\sigma\\varepsilon$ 后再对 $\\varepsilon$ 求期望，梯度才能对 $\\mu, \\sigma$ 反向传播。\n\n' +
          '**易错四**：变量替换时分子分母容易混淆——$f_Y(y) = f_X(g^{-1}(y)) \\cdot |dg^{-1}/dy|$，Jacobian 是 $g^{-1}$ 的导数绝对值，而不是 $g$ 的导数。',
        reveal: {
          q: '设 $p = \\mathcal{N}(0,1)$，$q = \\mathcal{N}(1,1)$（均值不同，方差相同）。$\\mathrm{KL}(p\\|q)$ 等于多少？',
          a: '用高斯 KL 公式：$\\ln(\\sigma_2/\\sigma_1) + (\\sigma_1^2 + (\\mu_1-\\mu_2)^2)/(2\\sigma_2^2) - 1/2$。代入 $\\mu_1=0,\\mu_2=1,\\sigma_1=\\sigma_2=1$：$\\ln 1 + (1+1)/2 - 1/2 = 0+1-1/2 = 1/2$。',
        },
      },
    ],
    questions: [
      // 题 1：judge — KL 散度非负
      {
        id: 'c30-b-q1',
        type: 'judge',
        prompt: '$\\mathrm{KL}(p\\|q)$ 永远大于或等于零，当且仅当 $p=q$ 时等于零。',
        answer: true,
        explain:
          'KL 散度的非负性由 Jensen 不等式推出：$\\ln$ 是凹函数，$-\\mathrm{KL}(p\\|q) = \\mathbb{E}_p[\\ln(q/p)] \\le \\ln\\mathbb{E}_p[q/p] = \\ln 1 = 0$，' +
          '故 $\\mathrm{KL}(p\\|q) \\ge 0$，等号当且仅当 $p=q$（a.e.）时成立。',
      },
      // 题 2：choice — KL 非对称
      {
        id: 'c30-b-q2',
        type: 'choice',
        prompt:
          '下列关于 KL 散度的说法，哪一条是**正确**的？',
        options: [
          '$\\mathrm{KL}(p\\|q) = \\mathrm{KL}(q\\|p)$ 恒成立',
          '$\\mathrm{KL}(p\\|q)$ 可以为负数',
          '$\\mathrm{KL}(p\\|q)$ 不对称，一般 $\\mathrm{KL}(p\\|q) \\ne \\mathrm{KL}(q\\|p)$',
          '$\\mathrm{KL}(p\\|q)=0$ 不能推出 $p=q$',
        ],
        answer: 2,
        explain:
          'KL 散度是非对称的——交换 $p$ 和 $q$ 通常结果不同，这使它不是真正意义上的"距离"（距离需满足对称性）。' +
          '它恒 $\\ge 0$（不能为负），且 $\\mathrm{KL}=0$ 当且仅当 $p=q$（a.e.）。',
      },
      // 题 3：input — 高斯 KL 计算
      {
        id: 'c30-b-q3',
        type: 'input',
        prompt:
          '设 $p = \\mathcal{N}(0,1)$，$q = \\mathcal{N}(1,1)$。利用高斯 KL 公式，$\\mathrm{KL}(p\\|q)$ 等于多少？（答案为 $0.5$）',
        accept: ['0.5', '1/2', '0.50'],
        explain:
          '高斯 KL 公式：$\\ln(\\sigma_2/\\sigma_1) + (\\sigma_1^2+(\\mu_1-\\mu_2)^2)/(2\\sigma_2^2) - 1/2$。' +
          '代入 $\\mu_1=0,\\mu_2=1,\\sigma_1=\\sigma_2=1$：$0 + (1+1)/2 - 1/2 = 1 - 0.5 = 0.5$。',
      },
      // 题 4：judge — 重参数化方向
      {
        id: 'c30-b-q4',
        type: 'judge',
        prompt:
          '重参数化技巧中，写 $z = \\mu + \\sigma\\varepsilon$（$\\varepsilon \\sim \\mathcal{N}(0,1)$），' +
          '其目的是使梯度能对参数 $\\mu$ 和 $\\sigma$ 反向传播。',
        answer: true,
        explain:
          '重参数化把随机性转移到与参数无关的 $\\varepsilon$ 上，使采样步骤变成确定的仿射变换 $z = \\mu + \\sigma\\varepsilon$。' +
          '期望 $\\mathbb{E}_{q}[f(z)] = \\mathbb{E}_{\\varepsilon}[f(\\mu+\\sigma\\varepsilon)]$ 对 $\\mu,\\sigma$ 可微，梯度可以反向传播。',
      },
      // 题 5：choice — ELBO 等号条件
      {
        id: 'c30-b-q5',
        type: 'choice',
        prompt:
          'ELBO 是 $\\ln p(x)$ 的下界。等号 $\\ln p(x) = \\mathrm{ELBO}$ 在什么条件下成立？',
        options: [
          '$q(z) = p(z)$（先验等于辅助分布）',
          '$q(z) = p(z|x)$（辅助分布等于真实后验）',
          '$p(x,z) = p(x)p(z)$（$x$ 与 $z$ 独立）',
          '$\\mathrm{KL}(q\\|p(z)) = 0$',
        ],
        answer: 1,
        explain:
          'ELBO 的推导中，$\\ln p(x) - \\mathrm{ELBO} = \\mathrm{KL}(q(z)\\|p(z|x))$。' +
          '等号成立 $\\Leftrightarrow$ KL 为零 $\\Leftrightarrow$ $q(z) = p(z|x)$（真实后验）。这正是变分推断的目标：让 $q$ 尽量逼近真实后验。',
      },
      // 题 6：match — 前向/反向 KL 特性
      {
        id: 'c30-b-q6',
        type: 'match',
        prompt: '将 KL 散度类型与其近似行为特征配对。',
        left: ['前向 KL：$\\mathrm{KL}(p\\|q)$', '反向 KL：$\\mathrm{KL}(q\\|p)$'],
        right: ['mode-covering（倾向覆盖所有 mode）', 'mode-seeking（倾向集中一个 mode）'],
      },
      // 题 7：choice — 变量替换 Jacobian
      {
        id: 'c30-b-q7',
        type: 'choice',
        prompt:
          '设 $X \\sim U[0,1]$，令 $Y = -\\ln X$。根据变量替换公式，$Y$ 服从什么分布？',
        options: [
          '$U[0,1]$',
          '$\\mathcal{N}(0,1)$',
          '$\\mathrm{Exp}(1)$（参数为 $1$ 的指数分布）',
          '$\\mathrm{Exp}(\\ln 2)$',
        ],
        answer: 2,
        explain:
          '$Y = -\\ln X$，逆变换 $X = e^{-Y}$，$dx/dy = -e^{-y}$，$|dx/dy| = e^{-y}$。' +
          '$f_Y(y) = f_X(e^{-y})\\cdot e^{-y} = 1\\cdot e^{-y}$（$y \\ge 0$），' +
          '即 $Y \\sim \\mathrm{Exp}(1)$。这是逆变换采样的经典示例。',
      },
    ],
  },
]
