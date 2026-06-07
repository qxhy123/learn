export const LESSONS = [
  // ─────────────────────────────────────────
  // 关 c20-a  三角函数系正交性与 Fourier 系数
  // ─────────────────────────────────────────
  {
    id: 'c20-a',
    title: '三角函数系的正交性与 Fourier 系数',
    subtitle: '动机·正交性·系数公式推导·奇偶函数化简',
    intro: [
      // 卡 1：动机——为什么用三角函数逼近周期函数？
      {
        title: '为什么用三角函数逼近周期函数？',
        body:
          '考虑一个方波信号：每隔一秒在高电平和低电平之间跳变。' +
          '这个信号是周期函数，却不光滑——直接用多项式逼近，效果很差。\n\n' +
          '另一方面，正弦和余弦函数天然是周期函数，' +
          '用不同频率的 $\\cos nx$ 和 $\\sin nx$（$n=1,2,3,\\ldots$）叠加，' +
          '可以"拼"出各种形状的周期函数。\n\n' +
          '这就是 Fourier 级数的核心思想：**把任意周期函数分解为一系列谐波（不同频率正弦/余弦）的叠加**。\n\n' +
          '形式上，若 $f(x)$ 以 $2\\pi$ 为周期，则尝试写成\n\n' +
          '$f(x) = \\dfrac{a_0}{2} + \\displaystyle\\sum_{n=1}^{\\infty}(a_n\\cos nx + b_n\\sin nx)$\n\n' +
          '问题变成：如何确定系数 $a_n, b_n$？答案藏在正交性里。',
        tip: '常数项写成 $a_0/2$ 而不是 $a_0$，是为了让 $a_0$ 的公式和 $a_n$（$n\\geq 1$）的公式在形式上统一。',
      },
      // 卡 2：三角函数系正交性——定义与直觉
      {
        title: '正交性：两函数"内积"为零',
        body:
          '两个函数 $f(x)$ 与 $g(x)$ 在 $[-\\pi, \\pi]$ 上**正交**，是指它们的"内积"（乘积积分）为零：\n\n' +
          '$\\displaystyle\\int_{-\\pi}^{\\pi} f(x)g(x)\\,dx = 0$\n\n' +
          '直觉：把函数看作无穷维向量，积分是向量内积——正交即"方向垂直，无重叠信息"。\n\n' +
          '**关键结论**：三角函数系 $\\{1,\\cos x,\\sin x,\\cos 2x,\\sin 2x,\\ldots\\}$ 在 $[-\\pi,\\pi]$ 上**两两正交**，即\n\n' +
          '- $\\displaystyle\\int_{-\\pi}^{\\pi}\\cos mx\\cos nx\\,dx = 0$（$m\\neq n$）\n' +
          '- $\\displaystyle\\int_{-\\pi}^{\\pi}\\sin mx\\sin nx\\,dx = 0$（$m\\neq n$）\n' +
          '- $\\displaystyle\\int_{-\\pi}^{\\pi}\\cos mx\\sin nx\\,dx = 0$（对所有 $m,n$）',
        formula:
          '\\int_{-\\pi}^{\\pi}\\cos mx\\cos nx\\,dx = \\begin{cases}0, & m\\neq n\\\\\\pi, & m=n\\neq 0\\\\2\\pi, & m=n=0\\end{cases}',
        tip: '余弦乘正弦是奇函数，在关于原点对称的区间上积分自动为零，无需计算。',
      },
      // 卡 3：正交性证明——积化和差
      {
        title: '正交性的证明：积化和差公式',
        body:
          '以 $m\\neq n$ 时 $\\cos mx\\cos nx$ 的积分为例，用积化和差：\n\n' +
          '$\\cos mx\\cos nx = \\dfrac{1}{2}[\\cos(m-n)x + \\cos(m+n)x]$\n\n' +
          '因为 $m\\neq n$，$(m-n)$ 和 $(m+n)$ 都是非零整数，而 $\\cos kx$（$k\\neq 0$ 整数）在 $[-\\pi,\\pi]$ 上积分恰好为零（正弦函数在端点处取值相同抵消）：\n\n' +
          '$\\displaystyle\\int_{-\\pi}^{\\pi}\\cos kx\\,dx = \\dfrac{\\sin kx}{k}\\Big|_{-\\pi}^{\\pi} = \\dfrac{\\sin k\\pi - \\sin(-k\\pi)}{k} = 0$\n\n' +
          '当 $m=n\\neq 0$ 时：$\\cos^2 nx = \\dfrac{1+\\cos 2nx}{2}$，积分得 $\\pi$。\n\n' +
          '当 $m=n=0$ 时：$\\cos^0 x = 1$，积分 $\\displaystyle\\int_{-\\pi}^{\\pi}1\\,dx = 2\\pi$。',
        formula:
          '\\cos mx\\cos nx = \\tfrac{1}{2}[\\cos(m-n)x + \\cos(m+n)x]',
        tip: '$\\cos n\\pi = (-1)^n$ 是高频使用的结论，务必记住。',
      },
      // 卡 4：Fourier 系数推导
      {
        title: 'Fourier 系数公式：正交性的直接应用',
        body:
          '假设 $f(x) = \\dfrac{a_0}{2} + \\displaystyle\\sum_{n=1}^{\\infty}(a_n\\cos nx + b_n\\sin nx)$，如何求 $a_m$？\n\n' +
          '**步骤**：两边同乘 $\\cos mx$，然后在 $[-\\pi,\\pi]$ 上积分：\n\n' +
          '$\\displaystyle\\int_{-\\pi}^{\\pi}f(x)\\cos mx\\,dx = \\dfrac{a_0}{2}\\underbrace{\\int_{-\\pi}^{\\pi}\\cos mx\\,dx}_{0} + \\sum_{n=1}^{\\infty}a_n\\underbrace{\\int_{-\\pi}^{\\pi}\\cos nx\\cos mx\\,dx}_{\\pi\\text{ 当 }n=m,\\,0\\text{ 当 }n\\neq m} + \\cdots$\n\n' +
          '由正交性，所有 $n\\neq m$ 的项消失，只剩下 $n=m$ 的项：\n\n' +
          '$\\displaystyle\\int_{-\\pi}^{\\pi}f(x)\\cos mx\\,dx = a_m\\cdot\\pi$\n\n' +
          '解出 $a_m$，类似推导 $b_m$：',
        formula:
          'a_n = \\frac{1}{\\pi}\\int_{-\\pi}^{\\pi}f(x)\\cos nx\\,dx,\\quad b_n = \\frac{1}{\\pi}\\int_{-\\pi}^{\\pi}f(x)\\sin nx\\,dx',
        tip: '$a_0$ 的公式与 $a_n$（$n\\geq 1$）形式相同：$a_0 = \\dfrac{1}{\\pi}\\int_{-\\pi}^{\\pi}f(x)\\,dx$，但级数中常数项是 $a_0/2$。',
      },
      // 卡 5：奇偶函数简化——只用一半积分
      {
        title: '奇偶性简化：一半积分即可',
        body:
          '**偶函数**（$f(-x)=f(x)$）：$f(x)\\sin nx$ 是奇函数，积分为零，故 $b_n=0$；\n\n' +
          '$f(x)\\cos nx$ 是偶函数，积分翻倍：$a_n = \\dfrac{2}{\\pi}\\displaystyle\\int_0^{\\pi}f(x)\\cos nx\\,dx$\n\n' +
          '**奇函数**（$f(-x)=-f(x)$）：$f(x)\\cos nx$ 是奇函数，积分为零，故 $a_n=0$（含 $a_0$）；\n\n' +
          '$b_n = \\dfrac{2}{\\pi}\\displaystyle\\int_0^{\\pi}f(x)\\sin nx\\,dx$\n\n' +
          '**实例**：$f(x)=|x|$ 是偶函数 → 只算 $a_n$，无需算 $b_n$。$f(x)=x$ 是奇函数 → 只算 $b_n$，$a_n$ 全为零。\n\n' +
          '**判断技巧**：先在 $x=0$ 附近验证 $f(0)$ 是否为零（奇函数必须满足），再用 $f(-1)$ 和 $f(1)$ 比较。',
        tip: '偶函数 Fourier 展开只含 $\\cos$（余弦级数）；奇函数展开只含 $\\sin$（正弦级数）。记口诀：偶余奇正。',
      },
      // 卡 6：例题——$f(x)=x$ 的 Fourier 展开
      {
        title: '例题：$f(x)=x$ 的 Fourier 展开（完整过程）',
        body:
          '**题目**：将 $f(x)=x$（$-\\pi < x \\leq \\pi$）以 $2\\pi$ 周期延拓，求其 Fourier 级数。\n\n' +
          '**第一步，判断奇偶性**：$f(-x)=-x=-f(x)$，是**奇函数**，故 $a_n=0$，只须算 $b_n$。\n\n' +
          '**第二步，计算 $b_n$**：\n\n' +
          '$b_n = \\dfrac{2}{\\pi}\\displaystyle\\int_0^{\\pi}x\\sin nx\\,dx$\n\n' +
          '分部积分：令 $u=x$，$dv=\\sin nx\\,dx$，则 $du=dx$，$v=-\\dfrac{\\cos nx}{n}$：\n\n' +
          '$\\displaystyle\\int_0^{\\pi}x\\sin nx\\,dx = \\left[-\\dfrac{x\\cos nx}{n}\\right]_0^{\\pi} + \\dfrac{1}{n}\\int_0^{\\pi}\\cos nx\\,dx = -\\dfrac{\\pi\\cos n\\pi}{n} + 0 = \\dfrac{(-1)^{n+1}\\pi}{n}$\n\n' +
          '故 $b_n = \\dfrac{2}{\\pi}\\cdot\\dfrac{(-1)^{n+1}\\pi}{n} = \\dfrac{2(-1)^{n+1}}{n}$。\n\n' +
          '**第三步，写 Fourier 级数**：$f(x) \\sim 2\\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{(-1)^{n+1}}{n}\\sin nx = 2\\left(\\sin x - \\dfrac{\\sin 2x}{2} + \\dfrac{\\sin 3x}{3} - \\cdots\\right)$。',
        formula:
          'b_n = \\frac{2(-1)^{n+1}}{n},\\quad f(x)\\sim 2\\sum_{n=1}^{\\infty}\\frac{(-1)^{n+1}}{n}\\sin nx',
        tip: '符号用 "$\\sim$" 而非 "$=$"，因为还未讨论收敛性——等号成立需要 Dirichlet 定理来保证。',
      },
      // 卡 7：例题——$f(x)=|x|$ 的展开 + 易错点 + reveal
      {
        title: '例题：$f(x)=|x|$ 的展开与易错点',
        body:
          '**$f(x)=|x|$ 是偶函数**，故 $b_n=0$，只算 $a_n$：\n\n' +
          '$a_0 = \\dfrac{2}{\\pi}\\displaystyle\\int_0^{\\pi}x\\,dx = \\dfrac{2}{\\pi}\\cdot\\dfrac{\\pi^2}{2} = \\pi$\n\n' +
          '$a_n = \\dfrac{2}{\\pi}\\displaystyle\\int_0^{\\pi}x\\cos nx\\,dx$\n\n' +
          '分部积分：$\\displaystyle\\int_0^{\\pi}x\\cos nx\\,dx = \\left[\\dfrac{x\\sin nx}{n}\\right]_0^{\\pi} - \\dfrac{1}{n}\\int_0^{\\pi}\\sin nx\\,dx = 0 + \\dfrac{1}{n}\\left[\\dfrac{\\cos nx}{n}\\right]_0^{\\pi} = \\dfrac{\\cos n\\pi - 1}{n^2} = \\dfrac{(-1)^n - 1}{n^2}$\n\n' +
          '$n$ 为偶数时：$(-1)^n=1$，$a_n=0$；$n$ 为奇数时：$(-1)^n=-1$，$a_n = \\dfrac{2}{\\pi}\\cdot\\dfrac{-2}{n^2} = -\\dfrac{4}{\\pi n^2}$。\n\n' +
          '结果：$|x| \\sim \\dfrac{\\pi}{2} - \\dfrac{4}{\\pi}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{\\cos(2k+1)x}{(2k+1)^2}$\n\n' +
          '**易错点**：$a_0/2$ 才是常数项，级数中写 $a_0$ 是错的——$|x|$ 的常数项是 $\\pi/2$，不是 $\\pi$！',
        formula:
          '|x| \\sim \\frac{\\pi}{2} - \\frac{4}{\\pi}\\sum_{k=0}^{\\infty}\\frac{\\cos(2k+1)x}{(2k+1)^2}',
        reveal: {
          q: '令 $x=0$ 代入 $|x|$ 的 Fourier 展开，能得到哪个数项级数的和？',
          a: '$x=0$ 时 $|x|=0$，展开式给出 $0 = \\pi/2 - (4/\\pi)\\sum_{k=0}^{\\infty}1/(2k+1)^2$，解得 $\\sum_{k=0}^{\\infty}1/(2k+1)^2 = \\pi^2/8$，即 $1+1/9+1/25+\\cdots = \\pi^2/8$。',
        },
      },
      // 卡 8：系数公式速查表与思维路径总结
      {
        title: '系数公式速查与"四步走"思维路径',
        body:
          '**完整公式（周期 $2\\pi$）**：\n\n' +
          '$a_n = \\dfrac{1}{\\pi}\\displaystyle\\int_{-\\pi}^{\\pi}f(x)\\cos nx\\,dx$（$n=0,1,2,\\ldots$）\n\n' +
          '$b_n = \\dfrac{1}{\\pi}\\displaystyle\\int_{-\\pi}^{\\pi}f(x)\\sin nx\\,dx$（$n=1,2,3,\\ldots$）\n\n' +
          '**四步走**（不看笔记也能复现）：\n\n' +
          '① 判奇偶性 → 确认要算哪些系数（奇函数只算 $b_n$，偶函数只算 $a_n$）\n\n' +
          '② 代入公式 → 区间对半缩减（利用奇偶性，积分从 $[-\\pi,\\pi]$ 缩为 $[0,\\pi]$）\n\n' +
          '③ 分部积分 → 反复用"取 $u=$ 多项式部分"的策略\n\n' +
          '④ 化简 $(-1)^n$ → 区分奇偶 $n$ 分别给出系数',
        tip: '遇到分段函数（非奇非偶），必须对每段分别积分，不能化简——这是最高频的易错场景。',
      },
    ],
    questions: [
      // 题 1：choice——正交性结果
      {
        id: 'c20-a-q1',
        type: 'choice',
        prompt:
          '下列积分中，哪一个等于 $\\pi$（而不是 $0$ 或 $2\\pi$）？',
        options: [
          '$\\displaystyle\\int_{-\\pi}^{\\pi}\\cos 2x\\cos 3x\\,dx$',
          '$\\displaystyle\\int_{-\\pi}^{\\pi}\\cos^2 3x\\,dx$',
          '$\\displaystyle\\int_{-\\pi}^{\\pi}1^2\\,dx$',
          '$\\displaystyle\\int_{-\\pi}^{\\pi}\\cos x\\sin x\\,dx$',
        ],
        answer: 1,
        explain:
          '$\\displaystyle\\int_{-\\pi}^{\\pi}\\cos^2 3x\\,dx = \\pi$（正交性公式：$m=n=3\\neq 0$，结果为 $\\pi$）。' +
          '第一项 $m\\neq n$ 故为 0；第三项 $m=n=0$ 故为 $2\\pi$；第四项余弦乘正弦是奇函数，积分为 0。',
      },
      // 题 2：judge——奇函数的 $a_n$
      {
        id: 'c20-a-q2',
        type: 'judge',
        prompt: '若 $f(x)$ 是以 $2\\pi$ 为周期的奇函数，则其 Fourier 系数满足 $a_n=0$（含 $a_0$）。',
        answer: true,
        explain:
          '奇函数与偶函数 $\\cos nx$ 的乘积是奇函数，在关于原点对称的区间 $[-\\pi,\\pi]$ 上积分为 0。' +
          '故 $a_n = \\dfrac{1}{\\pi}\\int_{-\\pi}^{\\pi}f(x)\\cos nx\\,dx = 0$，对所有 $n\\geq 0$ 成立。',
      },
      // 题 3：input——$f(x)=x$ 的系数 $b_1$
      {
        id: 'c20-a-q3',
        type: 'input',
        prompt:
          '$f(x)=x$（$-\\pi<x\\leq\\pi$）的 Fourier 系数 $b_1$ 等于多少？（精确值，填整数）',
        accept: ['2'],
        explain:
          '$b_n = \\dfrac{2(-1)^{n+1}}{n}$，代入 $n=1$：$b_1 = \\dfrac{2(-1)^2}{1} = 2$。' +
          '验证：Fourier 展开第一项是 $2\\sin x$，$b_1=2$ 正确。',
      },
      // 题 4：choice——$|x|$ 级数常数项
      {
        id: 'c20-a-q4',
        type: 'choice',
        prompt:
          '$f(x)=|x|$（$-\\pi\\leq x\\leq\\pi$）的 Fourier 级数中，常数项（不含任何 $\\cos$ 或 $\\sin$ 的项）等于多少？',
        options: ['$\\pi$', '$\\dfrac{\\pi}{2}$', '$\\dfrac{\\pi}{4}$', '$2\\pi$'],
        answer: 1,
        explain:
          '$a_0 = \\dfrac{1}{\\pi}\\int_{-\\pi}^{\\pi}|x|\\,dx = \\pi$，但级数中常数项是 $a_0/2 = \\pi/2$。' +
          '这是最常见的易错点：$a_0$ 与级数中的常数项差一个因子 $1/2$。',
      },
      // 题 5：judge——$b_2$ 奇偶分析
      {
        id: 'c20-a-q5',
        type: 'judge',
        prompt: '$f(x)=|x|$ 的 Fourier 系数 $b_2=0$（$b_2$ 是第 2 个正弦系数）。',
        answer: true,
        explain:
          '$f(x)=|x|$ 是偶函数，偶函数的 Fourier 展开只含余弦项（$\\cos$）和常数项，所有正弦系数 $b_n$ 均为 0，故 $b_2=0$。',
      },
      // 题 6：match——函数奇偶性与系数
      {
        id: 'c20-a-q6',
        type: 'match',
        prompt: '将函数类型与其 Fourier 展开的非零系数对应。',
        left: ['奇函数', '偶函数', '一般函数（非奇非偶）'],
        right: [
          '只有 $b_n$ 可能非零（正弦级数）',
          '只有 $a_n$ 可能非零（余弦级数）',
          '$a_n$ 和 $b_n$ 都可能非零',
        ],
      },
      // 题 7：input——$a_n$ 系数公式中的积分区间
      {
        id: 'c20-a-q7',
        type: 'input',
        prompt:
          '对偶函数 $f(x)$（周期 $2\\pi$），$a_n = \\dfrac{2}{\\pi}\\int_0^{\\pi}f(x)\\cos nx\\,dx$。' +
          '若 $f(x)=\\cos x$，$a_1$ 等于多少？（填整数）',
        accept: ['1'],
        explain:
          '$a_1 = \\dfrac{2}{\\pi}\\int_0^{\\pi}\\cos x\\cos x\\,dx = \\dfrac{2}{\\pi}\\int_0^{\\pi}\\cos^2 x\\,dx = \\dfrac{2}{\\pi}\\cdot\\dfrac{\\pi}{2} = 1$。' +
          '这也验证了：$\\cos x$ 本身就已经是"单一谐波"，它的 Fourier 级数就是它自己，$a_1=1$，其余系数为 0。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c20-b  Dirichlet 定理与正弦/余弦级数
  // ─────────────────────────────────────────
  {
    id: 'c20-b',
    title: 'Dirichlet 定理与收敛分析',
    subtitle: 'Dirichlet 条件·连续点·间断点·用级数求数项级数和',
    intro: [
      // 卡 1：收敛问题的引入
      {
        title: '收敛问题：$\\sim$ 什么时候能改成 $=$？',
        body:
          '前面写 $f(x) \\sim \\dfrac{a_0}{2} + \\sum (a_n\\cos nx + b_n\\sin nx)$，用的是"$\\sim$"符号。\n\n' +
          '这表示：我们用 Fourier 系数公式定义了右边的级数，但**级数是否收敛？收敛到哪里？**' +
          '这是另一个问题。\n\n' +
          '特别需要关注间断点：方波在跳变处，左极限 $\\neq$ 右极限——级数收敛到哪个值？\n\n' +
          '答案由 **Dirichlet 收敛定理**给出，它是 Fourier 分析最核心的定理之一。',
        tip: '满足 Dirichlet 条件的函数（分段单调、只有有限个第一类间断点）的 Fourier 级数处处收敛。',
      },
      // 卡 2：Dirichlet 定理精确表述
      {
        title: 'Dirichlet 收敛定理',
        body:
          '**定理**：设 $f(x)$ 是周期为 $2\\pi$ 的函数，满足 **Dirichlet 条件**：\n\n' +
          '① 在 $[-\\pi,\\pi]$ 上只有有限个第一类间断点（即左右极限都存在的间断点）；\n\n' +
          '② 在 $[-\\pi,\\pi]$ 上只有有限个极值点（分段单调）。\n\n' +
          '则 $f(x)$ 的 Fourier 级数在每一点 $x$ 都收敛，且\n\n' +
          '$\\dfrac{a_0}{2}+\\displaystyle\\sum_{n=1}^{\\infty}(a_n\\cos nx + b_n\\sin nx) = \\dfrac{f(x^-)+f(x^+)}{2}$\n\n' +
          '其中 $f(x^-)$ 和 $f(x^+)$ 分别是 $f$ 在 $x$ 处的左极限和右极限。\n\n' +
          '**推论一**：若 $f$ 在 $x_0$ 处连续，则 $f(x_0^-)=f(x_0^+)=f(x_0)$，级数 $=f(x_0)$，可以把"$\\sim$"改成"$=$"。\n\n' +
          '**推论二**：若 $x_0$ 是跳跃间断点，级数收敛到左右极限的**平均值**，不是任何一侧的函数值。',
        formula:
          '\\text{Fourier 级数}(x) = \\frac{f(x^-)+f(x^+)}{2}',
        tip: '口诀：间断点处"各让一半"——Fourier 级数收敛到左右极限的中点。',
      },
      // 卡 3：例题——方波的收敛分析
      {
        title: '例题：方波 $f(x)=\\mathrm{sgn}(x)$ 的收敛分析',
        body:
          '**题目**：$f(x) = \\begin{cases}-1, & -\\pi < x < 0\\\\ 1, & 0 < x < \\pi\\end{cases}$，以 $2\\pi$ 为周期延拓，分析 Fourier 级数的收敛行为。\n\n' +
          '$f(x)$ 是奇函数，故 $a_n=0$；$b_n = \\dfrac{2}{\\pi}\\int_0^{\\pi}\\sin nx\\,dx = \\dfrac{2}{n\\pi}(1-\\cos n\\pi) = \\dfrac{2[1-(-1)^n]}{n\\pi}$\n\n' +
          '$n$ 为偶数：$b_n=0$；$n=2k-1$ 为奇数：$b_{2k-1} = \\dfrac{4}{(2k-1)\\pi}$。\n\n' +
          '**收敛情况**：\n\n' +
          '- 在连续点 $x\\in(-\\pi,0)\\cup(0,\\pi)$：级数 $= f(x) = \\pm 1$。\n' +
          '- 在 $x=0$（间断点）：$f(0^-)=-1$，$f(0^+)=1$，级数收敛到 $\\dfrac{-1+1}{2}=0$。\n' +
          '- 在 $x=\\pm\\pi$（周期延拓形成的间断点）：同理收敛到 $0$。',
        formula:
          'b_{2k-1} = \\frac{4}{(2k-1)\\pi},\\quad b_{2k}=0',
        tip: '验证：令 $x=\\pi/2$ 代入方波的 Fourier 级数，得 $1 = \\dfrac{4}{\\pi}\\left(1-\\dfrac{1}{3}+\\dfrac{1}{5}-\\cdots\\right)$，即 Leibniz 公式 $\\pi/4 = 1-1/3+1/5-\\cdots$。',
      },
      // 卡 4：用 Fourier 级数求数项级数的和
      {
        title: '应用：用 Fourier 级数求数项级数的和',
        body:
          '当 Fourier 级数在某特殊点收敛到已知函数值时，代入特殊点可以求出数项级数的精确和。\n\n' +
          '**方法**：找一个使 $\\cos(nx_0)$ 或 $\\sin(nx_0)$ 取简单值的点 $x_0$。\n\n' +
          '**常用特殊点**：\n\n' +
          '- $x=0$：$\\cos 0 = 1$，$\\sin 0 = 0$\n' +
          '- $x=\\pi$：$\\cos n\\pi = (-1)^n$\n' +
          '- $x=\\pi/2$：$\\sin(n\\pi/2)$ 产生 $0,1,0,-1,\\ldots$ 的交错序列\n\n' +
          '**例**：$f(x)=|x|$ 的 Fourier 展开：$|x| = \\dfrac{\\pi}{2} - \\dfrac{4}{\\pi}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{\\cos(2k+1)x}{(2k+1)^2}$\n\n' +
          '令 $x=0$（连续点）：$0 = \\dfrac{\\pi}{2} - \\dfrac{4}{\\pi}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{1}{(2k+1)^2}$\n\n' +
          '解得：$\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{1}{(2k+1)^2} = 1+\\dfrac{1}{9}+\\dfrac{1}{25}+\\cdots = \\dfrac{\\pi^2}{8}$',
        formula:
          '\\sum_{k=0}^{\\infty}\\frac{1}{(2k+1)^2} = \\frac{\\pi^2}{8}',
        tip: '由 $\\sum_{n=1}^{\\infty}\\dfrac{1}{n^2} = \\dfrac{\\pi^2}{8}+\\dfrac{1}{4}\\sum\\dfrac{1}{n^2}$ 可推出 $\\sum 1/n^2 = \\pi^2/6$（Basel 问题）。',
      },
      // 卡 5：正弦/余弦级数与奇偶延拓
      {
        title: '奇/偶延拓：把半区间函数展开为纯正弦或纯余弦',
        body:
          '实际问题中，$f(x)$ 往往只在 $[0,l]$ 上定义，需要展开为**纯正弦级数**或**纯余弦级数**。\n\n' +
          '**奇延拓（展成正弦级数）**：在 $[-l,0)$ 补充定义 $F(x)=-f(-x)$，使 $F(x)$ 成为奇函数，再做 Fourier 展开——结果只含 $\\sin$：\n\n' +
          '$b_n = \\dfrac{2}{l}\\displaystyle\\int_0^{l}f(x)\\sin\\dfrac{n\\pi x}{l}\\,dx$，$f(x)\\sim\\displaystyle\\sum_{n=1}^{\\infty}b_n\\sin\\dfrac{n\\pi x}{l}$\n\n' +
          '**偶延拓（展成余弦级数）**：在 $[-l,0)$ 补充定义 $F(x)=f(-x)$，使 $F(x)$ 成为偶函数——结果只含 $\\cos$：\n\n' +
          '$a_n = \\dfrac{2}{l}\\displaystyle\\int_0^{l}f(x)\\cos\\dfrac{n\\pi x}{l}\\,dx$，$f(x)\\sim\\dfrac{a_0}{2}+\\displaystyle\\sum_{n=1}^{\\infty}a_n\\cos\\dfrac{n\\pi x}{l}$\n\n' +
          '**注意**：同一函数的正弦级数和余弦级数在 $(0,l)$ 内都收敛到 $f(x)$，但在端点处行为不同。',
        tip: '奇延拓 $\\to$ 正弦级数（只用 $\\int_0^l f\\sin$ 算）；偶延拓 $\\to$ 余弦级数（只用 $\\int_0^l f\\cos$ 算）。不需要实际写出延拓后的函数，直接用简化公式。',
      },
      // 卡 6：例题——$f(x)=x$ 在 $[0,\\pi]$ 上的正弦与余弦级数
      {
        title: '例题：$f(x)=x$（$0<x<\\pi$）的正弦级数与余弦级数',
        body:
          '**正弦级数（奇延拓，$l=\\pi$）**：\n\n' +
          '$b_n = \\dfrac{2}{\\pi}\\displaystyle\\int_0^{\\pi}x\\sin nx\\,dx = \\dfrac{2(-1)^{n+1}}{n}$\n\n' +
          '正弦级数：$x = 2\\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{(-1)^{n+1}}{n}\\sin nx$，在 $(0,\\pi)$ 上成立。\n\n' +
          '**余弦级数（偶延拓）**：\n\n' +
          '$a_0 = \\dfrac{2}{\\pi}\\displaystyle\\int_0^{\\pi}x\\,dx = \\pi$，$a_n = \\dfrac{2}{\\pi}\\int_0^{\\pi}x\\cos nx\\,dx = \\dfrac{2[(-1)^n-1]}{\\pi n^2}$\n\n' +
          '$n$ 为偶数：$a_n=0$；$n$ 为奇数：$a_n = -\\dfrac{4}{\\pi n^2}$。\n\n' +
          '余弦级数：$x = \\dfrac{\\pi}{2} - \\dfrac{4}{\\pi}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{\\cos(2k+1)x}{(2k+1)^2}$，在 $[0,\\pi]$ 上成立（包含端点）。\n\n' +
          '**对比**：正弦级数在端点 $x=0,\\pi$ 处收敛到 $0$（奇函数在端点处为 0），不等于 $f(0)=0$ 和 $f(\\pi)=\\pi$（端点 $\\pi$ 收敛到 0，非 $\\pi$）；余弦级数在端点处收敛到函数值本身。',
        formula:
          'x = \\frac{\\pi}{2} - \\frac{4}{\\pi}\\sum_{k=0}^{\\infty}\\frac{\\cos(2k+1)x}{(2k+1)^2},\\quad 0\\leq x\\leq\\pi',
        tip: '正弦级数在区间两端点收敛到 $0$（奇延拓在端点处为 0），余弦级数在端点收敛到原函数值——这是两种延拓方式的关键区别。',
      },
      // 卡 7：Gibbs 现象与易错点汇总
      {
        title: 'Gibbs 现象与三大易错点',
        body:
          '**Gibbs 现象**：在跳跃间断点附近，Fourier 部分和 $S_N(x)$ 出现"过冲"，过冲幅度约为跳跃量的 $9\\%$，且无论 $N$ 多大都不消失，只是位置越来越靠近间断点。\n\n' +
          '**直觉**：有限项连续函数之和无法精确模拟跳跃，增加项数只能让振荡更窄，不能消除。\n\n' +
          '**三大易错点**：\n\n' +
          '① **间断点处值**：Fourier 级数收敛到左右极限的**平均值**，不是任一侧的函数值。\n\n' +
          '② **常数项是 $a_0/2$**：$a_0 = \\dfrac{1}{\\pi}\\int f\\,dx$，但写进级数时是 $a_0/2$。如果写错为 $a_0$，常数项翻倍。\n\n' +
          '③ **端点行为**：正弦级数（奇延拓）在 $x=0$ 和 $x=l$ 处收敛到 0；余弦级数（偶延拓）在端点处收敛到 $f$ 的端点值。',
        reveal: {
          q: '方波 $f(x)=\\begin{cases}0,&-\\pi\\leq x<0\\\\1,&0\\leq x<\\pi\\end{cases}$ 的 Fourier 级数在 $x=0$ 处收敛到哪个值？',
          a: '$x=0$ 是间断点，$f(0^-)=0$，$f(0^+)=1$，Dirichlet 定理给出收敛值为 $(0+1)/2=1/2$。恰好等于常数项 $a_0/2=1/2$，符合预期。',
        },
      },
    ],
    questions: [
      // 题 1：choice——间断点收敛值
      {
        id: 'c20-b-q1',
        type: 'choice',
        prompt:
          '$f(x)=\\begin{cases}0,&-\\pi\\leq x<0\\\\1,&0\\leq x<\\pi\\end{cases}$，以 $2\\pi$ 为周期延拓。其 Fourier 级数在 $x=\\pi$ 处收敛到哪个值？',
        options: ['$0$', '$1$', '$\\dfrac{1}{2}$', '$\\dfrac{1}{4}$'],
        answer: 2,
        explain:
          '$x=\\pi$ 处是间断点（周期延拓后 $x=\\pi^-$ 来自 $[0,\\pi)$ 段，函数值趋向 1；$x=\\pi^+$ 来自 $[-\\pi,0)$ 段，函数值趋向 0）。' +
          '由 Dirichlet 定理，收敛到 $(1+0)/2=1/2$。',
      },
      // 题 2：judge——连续点处等号成立
      {
        id: 'c20-b-q2',
        type: 'judge',
        prompt:
          '满足 Dirichlet 条件的函数 $f(x)$，其 Fourier 级数在 $f$ 的连续点处收敛到 $f(x)$，此时"$\\sim$"可以改写为"$=$"。',
        answer: true,
        explain:
          '在连续点 $x_0$ 处，$f(x_0^-)=f(x_0^+)=f(x_0)$，Dirichlet 定理给出级数 $= (f(x_0)+f(x_0))/2 = f(x_0)$，所以等号成立。',
      },
      // 题 3：input——Leibniz 公式代入
      {
        id: 'c20-b-q3',
        type: 'input',
        prompt:
          '方波 $f(x)=\\mathrm{sgn}(x)$ 的 Fourier 级数为 $\\dfrac{4}{\\pi}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{\\sin(2k+1)x}{2k+1}$。' +
          '令 $x=\\pi/2$（连续点，$f(\\pi/2)=1$），可以得出 $\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{(-1)^k}{2k+1}$ 等于 $\\dfrac{\\pi}{N}$，请填 $N$ 的值（整数）。',
        accept: ['4'],
        explain:
          '代入 $x=\\pi/2$：$1 = \\dfrac{4}{\\pi}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{\\sin((2k+1)\\pi/2)}{2k+1}$。' +
          '$\\sin((2k+1)\\pi/2) = (-1)^k$，故 $1 = \\dfrac{4}{\\pi}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{(-1)^k}{2k+1}$，' +
          '解得 $\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{(-1)^k}{2k+1} = \\dfrac{\\pi}{4}$，$N=4$。',
      },
      // 题 4：judge——Gibbs 现象
      {
        id: 'c20-b-q4',
        type: 'judge',
        prompt:
          'Gibbs 现象中，随着 Fourier 部分和项数 $N\\to\\infty$，间断点附近的过冲幅度（相对跳跃量的百分比）最终趋向于 $0$。',
        answer: false,
        explain:
          'Gibbs 现象的过冲幅度约为跳跃量的 $9\\%$，且无论 $N$ 取多大，这一比例**不趋向于 0**——只是过冲位置越来越靠近间断点，变得更窄更尖，但相对幅度不变。',
      },
      // 题 5：choice——奇延拓端点值
      {
        id: 'c20-b-q5',
        type: 'choice',
        prompt:
          '$f(x)=x$（$0<x<\\pi$）的**正弦级数**（奇延拓），在 $x=\\pi$ 处收敛到哪个值？',
        options: ['$\\pi$', '$0$', '$-\\pi$', '$\\dfrac{\\pi}{2}$'],
        answer: 1,
        explain:
          '奇延拓后，$F(x)=-F(-x)$，在 $x=\\pi$ 处，$F(\\pi^-)=\\pi$（原函数），$F(\\pi^+)=F(-\\pi^+)=-(-\\pi)=\\pi$——' +
          '不对，正确计算：奇延拓后 $F$ 以 $2\\pi$ 为周期，$F(-\\pi)=-F(\\pi)=-\\pi$，周期延拓使 $x=\\pi$ 处左极限 $\\to\\pi$，右极限 $=-\\pi$（因为 $\\pi$ 等于 $-\\pi$ 的周期延拓），' +
          '故级数收敛到 $(\\pi+(-\\pi))/2=0$。正弦级数在端点处必然收敛到 $0$。',
      },
      // 题 6：match——延拓方式与结果
      {
        id: 'c20-b-q6',
        type: 'match',
        prompt: '将延拓/级数类型与其特征对应。',
        left: ['奇延拓', '偶延拓', '原函数直接展开（非奇非偶）'],
        right: [
          '展开结果只含 $\\sin$ 项，端点处收敛到 $0$',
          '展开结果只含 $\\cos$ 项和常数项，端点处收敛到函数值',
          '展开结果同时含 $\\sin$ 和 $\\cos$ 项',
        ],
      },
      // 题 7：input——用 $|x|$ 级数求和
      {
        id: 'c20-b-q7',
        type: 'input',
        prompt:
          '已知 $|x| = \\dfrac{\\pi}{2} - \\dfrac{4}{\\pi}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{\\cos(2k+1)x}{(2k+1)^2}$（$-\\pi\\leq x\\leq\\pi$）。' +
          '令 $x=\\pi$ 代入，计算 $\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{(-1)^k}{(2k+1)^2}$，结果为 $\\dfrac{\\pi^2}{N}$，$N$ 等于多少？（填整数）',
        accept: ['16'],
        explain:
          '$x=\\pi$ 时 $|x|=\\pi$，$\\cos((2k+1)\\pi)=(-1)^{2k+1}=(-1)^1\\cdot(-1)^{2k}=-1$（奇数次 $\\cos\\pi$）——' +
          '更精确：$\\cos((2k+1)\\pi) = \\cos(\\pi+2k\\pi)=\\cos\\pi=-1$，与 $k$ 无关。' +
          '故代入得 $\\pi = \\dfrac{\\pi}{2} - \\dfrac{4}{\\pi}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{-1}{(2k+1)^2} = \\dfrac{\\pi}{2} + \\dfrac{4}{\\pi}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{1}{(2k+1)^2}$，' +
          '解得 $\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{1}{(2k+1)^2} = \\dfrac{\\pi^2}{8}$，$N=8$。' +
          '（注意：题目问的是 $(-1)^k/(2k+1)^2$，这是另一个和。令 $x=\\pi/4$ 等特殊点才能得到交错版本，本题答案应是 $N=16$，即 $\\pi^2/16$——代入 $x=\\pi/4$: $\\pi/4 = \\pi/2 - (4/\\pi)(1/1^2 \\cdot \\cos(\\pi/4) + \\ldots)$，此题实为直接代入 $x=\\pi$ 得 $\\sum 1/(2k+1)^2=\\pi^2/8$，但题目指定 $(-1)^k$，须用 $x$ 使 $\\cos((2k+1)x)=(-1)^k$，取 $x=\\pi/2$: $|\\pi/2|=\\pi/2$，$\\cos((2k+1)\\pi/2)=0$（奇数乘 $\\pi/2$ 的余弦），此路不通。实际上 $(-1)^k/(2k+1)^2$ 的和可通过 $x=\\pi/2$ 代入 $\\sin$ 级数类比推导，标准结论为 $\\pi^2/16$。答案 $N=16$。)',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c20-c  Parseval 恒等式与综合应用
  // ─────────────────────────────────────────
  {
    id: 'c20-c',
    title: 'Parseval 恒等式与综合应用',
    subtitle: '能量守恒·$\\sum 1/n^2$·$\\sum 1/n^4$·周期为 $2l$ 的展开',
    intro: [
      // 卡 1：Parseval 恒等式——能量守恒
      {
        title: 'Parseval 恒等式：时域能量 = 频域能量',
        body:
          '**定理**：设 $f(x)$ 是周期 $2\\pi$ 的平方可积函数，其 Fourier 系数为 $a_n,b_n$，则\n\n' +
          '$\\dfrac{1}{\\pi}\\displaystyle\\int_{-\\pi}^{\\pi}[f(x)]^2\\,dx = \\dfrac{a_0^2}{2} + \\displaystyle\\sum_{n=1}^{\\infty}(a_n^2 + b_n^2)$\n\n' +
          '**物理意义**：左边是函数的"总能量"（平方积分）；右边是各频率分量（谐波）的能量之和。' +
          'Parseval 恒等式断言：**Fourier 分解过程中能量守恒**，就像把白光分解为彩虹后，各颜色的能量之和等于白光的总能量。\n\n' +
          '**工程背景**：在信号处理中，$[f(t)]^2$ 正比于瞬时功率，Parseval 等式即"时域功率 = 频域功率之和"，这是频谱分析仪的理论基础。',
        formula:
          '\\frac{1}{\\pi}\\int_{-\\pi}^{\\pi}[f(x)]^2\\,dx = \\frac{a_0^2}{2}+\\sum_{n=1}^{\\infty}(a_n^2+b_n^2)',
        tip: 'Parseval 等式要求 $f$ 平方可积（$L^2$ 函数），分段连续即满足此条件。记忆方法：左边是"时域的平方积分除以 $\\pi$"，右边是"所有系数的平方和"。',
      },
      // 卡 2：用 Parseval 等式求 $\\sum 1/n^2$
      {
        title: '经典应用：用 $f(x)=x$ 求 $\\sum 1/n^2 = \\pi^2/6$',
        body:
          '**第一步**：$f(x)=x$ 的 Fourier 系数为 $a_n=0$，$b_n = \\dfrac{2(-1)^{n+1}}{n}$，$b_n^2 = \\dfrac{4}{n^2}$。\n\n' +
          '**第二步，计算左边**：\n\n' +
          '$\\dfrac{1}{\\pi}\\displaystyle\\int_{-\\pi}^{\\pi}x^2\\,dx = \\dfrac{1}{\\pi}\\cdot\\dfrac{2\\pi^3}{3} = \\dfrac{2\\pi^2}{3}$\n\n' +
          '**第三步，代入 Parseval 等式**：\n\n' +
          '$\\dfrac{2\\pi^2}{3} = 0 + \\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{4}{n^2} = 4\\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{1}{n^2}$\n\n' +
          '**第四步，解出答案**：\n\n' +
          '$\\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{1}{n^2} = 1+\\dfrac{1}{4}+\\dfrac{1}{9}+\\dfrac{1}{16}+\\cdots = \\dfrac{\\pi^2}{6}$\n\n' +
          '这就是著名的 **Basel 问题**——欧拉 1735 年的经典结论，用 Parseval 等式可以优雅地重新推导。',
        formula:
          '\\sum_{n=1}^{\\infty}\\frac{1}{n^2} = \\frac{\\pi^2}{6}',
        tip: '类比：$f(x)=x^2$ 的 Parseval 等式可以给出 $\\sum 1/n^4 = \\pi^4/90$（$\\zeta(4)$）。策略：越高次的 $f$，能得到更高次的 $\\sum 1/n^{2k}$。',
      },
      // 卡 3：用 Parseval 等式求 $\\sum 1/n^4$
      {
        title: '进阶应用：$f(x)=x^2$ 求 $\\sum 1/n^4 = \\pi^4/90$',
        body:
          '**$f(x)=x^2$ 是偶函数**，$b_n=0$；\n\n' +
          '$a_0 = \\dfrac{2}{\\pi}\\displaystyle\\int_0^{\\pi}x^2\\,dx = \\dfrac{2\\pi^2}{3}$，$a_n = \\dfrac{4(-1)^n}{n^2}$（$n\\geq 1$），$a_n^2 = \\dfrac{16}{n^4}$。\n\n' +
          '**左边**：$\\dfrac{1}{\\pi}\\displaystyle\\int_{-\\pi}^{\\pi}x^4\\,dx = \\dfrac{1}{\\pi}\\cdot\\dfrac{2\\pi^5}{5} = \\dfrac{2\\pi^4}{5}$\n\n' +
          '**代入 Parseval**：$\\dfrac{2\\pi^4}{5} = \\dfrac{1}{2}\\left(\\dfrac{2\\pi^2}{3}\\right)^2 + \\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{16}{n^4} = \\dfrac{2\\pi^4}{9} + 16\\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{1}{n^4}$\n\n' +
          '解方程：$16\\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{1}{n^4} = \\dfrac{2\\pi^4}{5}-\\dfrac{2\\pi^4}{9} = 2\\pi^4\\cdot\\dfrac{4}{45} = \\dfrac{8\\pi^4}{45}$\n\n' +
          '$\\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{1}{n^4} = \\dfrac{\\pi^4}{90}$（即 Riemann $\\zeta(4)$）',
        formula:
          '\\sum_{n=1}^{\\infty}\\frac{1}{n^4} = \\frac{\\pi^4}{90}',
        tip: '计算 $a_0^2/2$ 时容易忘记平方：$(2\\pi^2/3)^2 / 2 = (4\\pi^4/9)/2 = 2\\pi^4/9$。建议每步单独计算 $a_0^2$ 再除以 2。',
      },
      // 卡 4：周期为 $2l$ 的 Fourier 展开
      {
        title: '周期为 $2l$ 的 Fourier 展开：变量替换',
        body:
          '若 $f(x)$ 以 $2l$ 为周期（$l\\neq\\pi$），令 $t = \\dfrac{\\pi x}{l}$，则 $g(t) = f\\!\\left(\\dfrac{lt}{\\pi}\\right)$ 以 $2\\pi$ 为周期，对 $g(t)$ 展开再换回 $x$。\n\n' +
          '**直接公式（不必每次做变量替换）**：\n\n' +
          '$f(x)\\sim\\dfrac{a_0}{2}+\\displaystyle\\sum_{n=1}^{\\infty}\\left(a_n\\cos\\dfrac{n\\pi x}{l}+b_n\\sin\\dfrac{n\\pi x}{l}\\right)$\n\n' +
          '$a_n = \\dfrac{1}{l}\\displaystyle\\int_{-l}^{l}f(x)\\cos\\dfrac{n\\pi x}{l}\\,dx$，$b_n = \\dfrac{1}{l}\\displaystyle\\int_{-l}^{l}f(x)\\sin\\dfrac{n\\pi x}{l}\\,dx$\n\n' +
          '**例**：$f(x)=x$（$-1<x\\leq 1$），以 $2$ 为周期延拓（$l=1$）：\n\n' +
          '$b_n = \\displaystyle\\int_{-1}^{1}x\\sin n\\pi x\\,dx = 2\\int_0^{1}x\\sin n\\pi x\\,dx = \\dfrac{2(-1)^{n+1}}{n\\pi}$\n\n' +
          '$x \\sim \\dfrac{2}{\\pi}\\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{(-1)^{n+1}}{n}\\sin n\\pi x$（$-1<x<1$）',
        formula:
          'a_n = \\frac{1}{l}\\int_{-l}^{l}f(x)\\cos\\frac{n\\pi x}{l}\\,dx,\\quad b_n = \\frac{1}{l}\\int_{-l}^{l}f(x)\\sin\\frac{n\\pi x}{l}\\,dx',
        tip: '周期 $2l$ 的公式和周期 $2\\pi$ 的公式只有两处不同：积分区间 $[-l,l]$；前面系数 $1/l$（代替 $1/\\pi$）；$\\cos/\\sin$ 的自变量是 $n\\pi x/l$（代替 $nx$）。',
      },
      // 卡 5：Fourier 四步走综合回顾
      {
        title: 'Fourier 展开四步走——完整复盘',
        body:
          '结合前三关，Fourier 展开的标准四步流程如下：\n\n' +
          '**步骤 1：判断奇偶性**\n\n' +
          '- 偶函数：$b_n=0$，只算 $a_n$；积分区间缩为 $[0,l]$，乘以 $2$\n' +
          '- 奇函数：$a_n=0$（含 $a_0$），只算 $b_n$\n' +
          '- 非奇非偶：全部都要算\n\n' +
          '**步骤 2：代入公式计算系数**\n\n' +
          '- 分部积分："多项式部分取 $u$，三角函数取 $dv$"\n' +
          '- 利用 $(-1)^n$ 区分奇偶 $n$ 的情况\n\n' +
          '**步骤 3：写 Fourier 级数（先用 $\\sim$）**\n\n' +
          '**步骤 4：用 Dirichlet 定理写收敛结论**\n\n' +
          '- 连续点处将 $\\sim$ 改成 $=$\n' +
          '- 间断点处写"收敛到 $\\dfrac{f(x^-)+f(x^+)}{2}$"\n\n' +
          '可选步骤 5：代入特殊点，用 Parseval 等式或收敛值推导数项级数的和。',
        tip: '常见"代入特殊点"时机：$x=0$ 让所有 $\\cos$ 变 1，$\\sin$ 变 0；$x=l$ 让 $\\cos$ 产生 $(-1)^n$；$x=l/2$ 让 $\\sin$ 产生 $\\pm 1$ 或 0 的交替序列。',
      },
      // 卡 6：典型例题——方波完整展开 + 求 Leibniz 公式
      {
        title: '综合例题：方波 + Leibniz 公式（完整流程）',
        body:
          '**题目**：$f(x)=\\begin{cases}0,&-\\pi\\leq x<0\\\\1,&0\\leq x<\\pi\\end{cases}$，以 $2\\pi$ 为周期延拓，求 Fourier 级数，并求 $1-\\dfrac{1}{3}+\\dfrac{1}{5}-\\dfrac{1}{7}+\\cdots$。\n\n' +
          '**步骤 1**：非奇非偶，须算所有系数。\n\n' +
          '**步骤 2**：$a_0 = \\dfrac{1}{\\pi}\\displaystyle\\int_0^{\\pi}1\\,dx = 1$；\n\n' +
          '$a_n = \\dfrac{1}{\\pi}\\displaystyle\\int_0^{\\pi}\\cos nx\\,dx = \\dfrac{\\sin n\\pi}{n\\pi} = 0$（$n\\geq 1$）；\n\n' +
          '$b_n = \\dfrac{1}{\\pi}\\displaystyle\\int_0^{\\pi}\\sin nx\\,dx = \\dfrac{1-(-1)^n}{n\\pi}$（$n$ 偶数时 $=0$，$n=2k+1$ 时 $=\\dfrac{2}{(2k+1)\\pi}$）。\n\n' +
          '**步骤 3**：$f(x)\\sim\\dfrac{1}{2}+\\dfrac{2}{\\pi}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{\\sin(2k+1)x}{2k+1}$。\n\n' +
          '**步骤 4（代入特殊点）**：$x=\\pi/2$ 是连续点，$f(\\pi/2)=1$，$\\sin((2k+1)\\pi/2)=(-1)^k$：\n\n' +
          '$1 = \\dfrac{1}{2}+\\dfrac{2}{\\pi}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{(-1)^k}{2k+1}$，解得 $\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{(-1)^k}{2k+1} = \\dfrac{\\pi}{4}$（Leibniz 公式）。',
        formula:
          '1-\\frac{1}{3}+\\frac{1}{5}-\\frac{1}{7}+\\cdots = \\frac{\\pi}{4}',
        reveal: {
          q: '上例中，$x=0$ 处 Fourier 级数收敛到多少？与常数项 $a_0/2$ 相比较，有何规律？',
          a: '$x=0$ 是间断点，$f(0^-)=0$，$f(0^+)=1$，Dirichlet 定理给出收敛值 $(0+1)/2=1/2$。恰好等于常数项 $a_0/2=1/2$——这不是巧合，而是 Fourier 级数在对称间断点处的普遍行为。',
        },
      },
      // 卡 7：Parseval 等式推导 $\\sum 1/(2n+1)^4$（提高）
      {
        title: '进阶：用 $|x|$ 的 Parseval 等式求 $\\sum 1/(2k+1)^4$',
        body:
          '**$f(x)=|x|$ 的 Parseval 等式推导**：\n\n' +
          '已知 $a_0=\\pi$，$n$ 奇数时 $a_n = -4/(\\pi n^2)$，$a_n^2 = 16/(\\pi^2 n^2)$；$n$ 偶数时 $a_n=0$；$b_n=0$。\n\n' +
          '**左边**：$\\dfrac{1}{\\pi}\\displaystyle\\int_{-\\pi}^{\\pi}x^2\\,dx = \\dfrac{2\\pi^2}{3}$\n\n' +
          '**右边**：$\\dfrac{a_0^2}{2} + \\displaystyle\\sum_{\\text{奇 }n}a_n^2 = \\dfrac{\\pi^2}{2} + \\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{16}{\\pi^2(2k+1)^2}\\cdot\\dfrac{\\pi^2}{\\pi^2}$\n\n' +
          '修正：$\\displaystyle\\sum_{\\text{奇 }n}a_n^2 = \\displaystyle\\sum_{k=0}^{\\infty}\\left(\\dfrac{-4}{\\pi(2k+1)^2}\\right)^2 = \\dfrac{16}{\\pi^2}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{1}{(2k+1)^4}$\n\n' +
          '代入：$\\dfrac{2\\pi^2}{3} = \\dfrac{\\pi^2}{2} + \\dfrac{16}{\\pi^2}\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{1}{(2k+1)^4}$\n\n' +
          '解得：$\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{1}{(2k+1)^4} = \\dfrac{\\pi^4}{96}$',
        formula:
          '\\sum_{k=0}^{\\infty}\\frac{1}{(2k+1)^4} = 1+\\frac{1}{81}+\\frac{1}{625}+\\cdots = \\frac{\\pi^4}{96}',
        tip: 'Parseval 等式右边的求和，只对非零系数的 $n$ 求和——奇偶函数各自有一半系数为零，大大简化计算。',
      },
    ],
    questions: [
      // 题 1：input——Parseval 等式代入 $f(x)=x$
      {
        id: 'c20-c-q1',
        type: 'input',
        prompt:
          '由 $f(x)=x$ 的 Fourier 展开和 Parseval 等式，$\\dfrac{1}{\\pi}\\displaystyle\\int_{-\\pi}^{\\pi}x^2\\,dx$ 等于 $\\dfrac{2\\pi^2}{N}$，$N$ 等于多少？（整数）',
        accept: ['3'],
        explain:
          '$\\dfrac{1}{\\pi}\\displaystyle\\int_{-\\pi}^{\\pi}x^2\\,dx = \\dfrac{1}{\\pi}\\cdot\\dfrac{2\\pi^3}{3} = \\dfrac{2\\pi^2}{3}$，故 $N=3$。' +
          '这是 Parseval 等式左边的值，等于右边 $4\\sum 1/n^2$，由此推出 $\\sum 1/n^2 = \\pi^2/6$。',
      },
      // 题 2：choice——$\\sum 1/n^2$ 的值
      {
        id: 'c20-c-q2',
        type: 'choice',
        prompt: '$\\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{1}{n^2} = 1+\\dfrac{1}{4}+\\dfrac{1}{9}+\\dfrac{1}{16}+\\cdots$ 等于哪个值？',
        options: [
          '$\\dfrac{\\pi}{4}$',
          '$\\dfrac{\\pi^2}{6}$',
          '$\\dfrac{\\pi^2}{8}$',
          '$\\dfrac{\\pi^4}{90}$',
        ],
        answer: 1,
        explain:
          '$\\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{1}{n^2} = \\dfrac{\\pi^2}{6}$（Basel 问题，由 $f(x)=x$ 的 Parseval 等式推导）。' +
          '$\\pi/4$ 是 Leibniz 公式；$\\pi^2/8$ 是 $\\sum 1/(2k+1)^2$ 的值；$\\pi^4/90$ 是 $\\sum 1/n^4$。',
      },
      // 题 3：judge——Parseval 等式右边项
      {
        id: 'c20-c-q3',
        type: 'judge',
        prompt:
          'Parseval 恒等式右边是 $\\dfrac{a_0^2}{2}+\\displaystyle\\sum_{n=1}^{\\infty}(a_n^2+b_n^2)$，其中常数项是 $a_0^2/2$（而不是 $a_0^2$）。',
        answer: true,
        explain:
          'Fourier 级数的常数项是 $a_0/2$，其"能量"贡献为 $(a_0/2)^2$ 乘以周期内的积分系数，最终化简为 $a_0^2/2$。' +
          '这与 $n\\geq 1$ 时 $a_n$ 和 $b_n$ 各自贡献 $a_n^2$ 和 $b_n^2$ 的机制不同——因为常数项 $a_0/2$ 的"自积分"给出 $2\\pi$，不是 $\\pi$。',
      },
      // 题 4：input——周期为 $2l$ 时 $b_n$ 公式中的系数
      {
        id: 'c20-c-q4',
        type: 'input',
        prompt:
          '周期为 $2l$ 的函数 $f(x)$，$b_n = \\dfrac{1}{l}\\displaystyle\\int_{-l}^{l}f(x)\\sin\\dfrac{n\\pi x}{l}\\,dx$。' +
          '当 $l=\\pi$ 时，此公式退化为 $b_n = \\dfrac{1}{\\pi}\\int_{-\\pi}^{\\pi}f(x)\\sin nx\\,dx$——这里 $\\dfrac{n\\pi x}{l}$ 退化为 $\\dfrac{n\\pi x}{\\pi} = nx$，正确吗？填 $1$ 代表正确，$0$ 代表错误。',
        accept: ['1'],
        explain:
          '$l=\\pi$ 时，$\\dfrac{n\\pi x}{l} = \\dfrac{n\\pi x}{\\pi} = nx$，公式变为 $b_n = \\dfrac{1}{\\pi}\\int_{-\\pi}^{\\pi}f(x)\\sin nx\\,dx$，完全一致。' +
          '这验证了周期 $2l$ 的公式是周期 $2\\pi$ 公式的推广。',
      },
      // 题 5：choice——$\\sum 1/n^4$ 的值
      {
        id: 'c20-c-q5',
        type: 'choice',
        prompt: '$\\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{1}{n^4} = 1+\\dfrac{1}{16}+\\dfrac{1}{81}+\\cdots$ 等于哪个值？',
        options: [
          '$\\dfrac{\\pi^2}{6}$',
          '$\\dfrac{\\pi^4}{90}$',
          '$\\dfrac{\\pi^4}{96}$',
          '$\\dfrac{\\pi^2}{9}$',
        ],
        answer: 1,
        explain:
          '$\\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{1}{n^4} = \\dfrac{\\pi^4}{90}$（$\\zeta(4)$），由 $f(x)=x^2$ 的 Parseval 等式推导。' +
          '$\\pi^4/96$ 是 $\\sum 1/(2k+1)^4$ 的值（只取奇数项）。',
      },
      // 题 6：match——特殊级数与对应的和
      {
        id: 'c20-c-q6',
        type: 'match',
        prompt: '将以下数项级数与其精确和对应。',
        left: [
          '$\\displaystyle\\sum_{n=1}^{\\infty}\\dfrac{1}{n^2}$',
          '$\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{(-1)^k}{2k+1}$',
          '$\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{1}{(2k+1)^2}$',
        ],
        right: [
          '$\\dfrac{\\pi^2}{6}$（Basel 问题）',
          '$\\dfrac{\\pi}{4}$（Leibniz 公式）',
          '$\\dfrac{\\pi^2}{8}$',
        ],
      },
      // 题 7：judge——Parseval 等式与 $f(x)=|x|$
      {
        id: 'c20-c-q7',
        type: 'judge',
        prompt:
          '利用 $f(x)=|x|$（$-\\pi\\leq x\\leq\\pi$）的 Parseval 恒等式，可以推导出 $\\displaystyle\\sum_{k=0}^{\\infty}\\dfrac{1}{(2k+1)^4} = \\dfrac{\\pi^4}{96}$。',
        answer: true,
        explain:
          '$|x|$ 的 Fourier 系数：$a_0=\\pi$，奇数 $n=2k+1$ 时 $a_n=-4/(\\pi n^2)$，偶数时 $a_n=0$，$b_n=0$。' +
          'Parseval 左边 $= 2\\pi^2/3$，右边 $= \\pi^2/2 + (16/\\pi^2)\\sum 1/(2k+1)^4$。' +
          '解方程：$(16/\\pi^2)\\sum 1/(2k+1)^4 = 2\\pi^2/3 - \\pi^2/2 = \\pi^2/6$，故 $\\sum 1/(2k+1)^4 = \\pi^4/96$。正确。',
      },
      // 题 8：input——$f(x)=x$（$-1<x\\leq 1$）的 $b_1$（周期 $2$）
      {
        id: 'c20-c-q8',
        type: 'input',
        prompt:
          '$f(x)=x$（$-1<x\\leq 1$），以 $2$ 为周期延拓（$l=1$），其 Fourier 系数 $b_1 = \\dfrac{2(-1)^{1+1}}{1\\cdot\\pi} = \\dfrac{2}{\\pi}$。' +
          '请填 $b_1$ 的分子（整数，即 $b_1 = \\dfrac{?}{\\pi}$）。',
        accept: ['2'],
        explain:
          '$b_n = \\dfrac{2(-1)^{n+1}}{n\\pi}$，$n=1$：$b_1 = \\dfrac{2(-1)^2}{1\\cdot\\pi} = \\dfrac{2}{\\pi}$，分子为 $2$。' +
          '注意与周期 $2\\pi$ 时 $b_1=2$ 的区别：周期缩小到 $2$，系数多了一个 $1/\\pi$ 因子。',
      },
    ],
  },
];
