import type { IntroCard } from '../../types'

// 第五部分（第15–17章）学新知卡片：复指数、双曲函数、矩阵指数
export const INTRO5: Record<string, IntroCard[]> = {
  // ===================== u15 复指数与欧拉公式 =====================
  'u15-l1': [
    {
      title: '动机：指数能进复数世界吗？',
      body: '第14章我们把指数函数写成了幂级数，它对任何实数 $x$ 都成立。一个大胆的念头：如果把 $x$ 换成虚数 $ix$，会发生什么？\n\n这不是“硬代入”——既然指数最深的定义之一就是幂级数，而幂级数本就能代入复数，那么复指数就是指数函数向复数域**最自然的延伸**。',
      formula: 'e^x=1+x+\\frac{x^2}{2!}+\\frac{x^3}{3!}+\\cdots',
    },
    {
      title: '核心：欧拉公式',
      body: '把 $x\\to i\\theta$ 代入级数，并用虚数单位的循环规律 $i^2=-1,\\ i^3=-i,\\ i^4=1$，级数自动分成两组：\n\n- 偶次项凑成 $\\cos\\theta$\n- 奇次项凑成 $i\\sin\\theta$\n\n于是得到统一指数、三角、复数三种语言的桥梁——**欧拉公式**。',
      formula: 'e^{i\\theta}=\\cos\\theta+i\\sin\\theta',
      image: 'exp-p5-15-1.svg',
      imageCaption: '$e^{i\\theta}$ 在复平面单位圆上：实部 $\\cos\\theta$、虚部 $\\sin\\theta$，随 $\\theta$ 增大逆时针旋转',
    },
    {
      title: '推导：为什么分成两组',
      body: '跟着虚数单位的幂走一遍，就能看清实部和虚部是怎么自然分开的。',
      steps: [
        '展开 $e^{i\\theta}=1+i\\theta+\\frac{(i\\theta)^2}{2!}+\\frac{(i\\theta)^3}{3!}+\\cdots$',
        '用 $i^2=-1,\\ i^3=-i,\\ i^4=1$ 替换各次幂',
        '偶次项不含 $i$：$1-\\frac{\\theta^2}{2!}+\\frac{\\theta^4}{4!}-\\cdots=\\cos\\theta$',
        '奇次项都含 $i$：$i\\left(\\theta-\\frac{\\theta^3}{3!}+\\cdots\\right)=i\\sin\\theta$',
        '合起来即 $e^{i\\theta}=\\cos\\theta+i\\sin\\theta$',
      ],
    },
    {
      title: '提示',
      body: '别把欧拉公式当成孤立的“魔法结论”去死记。它来自幂级数，记住来路你就永远忘不掉，也不会把实部虚部记反。',
      tip: '常见误区：以为复指数仍表示“单调增长”。其实纯虚指数 $e^{i\\theta}$ 模长恒为 1，描述的是**旋转**，不是放大。',
    },
    {
      title: '互动：先想一想',
      body: '在看答案前，自己用欧拉公式代一下这个最著名的角度。',
      reveal: {
        q: '把 $\\theta=\\pi$ 代入欧拉公式，$e^{i\\pi}$ 等于多少？',
        a: '$e^{i\\pi}=\\cos\\pi+i\\sin\\pi=-1+0=-1$。于是 $e^{i\\pi}+1=0$——把 $e,i,\\pi,1,0$ 压缩进一条恒等式，被誉为最美公式。',
      },
    },
  ],

  'u15-l2': [
    {
      title: '直觉：为什么是旋转，不是增长',
      body: '实数里 $e^x$ 最显眼的效果是放大或衰减——它描述“尺度怎样变化”。\n\n可一旦走到纯虚方向 $e^{i\\theta}$，模长不再增长，变的只是**角度**：点沿着单位圆运动。\n\n所以可以并排理解：$e^x$ 像沿实轴的放缩，$e^{i\\theta}$ 像复平面上的旋转。',
      formula: '|e^{i\\theta}|=\\sqrt{\\cos^2\\theta+\\sin^2\\theta}=1',
    },
    {
      title: '核心：极形式 $z=re^{i\\theta}$',
      body: '任意非零复数都能写成“模长 $\\times$ 方向”的形式：\n\n- $r=|z|>0$ 负责**有多长**（到原点的距离）\n- $e^{i\\theta}$ 负责**朝哪个方向**（与正实轴的夹角 $\\theta$，即辐角）\n\n结合欧拉公式也可写成 $z=r(\\cos\\theta+i\\sin\\theta)$。极形式是复平面上“长度 + 方向”最自然的坐标。',
      formula: 'z=re^{i\\theta}=r(\\cos\\theta+i\\sin\\theta)',
    },
    {
      title: '推导：乘法 = 模长相乘、角度相加',
      body: '复数乘法在极形式下会突然变简单，根源就是指数的加法法则 $e^{i\\theta_1}e^{i\\theta_2}=e^{i(\\theta_1+\\theta_2)}$。',
      steps: [
        '设 $z_1=r_1e^{i\\theta_1}$，$z_2=r_2e^{i\\theta_2}$',
        '相乘把模长与指数分开：$z_1z_2=r_1r_2\\,e^{i\\theta_1}e^{i\\theta_2}$',
        '用指数律合并角度：$e^{i\\theta_1}e^{i\\theta_2}=e^{i(\\theta_1+\\theta_2)}$',
        '得 $z_1z_2=r_1r_2\\,e^{i(\\theta_1+\\theta_2)}$：**模长相乘、角度相加**',
        '特例：乘 $e^{i\\theta}$ 时 $r$ 不变，只把辐角整体转过 $\\theta$——这就是纯旋转',
      ],
      formula: 'z_1z_2=r_1r_2\\,e^{i(\\theta_1+\\theta_2)}',
    },
    {
      title: '提示',
      body: '所以复数不只是“二维数对”，它更像在平面上执行变换的代数对象：乘正实数是纯放缩，乘 $e^{i\\theta}$ 是纯旋转，乘一般的 $re^{i\\theta}$ 则是“先放缩、再旋转”的组合。',
      tip: '做乘除时务必同时跟踪两件事：模长（相乘/相除）与辐角（相加/相减）。只盯一个就容易算错方向。',
    },
    {
      title: '互动：先算一算',
      body: '不必展开成实部虚部，直接用极形式的乘法规则。',
      reveal: {
        q: '已知 $z_1=\\sqrt2\\,e^{i\\pi/4}$，$z_2=2e^{i\\pi/6}$，求乘积 $z_1z_2$ 的模长与辐角。',
        a: '模长相乘 $\\sqrt2\\times2=2\\sqrt2$；辐角相加 $\\frac{\\pi}{4}+\\frac{\\pi}{6}=\\frac{5\\pi}{12}$。即 $z_1z_2=2\\sqrt2\\,e^{i5\\pi/12}$——结构比数值更值得记住。',
      },
    },
  ],

  'u15-l3': [
    {
      title: '动机：从一个数到一圈点',
      body: '幂运算在极形式里特别干净：由 $z=re^{i\\theta}$ 立刻得 $z^n=r^ne^{in\\theta}$——模长做 $n$ 次幂、角度乘以 $n$。\n\n反过来问“哪些 $z$ 满足 $z^n=1$”，答案就不止一个数，而是一**圈**均匀分布的点。这正是复指数把“解方程”和“看几何”统一起来的地方。',
      formula: 'z^n=r^ne^{in\\theta}\\ \\Longrightarrow\\ (\\cos\\theta+i\\sin\\theta)^n=\\cos n\\theta+i\\sin n\\theta',
    },
    {
      title: '核心：单位根等角分布',
      body: '因为 $1=e^{i2k\\pi}$（$k\\in\\mathbb Z$），方程 $z^n=1$ 的全部解为\n\n相邻两根的辐角差恒为 $\\frac{2\\pi}{n}$，所以它们都落在单位圆上，把整圆平均分成 $n$ 份，围成一个正 $n$ 边形的顶点集。',
      formula: 'z_k=e^{i\\frac{2k\\pi}{n}},\\qquad k=0,1,\\dots,n-1',
    },
    {
      title: '推导：四次单位根',
      body: '取 $n=4$ 走一遍标准流程：模长开 $n$ 次方、辐角除以 $n$ 并取齐所有分支。',
      steps: [
        '由公式 $z_k=e^{i\\frac{2k\\pi}{4}}$，$k=0,1,2,3$',
        '$z_0=e^{0}=1$',
        '$z_1=e^{i\\pi/2}=i$',
        '$z_2=e^{i\\pi}=-1$',
        '$z_3=e^{i3\\pi/2}=-i$',
        '全部解 $1,\\,i,\\,-1,\\,-i$ 落在单位圆上，正好是正方形四顶点',
      ],
    },
    {
      title: '提示',
      body: '把欧拉公式中的 $\\theta=\\pi$ 代入得 $e^{i\\pi}=-1$，于是 $e^{i\\pi}+1=0$。这条恒等式把指数 $e$、复数单位 $i$、圆周率 $\\pi$、算术基本元 $0$ 与 $1$ 压进一行，被誉为最美公式。',
      tip: '求单位根（或一般开方）时最常见的错误是只写主值。一定要把 $k=0,1,\\dots,n-1$ 所有角度分支取齐。',
    },
    {
      title: '互动：先想一想',
      body: '换个底数，看看根都落在哪条圆上。',
      reveal: {
        q: '方程 $z^3=1$ 的三个解是什么？它们在复平面上怎么分布？',
        a: '$z_k=e^{i\\frac{2k\\pi}{3}}$（$k=0,1,2$）：$1,\\ e^{i2\\pi/3}=-\\frac12+\\frac{\\sqrt3}{2}i,\\ e^{i4\\pi/3}=-\\frac12-\\frac{\\sqrt3}{2}i$。三点在单位圆上两两相差 $120^\\circ$，构成等边三角形。',
      },
    },
  ],

  // ===================== u16 双曲函数 =====================
  'u16-l1': [
    {
      title: '直觉：指数的对称拆分',
      body: '把 $e^x$ 和 $e^{-x}$ 并排看：一个向右快速放大，一个向右快速衰减。\n\n双曲函数做的，就是把这两支方向相反的指数做**加法**与**减法**组合：\n\n- $\\cosh x$ 是“对称平均”\n- $\\sinh x$ 是“反对称差值”\n\n所以它们不是另起炉灶的新对象，而是指数语言的延伸。',
      formula: '\\cosh x=\\frac{e^x+e^{-x}}{2},\\qquad \\sinh x=\\frac{e^x-e^{-x}}{2}',
    },
    {
      title: '核心：定义与图像',
      body: '三个双曲函数各有鲜明性格：\n\n- $\\cosh x$ 偶函数，呈 U 形，恒 $\\ge 1$\n- $\\sinh x$ 奇函数，过原点，左负右正\n- $\\tanh x=\\dfrac{\\sinh x}{\\cosh x}$，输出落在 $(-1,1)$，逐渐**饱和**\n\n与三角函数不同，它们通常不周期重复，更强调增长、夹逼与渐近。',
      image: 'exp-p5-16-1.svg',
      imageCaption: '$\\cosh x$ 是 $e^x$ 与 $e^{-x}$ 的半和（U 形），$\\sinh x$ 是两者的半差',
    },
    {
      title: '推导：把指数反拆回来',
      body: '双曲函数不只是由指数组合出来，反过来也能把 $e^x$ 拆成对称与反对称两个分量。',
      steps: [
        '直接相加：$\\cosh x+\\sinh x=\\frac{e^x+e^{-x}}{2}+\\frac{e^x-e^{-x}}{2}=e^x$',
        '直接相减：$\\cosh x-\\sinh x=\\frac{e^x+e^{-x}}{2}-\\frac{e^x-e^{-x}}{2}=e^{-x}$',
        '所以 $\\cosh x$ 抓住指数里的**对称部分**，$\\sinh x$ 抓住**反对称部分**',
        '由 $\\cosh(-x)=\\cosh x$ 知它是偶函数，由 $\\sinh(-x)=-\\sinh x$ 知它是奇函数',
      ],
      formula: 'e^x=\\cosh x+\\sinh x',
    },
    {
      title: '提示',
      body: '学双曲函数时不要只把它们当成“换了字母的三角函数”。三角函数强调周期旋转（圆结构），双曲函数强调两支指数的对称组合（双曲线结构）——背后的几何对象完全不同。',
      tip: '$\\tanh x$ 把任意实数压缩到 $(-1,1)$ 并逐渐饱和，常用作建模、神经网络里的“压缩映射”，渐近线为 $y=\\pm1$。',
    },
    {
      title: '互动：先算一算',
      body: '直接用定义代入原点，看看双曲函数和三角函数有没有相似的“起点”。',
      reveal: {
        q: '$\\sinh 0$ 与 $\\cosh 0$ 各等于多少？',
        a: '$\\sinh 0=\\frac{1-1}{2}=0$，$\\cosh 0=\\frac{1+1}{2}=1$。与 $\\sin 0=0,\\cos 0=1$ 形式相同，但几何背景是双曲线而非圆。',
      },
    },
  ],

  'u16-l2': [
    {
      title: '动机：和三角只差一个符号',
      body: '三角有 $\\cos^2 x+\\sin^2 x=1$（圆上的约束）。双曲函数有一条形式几乎一样、却把加号换成减号的恒等式。\n\n这个小小的符号差，正是“圆”与“双曲线”分道扬镳的地方，也让双曲函数的求导比三角更整齐。',
      formula: '\\cosh^2 x-\\sinh^2 x=1',
    },
    {
      title: '核心：整齐的导数',
      body: '双曲函数求导下彼此切换，而且**都不带负号**（不像 $(\\cos x)^{\\prime}=-\\sin x$）：\n\n- $(\\sinh x)^{\\prime}=\\cosh x$\n- $(\\cosh x)^{\\prime}=\\sinh x$\n- $(\\tanh x)^{\\prime}=\\dfrac{1}{\\cosh^2 x}=\\operatorname{sech}^2 x$\n\n所以微分方程 $y^{\\prime\\prime}=y$ 自然导向 $\\cosh,\\sinh$（试探 $e^{rx}$ 得 $r^2=1$），而 $y^{\\prime\\prime}=-y$ 才导向三角函数。',
      formula: '(\\sinh x)^{\\prime}=\\cosh x,\\qquad (\\cosh x)^{\\prime}=\\sinh x',
    },
    {
      title: '推导：为什么是减号',
      body: '把定义代回去平方，看中间项怎么抵消，就明白减号从何而来。',
      steps: [
        '$\\cosh^2 x=\\frac{(e^x+e^{-x})^2}{4}=\\frac{e^{2x}+2+e^{-2x}}{4}$',
        '$\\sinh^2 x=\\frac{(e^x-e^{-x})^2}{4}=\\frac{e^{2x}-2+e^{-2x}}{4}$',
        '相减：$\\cosh^2 x-\\sinh^2 x=\\frac{(e^{2x}+2+e^{-2x})-(e^{2x}-2+e^{-2x})}{4}$',
        '$e^{2x}$ 与 $e^{-2x}$ 互相抵消，只剩常数：$=\\frac{4}{4}=1$',
        '得 $\\cosh^2 x-\\sinh^2 x=1$（约束在双曲线上，而非圆上）',
      ],
    },
    {
      title: '提示',
      body: '$\\tanh x$ 的导数可由商法则现场推出：$(\\tanh x)^{\\prime}=\\dfrac{\\cosh^2 x-\\sinh^2 x}{\\cosh^2 x}=\\dfrac{1}{\\cosh^2 x}$，分子正好是基本恒等式。$|x|$ 越大 $\\cosh x$ 越大、导数趋于 0，曲线逐渐变平——这就是“饱和型响应”。',
      tip: '关键差别：$\\cosh x=\\frac{e^x+e^{-x}}{2}$ 是两个正数的平均，**恒为正**（$\\ge1$）。由 $\\cosh^2=1+\\sinh^2$ 开方时永远取正根，不必像三角那样讨论符号。',
    },
    {
      title: '互动：先想一想',
      body: '用刚才那条恒等式，从一个值反推另一个。',
      reveal: {
        q: '已知 $\\sinh x=\\frac34$，求 $\\cosh x$ 与 $\\tanh x$。',
        a: '$\\cosh^2 x=1+\\sinh^2 x=1+\\frac{9}{16}=\\frac{25}{16}$，取正根 $\\cosh x=\\frac54$；于是 $\\tanh x=\\frac{3/4}{5/4}=\\frac35$。',
      },
    },
  ],

  'u16-l3': [
    {
      title: '动机：从定义算到反函数',
      body: '双曲函数既能向前生成值，也能像对数那样把结构“反解”回来。\n\n本节把三件事串起来：先用定义直接算具体值，再看 $\\tanh$ 的饱和如何用作压缩映射，最后看反双曲函数为什么能写成对数——这正是指数与对数互逆关系的又一次现身。',
      formula: '\\operatorname{arsinh} x=\\ln\\!\\big(x+\\sqrt{x^2+1}\\big)',
    },
    {
      title: '核心：求导与求积分互逆',
      body: '双曲函数的积分与导数完全互逆，记住一对就够：\n\n- $\\int\\sinh x\\,dx=\\cosh x+C$\n- $\\int\\cosh x\\,dx=\\sinh x+C$\n- $\\int\\tanh x\\,dx=\\ln(\\cosh x)+C$\n\n反双曲函数则衔接到“$\\sqrt{x^2\\pm1}$ 型积分”，例如 $\\int\\dfrac{dx}{\\sqrt{x^2+1}}=\\operatorname{arsinh} x+C$，与三角代换形成漂亮类比。',
      formula: '\\int\\frac{dx}{\\sqrt{x^2+1}}=\\operatorname{arsinh} x+C=\\ln\\!\\big(x+\\sqrt{x^2+1}\\big)+C',
    },
    {
      title: '推导：反双曲正弦的对数式',
      body: '把“反解输入”这件事算到底，对数自然冒出来。',
      steps: [
        '设 $y=\\operatorname{arsinh} x$，即 $x=\\sinh y=\\frac{e^y-e^{-y}}{2}$',
        '令 $t=e^y>0$，则 $2x=t-\\frac1t$，两边乘 $t$ 得 $t^2-2xt-1=0$',
        '解二次方程：$t=x\\pm\\sqrt{x^2+1}$',
        '因 $t=e^y>0$ 而 $x-\\sqrt{x^2+1}<0$，必须取正根 $t=x+\\sqrt{x^2+1}$',
        '故 $y=\\ln t=\\ln\\!\\big(x+\\sqrt{x^2+1}\\big)$',
      ],
    },
    {
      title: '提示',
      body: '在 $x=\\ln a$ 这类点上双曲函数特别好算，因为 $e^{\\ln a}=a$ 把指数“解开”了。例如 $\\cosh(\\ln 2)=\\frac{2+\\frac12}{2}=\\frac54$，$\\sinh(\\ln 2)=\\frac{2-\\frac12}{2}=\\frac34$。',
      tip: '悬链线（自由悬挂的链条）的形状正是 $y=a\\cosh\\frac{x}{a}$。双曲函数描述的是“指数型对称增长”，而非周期振荡，这也是它在物理与工程中的常见来源。',
    },
    {
      title: '互动：先算一算',
      body: '用定义而不是查表，直接算出原点处的值。',
      reveal: {
        q: '$\\cosh 0$ 与 $\\tanh 0$ 各等于多少？',
        a: '$\\cosh 0=\\frac{1+1}{2}=1$；$\\tanh 0=\\frac{\\sinh 0}{\\cosh 0}=\\frac01=0$。$\\tanh$ 从原点出发，向两侧逐渐贴近 $y=\\pm1$。',
      },
    },
  ],

  // ===================== u17 矩阵指数与线性系统 =====================
  'u17-l1': [
    {
      title: '动机：指数也能作用在矩阵上？',
      body: '标量里 $e^{\\lambda t}$ 解一阶方程 $x^{\\prime}=\\lambda x$，解是 $x(t)=e^{\\lambda t}x_0$。\n\n现在把未知量换成向量 $\\mathbf x$、常数换成矩阵 $A$，就得到线性系统 $\\mathbf x^{\\prime}=A\\mathbf x$。自然要问：标量里的 $e^{\\lambda t}$，在矩阵世界里由谁替代？\n\n答案是**矩阵指数** $e^{At}$——它不是新记号，而是“指数作为系统演化算子”的推广。',
      formula: '\\mathbf x^{\\prime}(t)=A\\mathbf x(t)\\ \\Longrightarrow\\ \\mathbf x(t)=e^{At}\\mathbf x_0',
    },
    {
      title: '核心：用幂级数定义',
      body: '对任意**方阵** $A$，矩阵指数定义为与标量同源的幂级数，只是把数乘换成矩阵乘法：\n\n它对任意方阵都可用，不要求可对角化，也不需先知道特征值。最关键的是——它恰好给出线性系统的解。',
      formula: 'e^A=I+A+\\frac{A^2}{2!}+\\frac{A^3}{3!}+\\cdots',
      image: 'exp-p5-17-1.svg',
      imageCaption: '线性系统 $\\mathbf x^{\\prime}=A\\mathbf x$ 的相平面：复特征值 $\\lambda=-0.35\\pm i$（实部为负）时轨迹螺旋收敛到原点',
    },
    {
      title: '推导：为什么 $e^{At}$ 解系统',
      body: '把 $e^{At}$ 逐项对 $t$ 求导，就能看出它天然满足方程和初值。',
      steps: [
        '展开 $e^{At}=I+At+\\frac{(At)^2}{2!}+\\cdots$',
        '逐项求导：$\\frac{d}{dt}e^{At}=A+A^2 t+\\frac{A^3 t^2}{2!}+\\cdots$',
        '提出一个 $A$：$\\frac{d}{dt}e^{At}=A\\left(I+At+\\cdots\\right)=Ae^{At}$',
        '令 $\\mathbf x(t)=e^{At}\\mathbf x_0$，则 $\\mathbf x^{\\prime}(t)=A\\mathbf x(t)$ ✓',
        '又 $\\mathbf x(0)=e^{A\\cdot 0}\\mathbf x_0=I\\mathbf x_0=\\mathbf x_0$，初值也满足 ✓',
      ],
    },
    {
      title: '提示',
      body: '$e^A$ **不是**对矩阵每个元素分别取指数！它是幂级数。例如 $A=\\begin{pmatrix}0&1\\\\0&0\\end{pmatrix}$ 时，由 $A^2=0$ 得 $e^A=I+A=\\begin{pmatrix}1&1\\\\0&1\\end{pmatrix}$，而非 $\\begin{pmatrix}1&e\\\\1&1\\end{pmatrix}$。',
      tip: '另一个陷阱：$e^{A+B}=e^A e^B$ **不总成立**，只有当 $AB=BA$（可交换）时才对，因为矩阵乘法一般不可交换。',
    },
    {
      title: '互动：先想一想',
      body: '幂零矩阵能让无穷级数变成有限和。试着算一个。',
      reveal: {
        q: '设 $N=\\begin{pmatrix}0&1\\\\0&0\\end{pmatrix}$，满足 $N^2=0$，那 $e^{Nt}$ 是什么？',
        a: '级数在第二项后截断：$e^{Nt}=I+Nt=\\begin{pmatrix}1&t\\\\0&1\\end{pmatrix}$。这是一个**剪切变换**，不是指数爆炸——提醒我们 $e^{At}$ 不总是 $e^{\\lambda t}$ 的形状。',
      },
    },
  ],

  'u17-l2': [
    {
      title: '动机：把矩阵问题降成标量',
      body: '$e^{At}$ 给出了解 $\\mathbf x(t)=e^{At}\\mathbf x_0$，但怎么把这个矩阵真正算出来？\n\n直接展开无穷级数太累。窍门是：先把系统旋到**最自然的坐标轴**上（对角化），在那里各方向互不耦合，矩阵指数就退化成一堆标量指数 $e^{\\lambda t}$，算完再旋回来。',
      formula: '\\mathbf x(t)=e^{At}\\mathbf x_0',
    },
    {
      title: '核心：对角化求 $e^{At}$',
      body: '若 $A=PDP^{-1}$ 可对角化，$D=\\operatorname{diag}(\\lambda_1,\\dots,\\lambda_n)$，则\n\n而对角阵的指数逐个对角元取指数即可：$e^{Dt}=\\operatorname{diag}(e^{\\lambda_1 t},\\dots,e^{\\lambda_n t})$。各方向互不耦合，这正是对角化带来的便利。',
      formula: 'A=PDP^{-1}\\ \\Longrightarrow\\ e^{At}=Pe^{Dt}P^{-1}',
    },
    {
      title: '推导：为什么级数能这样收起来',
      body: '关键是相邻的 $P^{-1}P=I$ 把中间全部抵消，于是 $A^k$ 退化成 $PD^kP^{-1}$。',
      steps: [
        '$A^2=(PDP^{-1})(PDP^{-1})=PD(P^{-1}P)DP^{-1}=PD^2P^{-1}$',
        '同理对任意 $k$：$A^k=PD^kP^{-1}$',
        '代回级数：$e^{At}=\\sum_k\\frac{(At)^k}{k!}=P\\left(\\sum_k\\frac{(Dt)^k}{k!}\\right)P^{-1}$',
        '括号里正是 $e^{Dt}$，对角阵逐个对角元取指数',
        '得 $e^{At}=Pe^{Dt}P^{-1}$——矩阵指数降成标量指数',
      ],
    },
    {
      title: '提示',
      body: '完整算法四步走：(1) 求特征值与特征向量；(2) 拼出 $A=PDP^{-1}$；(3) 对角元取指数得 $e^{Dt}$；(4) 乘回得 $Pe^{Dt}P^{-1}$。本质是“先找最自然的坐标轴，在各轴上做标量演化，再变换回原坐标”。',
      tip: '算完一定用 $e^{A\\cdot 0}=I$ 自检：把 $t=0$ 代入结果矩阵，若不等于单位阵，说明哪一步算错了。',
    },
    {
      title: '互动：先想一想',
      body: '最简单的情形——矩阵本身已经是对角的。',
      reveal: {
        q: '设 $A=\\begin{pmatrix}1&0\\\\0&-2\\end{pmatrix}$，$e^{At}$ 是什么？',
        a: '对角阵直接逐个对角元取指数：$e^{At}=\\begin{pmatrix}e^{t}&0\\\\0&e^{-2t}\\end{pmatrix}$。第一方向按 $e^t$ 增长、第二方向按 $e^{-2t}$ 衰减——同一系统不同方向命运不同。',
      },
    },
  ],

  'u17-l3': [
    {
      title: '动机：从矩阵读出系统的命运',
      body: '$e^{At}$ 不只对应“放大”。当矩阵带有耦合或旋转结构时，它能产生剪切、振荡甚至旋转。\n\n这一节学会**不必把 $e^{At}$ 完整算出来**，只看特征值的实部，就能预判系统是衰减、爆发还是绕圈——这正是控制理论和微分方程里第一步总要做的事。',
      formula: 'e^{Jt}=\\begin{pmatrix}\\cos t&-\\sin t\\\\\\sin t&\\cos t\\end{pmatrix}',
    },
    {
      title: '核心：特征值实部定稳定性',
      body: '某个特征方向上的演化大致像 $e^{\\lambda t}$，于是：\n\n- $\\operatorname{Re}(\\lambda)>0$：该方向**指数增长**\n- $\\operatorname{Re}(\\lambda)<0$：该方向**指数衰减**\n- $\\operatorname{Re}(\\lambda)=0$（如纯虚 $\\pm\\omega i$）：常对应**无衰减振荡/旋转**\n\n全部实部为负则系统稳定（趋于原点）；只要有一个为正，通常沿某方向爆发增长。',
      formula: '\\operatorname{Re}(\\lambda)<0\\ (\\forall\\lambda)\\ \\Longrightarrow\\ \\text{稳定，}\\mathbf x(t)\\to\\mathbf 0',
    },
    {
      title: '推导：旋转矩阵的指数',
      body: '旋转矩阵 $J=\\begin{pmatrix}0&-1\\\\1&0\\end{pmatrix}$ 的指数把复指数的旋转推广到实平面。',
      steps: [
        '$J$ 的幂有循环：$J^2=-I,\\ J^3=-J,\\ J^4=I$',
        '把 $e^{Jt}=\\sum\\frac{(Jt)^k}{k!}$ 按偶/奇次项分组',
        '偶次项凑成 $\\left(1-\\frac{t^2}{2!}+\\cdots\\right)I=\\cos t\\,I$',
        '奇次项凑成 $\\left(t-\\frac{t^3}{3!}+\\cdots\\right)J=\\sin t\\,J$',
        '得 $e^{Jt}=\\cos t\\,I+\\sin t\\,J=\\begin{pmatrix}\\cos t&-\\sin t\\\\\\sin t&\\cos t\\end{pmatrix}$，恰是旋转矩阵',
      ],
    },
    {
      title: '提示',
      body: '$J$ 的特征值是纯虚数 $\\pm i$，实部为 0，对应无衰减旋转——轨迹是绕原点的圆。这与第 15 章 $e^{i\\theta}$ 在单位圆上旋转的图景完全呼应：矩阵指数把复指数的旋转搬到了二维实平面。',
      tip: '别把标量公式无脑搬来：$e^{A+B}=e^Ae^B$ 只在 $AB=BA$ 时成立。不可对角化（重特征值）时还会冒出 $te^{\\lambda t}$ 这类“多项式 × 指数”因子。',
    },
    {
      title: '互动：先想一想',
      body: '只看对角元（即特征值），就能说出两个方向各自的命运。',
      reveal: {
        q: '设 $A=\\begin{pmatrix}3&0\\\\0&-1\\end{pmatrix}$，系统第一分量按 $e^{3t}$ 演化，第二分量按什么演化？整体稳定吗？',
        a: '$e^{At}=\\begin{pmatrix}e^{3t}&0\\\\0&e^{-t}\\end{pmatrix}$：第二分量按 $e^{-t}$ 衰减。但特征值 $3>0$，第一方向爆发增长，所以系统**不稳定**（鞍点）。',
      },
    },
  ],
}
