import type { IntroCard } from '../../types'

// 学新知：第七部分（单元 19–21）各课的概念讲解卡片，按 lesson id 索引。
// 复平面与欧拉公式 / De Moivre 与单位根 / 傅里叶级数引论。
// 注意：TS 字符串内所有 LaTeX 反斜杠均双写；虚数单位写作 i，复指数写作 e^{i\\theta}；
// formula 字段写纯 LaTeX（不含 $）。
export const INTRO7: Record<string, IntroCard[]> = {
  // ============================== 单元 19：复平面与欧拉公式 ==============================
  'u19-l1': [
    {
      title: '为什么需要欧拉公式',
      body: '在初等三角里，**旋转**（转多少角度）和**函数值**（$\\cos,\\sin$ 是多少）一直是两件分开的事。欧拉公式做的事情，是把它们焊在同一个对象上：\n\n$$e^{i\\theta}=\\cos\\theta+i\\sin\\theta$$\n\n它一举统一了三件看起来毫不相干的东西——**指数函数**、**三角函数**、**平面旋转**。一旦接受它，很多「难记的公式」会突然变成「几何上显然的事实」，这也是它被称作连接初等三角与高等数学的桥梁的原因。',
      image: 'trig-p7-19-1.svg',
      imageCaption: '复平面单位圆上 $e^{i\\theta}$ 的位置：辐角为 $\\theta$，实部 $\\cos\\theta$、虚部 $\\sin\\theta$ 分别投影到两轴。',
    },
    {
      title: '核心公式：把点写成「长度 + 方向」',
      body: '复数 $z$ 既可写成「实部 + 虚部」$a+bi$，也可写成「长度 + 方向」。后者就是**极形式**：$z=r(\\cos\\theta+i\\sin\\theta)=re^{i\\theta}$。\n\n其中 $r=|z|$ 是**模长**（到原点的距离），$\\theta$ 是**辐角**（与正实轴的夹角）。欧拉公式正是连接这两种写法的纽带。',
      formula: 'z=re^{i\\theta}=r(\\cos\\theta+i\\sin\\theta)',
    },
    {
      title: '推导：从指数级数「长出」三角级数',
      body: '欧拉公式不是凭空规定的，它来自把 $e^x$ 的幂级数代入 $x=i\\theta$。关键全在 $i$ 的幂会**循环**：$i^2=-1,\\ i^3=-i,\\ i^4=1$。',
      steps: [
        '写出指数级数：$e^{i\\theta}=1+i\\theta+\\dfrac{(i\\theta)^2}{2!}+\\dfrac{(i\\theta)^3}{3!}+\\dfrac{(i\\theta)^4}{4!}+\\cdots$',
        '代入 $i^2=-1,\\ i^3=-i,\\ i^4=1$，让符号按 $+,-,+,-$ 循环。',
        '把**不带 $i$** 的项收在一起：$1-\\dfrac{\\theta^2}{2!}+\\dfrac{\\theta^4}{4!}-\\cdots$，这正是 $\\cos\\theta$ 的级数。',
        '把**带 $i$** 的项收在一起：$i\\left(\\theta-\\dfrac{\\theta^3}{3!}+\\dfrac{\\theta^5}{5!}-\\cdots\\right)$，括号里正是 $\\sin\\theta$ 的级数。',
        '合并即得 $e^{i\\theta}=\\cos\\theta+i\\sin\\theta$。',
      ],
    },
    {
      title: '最美的数学公式',
      body: '令 $\\theta=\\pi$：$e^{i\\pi}=\\cos\\pi+i\\sin\\pi=-1+0i=-1$，移项即得**欧拉恒等式**。它把数学里最重要的五个常数 $e,\\ i,\\ \\pi,\\ 1,\\ 0$ 串进同一个式子，被誉为「最美数学公式」。',
      formula: 'e^{i\\pi}+1=0',
      tip: '别把欧拉公式只当一个要背的式子。它的灵魂是「**乘以 $e^{i\\theta}$ = 逆时针旋转 $\\theta$**」；记住这个几何图像，后面单位根、傅里叶都会顺理成章。',
    },
    {
      title: '互动：试一个特殊值',
      body: '把欧拉公式当成「输入角度、输出单位圆上的点」的机器，亲手代一个值检验一下。',
      reveal: {
        q: '$e^{i\\cdot 0}=?$，$e^{i\\pi/2}=?$',
        a: '$e^{i\\cdot 0}=\\cos 0+i\\sin 0=1$；$e^{i\\pi/2}=\\cos\\dfrac{\\pi}{2}+i\\sin\\dfrac{\\pi}{2}=0+i=i$。',
      },
    },
  ],
  'u19-l2': [
    {
      title: '为什么 $e^{i\\theta}$ 永远落在单位圆上',
      body: '把欧拉公式的模长算出来：$|e^{i\\theta}|=\\sqrt{\\cos^2\\theta+\\sin^2\\theta}=\\sqrt{1}=1$。\n\n这意味着不管 $\\theta$ 取什么实数，$e^{i\\theta}$ 始终待在**半径为 1 的单位圆**上——它只改变方向，绝不改变长度。所以 $e^{i\\theta}$ 是「纯旋转」的代名词。',
      formula: '|e^{i\\theta}|=\\sqrt{\\cos^2\\theta+\\sin^2\\theta}=1',
    },
    {
      title: '模与辐角：复数的两个身份',
      body: '一个复数 $z=re^{i\\theta}$ 有两个互补的几何量：\n\n- **模长** $r=|z|$：它离原点多远（长度、大小）。\n- **辐角** $\\theta$：它朝哪个方向（角度、相位）。\n\n模长用勾股定理算，辐角靠 $\\cos\\theta,\\sin\\theta$ 的**正负号**定象限——这是不出错的关键。',
      formula: 'r=\\sqrt{a^2+b^2},\\qquad \\cos\\theta=\\frac{a}{r},\\ \\sin\\theta=\\frac{b}{r}',
    },
    {
      title: '例题：把 $1+\\sqrt3\\,i$ 化为极形式',
      body: '化极形式分两步走：先用勾股求模，再用实虚部符号定辐角。',
      steps: [
        '求模长：$r=|1+\\sqrt3\\,i|=\\sqrt{1^2+(\\sqrt3)^2}=\\sqrt{1+3}=2$。',
        '求辐角：$\\cos\\theta=\\dfrac{1}{2},\\ \\sin\\theta=\\dfrac{\\sqrt3}{2}$，两者皆正落在**第一象限**，故 $\\theta=\\dfrac{\\pi}{3}$。',
        '写出结果：$1+\\sqrt3\\,i=2\\left(\\cos\\dfrac{\\pi}{3}+i\\sin\\dfrac{\\pi}{3}\\right)=2e^{i\\pi/3}$。',
      ],
    },
    {
      title: '四个「整点」要记牢',
      body: '辐角 $0,\\dfrac{\\pi}{2},\\pi,\\dfrac{3\\pi}{2}$ 对应单位圆上四个特殊点 $1,\\ i,\\ -1,\\ -i$。它们是后面算单位根、判断象限的「锚点」。',
      tip: '只看 $\\tan\\theta$ 定辐角会出错——$\\tan$ 在相隔 $\\pi$ 的两个象限取值相同。必须**同时**看 $\\cos\\theta$ 与 $\\sin\\theta$ 的符号，辐角才唯一。',
    },
    {
      title: '互动：从辐角反推复数',
      body: '把「辐角 → 单位圆上的点」这条路走熟。',
      reveal: {
        q: '$e^{i\\pi}=?$，$e^{i\\cdot 3\\pi/2}=?$',
        a: '$e^{i\\pi}=\\cos\\pi+i\\sin\\pi=-1$；$e^{i\\cdot 3\\pi/2}=\\cos\\dfrac{3\\pi}{2}+i\\sin\\dfrac{3\\pi}{2}=-i$。',
      },
    },
  ],
  'u19-l3': [
    {
      title: '复数乘法的真正含义',
      body: '初见复数乘法 $(a+bi)(c+di)$ 像一团乱麻。但在极形式下它出奇地简单：\n\n$$r_1e^{i\\theta_1}\\cdot r_2e^{i\\theta_2}=r_1r_2\\,e^{i(\\theta_1+\\theta_2)}$$\n\n也就是 **「模长相乘、辐角相加」**。几何上，乘一个复数 = 先**缩放**（乘模长）再**旋转**（加辐角）。乘以 $e^{i\\theta}$ 这种模为 1 的复数，就是一次纯旋转。',
      formula: 'r_1e^{i\\theta_1}\\cdot r_2e^{i\\theta_2}=r_1r_2\\,e^{i(\\theta_1+\\theta_2)}',
    },
    {
      title: '指数律 = 旋转叠加',
      body: '指数函数的老规矩 $e^{i\\alpha}\\cdot e^{i\\beta}=e^{i(\\alpha+\\beta)}$ 在这里有了几何身体：**两次旋转叠加，等于一次总角度的旋转**。除法则相反——「模长相除、辐角相减」。',
      formula: '\\frac{z_1}{z_2}=\\frac{r_1}{r_2}\\,e^{i(\\theta_1-\\theta_2)}',
      reveal: {
        q: '$z_1=2e^{i\\pi/4},\\ z_2=3e^{i\\pi/6}$，求 $z_1z_2$ 的模与辐角。',
        a: '模 $=2\\cdot 3=6$；辐角 $=\\dfrac{\\pi}{4}+\\dfrac{\\pi}{6}=\\dfrac{3\\pi+2\\pi}{12}=\\dfrac{5\\pi}{12}$。',
      },
    },
    {
      title: '例题：用复数乘法推出和角公式',
      body: '和角公式不必死背，它就是「比较 $e^{i(\\alpha+\\beta)}=e^{i\\alpha}e^{i\\beta}$ 两边的实部、虚部」的副产品。',
      steps: [
        '左边按欧拉公式：$e^{i(\\alpha+\\beta)}=\\cos(\\alpha+\\beta)+i\\sin(\\alpha+\\beta)$。',
        '右边展开：$(\\cos\\alpha+i\\sin\\alpha)(\\cos\\beta+i\\sin\\beta)$。',
        '用 $i^2=-1$ 整理：$=(\\cos\\alpha\\cos\\beta-\\sin\\alpha\\sin\\beta)+i(\\sin\\alpha\\cos\\beta+\\cos\\alpha\\sin\\beta)$。',
        '比较**实部**：$\\cos(\\alpha+\\beta)=\\cos\\alpha\\cos\\beta-\\sin\\alpha\\sin\\beta$。',
        '比较**虚部**：$\\sin(\\alpha+\\beta)=\\sin\\alpha\\cos\\beta+\\cos\\alpha\\sin\\beta$。',
      ],
      tip: '「比较实部、虚部」是复数推三角恒等式的万能钥匙：一个复数等式，等价于实部相等**且**虚部相等两个实数等式。',
    },
    {
      title: '反推：用复指数表示 $\\cos,\\sin$',
      body: '把 $e^{i\\theta}=\\cos\\theta+i\\sin\\theta$ 和 $e^{-i\\theta}=\\cos\\theta-i\\sin\\theta$ 相加减，可以把三角函数「反解」成复指数。这是后面降幂、积分、傅里叶分析的常用利器。',
      formula: '\\cos\\theta=\\frac{e^{i\\theta}+e^{-i\\theta}}{2},\\qquad \\sin\\theta=\\frac{e^{i\\theta}-e^{-i\\theta}}{2i}',
      reveal: {
        q: '$e^{i\\theta}+e^{-i\\theta}=?$',
        a: '两式相加虚部抵消：$e^{i\\theta}+e^{-i\\theta}=2\\cos\\theta$。',
      },
    },
  ],

  // ============================== 单元 20：De Moivre 与单位根 ==============================
  'u20-l1': [
    {
      title: 'De Moivre：把角度乘以 $n$',
      body: '一旦接受「乘 $e^{i\\theta}$ = 旋转 $\\theta$」，复数**乘方**就有了直白的几何含义：连转 $n$ 次，总角度变成 $n\\theta$。这就是 De Moivre 公式：\n\n$$(\\cos\\theta+i\\sin\\theta)^n=\\cos n\\theta+i\\sin n\\theta$$\n\n它其实就是指数律 $\\left(e^{i\\theta}\\right)^n=e^{in\\theta}$ 换了个写法——把「角度乘 $n$」说得明明白白。',
      image: 'trig-p7-20-1.svg',
      imageCaption: '$z^5=1$ 的五个单位根 $e^{2\\pi ik/5}$ 在单位圆上等分圆周，构成正五边形，相邻辐角间隔 $\\dfrac{2\\pi}{5}$。',
    },
    {
      title: '带模长的完整版',
      body: '对一般复数 $z=r(\\cos\\theta+i\\sin\\theta)$ 取 $n$ 次幂，**模取 $n$ 次方、角度乘 $n$**：',
      formula: '[r(\\cos\\theta+i\\sin\\theta)]^n=r^n(\\cos n\\theta+i\\sin n\\theta)',
      tip: '高次幂别去硬展开二项式！先化极形式，再用 De Moivre 一步到位——「模 $r^n$、角 $n\\theta$」远比逐项相乘省力。',
    },
    {
      title: '例题：计算 $(1+i)^8$',
      body: '直接展开 $(1+i)^8$ 要算到天荒地老，化极形式后三步搞定。',
      steps: [
        '化极形式：$|1+i|=\\sqrt{1^2+1^2}=\\sqrt2$，辐角 $\\dfrac{\\pi}{4}$，即 $1+i=\\sqrt2\\,e^{i\\pi/4}$。',
        '模取 8 次方：$(\\sqrt2)^8=2^4=16$。',
        '角度乘 8：$8\\cdot\\dfrac{\\pi}{4}=2\\pi$，转回正实轴，$\\cos 2\\pi+i\\sin 2\\pi=1$。',
        '合并：$(1+i)^8=16\\cdot 1=16$。',
      ],
    },
    {
      title: '副产品：倍角公式',
      body: '令 $n=2$，把 $(\\cos\\theta+i\\sin\\theta)^2$ 展开再比较实虚部，立刻得到二倍角公式。同理 $n=3$ 给出三倍角 $\\cos3\\theta=4\\cos^3\\theta-3\\cos\\theta$——这正是 Chebyshev 多项式的雏形。',
      formula: '\\cos 2\\theta=\\cos^2\\theta-\\sin^2\\theta,\\quad \\sin 2\\theta=2\\sin\\theta\\cos\\theta',
      reveal: {
        q: '$\\left(\\cos\\dfrac{\\pi}{6}+i\\sin\\dfrac{\\pi}{6}\\right)^3=?$',
        a: '角度乘 3：$\\cos\\dfrac{\\pi}{2}+i\\sin\\dfrac{\\pi}{2}=i$。',
      },
    },
  ],
  'u20-l2': [
    {
      title: '方程 $z^n=1$ 有几个根',
      body: '在实数里 $z^n=1$ 顶多两个解；但在复数里，**代数基本定理**保证它恰有 $n$ 个根，称作 $n$ 次**单位根**。\n\n直觉是：旋转 $n$ 次正好转回原点（绕回 $2\\pi$ 的整数倍），有 $n$ 种不同的「转法」，对应 $n$ 个根。',
      formula: 'z^n=1 \\ \\Longrightarrow\\ z_k=e^{2\\pi i k/n},\\quad k=0,1,\\dots,n-1',
    },
    {
      title: '推导：辐角被均匀切成 $n$ 份',
      body: '设 $z=e^{i\\theta}$（模必为 1，否则 $|z|^n\\ne 1$），代入方程解辐角。',
      steps: [
        '设 $z=e^{i\\theta}$，方程变成 $e^{in\\theta}=1$。',
        '而 $1=e^{2k\\pi i}$（绕 $k$ 圈都回到 1），故 $e^{in\\theta}=e^{2k\\pi i}$。',
        '比较辐角：$n\\theta=2k\\pi$，即 $\\theta=\\dfrac{2k\\pi}{n}$。',
        '取 $k=0,1,\\dots,n-1$ 得 $n$ 个互不相同的根（再大就与已有根重合）。',
      ],
    },
    {
      title: '几何图像：正 $n$ 边形',
      body: '$n$ 个单位根模长都是 1、辐角间隔都是 $\\dfrac{2\\pi}{n}$，于是它们**等分单位圆周**，恰好是一个**正 $n$ 边形**的顶点。例如 $z^4=1$ 的四根 $1,i,-1,-i$ 构成正方形；$z^5=1$ 的五根构成正五边形。',
      tip: '常见错误：只写出 $z=1$ 一个「显然根」就收手。$n$ 次方程必须凑齐 $n$ 个根，几何上少一个顶点就不是正多边形了。',
    },
    {
      title: '一个漂亮恒等式',
      body: '把正 $n$ 边形的所有顶点向量相加，由对称性它们**两两抵消**，和为零。代数上这来自 $z^n-1$ 缺一次项（Vieta 定理：根之和 = 0）。',
      formula: '1+\\omega+\\omega^2+\\cdots+\\omega^{n-1}=0,\\quad \\omega=e^{2\\pi i/n}',
      reveal: {
        q: '$z^3=1$ 的三个根是？',
        a: '$1,\\ e^{2\\pi i/3},\\ e^{4\\pi i/3}$，在复平面构成正三角形，且三者之和为 $0$。',
      },
    },
  ],
  'u20-l3': [
    {
      title: '开方 = 圆周分点问题',
      body: '求 $w^n=z$ 就是给复数「开 $n$ 次方」。和单位根同理，它一般有 $n$ 个根，本质是一道**圆周分点**问题：\n\n$$w_k=\\sqrt[n]{r}\\,e^{i(\\theta+2k\\pi)/n},\\quad k=0,1,\\dots,n-1$$\n\n口诀是 **「模开方、角等分」**：模长开 $n$ 次方根，辐角加上整周期 $2k\\pi$ 再除以 $n$。',
      formula: 'w_k=\\sqrt[n]{r}\\,e^{i(\\theta+2k\\pi)/n},\\quad k=0,1,\\dots,n-1',
    },
    {
      title: '为什么必须加 $2k\\pi$',
      body: '若只写 $\\sqrt[n]{r}\\,e^{i\\theta/n}$ 就只得到**一个**根，漏掉其余 $n-1$ 个。原复数辐角其实是 $\\theta+2k\\pi$（绕几圈都一样），除以 $n$ 后这些「绕圈」会分裂出不同的根。',
      tip: '开方最常见的丢分点就是**漏根**。务必让 $k$ 跑遍 $0$ 到 $n-1$，凑齐 $n$ 个等角分布的根。',
    },
    {
      title: '例题：解 $w^3=-8$',
      body: '实数也要先化极形式才能在复数域开方。',
      steps: [
        '化极形式：$-8=8e^{i\\pi}$（模 8，辐角 $\\pi$）。',
        '套公式：$w_k=\\sqrt[3]{8}\\,e^{i(\\pi+2k\\pi)/3}=2e^{i(\\pi+2k\\pi)/3},\\ k=0,1,2$。',
        '$k=0$：$w_0=2e^{i\\pi/3}=1+\\sqrt3\\,i$。',
        '$k=1$：$w_1=2e^{i\\pi}=-2$（我们熟悉的那个实根）。',
        '$k=2$：$w_2=2e^{i5\\pi/3}=1-\\sqrt3\\,i$。三根模长都是 2，等角分布。',
      ],
    },
    {
      title: '单位根之和的妙用',
      body: '所有 $n$ 次单位根之和为 0（$n\\ge2$）。取它的**实部**，就能把纯几何对称翻译成具体的三角恒等式——这正是尺规作正多边形的代数基础。',
      reveal: {
        q: '由五次单位根之和为零，求 $\\cos\\dfrac{2\\pi}{5}+\\cos\\dfrac{4\\pi}{5}$。',
        a: '实部和 $1+2\\cos\\dfrac{2\\pi}{5}+2\\cos\\dfrac{4\\pi}{5}=0$（用 $\\cos(2\\pi-x)=\\cos x$ 配对），故和为 $-\\dfrac{1}{2}$。',
      },
    },
  ],

  // ============================== 单元 21：傅里叶级数引论 ==============================
  'u21-l1': [
    {
      title: '傅里叶的核心思想',
      body: '方波、锯齿波、周期温度变化……这些复杂的周期现象，傅里叶给出一个惊人的论断：\n\n> 任何足够「规整」的周期函数，都能拆成不同频率的 $\\sin$ 与 $\\cos$ 的**叠加**。\n\n于是三角函数在这里**改了角色**——不再是被求值的「目标」，而是搭建一切周期信号的「**频率基底**」，就像积木一样。',
      image: 'trig-p7-21-1.svg',
      imageCaption: '方波的傅里叶部分和逼近：$N=1,3,5$ 三条部分和曲线与方波叠加，随项数增加逐步逼近。',
    },
    {
      title: '级数长什么样',
      body: '周期为 $2\\pi$ 的函数，其傅里叶级数是一个常数项加上各次谐波（$\\cos nx,\\ \\sin nx$）的无穷叠加：',
      formula: 'f(x)\\sim\\frac{a_0}{2}+\\sum_{n=1}^{\\infty}\\left(a_n\\cos nx+b_n\\sin nx\\right)',
    },
    {
      title: '为什么能逐个提取系数：正交性',
      body: '能把每个频率的「分量」单独拎出来，靠的是不同频率三角函数的**正交性**——不同频率「互不干扰」，就像线性代数里不同基向量彼此独立。',
      steps: [
        '想求系数 $a_m$，就在级数两边同乘 $\\cos mx$。',
        '对整周期 $[-\\pi,\\pi]$ 积分。',
        '正交性使得**所有 $n\\ne m$ 的项积分为零**，只有 $n=m$ 那一项存活。',
        '于是 $a_m=\\dfrac{1}{\\pi}\\displaystyle\\int_{-\\pi}^{\\pi}f(x)\\cos mx\\,dx$，系数被「投影」出来。',
      ],
      tip: '验证正交性的工具就是老朋友**积化和差**：$\\cos A\\cos B=\\dfrac12[\\cos(A-B)+\\cos(A+B)]$，把两个频率的乘积拆成单频率，整周期积分自然归零。',
    },
    {
      title: '系数到底代表什么',
      body: '傅里叶系数不是神秘常数：$a_n,b_n$ 衡量的是**某个频率分量在原函数中所占的权重**——哪些频率强、信号偏低频还是高频。它本质上是一份「频率内容清单」。',
      reveal: {
        q: '$\\displaystyle\\int_{-\\pi}^{\\pi}\\cos x\\,\\cos 2x\\,dx=?$',
        a: '积化和差为 $\\tfrac12(\\cos x+\\cos 3x)$，两个非零频率整周期积分都为 0，故结果为 $0$（正交）。',
      },
    },
  ],
  'u21-l2': [
    {
      title: '先看对称性，能省一半功夫',
      body: '在动手算积分之前，先问一句：函数是奇还是偶？这个简单判断能直接**砍掉一半系数**：\n\n- **偶函数** → 只含余弦项（与常数项），所有 $b_n=0$。\n- **奇函数** → 只含正弦项，所有 $a_n=0$。',
      formula: 'f\\text{ 偶}\\Rightarrow b_n=0;\\qquad f\\text{ 奇}\\Rightarrow a_n=0',
    },
    {
      title: '为什么？被积函数的奇偶在作怪',
      body: '关键是「奇×偶=奇」「奇×奇=偶」，而**奇函数在对称区间上积分为零**。',
      steps: [
        '若 $f$ 偶，则 $f(x)\\sin nx$ = 偶×奇 = **奇**，在 $[-\\pi,\\pi]$ 上积分为 0，故 $b_n=0$。',
        '若 $f$ 奇，则 $f(x)\\cos nx$ = 奇×偶 = **奇**，积分为 0，故 $a_n=0$。',
        '所以偶函数只剩余弦谐波，奇函数只剩正弦谐波。',
      ],
    },
    {
      title: '例题：判断只含哪类项',
      body: '只需判断奇偶性，记住「奇×偶=奇」即可。',
      steps: [
        '$f(x)=x^2$：$f(-x)=x^2=f(x)$，**偶** → 只含余弦项（含常数项）。',
        '$g(x)=x^3$：$g(-x)=-x^3=-g(x)$，**奇** → 只含正弦项。',
        '$h(x)=x\\cos x$：$x$ 奇、$\\cos x$ 偶，乘积为**奇** → 只含正弦项。',
      ],
      tip: '锯齿波 $f(x)=x$ 是奇函数，故 $a_n=0$，只需算 $b_n=\\dfrac{2(-1)^{n+1}}{n}$。先用对称性，再分部积分，工作量减半。',
    },
    {
      title: '互动：对称性速判',
      body: '把「看一眼对称性」练成条件反射。',
      reveal: {
        q: '$f(x)=|x|$（周期 $2\\pi$）的傅里叶级数只含哪类项？',
        a: '$|{-x}|=|x|$ 为偶函数，故只含余弦项与常数项（$b_n=0$）。',
      },
    },
  ],
  'u21-l3': [
    {
      title: '经典样板：方波展开',
      body: '方波 $f(x)=1\\,(0<x<\\pi),\\ -1\\,(-\\pi<x<0)$ 是奇函数，故 $a_n=0$。算出 $b_n=\\dfrac{2}{n\\pi}(1-\\cos n\\pi)$，偶次项全为 0，只剩**奇次正弦谐波**：\n\n$$f(x)=\\frac{4}{\\pi}\\left(\\sin x+\\frac{\\sin 3x}{3}+\\frac{\\sin 5x}{5}+\\cdots\\right)$$',
      formula: 'f(x)=\\frac{4}{\\pi}\\sum_{k=0}^{\\infty}\\frac{\\sin(2k+1)x}{2k+1}',
    },
    {
      title: '部分和如何逐步逼近',
      body: '只取前几项 $N=1,3,5,\\dots$ 求和，得到的「部分和」曲线会**越来越像方波**。项数越多，平的地方越平、跳变越陡。这就是「无穷个简单波叠出复杂波形」的直观演示（见本课配图）。',
      tip: '在**间断点**处，傅里叶级数收敛到左右极限的**平均值** $\\dfrac{f(x^+)+f(x^-)}{2}$（Dirichlet 条件），而非任意一边的值。方波跳变处恰好收敛到 0。',
    },
    {
      title: 'Gibbs 现象：永不消失的过冲',
      body: '有个反直觉的事实：无论取多少项，间断点附近**总有约 9% 的过冲尖角**，且它不会随项数增加而消失，只会越挤越窄。这叫 **Gibbs 现象**，是傅里叶逼近间断函数的固有特征。',
      reveal: {
        q: '方波只含奇次还是偶次谐波？为什么？',
        a: '只含奇次。偶次系数 $b_n=\\dfrac{2}{n\\pi}(1-\\cos n\\pi)$，当 $n$ 偶时 $\\cos n\\pi=1$，括号为 0。',
      },
    },
    {
      title: '意外彩蛋：算出 $\\pi$',
      body: '在方波展开里代入特殊点 $x=\\dfrac{\\pi}{2}$，左边 $f=1$，右边各 $\\sin$ 取值变成 $1,-\\tfrac13,\\tfrac15,\\dots$，整理就得到著名的 **Leibniz 公式**——傅里叶级数顺手给了我们一个算 $\\pi$ 的级数。',
      formula: '\\frac{\\pi}{4}=1-\\frac{1}{3}+\\frac{1}{5}-\\frac{1}{7}+\\cdots',
      reveal: {
        q: '由 $1=\\dfrac{4}{\\pi}\\left(1-\\dfrac13+\\dfrac15-\\cdots\\right)$ 解出括号内的和。',
        a: '两边乘 $\\dfrac{\\pi}{4}$ 即得 $1-\\dfrac13+\\dfrac15-\\cdots=\\dfrac{\\pi}{4}$。',
      },
    },
  ],
}
