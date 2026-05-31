import type { IntroCard } from '../../types'

// 学新知讲解卡：Part 2（第 4–6 章）。每课 4–5 张，按「动机 → 核心概念 → 推导/例题 → 易错点 → 互动」节奏。
// 注意：本文件中所有 LaTeX 反斜杠均写成双反斜杠，避免破坏 TS 构建。
export const INTRO2: Record<string, IntroCard[]> = {
  // ============================== 单元 4：基本恒等式 ==============================
  'u4-l1': [
    {
      title: '为什么',
      body:
        '很多人学三角函数时，把恒等式当成“一大堆要背的公式”，越背越乱。其实真正的**根**只有一条：\n\n' +
        '$$\\sin^2\\theta+\\cos^2\\theta=1$$\n\n' +
        '它不是凭空规定的技巧，而是**单位圆这一几何事实的函数化表达**。掌握它的来源后，大量复杂化简都只是“不断回到这条根”。',
      tip: '记住一句话：看到复杂三角式，先想办法把它化归回 $\\sin^2\\theta+\\cos^2\\theta=1$。',
    },
    {
      title: '核心概念',
      body:
        '在单位圆上取点 $P(\\cos\\theta,\\sin\\theta)$。因为单位圆满足 $x^2+y^2=1$，把坐标代入立刻得到平方关系。\n\n' +
        '换个角度看：半径为 $1$ 的直角三角形，两条直角边正是 $\\cos\\theta$ 与 $\\sin\\theta$，斜边为 $1$，于是**勾股定理**直接给出这条恒等式。',
      formula: '\\sin^2\\theta+\\cos^2\\theta=1',
      image: 'trig-p2-04-1.svg',
      imageCaption: '单位圆上半径为 1 的直角三角形：两直角边 $\\cos\\theta$、$\\sin\\theta$，斜边 1，勾股定理给出 $\\cos^2\\theta+\\sin^2\\theta=1$。',
    },
    {
      title: '例题精讲',
      body:
        '**题**：已知 $\\sin\\theta=\\dfrac35$ 且 $\\theta$ 在第二象限，求 $\\cos\\theta$。\n\n' +
        '关键在于：平方关系给出的是**结构**（绝对值），象限给出的是**符号**。',
      steps: [
        '由平方关系移项：$\\cos^2\\theta=1-\\sin^2\\theta=1-\\dfrac{9}{25}=\\dfrac{16}{25}$。',
        '开平方只得绝对值：$|\\cos\\theta|=\\dfrac45$。',
        '第二象限余弦为**负**，故 $\\cos\\theta=-\\dfrac45$。',
      ],
      tip: '开平方那一步只给绝对值！符号必须另由象限判断，绝不能随手取正。',
    },
    {
      title: '想一想',
      body: '先自己判断，再点开看答案。',
      reveal: {
        q: '由 $\\sin^2\\theta=\\dfrac14$，能直接断定 $\\sin\\theta=\\dfrac12$ 吗？',
        a: '不能。开平方只给绝对值 $|\\sin\\theta|=\\dfrac12$，所以 $\\sin\\theta$ 也可能是 $-\\dfrac12$，到底取哪个由 $\\theta$ 所在象限决定。',
      },
    },
  ],
  'u4-l2': [
    {
      title: '为什么',
      body:
        '六个三角函数看似各自为政，其实**不是六套独立对象**，而是一张靠两类关系连起来的网：**商数关系**与**倒数关系**。\n\n' +
        '理解这张网，就能把任何函数随时拉回 $\\sin,\\cos$ 来思考，这是化简的总抓手。',
    },
    {
      title: '核心概念',
      body:
        '**商数关系**直接来自定义（纵坐标 / 横坐标）：\n\n' +
        '$$\\tan\\theta=\\frac{\\sin\\theta}{\\cos\\theta},\\qquad \\cot\\theta=\\frac{\\cos\\theta}{\\sin\\theta}$$\n\n' +
        '**倒数关系**把另外三个函数定义出来：\n\n' +
        '$$\\sec\\theta=\\frac{1}{\\cos\\theta},\\quad \\csc\\theta=\\frac{1}{\\sin\\theta},\\quad \\cot\\theta=\\frac{1}{\\tan\\theta}$$',
      formula: '\\tan\\theta=\\dfrac{\\sin\\theta}{\\cos\\theta},\\quad \\sec\\theta=\\dfrac{1}{\\cos\\theta},\\quad \\csc\\theta=\\dfrac{1}{\\sin\\theta}',
      tip: '配对口诀：$\\sec\\leftrightarrow\\cos$、$\\csc\\leftrightarrow\\sin$、$\\cot\\leftrightarrow\\tan$。注意 $\\sec$ 配的是 $\\cos$ 不是 $\\sin$，最易记反。',
    },
    {
      title: '例题精讲',
      body:
        '**题**：已知 $\\tan\\theta=2$，求 $\\dfrac{\\sin\\theta+\\cos\\theta}{2\\sin\\theta-\\cos\\theta}$。\n\n' +
        '分子分母都是关于 $\\sin\\theta,\\cos\\theta$ 的**一次齐次式**，可同除以 $\\cos\\theta$ 化成只含 $\\tan\\theta$ 的式子。',
      steps: [
        '分子分母同除 $\\cos\\theta$（$\\tan\\theta=2$ 保证 $\\cos\\theta\\ne0$）：$\\dfrac{\\tan\\theta+1}{2\\tan\\theta-1}$。',
        '代入 $\\tan\\theta=2$：$\\dfrac{2+1}{2\\cdot2-1}=\\dfrac{3}{3}$。',
        '结果为 $1$。无需求出 $\\sin\\theta,\\cos\\theta$ 各自的值，也省去讨论象限。',
      ],
      tip: '齐次式求值的招牌技巧：分子分母同除以 $\\cos\\theta$（或它的幂），整个式子瞬间“只剩 $\\tan$”。',
    },
    {
      title: '想一想',
      body: '先自己判断，再点开看答案。',
      reveal: {
        q: '$\\tan\\theta$ 在 $\\cos\\theta=0$（如 $\\theta=\\dfrac{\\pi}{2}$）处有定义吗？',
        a: '没有。$\\tan\\theta=\\dfrac{\\sin\\theta}{\\cos\\theta}$，分母为 $0$ 时无定义——这也是正切图像在那里出现竖直渐近线的原因。',
      },
    },
  ],
  'u4-l3': [
    {
      title: '为什么',
      body:
        '$1+\\tan^2\\theta=\\sec^2\\theta$、$1+\\cot^2\\theta=\\csc^2\\theta$ 这些常用结论，**根本不必单独背**——它们全是平方关系的“一次除法”变形。\n\n' +
        '本课目标：把公式从“死记硬背”升级成**模式识别**。',
    },
    {
      title: '推导',
      body:
        '从平方关系 $\\sin^2\\theta+\\cos^2\\theta=1$ 出发，两边除以 $\\cos^2\\theta$：',
      steps: [
        '两边除以 $\\cos^2\\theta$：$\\dfrac{\\sin^2\\theta}{\\cos^2\\theta}+1=\\dfrac{1}{\\cos^2\\theta}$。',
        '用商数与倒数关系：$\\tan^2\\theta+1=\\sec^2\\theta$。',
        '改除以 $\\sin^2\\theta$，同理得：$1+\\cot^2\\theta=\\csc^2\\theta$。',
      ],
      formula: '1+\\tan^2\\theta=\\sec^2\\theta,\\qquad 1+\\cot^2\\theta=\\csc^2\\theta',
      tip: '看到 $1+\\tan^2$ 立刻想 $\\sec^2$；看到 $1-\\sin^2$ 立刻想 $\\cos^2$。这种“一眼到位”就是模式识别。',
    },
    {
      title: '例题精讲',
      body:
        '**题**：化简 $\\sin^4\\theta+\\cos^4\\theta$。\n\n' +
        '思路：凑出 $(\\sin^2\\theta+\\cos^2\\theta)^2$，凡见到 $\\sin^2\\theta+\\cos^2\\theta$ 一律当 $1$。',
      steps: [
        '配方：$\\sin^4\\theta+\\cos^4\\theta=(\\sin^2\\theta+\\cos^2\\theta)^2-2\\sin^2\\theta\\cos^2\\theta$。',
        '把 $\\sin^2\\theta+\\cos^2\\theta$ 视为 $1$：$=1-2\\sin^2\\theta\\cos^2\\theta$。',
        '这是常用的降幂结论；同法可得 $\\sin^6\\theta+\\cos^6\\theta=1-3\\sin^2\\theta\\cos^2\\theta$。',
      ],
      tip: '证恒等式时别忘**定义域**：约分若忽略分母为零，可能推出非法结论。',
    },
    {
      title: '想一想',
      body: '先自己判断，再点开看答案。',
      reveal: {
        q: '“证明恒等式”和“解方程”逻辑一样吗？',
        a: '不一样。恒等式要证对定义域内**所有** $x$ 都成立（如 $\\sin^2x+\\cos^2x=1$）；方程只需找出**让式子成立的那些** $x$（如 $\\sin x=\\cos x$ 的解 $x=\\dfrac{\\pi}{4}+k\\pi$）。',
      },
    },
  ],

  // ============================== 单元 5：和差·倍角·半角 ==============================
  'u5-l1': [
    {
      title: '为什么',
      body:
        '和差公式是整个三角变换的**发动机**。倍角只是“两角取相等”的特例，半角是对倍角的反向整理，万能代换又由半角而来：\n\n' +
        '**和差公式 → 倍角 → 半角/降幂 → 万能代换**\n\n' +
        '所以这一条链的源头务必吃透。',
    },
    {
      title: '核心概念',
      body:
        '余弦差角公式可由**单位圆两点距离**导出：取 $P(\\cos\\alpha,\\sin\\alpha)$、$Q(\\cos\\beta,\\sin\\beta)$，它们夹角为 $\\alpha-\\beta$，把 $|PQ|^2$ 用两种方式算出再令其相等，即得 $\\cos(\\alpha-\\beta)$。其余三条由它替换符号推出。\n\n' +
        '$$\\sin(\\alpha\\pm\\beta)=\\sin\\alpha\\cos\\beta\\pm\\cos\\alpha\\sin\\beta$$\n' +
        '$$\\cos(\\alpha\\pm\\beta)=\\cos\\alpha\\cos\\beta\\mp\\sin\\alpha\\sin\\beta$$\n' +
        '$$\\tan(\\alpha\\pm\\beta)=\\frac{\\tan\\alpha\\pm\\tan\\beta}{1\\mp\\tan\\alpha\\tan\\beta}$$',
      formula: '\\cos(\\alpha\\pm\\beta)=\\cos\\alpha\\cos\\beta\\mp\\sin\\alpha\\sin\\beta',
      image: 'trig-p2-05-1.svg',
      imageCaption: '单位圆上两点夹角为 $\\alpha-\\beta$，用 $|P_1P_2|^2=2-2\\cos(\\alpha-\\beta)$ 导出余弦差角公式。',
    },
    {
      title: '例题精讲',
      body:
        '**题**：不查表求 $\\cos 75^\\circ$。\n\n' +
        '非特殊角往往能**拆成两个特殊角之和或差**，这就是拆角法。',
      steps: [
        '拆角：$75^\\circ=45^\\circ+30^\\circ$，用余弦和角公式（符号相反）。',
        '$\\cos 75^\\circ=\\cos 45^\\circ\\cos 30^\\circ-\\sin 45^\\circ\\sin 30^\\circ$。',
        '代值：$=\\dfrac{\\sqrt2}{2}\\cdot\\dfrac{\\sqrt3}{2}-\\dfrac{\\sqrt2}{2}\\cdot\\dfrac12=\\dfrac{\\sqrt6-\\sqrt2}{4}$。',
      ],
      tip: '符号陷阱：正弦和差“符号同号”，余弦和差“符号相反”。即 $\\sin(\\alpha+\\beta)$ 中间是 $+$，$\\cos(\\alpha+\\beta)$ 中间却是 $-$。',
    },
    {
      title: '想一想',
      body: '先自己判断，再点开看答案。',
      reveal: {
        q: '$\\sin 15^\\circ$ 和 $\\cos 75^\\circ$ 谁大？',
        a: '一样大。$\\sin 15^\\circ=\\sin(45^\\circ-30^\\circ)=\\dfrac{\\sqrt6-\\sqrt2}{4}=\\cos 75^\\circ$，这正是余函数关系 $\\sin\\theta=\\cos(90^\\circ-\\theta)$。',
      },
    },
  ],
  'u5-l2': [
    {
      title: '为什么',
      body:
        '倍角公式不是“另一张要背的表”。在和差公式里令 $\\alpha=\\beta=x$，倍角公式就**自动掉出来**。理解这一点，记忆量直接减半。',
    },
    {
      title: '核心概念',
      body:
        '令 $\\alpha=\\beta=x$ 代入和差公式：\n\n' +
        '$$\\sin 2x=2\\sin x\\cos x$$\n' +
        '$$\\cos 2x=\\cos^2x-\\sin^2x=1-2\\sin^2x=2\\cos^2x-1$$\n' +
        '$$\\tan 2x=\\frac{2\\tan x}{1-\\tan^2x}$$\n\n' +
        '$\\cos 2x$ 有三种形式：用平方关系把 $\\sin^2x$ 或 $\\cos^2x$ 换掉即可互相转化。',
      formula: '\\sin 2x=2\\sin x\\cos x,\\quad \\cos 2x=1-2\\sin^2x',
      tip: '$\\cos 2x$ 三形式按“题目给了谁就选谁”：给 $\\sin x$ 用 $1-2\\sin^2x$，给 $\\cos x$ 用 $2\\cos^2x-1$，最省事。',
    },
    {
      title: '例题精讲',
      body:
        '**题**：已知 $\\sin x=\\dfrac35$ 且 $x$ 为锐角，求 $\\sin 2x$ 与 $\\cos 2x$。',
      steps: [
        '先补 $\\cos x$：$x$ 锐角余弦为正，$\\cos x=\\sqrt{1-\\dfrac{9}{25}}=\\dfrac45$。',
        '$\\sin 2x=2\\sin x\\cos x=2\\cdot\\dfrac35\\cdot\\dfrac45=\\dfrac{24}{25}$。',
        '$\\cos 2x=1-2\\sin^2x=1-2\\cdot\\dfrac{9}{25}=\\dfrac{7}{25}$。',
      ],
      tip: '标准两步：先用平方关系补齐缺的函数值（并由象限定符号），再代入倍角公式。',
    },
    {
      title: '想一想',
      body: '先自己回忆，再点开看答案。',
      reveal: {
        q: '$\\sin 3x$ 的三倍角公式是什么？怎么记？',
        a: '$\\sin 3x=3\\sin x-4\\sin^3x$，口诀“**正弦 3 减 4 立方**”。对应地 $\\cos 3x=4\\cos^3x-3\\cos x$，“余弦 4 立方减 3”。它由 $3x=2x+x$ 用和角＋倍角推出。',
      },
    },
  ],
  'u5-l3': [
    {
      title: '为什么',
      body:
        '半角公式是把倍角公式**反向整理**得来的：既能降幂（把平方拆成一次），又能在积分、解方程里大显身手。而把它推到极致就是**万能代换**——一把统一有理化的钥匙。',
    },
    {
      title: '核心概念',
      body:
        '从 $\\cos x=1-2\\sin^2\\dfrac{x}{2}$ 与 $\\cos x=2\\cos^2\\dfrac{x}{2}-1$ 反解：\n\n' +
        '$$\\sin^2\\frac{x}{2}=\\frac{1-\\cos x}{2},\\qquad \\cos^2\\frac{x}{2}=\\frac{1+\\cos x}{2}$$\n\n' +
        '设 $t=\\tan\\dfrac{x}{2}$，则有**万能代换**：\n\n' +
        '$$\\sin x=\\frac{2t}{1+t^2},\\qquad \\cos x=\\frac{1-t^2}{1+t^2}$$',
      formula: '\\sin^2\\dfrac{x}{2}=\\dfrac{1-\\cos x}{2},\\qquad \\cos^2\\dfrac{x}{2}=\\dfrac{1+\\cos x}{2}',
      tip: '区分两个分子：$\\sin^2$ 用 $1-\\cos x$，$\\cos^2$ 用 $1+\\cos x$。“正弦减、余弦加”，别记反。',
    },
    {
      title: '例题精讲',
      body:
        '**题**：用万能代换化简 $\\dfrac{\\sin x}{1+\\cos x}$（设 $t=\\tan\\dfrac{x}{2}$）。',
      steps: [
        '代入：分子 $\\dfrac{2t}{1+t^2}$，分母 $1+\\dfrac{1-t^2}{1+t^2}=\\dfrac{2}{1+t^2}$。',
        '相除约去 $\\dfrac{1}{1+t^2}$：$\\dfrac{2t}{2}=t$。',
        '即得漂亮的半角恒等式 $\\dfrac{\\sin x}{1+\\cos x}=\\tan\\dfrac{x}{2}$。',
      ],
      tip: '万能代换把同时含 $\\sin x,\\cos x$ 的式子统一成关于 $t$ 的有理式——这是它“万能”之处。',
    },
    {
      title: '想一想',
      body: '先自己判断，再点开看答案。',
      reveal: {
        q: '写 $\\sin\\dfrac{x}{2}=\\sqrt{\\dfrac{1-\\cos x}{2}}$ 时，永远取正号对吗？',
        a: '不对。正负号取决于 $\\dfrac{x}{2}$ 所在象限。例如 $x\\in\\left(\\dfrac{\\pi}{2},\\pi\\right)$ 时 $\\dfrac{x}{2}\\in\\left(\\dfrac{\\pi}{4},\\dfrac{\\pi}{2}\\right)$ 在第一象限取正；但若 $\\dfrac{x}{2}$ 落到第三、四象限就要取负。务必先由 $x$ 范围推出 $\\dfrac{x}{2}$ 范围。',
      },
    },
  ],

  // ============================== 单元 6：积化和差·和差化积 ==============================
  'u6-l1': [
    {
      title: '为什么',
      body:
        '很多三角式难算，不是函数本身难，而是**写法不适合当前问题**：求和或积分时，乘积往往碍手碍脚。\n\n' +
        '积化和差就是把“乘积结构”翻译成“频率和与差”，让难题换一种表示就迎刃而解。',
    },
    {
      title: '核心概念',
      body:
        '四组公式都由和差公式加减组合而来，不必死记：\n\n' +
        '$$\\sin A\\cos B=\\tfrac12[\\sin(A+B)+\\sin(A-B)]$$\n' +
        '$$\\cos A\\cos B=\\tfrac12[\\cos(A+B)+\\cos(A-B)]$$\n' +
        '$$\\sin A\\sin B=-\\tfrac12[\\cos(A+B)-\\cos(A-B)]$$\n\n' +
        '例如把 $\\sin(A+B)$ 与 $\\sin(A-B)$ 相加除以 $2$，立刻得到第一条。',
      image: 'trig-p2-06-1.svg',
      imageCaption: '积化和差把两个频率之积写成“和频 + 差频”的组合，是频率分析的语言。',
      formula: '\\cos A\\cos B=\\tfrac12[\\cos(A+B)+\\cos(A-B)]',
      tip: '$\\sin A\\sin B$ 那条**前面有负号**，是最容易记错的一组，写时务必先打负号。',
    },
    {
      title: '例题精讲',
      body:
        '**题**：求 $\\cos 75^\\circ\\cos 15^\\circ$ 的精确值。\n\n' +
        '比起分别求两个余弦再相乘，积化和差要干净得多。',
      steps: [
        '取 $A=75^\\circ,\\,B=15^\\circ$，用 $\\cos A\\cos B=\\tfrac12[\\cos(A+B)+\\cos(A-B)]$。',
        '$=\\tfrac12[\\cos 90^\\circ+\\cos 60^\\circ]$。',
        '$\\cos 90^\\circ=0$ 直接清掉，$\\cos 60^\\circ=\\tfrac12$，得 $\\tfrac12\\cdot\\tfrac12=\\dfrac14$。',
      ],
      tip: '拆成“和频 + 差频”后，常有一项（如 $\\cos 90^\\circ$）直接归零，求值一步到位。',
    },
    {
      title: '想一想',
      body: '先自己想想，再点开看答案。',
      reveal: {
        q: '为什么 $\\cos 20^\\circ\\cos 40^\\circ\\cos 80^\\circ=\\dfrac18$？',
        a: '逐层用积化和差降级：先 $\\cos 40^\\circ\\cos 80^\\circ=-\\tfrac14+\\tfrac12\\cos 40^\\circ$，再展开 $\\cos 20^\\circ\\cos 40^\\circ=\\tfrac14+\\tfrac12\\cos 20^\\circ$，代回后含 $\\cos 20^\\circ$ 的项神奇地两两抵消，只剩常数 $\\dfrac18$。',
      },
    },
  ],
  'u6-l2': [
    {
      title: '为什么',
      body:
        '和差化积是积化和差的**逆向**：把“叠加结构”写成“包络 × 波形”。在分析拍频、调制、求和与最值时，这种形式比原来的和差更有解释力。',
    },
    {
      title: '核心概念',
      body:
        '常见四组（记 $s=\\dfrac{A+B}{2}$ 为“半和”，$d=\\dfrac{A-B}{2}$ 为“半差”）：\n\n' +
        '$$\\sin A+\\sin B=2\\sin s\\cos d$$\n' +
        '$$\\sin A-\\sin B=2\\cos s\\sin d$$\n' +
        '$$\\cos A+\\cos B=2\\cos s\\cos d$$\n' +
        '$$\\cos A-\\cos B=-2\\sin s\\sin d$$',
      formula: '\\sin A+\\sin B=2\\sin\\dfrac{A+B}{2}\\cos\\dfrac{A-B}{2}',
      tip: '$\\cos A-\\cos B$ 那条**带负号**，且变成两个正弦之积——这是全组最容易记错符号的一条。',
    },
    {
      title: '例题精讲',
      body:
        '**题**：化简 $\\dfrac{\\sin 5x+\\sin x}{\\cos 5x+\\cos x}$。\n\n' +
        '凡是“同名函数之和 / 同名函数之和”的比值，都值得先试和差化积。',
      steps: [
        '分子：$\\sin 5x+\\sin x=2\\sin 3x\\cos 2x$（$s=3x,\\,d=2x$）。',
        '分母：$\\cos 5x+\\cos x=2\\cos 3x\\cos 2x$。',
        '约去公因子 $2\\cos 2x$：$\\dfrac{\\sin 3x}{\\cos 3x}=\\tan 3x$。',
      ],
      tip: '关键是分子分母都析出了**同一个公因子** $\\cos 2x$，约掉后立刻露出 $\\tan 3x$。',
    },
    {
      title: '想一想',
      body: '先自己想想，再点开看答案。',
      reveal: {
        q: '拍频 $\\sin 101t+\\sin 99t=2\\sin 100t\\cos t$ 中，谁是慢变包络？',
        a: '慢变包络是 $2\\cos t$（频率只有 $1$，变化慢），快速振荡是 $\\sin 100t$（频率 $100$）。耳朵听到的“一强一弱周期起伏”正是包络 $2\\cos t$ 造成的拍。',
      },
    },
  ],
  'u6-l3': [
    {
      title: '为什么',
      body:
        '$a\\sin x+b\\cos x$ 看似两个波，其实可以**合成一个**正弦 $R\\sin(x+\\varphi)$。一旦合一，求最值、周期、相位全都一目了然——这是辅助角公式的威力，本质是和角公式的**逆用**。',
    },
    {
      title: '核心概念',
      body:
        '设 $a\\sin x+b\\cos x=R\\sin(x+\\varphi)=R\\sin x\\cos\\varphi+R\\cos x\\sin\\varphi$。对照系数得 $R\\cos\\varphi=a,\\ R\\sin\\varphi=b$。两式平方相加（用平方关系消 $\\varphi$）即得 $R$：\n\n' +
        '$$R=\\sqrt{a^2+b^2},\\qquad \\tan\\varphi=\\frac{b}{a}$$\n\n' +
        '于是 $a\\sin x+b\\cos x$ 的**最大值为 $\\sqrt{a^2+b^2}$**（正弦最大为 $1$）。',
      formula: 'a\\sin x+b\\cos x=\\sqrt{a^2+b^2}\\,\\sin(x+\\varphi)',
      tip: '$\\varphi$ 要由 $\\cos\\varphi=\\dfrac{a}{R}$ 与 $\\sin\\varphi=\\dfrac{b}{R}$ **联立**确定象限；只看 $\\tan\\varphi$ 会丢象限信息。',
    },
    {
      title: '例题精讲',
      body:
        '**题**：求 $f(x)=\\sqrt3\\sin x+\\cos x$ 的最大值，并写成 $R\\sin(x+\\varphi)$。',
      steps: [
        '$R=\\sqrt{(\\sqrt3)^2+1^2}=\\sqrt{4}=2$。',
        '$\\cos\\varphi=\\dfrac{\\sqrt3}{2},\\ \\sin\\varphi=\\dfrac12$，均为正，故 $\\varphi=\\dfrac{\\pi}{6}$（第一象限）。',
        '$f(x)=2\\sin\\left(x+\\dfrac{\\pi}{6}\\right)$，最大值为 $2$。',
      ],
      tip: '选公式的总策略：看到**乘积**先想积化和差，看到**和差**先想和差化积；要合成一个波则用辅助角公式。',
    },
    {
      title: '想一想',
      body: '先自己算算，再点开看答案。',
      reveal: {
        q: '$f(x)=3\\sin x+4\\cos x$ 的最大值是多少？',
        a: '$R=\\sqrt{3^2+4^2}=\\sqrt{25}=5$，所以 $f(x)=5\\sin(x+\\varphi)$，最大值为 $5$（$3$-$4$-$5$ 直角三角形在这里又出现了）。',
      },
    },
  ],
}
