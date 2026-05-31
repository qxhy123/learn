import type { IntroCard } from '../../types'

// 学新知：第八部分（单元 22–24）各课的概念讲解卡片，按 lesson id 索引。
// 结构：动机/应用 → 核心(+公式/图) → 例题或推导(steps) → 提示(tip) → 互动(reveal)。
// 每单元首张概念卡配一张图：u22→trig-p8-22-1.svg，u23→trig-p8-23-1.svg，u24→trig-p8-24-1.svg。
export const INTRO8: Record<string, IntroCard[]> = {
  // ======================= 单元 22：信号·波与相量 =======================
  'u22-l1': [
    {
      title: '三角函数的“工程身份”',
      body:
        '当三角函数走出课本、进入工程，它最重要的身份就不再是“一个可以求值的函数”，而是描述**振动、波、交流电、相位关系**的语言。\n\n' +
        '现实里几乎所有周期性、平滑变化的量——交流电压、声波、无线电信号、机械振动——都能写成统一的标准形式，关键就在于读懂它的几个参数。',
      formula: 's(t)=A\\cos(\\omega t+\\varphi)',
      image: 'trig-p8-22-1.svg',
      imageCaption: '相量：复平面上以 $\\omega$ 旋转的向量，其在实轴的投影画出正弦信号 $A\\cos(\\omega t+\\varphi)$',
    },
    {
      title: '四个参数各管什么',
      body:
        '把任意正弦信号对照 $A\\cos(\\omega t+\\varphi)$，逐项读出含义：\n\n' +
        '- **振幅 $A$**：波的峰值，决定“多强”；\n' +
        '- **角频率 $\\omega$**（单位 rad/s）：每秒转过多少弧度，决定“多快”；\n' +
        '- **初相位 $\\varphi$**：$t=0$ 时刻的“起跑位置”，决定峰值何时出现。\n\n' +
        '相位不是无关紧要的小修正：它直接决定两个同频波叠加时是**相长**还是**相消**。',
    },
    {
      title: '周期与频率：由 $\\omega$ 一锤定音',
      body:
        '角频率定了，周期就定了——转满一圈 $2\\pi$ 弧度所需的时间就是**周期 $T$**。每秒转的圈数就是**频率 $f$**（单位 Hz）。三者互相换算：',
      formula: 'T=\\dfrac{2\\pi}{\\omega},\\qquad \\omega=2\\pi f,\\qquad f=\\dfrac{1}{T}',
      tip: '务必区分角频率 $\\omega$（每秒**弧度**）与频率 $f$（每秒**圈**），二者恒差一个 $2\\pi$：$\\omega=2\\pi f$。混用是这一节最常见的错。',
    },
    {
      title: '例题精讲：读懂一段交流电压',
      body: '某交流电压 $u(t)=12\\sin\\!\\big(100\\pi t+\\tfrac{\\pi}{6}\\big)$（伏，$t$ 秒）。逐项对照标准形式即可读出全部参数。',
      steps: [
        '振幅：直接读系数，$A=12$ 伏。',
        '角频率：$\\omega=100\\pi\\ \\text{rad/s}$。',
        '频率：$f=\\dfrac{\\omega}{2\\pi}=\\dfrac{100\\pi}{2\\pi}=50$ Hz——正是我国工频。',
        '周期：$T=\\dfrac{2\\pi}{\\omega}=\\dfrac{2\\pi}{100\\pi}=0.02$ 秒。',
        '初相位：$\\varphi=\\dfrac{\\pi}{6}$，相当于波形在时间上超前了 $\\dfrac{\\varphi}{\\omega}=\\dfrac{1}{600}$ 秒。',
      ],
    },
    {
      title: '试一试',
      body: '套公式 $f=\\dfrac{\\omega}{2\\pi}$ 即可，无需任何展开。',
      reveal: { q: '$u(t)=12\\sin(100\\pi t+\\tfrac{\\pi}{6})$ 的频率 $f=?$', a: '$f=\\dfrac{100\\pi}{2\\pi}=50$ Hz（我国民用工频）' },
    },
  ],

  'u22-l2': [
    {
      title: '波叠加是“结构问题”，不是死算',
      body:
        '把多个正弦波相加，与其硬展开，不如先看**结构**：它们是**同频**还是**频率相近**？这决定了用哪把工具，结果也呈现完全不同的样貌。\n\n' +
        '本课用一个老朋友——**和差化积**——把叠加拆成看得懂的几何与物理图景。',
    },
    {
      title: '同频叠加：仍是同一频率的正弦波',
      body:
        '两个**同频**正弦波相加，结果**还是同频正弦波**，只是振幅、相位变了。用辅助角公式把“正弦 + 余弦”合成单一正弦，振幅就是两系数的平方和开根：',
      formula: 'a\\cos\\theta+b\\sin\\theta=A\\sin(\\theta+\\phi),\\qquad A=\\sqrt{a^2+b^2}',
      tip: '确定相位 $\\phi$ 时要看两个分量的**符号**定象限，单凭 $\\arctan$ 会丢象限信息。',
    },
    {
      title: '拍频：快振荡被慢包络调制',
      body:
        '两个**频率相近**的波叠加，和差化积会把它分成“快振荡 × 慢包络”。以 $\\sin 101t+\\sin 99t$ 为例：',
      formula: '\\sin 101t+\\sin 99t=2\\sin 100t\\,\\cos t',
      steps: [
        '$\\sin\\alpha+\\sin\\beta=2\\sin\\tfrac{\\alpha+\\beta}{2}\\cos\\tfrac{\\alpha-\\beta}{2}$；',
        '$\\sin 100t$ 是**快振荡**（载波）；',
        '$2\\cos t$ 是**慢包络**，缓慢地把振幅“涨落”——这就是听到的强弱起伏。',
      ],
    },
    {
      title: '应用：调音师靠拍频对准音高',
      body:
        '两音 $440$ Hz 与 $444$ Hz 叠加，包络让响度周期性涨落。响度由振幅的**绝对值**决定，正负峰都“响”，所以每秒听到的起伏次数（拍频）等于两频率之差 $|f_1-f_2|$。\n\n' +
        '两根弦越接近，拍越慢；调到完全听不出起伏，就对准了。',
    },
    {
      title: '试一试',
      body: '拍频公式：$f_{\\text{beat}}=|f_1-f_2|$。',
      reveal: { q: '$440$ Hz 与 $444$ Hz 叠加，每秒几次强弱起伏？', a: '$|444-440|=4$ 次/秒' },
    },
  ],

  'u22-l3': [
    {
      title: '相量：把“转动的波”冻结成向量',
      body:
        '直接对多个正弦波做三角运算很累。**相量法**换一个表示系统：把 $A\\cos(\\omega t+\\varphi)$ 看成复平面上一个以角速度 $\\omega$ 旋转的向量——它在实轴上的**投影**正好画出这条正弦波。\n\n' +
        '向量的**模**就是振幅 $A$，**辐角**就是初相 $\\varphi$。',
      formula: 'A\\cos(\\omega t+\\varphi)\\ \\longleftrightarrow\\ \\mathbf V=A\\angle\\varphi',
      image: 'trig-p8-22-1.svg',
      imageCaption: '旋转向量（模 $A$、辐角 $\\omega t+\\varphi$）在实轴投影即得正弦信号',
    },
    {
      title: '为什么省事：公共因子 $e^{i\\omega t}$ 可约去',
      body:
        '同频的所有波都带着同一个旋转因子 $e^{i\\omega t}$。既然大家一起转，就可以把它**约去**，只留下一个不随时间变的复数（相量）。\n\n' +
        '于是“波的叠加”降级为“向量相加”——一道反复用和差化积的题，变成复平面上的几何加法。',
    },
    {
      title: '例题精讲：用相量合成两个余弦',
      body: '求 $\\cos t+\\cos\\!\\big(t+\\tfrac{\\pi}{3}\\big)$ 的合成振幅与相位。',
      steps: [
        '写成相量：$\\mathbf V_1=1\\angle 0$，$\\mathbf V_2=1\\angle\\tfrac{\\pi}{3}$；',
        '化直角坐标相加：$\\mathbf V_1+\\mathbf V_2=(1+\\cos 60^\\circ)+i\\sin 60^\\circ=\\tfrac32+i\\tfrac{\\sqrt3}{2}$；',
        '合成振幅 $A=\\sqrt{(\\tfrac32)^2+(\\tfrac{\\sqrt3}{2})^2}=\\sqrt3$；',
        '合成相位 $\\varphi=\\arctan\\dfrac{\\sqrt3/2}{3/2}=\\dfrac{\\pi}{6}$，得 $\\sqrt3\\cos\\!\\big(t+\\tfrac{\\pi}{6}\\big)$。',
      ],
    },
    {
      title: '提示与互动',
      body: '相量 $A\\angle\\varphi$ 的位置由辐角符号决定：实部正、虚部负 → 第四象限。',
      tip: '相量法在同频叠加里特别高效——超过两个波时优势更明显，全部归结为复平面上的加减。',
      reveal: { q: '用相量求 $\\sin\\theta+\\sqrt3\\cos\\theta$ 的振幅', a: '$1\\angle 0+\\sqrt3\\angle\\tfrac{\\pi}{2}=1+\\sqrt3\\,i$，模 $=\\sqrt{1+3}=2$' },
    },
  ],

  // ======================= 单元 23：旋转与向量 =======================
  'u23-l1': [
    {
      title: '旋转里为什么一定冒出三角函数',
      body:
        '把平面上的点绕原点逆时针旋转 $\\theta$，可以用**一个矩阵乘法**一次完成。这个矩阵不是死记的公式，而是**基向量旋转后坐标拼起来**的自然结果。\n\n' +
        '看清这一点，整章的旋转矩阵就再也不用背了。',
      formula: 'R(\\theta)=\\begin{pmatrix}\\cos\\theta&-\\sin\\theta\\\\\\\\\\sin\\theta&\\cos\\theta\\end{pmatrix}',
      image: 'trig-p8-23-1.svg',
      imageCaption: '旋转矩阵 $R(\\theta)$ 把向量 $\\mathbf v$ 旋转 $\\theta$ 得 $\\mathbf v\'$，模长不变',
    },
    {
      title: '矩阵的两列 = 基向量的去向',
      body:
        '把标准基向量旋转 $\\theta$：\n\n' +
        '- $(1,0)$ 转到 $(\\cos\\theta,\\sin\\theta)$——这就是单位圆上的点，作矩阵**第一列**；\n' +
        '- $(0,1)$ 转到 $(-\\sin\\theta,\\cos\\theta)$——作矩阵**第二列**。\n\n' +
        '把两列拼起来，正好得到 $R(\\theta)$。所以“单位圆上的点坐标”其实就是“旋转后的标准基向量”。',
      formula: '(1,0)\\to(\\cos\\theta,\\sin\\theta),\\qquad (0,1)\\to(-\\sin\\theta,\\cos\\theta)',
      tip: '负号在**右上角**：$-\\sin\\theta$ 在第一行第二列。写错位置会变成顺时针旋转。',
    },
    {
      title: '例题精讲：把 $(3,4)$ 逆时针转 $90^\\circ$',
      body: '代 $\\theta=90^\\circ$，$\\cos 90^\\circ=0$、$\\sin 90^\\circ=1$。',
      steps: [
        '$R(90^\\circ)=\\begin{pmatrix}0&-1\\\\\\\\1&0\\end{pmatrix}$；',
        '$R(90^\\circ)\\begin{pmatrix}3\\\\\\\\4\\end{pmatrix}=\\begin{pmatrix}0\\cdot 3+(-1)\\cdot 4\\\\\\\\1\\cdot 3+0\\cdot 4\\end{pmatrix}=\\begin{pmatrix}-4\\\\\\\\3\\end{pmatrix}$；',
        '即 $90^\\circ$ 旋转的效果是 $(x,y)\\mapsto(-y,x)$；',
        '验证模长不变：$\\sqrt{3^2+4^2}=\\sqrt{(-4)^2+3^2}=5$。',
      ],
    },
    {
      title: '试一试',
      body: '旋转是正交变换，**不改变模长**。极坐标视角：把 $(r,0)$ 转 $\\theta$ 得 $(r\\cos\\theta,r\\sin\\theta)$。',
      reveal: { q: '把 $(1,0)$ 逆时针转 $90^\\circ$？', a: '$R(90^\\circ)\\binom{1}{0}=\\binom{0}{1}$，即 $(1,0)\\to(0,1)$' },
    },
  ],

  'u23-l2': [
    {
      title: '点积：余弦量“方向相似度”',
      body:
        '余弦不是孤立的函数，它直接藏在向量的**点积**里：两向量越同向，点积越大；垂直时点积为零；反向时为负。这就把“方向有多接近”量化成了一个余弦。',
      formula: '\\mathbf a\\cdot\\mathbf b=|\\mathbf a||\\mathbf b|\\cos\\theta\\quad\\Longrightarrow\\quad \\cos\\theta=\\dfrac{\\mathbf a\\cdot\\mathbf b}{|\\mathbf a||\\mathbf b|}',
    },
    {
      title: '旋转矩阵把和角公式“装进矩阵”',
      body:
        '把 $R_\\alpha$ 与 $R_\\beta$ 相乘，逐元素用和角公式，会发现结果恰好是 $R_{\\alpha+\\beta}$——“先转 $\\beta$ 再转 $\\alpha$”等于“一次转 $\\alpha+\\beta$”。矩阵恒等式与三角和角公式互为表里。',
      formula: 'R_\\alpha R_\\beta=R_{\\alpha+\\beta}',
      steps: [
        '左上元：$\\cos\\alpha\\cos\\beta-\\sin\\alpha\\sin\\beta=\\cos(\\alpha+\\beta)$；',
        '左下元：$\\sin\\alpha\\cos\\beta+\\cos\\alpha\\sin\\beta=\\sin(\\alpha+\\beta)$；',
        '四个元素拼好，正是 $R_{\\alpha+\\beta}$。',
      ],
    },
    {
      title: '旋转的另外两条性质',
      body:
        '由 $R_\\alpha R_\\beta=R_{\\alpha+\\beta}$ 取 $\\beta=-\\alpha$，立刻得到：\n\n' +
        '- **逆 = 反向旋转 = 转置**：$R_\\theta^{-1}=R_{-\\theta}=R_\\theta^{T}$（正交矩阵，求逆不必消元，直接转置）；\n' +
        '- **行列式恒为 $1$**：$\\det R_\\theta=\\cos^2\\theta+\\sin^2\\theta=1$，说明旋转不缩放、不翻面。',
      formula: 'R_\\theta^{-1}=R_{-\\theta}=R_\\theta^{T},\\qquad \\det R_\\theta=1',
    },
    {
      title: '注意维度：三维不可交换',
      body: '二维旋转可交换（$R_\\alpha R_\\beta=R_\\beta R_\\alpha$），但**三维一般不可交换**——绕不同轴先后旋转，顺序会改变结果，$R_x(\\alpha)R_z(\\beta)\\neq R_z(\\beta)R_x(\\alpha)$。',
      tip: '魔方、机械臂、航天器姿态都受此限制：旋转顺序是有意义的信息，不能随意交换。',
      reveal: { q: '$\\mathbf a=(1,\\sqrt3),\\ \\mathbf b=(\\sqrt3,1)$ 的夹角？', a: '$\\cos\\theta=\\dfrac{2\\sqrt3}{2\\cdot 2}=\\dfrac{\\sqrt3}{2}\\Rightarrow\\theta=30^\\circ$' },
    },
  ],

  'u23-l3': [
    {
      title: '复数乘法就是旋转',
      body:
        '在复平面里，乘以单位复数 $e^{i\\theta}$ 等于把整个平面**逆时针旋转 $\\theta$**。这与上一课的旋转矩阵讲的是**同一件事**——只是换了一套记号。\n\n' +
        '若再带上模长 $r$，乘 $re^{i\\theta}$ 就同时完成“旋转 + 缩放”，比矩阵更紧凑。',
      formula: 'e^{i\\theta}=\\cos\\theta+i\\sin\\theta',
    },
    {
      title: '两种语言，同一动作',
      body:
        '把复数 $x+iy$ 乘 $e^{i\\theta}$，展开后实部、虚部恰好就是 $R_\\theta\\binom{x}{y}$ 的两个分量。**欧拉公式**正是连接矩阵世界与复数世界的那座桥。',
      formula: 'e^{i\\theta}(x+iy)=(x\\cos\\theta-y\\sin\\theta)+i(x\\sin\\theta+y\\cos\\theta)',
    },
    {
      title: '例题精讲：把 $1+i$ 旋转 $45^\\circ$',
      body: '复数相乘的口诀：**模相乘、辐角相加**。',
      steps: [
        '$z=1+i$ 的模 $\\sqrt2$、辐角 $45^\\circ$；',
        '乘 $e^{i45^\\circ}$：模仍 $\\sqrt2$（因 $|e^{i\\theta}|=1$），辐角变 $45^\\circ+45^\\circ=90^\\circ$；',
        '辐角 $90^\\circ$、模 $\\sqrt2$ 的复数就是 $\\sqrt2\\,i$；',
        '矩阵验证：$R_{45^\\circ}\\binom{1}{1}=\\binom{0}{\\sqrt2}$，对应复数 $\\sqrt2\\,i$，两法一致。',
      ],
    },
    {
      title: '提示与互动',
      body: '三维里绕 $z$ 轴旋转就是让 $z$ 不动、对 $(x,y)$ 做二维旋转：$(1,0,0)\\to(0,1,0)$。',
      tip: '凡是涉及旋转、幂、单位根的问题，复数/欧拉公式往往比堆三角恒等式一步到位。',
      reveal: { q: '$z=1+i$ 乘 $e^{i45^\\circ}$ 等于？', a: '模 $\\sqrt2$、辐角 $90^\\circ$，即 $\\sqrt2\\,i$' },
    },
  ],

  // ======================= 单元 24：高级综合与展望 =======================
  'u24-l1': [
    {
      title: '高阶题最难的是“选方法”',
      body:
        '真正高阶的三角题，难点常常不在公式复杂，而在于能否**几秒内判断**：这题该用单位圆、图像、恒等式、几何、复数还是信号的语言来描述。\n\n' +
        '常见的失败不是不会算，而是“在代数里硬算一个本该画图的题”“在单位圆里死抠一个本该用复数的题”。所以第一步永远是：**先判断表示系统，再动手计算。**',
      image: 'trig-p8-24-1.svg',
      imageCaption: '李萨如图形 $x=\\sin 3t,\\ y=\\sin 2t$：一条曲线里同时藏着旋转、投影与周期三条主线',
    },
    {
      title: '题型信号 → 优先方法',
      body:
        '熟练者看到题目结构，往往就知道走哪条路：\n\n' +
        '- 角位置、象限、符号 → **单位圆**；\n' +
        '- 周期、最值、区间 → **图像**；\n' +
        '- $a\\sin x+b\\cos x$ → **辅助角压缩**；\n' +
        '- 一般三角形 → **正弦/余弦定理**；\n' +
        '- 旋转、幂、单位根 → **复数 / 欧拉公式**；\n' +
        '- 周期叠加、交流信号 → **相量 / 拍频**。',
    },
    {
      title: '最常用入口：辅助角',
      body:
        '看到 $a\\sin x+b\\cos x$ 求最值，第一反应就是辅助角——一步合成单一正弦，最值立刻是振幅 $\\sqrt{a^2+b^2}$，不必平方、求导。',
      formula: 'a\\sin x+b\\cos x=\\sqrt{a^2+b^2}\\,\\sin(x+\\phi)',
      steps: [
        '$\\sin x+\\cos x=\\sqrt2\\sin\\!\\big(x+\\tfrac{\\pi}{4}\\big)$，最大值 $\\sqrt2$；',
        '$\\sin x+\\sqrt3\\cos x=2\\sin\\!\\big(x+\\tfrac{\\pi}{3}\\big)$，最大值 $2$。',
      ],
    },
    {
      title: '提示与互动',
      body: '“正确方法让题目直接变短，错误方法让题目显得很难”——这是本章贯穿始终的主张。',
      tip: '“两边及夹角求第三边”是余弦定理的招牌信号：$a^2=b^2+c^2-2bc\\cos A$，无需先求别的角。',
      reveal: { q: '$\\triangle ABC$ 中 $b=7,\\ c=5,\\ A=120^\\circ$，求 $a$', a: '$a^2=49+25-2\\cdot 7\\cdot 5\\cos 120^\\circ=74+35=109$，$a=\\sqrt{109}$' },
    },
  ],

  'u24-l2': [
    {
      title: '复数统一旋转与多倍角',
      body:
        '当题目涉及**多倍角、幂、单位根**时，与其堆叠三角恒等式，不如搬进复数。**棣莫弗定理**把“求 $n$ 倍角”变成“求 $n$ 次幂”：',
      formula: '(\\cos\\theta+i\\sin\\theta)^n=\\cos n\\theta+i\\sin n\\theta',
    },
    {
      title: '例题精讲：一步推出三倍角',
      body: '取 $n=3$，对 $(\\cos\\theta+i\\sin\\theta)^3$ 做二项展开再取实部。',
      steps: [
        '展开：$(\\cos\\theta+i\\sin\\theta)^3=\\cos^3\\theta+3i\\cos^2\\theta\\sin\\theta-3\\cos\\theta\\sin^2\\theta-i\\sin^3\\theta$；',
        '取实部：$\\cos 3\\theta=\\cos^3\\theta-3\\cos\\theta\\sin^2\\theta$；',
        '代 $\\sin^2\\theta=1-\\cos^2\\theta$：$\\cos 3\\theta=4\\cos^3\\theta-3\\cos\\theta$；',
        '这正是切比雪夫多项式 $T_3$，后续在数值分析与逼近论中反复出现。',
      ],
    },
    {
      title: '值域题也偏爱结构化方法',
      body:
        '求 $g(x)=\\dfrac{\\sin x}{2+\\cos x}$ 的值域，与其逐点求导，不如设 $y=g(x)$ 整理成 $\\sin x-y\\cos x=2y$。左边是 $a\\sin x+b\\cos x$ 型，振幅 $\\sqrt{1+y^2}$，方程有解须 $|2y|\\le\\sqrt{1+y^2}$，解得 $3y^2\\le 1$，值域 $\\left[-\\tfrac{\\sqrt3}{3},\\tfrac{\\sqrt3}{3}\\right]$。',
      formula: '|a\\sin x+b\\cos x|\\le\\sqrt{a^2+b^2}',
      tip: '把“求值域”转化为“含参三角方程何时有解”，再用振幅不等式卡参数——比硬求导更看清结构。',
    },
    {
      title: '最美的恒等式',
      body: '欧拉公式在 $\\theta=\\pi$ 处把 $e,\\ i,\\ \\pi,\\ 1,\\ 0$ 五个最重要的常数串成一行——常被誉为数学中最美的等式。',
      formula: 'e^{i\\pi}+1=0',
      reveal: { q: '$e^{i\\pi}+1=?$', a: '$e^{i\\pi}=-1\\Rightarrow e^{i\\pi}+1=0$（欧拉恒等式）' },
    },
  ],

  'u24-l3': [
    {
      title: '一道题，串起所有主线',
      body:
        '本课是全教程的收束：把**辅助角合成、正/余弦定理、相量叠加、旋转矩阵、复数/欧拉**这几条主线汇到一起。\n\n' +
        '学完后你应具备三种能力——**快速归约**（任意角归到参考角与象限）、**结构识别**（判断题目属于哪类表示）、**表示切换**（在图像、代数、几何、复数、频率之间自由换轨）。',
      formula: 'R(\\theta)=\\begin{pmatrix}\\cos\\theta&-\\sin\\theta\\\\\\\\\\sin\\theta&\\cos\\theta\\end{pmatrix},\\quad T=\\dfrac{2\\pi}{\\omega},\\quad e^{i\\theta}=\\cos\\theta+i\\sin\\theta',
    },
    {
      title: '三相为零：同一现象的三副面孔',
      body:
        '三个等幅、相位互差 $120^\\circ$ 的同频波之和**恒为零**——这一个事实，可以从三个角度看：',
      formula: '\\cos\\omega t+\\cos\\!\\big(\\omega t+\\tfrac{2\\pi}{3}\\big)+\\cos\\!\\big(\\omega t+\\tfrac{4\\pi}{3}\\big)\\equiv 0',
      steps: [
        '**相量视角**：三个等长向量两两成 $120^\\circ$，首尾相接闭合成正三角形，合矢量为零；',
        '**旋转视角**：每个相量由前一个绕原点作 $R_{120^\\circ}$ 得到，三步回到原点；',
        '**单位根视角**：等价于 $1+\\omega+\\omega^2=0$（$\\omega=e^{i2\\pi/3}$）——也正是三相交流电平衡、中性线电流为零的根据。',
      ],
    },
    {
      title: '综合演练：方法选择',
      body: '同一份知识，换不同信号就调不同武器。',
      steps: [
        '信号周期：$\\omega=4\\pi\\Rightarrow T=\\dfrac{2\\pi}{4\\pi}=\\dfrac12$；',
        '旋转：$(0,1)$ 逆时针 $90^\\circ\\Rightarrow(x,y)\\mapsto(-y,x)\\Rightarrow(-1,0)$；',
        '辅助角：$2\\sin x+2\\cos x$ 振幅 $\\sqrt{2^2+2^2}=2\\sqrt2$；',
        '解三角形：“两边夹角求第三边”用余弦定理。',
      ],
    },
    {
      title: '展望：三条延伸路线',
      body:
        '三角函数是后续三条路的**共同语言**：\n\n' +
        '- **微积分**——极限、导数与 Fourier 分析；\n' +
        '- **线性代数 / 几何**——旋转矩阵与正交变换；\n' +
        '- **信号处理**——拍频、调制、频谱与相量。\n\n' +
        '把它们贯通的，始终是单位圆上那一对 $\\cos\\theta,\\ \\sin\\theta$。',
      reveal: { q: '把 $(0,1)$ 逆时针转 $90^\\circ$？', a: '$(x,y)\\mapsto(-y,x)$，故 $(0,1)\\to(-1,0)$' },
    },
  ],
}
