import type { IntroCard } from '../../types'

// 学新知：第八部分（单元 22–24）各课的概念讲解卡片，按 lesson id 索引。
export const INTRO8: Record<string, IntroCard[]> = {
  'u22-l1': [
    {
      title: '正弦信号的四个参数',
      body: '一个正弦信号写成 $A\\cos(\\omega t+\\varphi)$：$A$ 是**振幅**（峰值），$\\omega$ 是**角频率**（每秒转过多少弧度），$\\varphi$ 是**初相位**（起跑时刻）。',
      formula: 's(t)=A\\cos(\\omega t+\\varphi)',
    },
    {
      title: '周期与频率',
      body: '角频率定下来，周期就定了：转满一圈 $2\\pi$ 需要的时间就是周期 $T$。频率 $f$（每秒几圈）与 $\\omega$ 差一个 $2\\pi$。',
      formula: 'T=\\frac{2\\pi}{\\omega},\\qquad \\omega=2\\pi f',
    },
    {
      title: '试一试',
      body: '直接套公式 $T=\\dfrac{2\\pi}{\\omega}$。',
      reveal: { q: '$u(t)=12\\sin(100\\pi t+\\tfrac{\\pi}{6})$ 的频率？', a: '$f=\\dfrac{\\omega}{2\\pi}=\\dfrac{100\\pi}{2\\pi}=50$ Hz（工频）' },
    },
  ],
  'u22-l2': [
    {
      title: '同频叠加',
      body: '两个**同频**正弦波相加，结果仍是同频正弦波，只是振幅、相位变了。用辅助角公式 $a\\cos\\theta+b\\sin\\theta=A\\sin(\\theta+\\phi)$ 合成，$A=\\sqrt{a^2+b^2}$。',
      formula: 'A=\\sqrt{a^2+b^2}',
    },
    {
      title: '拍频：慢包络调制快振荡',
      body: '两个**频率相近**的波叠加，用和差化积会分出“快振荡 × 慢包络”，听上去就是强弱起伏的**拍**。拍频 $=|f_1-f_2|$。',
      formula: '\\sin\\alpha+\\sin\\beta=2\\sin\\tfrac{\\alpha+\\beta}{2}\\cos\\tfrac{\\alpha-\\beta}{2}',
    },
    {
      title: '听一听',
      body: '调音时就靠拍频：两音越接近，起伏越慢。',
      reveal: { q: '$440$ Hz 与 $444$ Hz 叠加，每秒几次起伏？', a: '拍频 $=|444-440|=4$ 次/秒' },
    },
  ],
  'u22-l3': [
    {
      title: '把波“冻结”成向量',
      body: '**相量**法：把 $A\\cos(\\omega t+\\varphi)$ 看成复平面上一个以 $\\omega$ 旋转的向量，**模**是振幅 $A$、**辐角**是初相 $\\varphi$。记作 $A\\angle\\varphi$。',
      formula: 'A\\cos(\\omega t+\\varphi)\\ \\longleftrightarrow\\ A\\angle\\varphi',
    },
    {
      title: '为什么省事',
      body: '同频的波都带公共因子 $e^{i\\omega t}$，可以**约去**。于是“波的叠加”就变成不随时间变的**向量相加**——三角难题降级成几何加法。',
    },
    {
      title: '试一试',
      body: '把每项写成相量再相加，最后取模。',
      reveal: { q: '用相量求 $\\sin\\theta+\\sqrt3\\cos\\theta$ 的振幅', a: '$1\\angle 0+\\sqrt3\\angle\\tfrac{\\pi}{2}=1+\\sqrt3\\,i$，模 $=\\sqrt{1+3}=2$' },
    },
  ],
  'u23-l1': [
    {
      title: '二维旋转矩阵',
      body: '把平面上的点绕原点逆时针旋转 $\\theta$，可以用一个矩阵乘法实现。这个矩阵的两列，正是两个基向量旋转后的去向。',
      formula: 'R(\\theta)=\\begin{pmatrix}\\cos\\theta&-\\sin\\theta\\\\\\sin\\theta&\\cos\\theta\\end{pmatrix}',
    },
    {
      title: '列 = 基向量的去向',
      body: '第一列是 $(1,0)$ 旋转后的坐标 $(\\cos\\theta,\\sin\\theta)$，第二列是 $(0,1)$ 旋转后的 $(-\\sin\\theta,\\cos\\theta)$。旋转**不改变模长**。',
      formula: '(1,0)\\ \\to\\ (\\cos\\theta,\\sin\\theta)',
    },
    {
      title: '试一试 $90^\\circ$',
      body: '代 $\\theta=90^\\circ$：$R=\\begin{pmatrix}0&-1\\\\1&0\\end{pmatrix}$，效果是 $(x,y)\\mapsto(-y,x)$。',
      reveal: { q: '把 $(1,0)$ 逆时针转 $90^\\circ$？', a: '$(1,0)\\to(0,1)$' },
    },
  ],
  'u23-l2': [
    {
      title: '点积量“方向相似度”',
      body: '两个向量的**点积**含一个余弦：$\\mathbf a\\cdot\\mathbf b=|\\mathbf a||\\mathbf b|\\cos\\theta$。夹角越小余弦越大，方向越接近；垂直时点积为 $0$。',
      formula: '\\cos\\theta=\\frac{\\mathbf a\\cdot\\mathbf b}{|\\mathbf a||\\mathbf b|}',
    },
    {
      title: '旋转的三条性质',
      body: '旋转矩阵把和角公式“装进了矩阵”：连转两次等于一次合角；逆矩阵就是反向旋转；行列式恒为 $1$（不缩放、不翻面）。',
      formula: 'R_\\alpha R_\\beta=R_{\\alpha+\\beta},\\quad R_\\theta^{-1}=R_{-\\theta},\\quad \\det R_\\theta=1',
    },
    {
      title: '注意维度',
      body: '二维旋转可交换，三维一般**不可交换**——绕不同轴先后旋转，顺序会影响结果。',
      reveal: { q: '$\\mathbf a=(1,\\sqrt3),\\ \\mathbf b=(\\sqrt3,1)$ 的夹角？', a: '$\\cos\\theta=\\dfrac{2\\sqrt3}{2\\cdot2}=\\dfrac{\\sqrt3}{2}\\Rightarrow\\theta=30^\\circ$' },
    },
  ],
  'u23-l3': [
    {
      title: '复数乘法 = 旋转',
      body: '在复平面里，乘以单位复数 $e^{i\\theta}$ 就是把整个平面**逆时针旋转 $\\theta$**。这与旋转矩阵讲的是同一件事。',
      formula: 'e^{i\\theta}=\\cos\\theta+i\\sin\\theta',
    },
    {
      title: '两种语言，同一动作',
      body: '把复数 $x+iy$ 乘 $e^{i\\theta}$，得到的实部、虚部恰好就是 $R_\\theta\\binom{x}{y}$ 的两个分量。欧拉公式正是连接矩阵与复数的桥梁。',
      formula: 'e^{i\\theta}(x+iy)\\ \\longleftrightarrow\\ R_\\theta\\binom{x}{y}',
    },
    {
      title: '试一试',
      body: '复数相乘：模相乘、辐角相加。',
      reveal: { q: '$z=1+i$ 乘 $e^{i45^\\circ}$？', a: '模 $\\sqrt2$、辐角 $45^\\circ$，再转 $45^\\circ$ 到 $90^\\circ$，得 $\\sqrt2\\,i$' },
    },
  ],
  'u24-l1': [
    {
      title: '先选系统，再动手',
      body: '高阶三角题的第一步不是硬算，而是**判断该用哪种表示系统**：单位圆、图像、恒等式、几何（正/余弦定理）、复数、信号。选对了，题目往往直接变短。',
    },
    {
      title: '常用入口：辅助角',
      body: '看到 $a\\sin x+b\\cos x$ 求最值，第一反应应是辅助角公式，一步合成一个正弦，最值就是振幅 $\\sqrt{a^2+b^2}$。',
      formula: 'a\\sin x+b\\cos x=\\sqrt{a^2+b^2}\\,\\sin(x+\\phi)',
    },
    {
      title: '试一试',
      body: '“两边及夹角求第三边”是余弦定理的招牌信号。',
      reveal: { q: '$b=7,c=5,A=120^\\circ$，求 $a$', a: '$a^2=49+25-2\\cdot7\\cdot5\\cos120^\\circ=109,\\ a=\\sqrt{109}$' },
    },
  ],
  'u24-l2': [
    {
      title: '复数统一旋转与多倍角',
      body: '**棣莫弗定理** $(\\cos\\theta+i\\sin\\theta)^n=\\cos n\\theta+i\\sin n\\theta$。展开取实部/虚部，就能一举得到多倍角公式，比堆恒等式高效得多。',
      formula: '(\\cos\\theta+i\\sin\\theta)^n=\\cos n\\theta+i\\sin n\\theta',
    },
    {
      title: '例：三倍角',
      body: '取 $n=3$ 展开 $(\\cos\\theta+i\\sin\\theta)^3$ 的实部，代 $\\sin^2=1-\\cos^2$，得到的正是切比雪夫多项式 $T_3$。',
      formula: '\\cos 3\\theta=4\\cos^3\\theta-3\\cos\\theta',
    },
    {
      title: '最美的恒等式',
      body: '欧拉公式在 $\\theta=\\pi$ 处串起 $e,i,\\pi,1,0$ 五个常数。',
      reveal: { q: '$e^{i\\pi}+1=?$', a: '$e^{i\\pi}=-1\\Rightarrow e^{i\\pi}+1=0$（欧拉恒等式）' },
    },
  ],
  'u24-l3': [
    {
      title: '一道题，多条主线',
      body: '本课把整套教程的主线串起来：**辅助角**合成、**正/余弦定理**解三角形、**相量**叠加、**旋转矩阵**与**复数/欧拉**。识别结构，调用对应武器。',
      formula: 'R(\\theta)=\\begin{pmatrix}\\cos\\theta&-\\sin\\theta\\\\\\sin\\theta&\\cos\\theta\\end{pmatrix},\\quad T=\\frac{2\\pi}{\\omega},\\quad e^{i\\theta}=\\cos\\theta+i\\sin\\theta',
    },
    {
      title: '三相为零：同一现象的三副面孔',
      body: '三个等幅、互差 $120^\\circ$ 的波之和为零——既是相量正三角形闭合，也是单位根 $1+\\omega+\\omega^2=0$，还是三相交流平衡。',
      formula: '\\cos\\omega t+\\cos(\\omega t+\\tfrac{2\\pi}{3})+\\cos(\\omega t+\\tfrac{4\\pi}{3})\\equiv 0',
    },
    {
      title: '展望',
      body: '三角函数是后续三条路的共同语言：**微积分**（导数与 Fourier）、**线性代数/几何**（旋转与正交变换）、**信号处理**（拍频、相量）。',
      reveal: { q: '把 $(0,1)$ 逆时针转 $90^\\circ$？', a: '$(x,y)\\mapsto(-y,x)$，故 $(0,1)\\to(-1,0)$' },
    },
  ],
}
