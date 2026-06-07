// 第29章  矩阵微积分（第26章知识点）
// 关 c29-a：向量/矩阵求导与 Jacobian（8 卡 8 题）
// 关 c29-b：链式法则与反向传播（7 卡 7 题）
// 关 c29-c：自动微分与 Softmax 梯度（6 卡 6 题）

export const LESSONS = [
  // ─────────────────────────────────────────
  // 关 c29-a  向量/矩阵求导与 Jacobian
  // ─────────────────────────────────────────
  {
    id: 'c29-a',
    title: '向量、矩阵求导与 Jacobian',
    subtitle: '动机·求导对象类型·线性型·二次型·Jacobian 定义',
    intro: [
      // 卡 1：动机——反向传播需要什么？
      {
        title: '为什么需要矩阵微积分？',
        body:
          '训练神经网络的核心操作是**反向传播**：给定损失函数 $L$，对每个参数矩阵 $W$ 计算梯度 $\\partial L/\\partial W$，再用梯度下降更新参数。\n\n' +
          '一元微积分的导数 $df/dx$ 只处理"标量对标量"，但神经网络的参数动辄是数百万维的向量或矩阵。\n\n' +
          '**问题**：如果 $L$ 是标量，$W$ 是一个 $1000\\times 1000$ 的矩阵，那"$L$ 对 $W$ 的导数"是什么形状？\n\n' +
          '答案：它和 $W$ 形状相同——也是 $1000\\times 1000$ 的矩阵，每个元素 $(i,j)$ 对应 $\\partial L/\\partial W_{ij}$。\n\n' +
          '矩阵微积分就是把这些"高维导数"系统化地写出来，让链式法则依然可以用。',
        tip: '核心动机：让"参数是矩阵"的偏导数也能像一维导数一样链式计算。',
      },
      // 卡 2：四种求导对象
      {
        title: '求导对象的四种类型',
        body:
          '根据**分子**（被求导量）和**分母**（求导变量）的形态，矩阵求导分四种：\n\n' +
          '① **标量对标量**：$df/dx$，普通导数，大家熟悉。\n\n' +
          '② **标量对向量**（梯度）：$f$ 是标量，$x\\in\\mathbb{R}^n$，结果 $\\nabla_x f\\in\\mathbb{R}^n$ 是列向量，每个分量 $= \\partial f/\\partial x_i$。\n\n' +
          '③ **向量对向量**（Jacobian）：$y\\in\\mathbb{R}^m$，$x\\in\\mathbb{R}^n$，结果是 $m\\times n$ 矩阵，$(i,j)$ 元素 $= \\partial y_i/\\partial x_j$。\n\n' +
          '④ **标量对矩阵**：$f$ 是标量，$A\\in\\mathbb{R}^{m\\times n}$，结果是 $m\\times n$ 矩阵，$(i,j)$ 元素 $= \\partial f/\\partial A_{ij}$。\n\n' +
          '**易错点**：矩阵微积分最大的坑不是"不会求导"，而是**布局约定混用**——不同教材对梯度是行向量还是列向量并不一致，混用后链式法则顺序会出错。',
        formula: 'J_{ij}=\\frac{\\partial y_i}{\\partial x_j},\\quad \\nabla_x f=\\left[\\frac{\\partial f}{\\partial x_1},\\ldots,\\frac{\\partial f}{\\partial x_n}\\right]^{\\top}',
        tip: '本课统一采用**分母布局**：梯度 $\\nabla_x f$ 是列向量，Jacobian $(i,j)$ 元 $=\\partial y_i/\\partial x_j$。做题前先确认约定！',
      },
      // 卡 3：线性型求导——最基础公式
      {
        title: '线性型求导：$\\nabla_x(a^{\\top}x) = a$',
        body:
          '设 $a,x\\in\\mathbb{R}^n$，$f(x)=a^{\\top}x=\\sum_{i=1}^n a_i x_i$。\n\n' +
          '对第 $k$ 个分量求偏导：$\\dfrac{\\partial f}{\\partial x_k}=a_k$。\n\n' +
          '写成梯度向量：$\\nabla_x f=[a_1,a_2,\\ldots,a_n]^{\\top}=a$。\n\n' +
          '**直觉**：$a^{\\top}x$ 对 $x$ 是线性的，就像 $f(x)=3x$ 的导数是 $3$——系数即梯度。\n\n' +
          '**扩展**：若 $y=Ax$，$A\\in\\mathbb{R}^{m\\times n}$，$x\\in\\mathbb{R}^n$，则 $\\partial(Ax)/\\partial x=A$（Jacobian 就是矩阵 $A$ 本身）。\n\n' +
          '**数字例**：$f(x)=2x_1-3x_2+x_3$，则 $\\nabla_x f=[2,-3,1]^{\\top}$，等于系数向量 $a=[2,-3,1]^{\\top}$。',
        formula: '\\nabla_x(a^{\\top}x)=a,\\quad \\frac{\\partial(Ax)}{\\partial x}=A',
        tip: '线性函数的梯度 $=$ 系数，这是所有矩阵求导公式中最基础的一条。',
      },
      // 卡 4：二次型求导——分量推导
      {
        title: '二次型求导：$\\nabla_x(x^{\\top}Ax)=(A+A^{\\top})x$',
        body:
          '设 $A\\in\\mathbb{R}^{n\\times n}$，$x\\in\\mathbb{R}^n$，$f(x)=x^{\\top}Ax=\\sum_{j,k}A_{jk}x_jx_k$。\n\n' +
          '我们用**分量法**推导，避免凭空记公式：\n\n' +
          '对第 $i$ 个分量 $x_i$ 求偏导，$x_i$ 出现在两类项中：\n\n' +
          '（1）$j=i$ 的项 $\\sum_k A_{ik}x_ix_k$，对 $x_i$ 求导得 $\\sum_k A_{ik}x_k=(Ax)_i$；\n\n' +
          '（2）$k=i$ 的项 $\\sum_j A_{ji}x_jx_i$，对 $x_i$ 求导得 $\\sum_j A_{ji}x_j=(A^{\\top}x)_i$。\n\n' +
          '合并：$\\dfrac{\\partial f}{\\partial x_i}=(Ax)_i+(A^{\\top}x)_i$，写成向量形式就是 $\\nabla_x f=(A+A^{\\top})x$。\n\n' +
          '**当 $A$ 对称时**（$A=A^{\\top}$）：$\\nabla_x f=2Ax$。',
        formula: '\\nabla_x(x^{\\top}Ax)=(A+A^{\\top})x\\quad(A\\text{ 对称时}=2Ax)',
        steps: [
          '写出 $f=\\sum_{j,k}A_{jk}x_jx_k$。',
          '对 $x_i$ 求偏导，找包含 $x_i$ 的项：$j=i$ 一类，$k=i$ 一类。',
          '$j=i$ 类贡献 $(Ax)_i$；$k=i$ 类贡献 $(A^{\\top}x)_i$。',
          '合并得 $\\partial f/\\partial x_i=(Ax+A^{\\top}x)_i$，向量形式为 $(A+A^{\\top})x$。',
        ],
        tip: '两类项来自"$x_i$ 在前"和"$x_i$ 在后"——这就是为什么结果有 $A$ 和 $A^{\\top}$ 两项。$A$ 对称时两项相同，合并为 $2Ax$。',
      },
      // 卡 5：二次型应用——最小二乘
      {
        title: '应用：最小二乘正规方程',
        body:
          '最小二乘问题：$\\min_x f(x)=\\|Ax-b\\|_2^2$，$A\\in\\mathbb{R}^{m\\times n}$，$b\\in\\mathbb{R}^m$。\n\n' +
          '**展开为二次型**：\n\n' +
          '$f=(Ax-b)^{\\top}(Ax-b)=x^{\\top}A^{\\top}Ax-2b^{\\top}Ax+b^{\\top}b$\n\n' +
          '逐项求梯度：\n\n' +
          '• 二次型 $x^{\\top}(A^{\\top}A)x$：$A^{\\top}A$ 对称，梯度 $=2A^{\\top}Ax$；\n\n' +
          '• 线性项 $-2b^{\\top}Ax=-2(A^{\\top}b)^{\\top}x$：梯度 $=-2A^{\\top}b$；\n\n' +
          '• 常数项：梯度 $=0$。\n\n' +
          '合并：$\\nabla_x f=2A^{\\top}Ax-2A^{\\top}b=2A^{\\top}(Ax-b)$。\n\n' +
          '令梯度为零，得**正规方程** $A^{\\top}Ax=A^{\\top}b$，解为 $x^{\\star}=(A^{\\top}A)^{-1}A^{\\top}b$。',
        formula: '\\nabla_x\\|Ax-b\\|_2^2=2A^{\\top}(Ax-b)',
        tip: '最小二乘梯度 $=2A^{\\top}(Ax-b)$，令其为零即得正规方程。这是机器学习线性回归的基础。',
      },
      // 卡 6：迹技巧——标量对矩阵求导
      {
        title: '迹技巧：标量对矩阵求导',
        body:
          '**迹（trace）**是方阵对角元之和：$\\mathrm{tr}(A)=\\sum_i A_{ii}$。\n\n' +
          '关键恒等式：任何标量 $s=\\mathrm{tr}(s)$，且 $x^{\\top}Ax=\\mathrm{tr}(x^{\\top}Ax)=\\mathrm{tr}(Axx^{\\top})$（利用迹的循环不变性 $\\mathrm{tr}(ABC)=\\mathrm{tr}(CAB)$）。\n\n' +
          '标量对矩阵 $X$ 求导时，迹技巧把问题化为：若 $df=\\mathrm{tr}(G^{\\top}dX)$，则 $\\partial f/\\partial X=G$。\n\n' +
          '**最常用的公式**：\n\n' +
          '$\\dfrac{\\partial\\, \\mathrm{tr}(AX)}{\\partial X}=A^{\\top}$\n\n' +
          '$\\dfrac{\\partial\\, \\mathrm{tr}(X^{\\top}AX)}{\\partial X}=(A+A^{\\top})X$（$A$ 对称时为 $2AX$）\n\n' +
          '**数字体会**：$\\mathrm{tr}\\begin{pmatrix}2&1\\\\0&3\\end{pmatrix}=2+3=5$；迹的循环性：$\\mathrm{tr}(ABC)=\\mathrm{tr}(BCA)=\\mathrm{tr}(CAB)$，顺序可以循环但不能随意交换。',
        formula: '\\frac{\\partial\\,\\mathrm{tr}(AX)}{\\partial X}=A^{\\top},\\quad \\frac{\\partial\\,\\mathrm{tr}(X^{\\top}AX)}{\\partial X}=(A+A^{\\top})X',
        tip: '迹技巧三步：① 把标量写成 $\\mathrm{tr}(\\cdot)$；② 用循环不变性调顺序；③ 对比 $\\mathrm{tr}(G^{\\top}dX)$ 读出梯度 $G$。',
      },
      // 卡 7：对数行列式梯度
      {
        title: '对数行列式：$\\partial\\ln|A|/\\partial A=A^{-\\top}$',
        body:
          '高斯概率模型、normalizing flow 等场景中，经常对 $\\ln|A|$（$A$ 为协方差矩阵）求梯度。\n\n' +
          '**结论**：$\\dfrac{\\partial\\ln|A|}{\\partial A}=A^{-\\top}=(A^{-1})^{\\top}$。\n\n' +
          '**推导思路**（用矩阵微分）：\n\n' +
          '有标准恒等式 $d\\ln|A|=\\mathrm{tr}(A^{-1}dA)$（可由行列式的 Leibniz 展开验证）。\n\n' +
          '另一方面，对任意标量函数 $f(A)$，$df=\\mathrm{tr}\\!\\left[(\\partial f/\\partial A)^{\\top}dA\\right]$。\n\n' +
          '对比两式：$(\\partial\\ln|A|/\\partial A)^{\\top}=A^{-1}$，故 $\\partial\\ln|A|/\\partial A=A^{-\\top}$。\n\n' +
          '**逆矩阵的微分**（同类公式）：$d(A^{-1})=-A^{-1}dA\\,A^{-1}$，来自对 $AA^{-1}=I$ 两边取微分。',
        formula: '\\frac{\\partial\\ln|A|}{\\partial A}=A^{-\\top},\\quad d(A^{-1})=-A^{-1}dA\\,A^{-1}',
        tip: '逆矩阵微分公式 $d(A^{-1})=-A^{-1}dA\\,A^{-1}$ 的推导：对 $AA^{-1}=I$ 两边微分，得 $dA\\cdot A^{-1}+A\\cdot d(A^{-1})=0$，解出即可。',
      },
      // 卡 8：易错点汇总 + reveal
      {
        title: '易错点汇总与自测',
        body:
          '**易错一（布局约定）**：分子布局与分母布局下，梯度互为转置。混用后链式法则顺序会整体出错，导致结果差一个转置。做题前必须先锁定约定。\n\n' +
          '**易错二（矩阵积导数）**：$\\partial(AB)/\\partial A\\ne B^{\\top}$（一般情形）。若 $A$ 和 $B$ 都含参数，必须用矩阵微分 $d(AB)=dA\\cdot B+A\\cdot dB$，再读出梯度，不能直接套标量链式法则。\n\n' +
          '**易错三（逆矩阵微分）**：$d(X^{-1})\\ne -X^{-2}dX$。正确形式是 $-X^{-1}dX\\,X^{-1}$，两边各乘一次逆矩阵，且顺序不可交换。\n\n' +
          '**易错四（对称约束）**：若参数矩阵被约束为对称矩阵 $X=X^{\\top}$，则非对角元素的有效自由度 $X_{ij}=X_{ji}$ 只算一个，导数需要对称化修正：$\\frac{1}{2}(G+G^{\\top})$。',
        reveal: {
          q: '设 $A$ 对称正定，$w\\in\\mathbb{R}^n$。Ridge 回归损失 $L(w)=\\frac{1}{2}\\|Xw-y\\|_2^2+\\frac{\\lambda}{2}\\|w\\|_2^2$。写出 $\\nabla_w L$，并说明最优解形式。',
          a: '$\\nabla_w L=X^{\\top}(Xw-y)+\\lambda w$。令梯度为零：$(X^{\\top}X+\\lambda I)w^{\\star}=X^{\\top}y$，解为 $w^{\\star}=(X^{\\top}X+\\lambda I)^{-1}X^{\\top}y$。因为 $X^{\\top}X$ 半正定，$\\lambda I$ 正定，所以 $X^{\\top}X+\\lambda I$ 严格正定、可逆，保证唯一最优解。',
        },
      },
    ],
    questions: [
      // 题 1：choice — 梯度的形状
      {
        id: 'c29-a-q1',
        type: 'choice',
        prompt:
          '设 $f$ 是标量，$x\\in\\mathbb{R}^5$。在分母布局下，$\\nabla_x f$ 的形状是什么？',
        options: [
          '标量（$1\\times 1$）',
          '$5\\times 1$ 列向量',
          '$1\\times 5$ 行向量',
          '$5\\times 5$ 矩阵',
        ],
        answer: 1,
        explain:
          '分母布局下，标量对 $n$ 维向量的梯度是 $n\\times 1$ 列向量，每个分量 $=\\partial f/\\partial x_i$。这里 $n=5$，所以是 $5\\times 1$ 列向量。',
      },
      // 题 2：input — 线性型梯度
      {
        id: 'c29-a-q2',
        type: 'input',
        prompt:
          '设 $a=[1,2,3]^{\\top}$，$x\\in\\mathbb{R}^3$，$f(x)=a^{\\top}x$。$\\nabla_x f$ 的第二个分量等于多少？',
        accept: ['2'],
        placeholder: '输入整数',
        explain:
          '$\\nabla_x(a^{\\top}x)=a$，故 $\\nabla_x f=[1,2,3]^{\\top}$，第二个分量为 $2$。线性型的梯度就是系数向量本身。',
      },
      // 题 3：judge — Jacobian 的维度
      {
        id: 'c29-a-q3',
        type: 'judge',
        prompt:
          '若 $y=Ax$，$A\\in\\mathbb{R}^{3\\times 4}$，$x\\in\\mathbb{R}^4$，则 $y$ 对 $x$ 的 Jacobian 是 $4\\times 3$ 矩阵。',
        answer: false,
        explain:
          'Jacobian $J=\\partial y/\\partial x$，$J_{ij}=\\partial y_i/\\partial x_j$。$y\\in\\mathbb{R}^3$，$x\\in\\mathbb{R}^4$，所以 Jacobian 是 $3\\times 4$ 矩阵（行数等于输出维度，列数等于输入维度）。对线性映射 $y=Ax$，Jacobian $=A$ 本身，即 $3\\times 4$。',
      },
      // 题 4：choice — 二次型梯度
      {
        id: 'c29-a-q4',
        type: 'choice',
        prompt:
          '设 $A$ 是 $n\\times n$ 对称矩阵，$x\\in\\mathbb{R}^n$，则 $\\nabla_x(x^{\\top}Ax)$ 等于哪个？',
        options: [
          '$Ax$',
          '$2Ax$',
          '$(A+A^{\\top})$',
          '$x^{\\top}A$',
        ],
        answer: 1,
        explain:
          '$\\nabla_x(x^{\\top}Ax)=(A+A^{\\top})x$，当 $A$ 对称（$A=A^{\\top}$）时化简为 $2Ax$。注意结果是向量（$n\\times 1$），而非矩阵。',
      },
      // 题 5：input — 最小二乘梯度
      {
        id: 'c29-a-q5',
        type: 'input',
        prompt:
          '设 $L(x)=\\|Ax-b\\|_2^2$，令 $\\nabla_x L=0$，最优解满足正规方程 $A^{\\top}Ax=A^{\\top}b$。若 $A^{\\top}A$ 可逆，$x^{\\star}=(A^{\\top}A)^{-1}A^{\\top}b$。当 $A=[1,2]^{\\top}$，$b=[5]$ 时，$A^{\\top}A=$ 多少？',
        accept: ['5', '[5]'],
        placeholder: '输入整数',
        explain:
          '$A=[1,2]^{\\top}$ 是 $2\\times 1$ 列向量，$A^{\\top}A=[1,2]\\begin{bmatrix}1\\\\2\\end{bmatrix}=1\\times1+2\\times2=5$（一个标量）。$x^{\\star}=A^{\\top}b/5=[1,2]\\cdot[5]/5=1$（投影到 $A$ 方向）。',
      },
      // 题 6：match — 公式与名称
      {
        id: 'c29-a-q6',
        type: 'match',
        prompt: '将求导对象与对应结果配对。',
        left: [
          '$\\nabla_x(a^{\\top}x)$',
          '$\\nabla_x(x^{\\top}Ax)$（$A$ 对称）',
          '$\\partial\\,\\mathrm{tr}(AX)/\\partial X$',
        ],
        right: ['$a$', '$2Ax$', '$A^{\\top}$'],
      },
      // 题 7：judge — 迹的循环不变性
      {
        id: 'c29-a-q7',
        type: 'judge',
        prompt: '$\\mathrm{tr}(ABC)=\\mathrm{tr}(BAC)$（迹的循环不变性）。',
        answer: false,
        explain:
          '迹的循环不变性是 $\\mathrm{tr}(ABC)=\\mathrm{tr}(CAB)=\\mathrm{tr}(BCA)$，只能**循环移动**，不能任意交换位置。$\\mathrm{tr}(ABC)\\ne\\mathrm{tr}(BAC)$，因为 $B$ 和 $A$ 的位置直接对调（不是循环移动），一般不相等。',
      },
      // 题 8：choice — 对数行列式梯度
      {
        id: 'c29-a-q8',
        type: 'choice',
        prompt:
          '设 $A\\in\\mathbb{R}^{n\\times n}$ 可逆，则 $\\partial\\ln|A|/\\partial A$ 等于哪个？',
        options: ['$A^{-1}$', '$A^{\\top}$', '$A^{-\\top}$', '$|A|^{-1}$'],
        answer: 2,
        explain:
          '$\\partial\\ln|A|/\\partial A=A^{-\\top}=(A^{-1})^{\\top}=(A^{\\top})^{-1}$。推导利用微分恒等式 $d\\ln|A|=\\mathrm{tr}(A^{-1}dA)$，对比 $df=\\mathrm{tr}[(\\partial f/\\partial A)^{\\top}dA]$，读出 $(\\partial\\ln|A|/\\partial A)^{\\top}=A^{-1}$，转置后得 $A^{-\\top}$。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c29-b  链式法则与反向传播
  // ─────────────────────────────────────────
  {
    id: 'c29-b',
    title: '矩阵链式法则与全连接层反向传播',
    subtitle: '向量链式法则·线性层三个梯度·标准四步·Softmax Jacobian',
    intro: [
      // 卡 1：动机——深度网络怎么训练？
      {
        title: '动机：深度网络如何逐层传梯度？',
        body:
          '一个三层神经网络的前向传播：$x\\to h_1=W_1x+b_1\\to h_2=\\sigma(h_1)\\to y=W_2h_2+b_2\\to L$。\n\n' +
          '损失 $L$ 对 $W_1$ 的梯度，需要通过 $L\\to y\\to h_2\\to h_1\\to W_1$ 的链条逐层回传。\n\n' +
          '如果每一层只知道"上游传来的梯度"和"自己的局部 Jacobian"，那它就能计算"自己的参数梯度"和"向前传递的梯度"——这就是**反向传播**的本质：\n\n' +
          '每个节点只需做三件事：① 接收上游梯度；② 乘以本地 Jacobian 的转置；③ 传给前一层。\n\n' +
          '不需要知道整个网络结构，每层只需关心自己的输入输出关系。这就是反向传播高效的原因。',
        tip: '反向传播 $=$ 链式法则 $+$ 模块化。每层只计算本地 Jacobian，整体梯度由链式法则自动拼接。',
      },
      // 卡 2：向量链式法则
      {
        title: '向量链式法则：上游梯度乘 Jacobian 转置',
        body:
          '设 $y=y(x)$，$L=L(y)$，则在分母布局下：\n\n' +
          '$\\nabla_x L=\\left(\\dfrac{\\partial y}{\\partial x}\\right)^{\\top}\\nabla_y L$\n\n' +
          '其中 $\\dfrac{\\partial y}{\\partial x}$ 是 Jacobian 矩阵（$m\\times n$），$\\nabla_y L$ 是上游梯度（$m\\times 1$），$(\\cdot)^{\\top}$ 后变为 $n\\times m$，乘以 $m\\times 1$ 得到 $n\\times 1$ 的 $\\nabla_x L$。\n\n' +
          '**与一维对比**：一维链式法则 $dL/dx=(dL/dy)(dy/dx)$，只是标量相乘；向量版是"上游梯度乘以 Jacobian 的转置"，多了一个转置。\n\n' +
          '**为什么是转置？**：这是分母布局约定的结果，保证梯度维度和参数维度始终一致。',
        formula: '\\nabla_x L=\\left(\\frac{\\partial y}{\\partial x}\\right)^{\\top}\\nabla_y L',
        tip: '记忆口诀："上游梯度左乘 Jacobian 转置"。维度检验：$(n\\times m)\\times(m\\times 1)=(n\\times 1)$，和 $x$ 的维度匹配。',
      },
      // 卡 3：全连接层的三个梯度（推导）
      {
        title: '全连接层：推导三个梯度',
        body:
          '线性层 $y=Wx+b$，$W\\in\\mathbb{R}^{m\\times n}$，$x\\in\\mathbb{R}^n$，$b\\in\\mathbb{R}^m$，$L$ 是标量损失。\n\n' +
          '设上游梯度 $g=\\partial L/\\partial y\\in\\mathbb{R}^m$（与 $y$ 同维度）。\n\n' +
          '**梯度一，对 $b$**：$y$ 对 $b$ 是恒等映射（Jacobian $=I_m$），故 $\\partial L/\\partial b=I_m^{\\top}g=g$。\n\n' +
          '**梯度二，对 $x$**：$y=Wx+b$，故 $\\partial y/\\partial x=W$（Jacobian 是 $W$），$\\partial L/\\partial x=W^{\\top}g$。\n\n' +
          '**梯度三，对 $W$**：$\\partial y_i/\\partial W_{ij}=x_j$，所以 $(\\partial L/\\partial W)_{ij}=g_i x_j$，矩阵形式为 $\\partial L/\\partial W=gx^{\\top}$（上游梯度与输入的**外积**）。\n\n' +
          '结果总结：$\\partial L/\\partial W=gx^{\\top}$，$\\partial L/\\partial x=W^{\\top}g$，$\\partial L/\\partial b=g$。',
        formula:
          '\\frac{\\partial L}{\\partial W}=g\\,x^{\\top},\\quad\\frac{\\partial L}{\\partial x}=W^{\\top}g,\\quad\\frac{\\partial L}{\\partial b}=g',
        steps: [
          '记上游梯度 $g=\\partial L/\\partial y\\in\\mathbb{R}^m$。',
          '对 $b$：Jacobian $=I_m$，$\\partial L/\\partial b=g$（无转置，因为 $I^{\\top}=I$）。',
          '对 $x$：Jacobian $=W$，$\\partial L/\\partial x=W^{\\top}g$（$W^{\\top}$ 是 $n\\times m$，乘 $g$ 得 $n\\times 1$）。',
          '对 $W$：$(\\partial L/\\partial W)_{ij}=g_ix_j$，写成矩阵 $=gx^{\\top}$（$m\\times 1$ 乘 $1\\times n$，得 $m\\times n$）。',
        ],
        tip: '最重要的一条：$\\partial L/\\partial W=gx^{\\top}$（上游梯度 $\\times$ 输入转置），梯度与 $W$ 维度相同（$m\\times n$）。这是所有全连接层反向传播的核心。',
      },
      // 卡 4：反向传播标准四步
      {
        title: '反向传播标准四步法',
        body:
          '拿到任意一层网络，计算梯度的固定流程：\n\n' +
          '**第一步（前向）**：写清楚该层的映射关系，例如 $y=Wx+b$，记录维度。\n\n' +
          '**第二步（接收上游）**：定义 $g=\\partial L/\\partial y$（上游反传来的梯度，与输出同维度）。\n\n' +
          '**第三步（参数梯度）**：$\\partial L/\\partial W=g x^{\\top}$（上游梯度与**输入**的外积）。维度：$(m\\times 1)\\times(1\\times n)=m\\times n$，与 $W$ 一致。\n\n' +
          '**第四步（输入梯度）**：$\\partial L/\\partial x=W^{\\top}g$（权重矩阵转置乘上游梯度）。维度：$(n\\times m)\\times(m\\times 1)=n\\times 1$，与 $x$ 一致。\n\n' +
          '注意：参数梯度（第三步）用于更新 $W$；输入梯度（第四步）继续往前一层传递。',
        tip: '用维度检验法排查错误：$gx^{\\top}$ 必须和 $W$ 形状相同；$W^{\\top}g$ 必须和 $x$ 形状相同。维度不匹配时，优先检查转置方向。',
      },
      // 卡 5：具体数字例——验证公式
      {
        title: '数字例：验证反向传播公式',
        body:
          '设 $W=\\begin{bmatrix}1&2\\\\3&4\\end{bmatrix}$（$2\\times 2$），$x=\\begin{bmatrix}1\\\\-1\\end{bmatrix}$，$b=\\begin{bmatrix}0\\\\0\\end{bmatrix}$。\n\n' +
          '前向：$y=Wx=\\begin{bmatrix}1\\cdot1+2\\cdot(-1)\\\\3\\cdot1+4\\cdot(-1)\\end{bmatrix}=\\begin{bmatrix}-1\\\\-1\\end{bmatrix}$。\n\n' +
          '设损失 $L=\\|y\\|_2^2=(-1)^2+(-1)^2=2$，则上游梯度 $g=\\partial L/\\partial y=2y=\\begin{bmatrix}-2\\\\-2\\end{bmatrix}$。\n\n' +
          '**对 $W$ 的梯度**：$\\partial L/\\partial W=gx^{\\top}=\\begin{bmatrix}-2\\\\-2\\end{bmatrix}[1,-1]=\\begin{bmatrix}-2&2\\\\-2&2\\end{bmatrix}$。\n\n' +
          '**对 $x$ 的梯度**：$\\partial L/\\partial x=W^{\\top}g=\\begin{bmatrix}1&3\\\\2&4\\end{bmatrix}\\begin{bmatrix}-2\\\\-2\\end{bmatrix}=\\begin{bmatrix}-8\\\\-12\\end{bmatrix}$。\n\n' +
          '可以手动验证（对 $x_1$ 做有限差分）：$x_1$ 增大 $\\varepsilon$ 后 $y=\\begin{bmatrix}-1+\\varepsilon\\\\-1+3\\varepsilon\\end{bmatrix}$，$L=(1-\\varepsilon)^2+(1-3\\varepsilon)^2$，对 $\\varepsilon$ 求导在 $0$ 处得 $-2-6=-8$。与 $-8$ 一致。',
        tip: '数字验证是排查反向传播实现错误最高效的方法：先手算，再用有限差分 $(f(x+h)-f(x-h))/(2h)$ 近似，两者应高度吻合。',
      },
      // 卡 6：Softmax 的 Jacobian——耦合效应
      {
        title: 'Softmax 的 Jacobian：分量之间相互耦合',
        body:
          'Softmax 函数：$p_i=e^{z_i}/\\sum_k e^{z_k}$，将 $z\\in\\mathbb{R}^n$ 映射到概率分布 $p\\in\\mathbb{R}^n$。\n\n' +
          '**关键性质**：$p_i$ 依赖所有的 $z_j$（通过分母的归一化），所以 Jacobian 不是对角矩阵。\n\n' +
          '**推导**（两种情况）：\n\n' +
          '若 $i=j$：$\\dfrac{\\partial p_i}{\\partial z_i}=\\dfrac{e^{z_i}\\cdot\\sum_k e^{z_k}-e^{z_i}\\cdot e^{z_i}}{(\\sum_k e^{z_k})^2}=p_i(1-p_i)$。\n\n' +
          '若 $i\\ne j$：$\\dfrac{\\partial p_i}{\\partial z_j}=\\dfrac{0-e^{z_i}\\cdot e^{z_j}}{(\\sum_k e^{z_k})^2}=-p_ip_j$。\n\n' +
          '统一写成：$\\dfrac{\\partial p_i}{\\partial z_j}=p_i(\\delta_{ij}-p_j)$，矩阵形式 $J_{\\mathrm{softmax}}=\\mathrm{diag}(p)-pp^{\\top}$。',
        formula: '\\frac{\\partial p_i}{\\partial z_j}=p_i(\\delta_{ij}-p_j),\\quad J_{\\mathrm{softmax}}=\\mathrm{diag}(p)-pp^{\\top}',
        tip: '易错点：Softmax 的导数不是对每个分量单独求 sigmoid 导数。归一化分母使所有分量耦合，当 $i\\ne j$ 时 $\\partial p_i/\\partial z_j=-p_ip_j\\ne0$。',
      },
      // 卡 7：易错点 + reveal
      {
        title: '易错点汇总与自测',
        body:
          '**易错一（链式法则顺序）**：矩阵乘法不可交换。$\\nabla_x L=J^{\\top}\\nabla_y L$ 中，$J^{\\top}$ 必须在左边乘（分母布局），不能写成 $\\nabla_y L\\cdot J^{\\top}$（维度会不对）。\n\n' +
          '**易错二（参数梯度方向）**：$\\partial L/\\partial W=gx^{\\top}$，是"列向量 $g$ 乘行向量 $x^{\\top}$"（外积），结果是矩阵。不要写成 $g^{\\top}x$（内积，是标量）。\n\n' +
          '**易错三（Softmax 对角化谬误）**：Softmax 的 Jacobian 不是对角矩阵。若把它当对角矩阵处理，会漏掉 $-pp^{\\top}$ 项，导致梯度计算错误。\n\n' +
          '**易错四（批量维度）**：实际训练时输入是批量 $X\\in\\mathbb{R}^{B\\times n}$，梯度 $\\partial L/\\partial W=(\\partial L/\\partial Y)^{\\top}X$ 会在批量维度上自动求和。单样本公式不能直接用，需要注意 batch 维度的处理。',
        reveal: {
          q: '线性层 $y=Wx+b$，$W\\in\\mathbb{R}^{3\\times 2}$，$x\\in\\mathbb{R}^2$，$b\\in\\mathbb{R}^3$，上游梯度 $g\\in\\mathbb{R}^3$。$\\partial L/\\partial x$ 的形状是什么？',
          a: '$\\partial L/\\partial x=W^{\\top}g$，$W^{\\top}\\in\\mathbb{R}^{2\\times 3}$，$g\\in\\mathbb{R}^3$，乘积是 $\\mathbb{R}^2$ 的向量，与 $x$ 维度一致。若写成 $Wg$ 则维度为 $3\\times 3\\times 1=3\\times 1$，与 $x\\in\\mathbb{R}^2$ 不符，这就是转置错误的典型表现。',
        },
      },
    ],
    questions: [
      // 题 1：choice — 向量链式法则
      {
        id: 'c29-b-q1',
        type: 'choice',
        prompt:
          '设 $y\\in\\mathbb{R}^3$，$x\\in\\mathbb{R}^4$，Jacobian $J=\\partial y/\\partial x\\in\\mathbb{R}^{3\\times 4}$，上游梯度 $\\nabla_y L\\in\\mathbb{R}^3$。$\\nabla_x L$ 应该用哪个公式计算？',
        options: [
          '$J\\cdot\\nabla_y L$（$3\\times 4$ 乘 $3\\times 1$，维度错误）',
          '$J^{\\top}\\cdot\\nabla_y L$（$4\\times 3$ 乘 $3\\times 1$，得 $4\\times 1$）',
          '$\\nabla_y L\\cdot J$（$3\\times 1$ 乘 $3\\times 4$，维度错误）',
          '$\\nabla_y L^{\\top}\\cdot J^{\\top}$（$1\\times 3$ 乘 $4\\times 3$，维度错误）',
        ],
        answer: 1,
        explain:
          '分母布局下，$\\nabla_x L=J^{\\top}\\nabla_y L$。$J^{\\top}\\in\\mathbb{R}^{4\\times 3}$ 乘 $\\nabla_y L\\in\\mathbb{R}^3$ 得 $\\nabla_x L\\in\\mathbb{R}^4$，与 $x$ 维度一致。其他选项维度都不匹配。',
      },
      // 题 2：judge — 参数梯度公式
      {
        id: 'c29-b-q2',
        type: 'judge',
        prompt:
          '全连接层 $y=Wx+b$，上游梯度 $g=\\partial L/\\partial y$，则 $\\partial L/\\partial W=x^{\\top}g$（行向量 $x^{\\top}$ 乘列向量 $g$，结果是标量）。',
        answer: false,
        explain:
          '$\\partial L/\\partial W=gx^{\\top}$（列向量 $g$ 乘行向量 $x^{\\top}$，外积，结果是矩阵）。若写成 $x^{\\top}g$，则是内积（标量），维度与 $W$ 完全不符。外积和内积的顺序不能颠倒。',
      },
      // 题 3：input — 对 b 的梯度
      {
        id: 'c29-b-q3',
        type: 'input',
        prompt:
          '线性层 $y=Wx+b$，上游梯度 $g=[1,2,3]^{\\top}$。$\\partial L/\\partial b$ 的第二个分量等于多少？',
        accept: ['2'],
        placeholder: '输入整数',
        explain:
          '$\\partial L/\\partial b=g=[1,2,3]^{\\top}$，偏置梯度就等于上游梯度，第二个分量为 $2$。因为 $y$ 对 $b$ 是恒等映射，Jacobian $=I$，$I^{\\top}g=g$。',
      },
      // 题 4：choice — Softmax 的 Jacobian
      {
        id: 'c29-b-q4',
        type: 'choice',
        prompt:
          'Softmax 函数 $p=\\mathrm{softmax}(z)$，$p\\in\\mathbb{R}^n$，其 Jacobian $\\partial p/\\partial z$ 是什么形式？',
        options: [
          '$\\mathrm{diag}(p)$（对角矩阵）',
          '$\\mathrm{diag}(p)-pp^{\\top}$（对角矩阵减外积）',
          '$pp^{\\top}$（外积）',
          '$I-\\mathrm{diag}(p)$',
        ],
        answer: 1,
        explain:
          '$J_{\\mathrm{softmax}}=\\mathrm{diag}(p)-pp^{\\top}$。对角线元素 $\\partial p_i/\\partial z_i=p_i(1-p_i)$，非对角元素 $\\partial p_i/\\partial z_j=-p_ip_j$（$i\\ne j$）。因为归一化分母耦合了所有分量，所以不是对角矩阵。',
      },
      // 题 5：judge — 每层只需本地 Jacobian
      {
        id: 'c29-b-q5',
        type: 'judge',
        prompt:
          '反向传播时，每一层只需要知道自己的局部 Jacobian 和上游梯度，不需要了解整个网络结构。',
        answer: true,
        explain:
          '这正是反向传播高效的原因：每层接收上游梯度 $g$，计算本地 Jacobian $J$，输出 $J^{\\top}g$ 给前一层。网络结构的"全局信息"已经被自动编码在传入的梯度里，每层只做局部计算。',
      },
      // 题 6：match — 梯度公式对应
      {
        id: 'c29-b-q6',
        type: 'match',
        prompt: '线性层 $y=Wx+b$，上游梯度 $g$，将参数与对应梯度配对。',
        left: [
          '对权重 $W$ 的梯度',
          '对输入 $x$ 的梯度',
          '对偏置 $b$ 的梯度',
        ],
        right: ['$gx^{\\top}$', '$W^{\\top}g$', '$g$'],
      },
      // 题 7：input — 矩阵乘法验证
      {
        id: 'c29-b-q7',
        type: 'input',
        prompt:
          '设 $W=\\begin{bmatrix}2&0\\\\0&3\\end{bmatrix}$，上游梯度 $g=\\begin{bmatrix}1\\\\1\\end{bmatrix}$。$\\partial L/\\partial x=W^{\\top}g$ 的第一个分量等于多少？',
        accept: ['2'],
        placeholder: '输入整数',
        explain:
          '$W$ 是对角矩阵，$W^{\\top}=W$。$W^{\\top}g=\\begin{bmatrix}2&0\\\\0&3\\end{bmatrix}\\begin{bmatrix}1\\\\1\\end{bmatrix}=\\begin{bmatrix}2\\\\3\\end{bmatrix}$，第一个分量为 $2$。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c29-c  Softmax 交叉熵梯度 + 自动微分
  // ─────────────────────────────────────────
  {
    id: 'c29-c',
    title: 'Softmax 交叉熵梯度与自动微分原理',
    subtitle: '交叉熵梯度化简·自动微分三类·前向/反向模式·梯度检验',
    intro: [
      // 卡 1：动机——分类损失梯度为何如此简洁？
      {
        title: '动机：为什么分类模型的梯度这么简洁？',
        body:
          '深度学习分类任务的最后一步通常是：$z\\to p=\\mathrm{softmax}(z)\\to L=-\\sum_i y_i\\log p_i$（交叉熵损失，$y$ 是 one-hot 标签）。\n\n' +
          '从数学上看，要对 $z$ 求梯度，需要先对 $\\log p_i$ 求导，再乘以 Softmax 的 Jacobian，这看起来相当复杂。\n\n' +
          '但最终结果极其简洁：$\\nabla_z L=p-y$（预测概率减去标签）。\n\n' +
          '**这不是巧合**，而是 Softmax 与交叉熵配对后，Softmax Jacobian 与 $\\log$ 的导数恰好相互抵消的结果。\n\n' +
          '理解这个推导，是理解分类模型反向传播的关键一步。',
        tip: '结论先记住：交叉熵 $+$ Softmax 的输出层梯度 $=$ 预测概率 $-$ 真实标签，$\\nabla_z L=p-y$。',
      },
      // 卡 2：交叉熵 + Softmax 梯度推导
      {
        title: '推导：$\\nabla_z L=p-y$',
        body:
          '交叉熵损失 $L=-\\sum_i y_i\\log p_i$，$p=\\mathrm{softmax}(z)$，$y$ 是 one-hot（只有一个分量为 $1$，其余为 $0$）。\n\n' +
          '由链式法则：$\\dfrac{\\partial L}{\\partial z_j}=\\sum_i\\dfrac{\\partial L}{\\partial p_i}\\cdot\\dfrac{\\partial p_i}{\\partial z_j}$。\n\n' +
          '代入两个导数：$\\dfrac{\\partial L}{\\partial p_i}=-\\dfrac{y_i}{p_i}$，$\\dfrac{\\partial p_i}{\\partial z_j}=p_i(\\delta_{ij}-p_j)$。\n\n' +
          '展开：$\\dfrac{\\partial L}{\\partial z_j}=\\sum_i\\left(-\\dfrac{y_i}{p_i}\\right)p_i(\\delta_{ij}-p_j)=\\sum_i(-y_i)(\\delta_{ij}-p_j)$\n\n' +
          '$=-y_j+\\left(\\sum_i y_i\\right)p_j$。\n\n' +
          '对 one-hot 标签，$\\sum_i y_i=1$，故 $\\dfrac{\\partial L}{\\partial z_j}=p_j-y_j$，向量形式即 $\\nabla_z L=p-y$。',
        formula: '\\nabla_z L=p-y\\quad(y\\text{ 为 one-hot 标签，}p=\\mathrm{softmax}(z))',
        steps: [
          '链式法则：$\\partial L/\\partial z_j=\\sum_i(\\partial L/\\partial p_i)(\\partial p_i/\\partial z_j)$。',
          '代入 $\\partial L/\\partial p_i=-y_i/p_i$ 和 Softmax Jacobian $p_i(\\delta_{ij}-p_j)$。',
          '展开求和，$p_i/p_i=1$，得 $\\sum_i(-y_i)(\\delta_{ij}-p_j)$。',
          '利用 $\\sum_i y_i=1$（one-hot），化简得 $p_j-y_j$。',
          '向量形式：$\\nabla_z L=p-y$。',
        ],
        tip: 'Softmax 与交叉熵的"绝配"：Softmax Jacobian 中的 $p_i$ 和 $\\log$ 导数中的 $1/p_i$ 恰好抵消，最终结果只剩 $p-y$。这就是为什么二者总是配对使用。',
      },
      // 卡 3：三种微分方法的对比
      {
        title: '三种微分方法：符号、数值、自动',
        body:
          '**符号微分**：对数学表达式直接做代数求导。优点：精确；缺点：表达式可能爆炸（如求 $\\sin(\\tan(\\cdots))$ 的高阶导数，结果极其复杂）。适合简单闭合表达式。\n\n' +
          '**数值微分（有限差分）**：$f\'(x)\\approx\\dfrac{f(x+h)-f(x)}{h}$（前向差分）或 $\\dfrac{f(x+h)-f(x-h)}{2h}$（中心差分，精度更高）。优点：实现简单；缺点：同时受截断误差（$h$ 太大）和浮点舍入误差（$h$ 太小）的影响，无法做到机器精度。\n\n' +
          '**自动微分（AD）**：把复合函数拆成基本运算，对每步应用链式法则。不是数值近似，而是在机器精度范围内精确计算梯度。PyTorch、JAX 都使用这种方法。\n\n' +
          '**本质区别**：数值微分是近似，自动微分是精确——这就是为什么深度学习框架选择 AD 而非有限差分。',
        tip: '数值微分（有限差分）不是自动微分！有限差分有误差，自动微分在机器精度下精确。实践中有限差分用来**验证**自动微分实现是否正确（梯度检验）。',
      },
      // 卡 4：前向模式与反向模式
      {
        title: '前向模式与反向模式：何时用哪种？',
        body:
          '自动微分有两种模式，区别在于链式法则的计算方向：\n\n' +
          '**前向模式**（Forward mode）：从输入到输出，一次计算所有输出对某一个输入的导数。代价约等于一次前向传播；适合**输入维度小**（$n\\ll m$）的场景，如 $f:\\mathbb{R}^1\\to\\mathbb{R}^m$。\n\n' +
          '**反向模式**（Reverse mode）：从输出到输入，一次计算某一个输出对所有输入的梯度。代价约等于一次前向传播；适合**输出维度小**（$m\\ll n$）的场景，如 $f:\\mathbb{R}^n\\to\\mathbb{R}^1$。\n\n' +
          '**神经网络的情形**：参数 $n$ 可达数十亿，损失 $L$ 是标量（$m=1$）。这正是反向模式的理想场景：一次反向传播就能得到所有参数的梯度。\n\n' +
          '反向模式就是反向传播（backpropagation）的理论基础。',
        formula: '\\text{反向模式代价}\\approx O(1)\\times\\text{前向传播}\\quad(m=1\\text{ 时})',
        tip: '判断标准：输出维度 $m$ 和输入维度 $n$ 哪个更小？$m$ 小用反向模式；$n$ 小用前向模式。神经网络 $m=1$（损失是标量），所以用反向模式。',
      },
      // 卡 5：梯度检验——有限差分验证 AD
      {
        title: '梯度检验：用有限差分验证自动微分',
        body:
          '实现反向传播后，如何验证梯度是否正确？\n\n' +
          '**中心差分近似**：对第 $i$ 个输入坐标方向 $e_i$，\n\n' +
          '$g_{\\mathrm{fd},i}=\\dfrac{f(x+h e_i)-f(x-h e_i)}{2h}$\n\n' +
          '选取合适步长（如 $h=10^{-5}$），计算数值梯度 $g_{\\mathrm{fd}}$，与 AD 梯度 $g_{\\mathrm{ad}}$ 比较：\n\n' +
          '$\\mathrm{RelErr}=\\dfrac{\\|g_{\\mathrm{fd}}-g_{\\mathrm{ad}}\\|}{\\|g_{\\mathrm{fd}}\\|+\\|g_{\\mathrm{ad}}\\|}$\n\n' +
          '若相对误差 $<10^{-4}$，通常说明实现正确。\n\n' +
          '**步长选择**：$h$ 太大，截断误差主导（差分近似不精确）；$h$ 太小，浮点舍入误差放大。$h\\approx10^{-5}$ 是常见折中选择（对 float64）。\n\n' +
          '梯度检验是排查反向传播 bug 最标准的方法，也是机器学习实践中的重要工程技能。',
        formula: 'g_{\\mathrm{fd},i}=\\frac{f(x+he_i)-f(x-he_i)}{2h}',
        tip: '梯度检验中，$h$ 不要选太小（$h<10^{-7}$ 时 float64 的舍入误差开始主导）。中心差分比前向差分精度高一阶（误差从 $O(h)$ 降为 $O(h^2)$）。',
      },
      // 卡 6：易错点 + reveal
      {
        title: '易错点汇总与自测',
        body:
          '**易错一（有限差分 $\\ne$ 自动微分）**：两者都能计算梯度，但有限差分是数值近似（有误差），自动微分是精确计算（机器精度）。实践中有限差分只用于**验证**，不用于训练。\n\n' +
          '**易错二（Softmax 梯度的 one-hot 前提）**：$\\nabla_z L=p-y$ 成立的条件是 $y$ 是 one-hot（$\\sum_i y_i=1$）。若 $y$ 是软标签（soft label，$\\sum_i y_i=1$ 但各分量不是 $0/1$），结论形式相同；若 $y$ 是回归目标（$\\sum_i y_i\\ne1$），则公式不成立。\n\n' +
          '**易错三（高阶 Hessian）**：$H=\\nabla^2 f$ 是 $n\\times n$ 矩阵，显式构造代价是 $O(n^2)$。实践中用 Hessian-vector product $Hv$（代价仅约等于两次前向传播）代替，支持二阶优化、Fisher 信息矩阵近似。\n\n' +
          '**易错四（数值梯度只做检验）**：梯度检验时每计算一个坐标的数值梯度需要两次前向传播，总代价是 $2n$ 次。对 $n=10^6$ 的模型完全不可行，只在小模型或小参数块上做抽样检验。',
        reveal: {
          q: '用交叉熵损失训练分类模型，最后一层 logit 为 $z=[2,1,0]^{\\top}$，标签 $y=[1,0,0]^{\\top}$（第一类）。Softmax 概率约为 $p=[0.665,0.245,0.090]^{\\top}$。输出层梯度 $\\nabla_z L=p-y$ 等于多少？',
          a: '$\\nabla_z L=p-y=[0.665-1,\\,0.245-0,\\,0.090-0]^{\\top}=[-0.335,\\,0.245,\\,0.090]^{\\top}$。第一类是正确类，预测概率 $0.665$ 低于目标 $1$，梯度为负（需要增大该 logit）；其余类梯度为正（需要减小对应 logit）。',
        },
      },
    ],
    questions: [
      // 题 1：choice — 交叉熵 Softmax 梯度
      {
        id: 'c29-c-q1',
        type: 'choice',
        prompt:
          '分类模型最后一层：$p=\\mathrm{softmax}(z)$，$L=-\\sum_i y_i\\log p_i$（$y$ 是 one-hot）。$\\nabla_z L$ 等于哪个？',
        options: ['$-y/p$', '$p-y$', '$y-p$', '$\\log p$'],
        answer: 1,
        explain:
          '$\\nabla_z L=p-y$。推导：链式法则 $+$ Softmax Jacobian $+$ $\\sum_i y_i=1$。结果：对正确类，梯度 $=p_c-1<0$（需增大 logit）；对错误类，梯度 $=p_j>0$（需减小 logit）。',
      },
      // 题 2：judge — 自动微分的精度
      {
        id: 'c29-c-q2',
        type: 'judge',
        prompt:
          '自动微分（如 PyTorch 的 autograd）和有限差分法都是对梯度的数值近似，精度类似。',
        answer: false,
        explain:
          '自动微分不是数值近似，而是把链式法则系统化执行，在机器精度（float64 约 $10^{-16}$）范围内给出精确梯度。有限差分是近似，误差约为 $O(h)$ 或 $O(h^2)$（中心差分），且受浮点舍入误差限制无法任意减小。',
      },
      // 题 3：choice — 反向模式适用场景
      {
        id: 'c29-c-q3',
        type: 'choice',
        prompt:
          '自动微分的反向模式（reverse mode）最适合哪种场景？',
        options: [
          '输入维度 $n=1$，输出维度 $m=10^6$',
          '输入维度 $n=10^6$，输出维度 $m=1$（标量损失）',
          '输入维度和输出维度相同',
          '只有纯线性函数时',
        ],
        answer: 1,
        explain:
          '反向模式一次传播可以得到标量输出对所有输入的梯度，代价约为一次前向传播。当 $m=1$（损失是标量，$n$ 很大）时最高效。神经网络训练正是这种情形，所以使用反向模式（即反向传播）。',
      },
      // 题 4：input — 梯度检验步长
      {
        id: 'c29-c-q4',
        type: 'input',
        prompt:
          '中心差分公式 $g_{\\mathrm{fd},i}=(f(x+he_i)-f(x-he_i))/(2h)$ 的截断误差阶数是 $O(h^k)$，$k$ 等于多少？',
        accept: ['2'],
        placeholder: '输入整数',
        explain:
          '中心差分的截断误差是 $O(h^2)$（二阶精度）：将 $f(x\\pm h)$ 做 Taylor 展开，一阶项抵消，误差从主项 $f\'\'\'h^2/6$ 开始。相比之下，前向差分只有一阶精度 $O(h)$。因此梯度检验优先用中心差分。',
      },
      // 题 5：judge — 梯度检验可行性
      {
        id: 'c29-c-q5',
        type: 'judge',
        prompt:
          '梯度检验（用有限差分验证 AD 梯度）对 $10^8$ 个参数的大模型也可以逐个坐标全量验证。',
        answer: false,
        explain:
          '逐坐标梯度检验每个参数需要两次前向传播，$10^8$ 个参数需要 $2\\times10^8$ 次前向传播，完全不可行。实践中只对小模型或随机抽取的少量参数做梯度检验（抽样检验）。',
      },
      // 题 6：match — 微分方法配对
      {
        id: 'c29-c-q6',
        type: 'match',
        prompt: '将微分方法与其核心特点配对。',
        left: [
          '符号微分',
          '有限差分',
          '自动微分（反向模式）',
        ],
        right: [
          '精确但表达式可能爆炸',
          '简单但有截断和舍入误差',
          '机器精度精确，代价约一次前向传播',
        ],
      },
    ],
  },
]
