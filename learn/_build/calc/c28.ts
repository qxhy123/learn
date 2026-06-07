// 第25章  凸优化基础
// 关 c28-a：凸集、凸函数与 Hessian 判别（8 卡 7 题）
// 关 c28-b：KKT 条件与机器学习中的凸优化（7 卡 7 题）
// 关 c28-c：Jensen 不等式与信息论应用（6 卡 6 题）

export const LESSONS = [
  // ─────────────────────────────────────────
  // 关 c28-a  凸集、凸函数与 Hessian 判别
  // ─────────────────────────────────────────
  {
    id: 'c28-a',
    title: '凸集、凸函数与 Hessian 判别',
    subtitle: '机器学习为何需要凸性·凸集·凸函数·一阶/二阶判别·Hessian 正定',
    intro: [
      // 卡 1：动机——为什么机器学习需要凸优化？
      {
        title: '为什么机器学习需要凸优化？',
        body:
          '训练一个机器学习模型，本质上是在**最小化损失函数**——找到使预测误差最小的参数。\n\n' +
          '但"找最小值"在一般情况下极其困难：函数可能有无数个局部低谷（局部极小值），梯度下降容易陷入其中，找不到真正最好的解。\n\n' +
          '**凸优化**给了我们一个关键保证：对凸函数来说，**任何局部最小值都是全局最小值**。梯度下降一定能找到最优解，不会被局部极小值欺骗。\n\n' +
          '线性回归（含 $L_2$ 正则化）、支持向量机（SVM）、逻辑回归——这些经典算法的目标函数都是凸的，这正是它们理论完善、工程可靠的根本原因。',
        tip: '核心直觉：凸函数的图像像一个碗，无论从哪里出发，梯度总指向同一个最低点。',
      },
      // 卡 2：凸集的定义——线段封闭性
      {
        title: '凸集：任意两点连线仍在集合内',
        body:
          '**定义**：集合 $S \\subseteq \\mathbb{R}^n$ 称为凸集，若对任意 $x, y \\in S$ 与任意 $\\lambda \\in [0,1]$，都有\n\n' +
          '$\\lambda x + (1-\\lambda)y \\in S$。\n\n' +
          '**白话翻译**：集合内任意两点之间的线段，完整地落在集合里。\n\n' +
          '**凸集举例**：圆盘（球内部）、三角形内部、直线、半空间（$ax+by \\leq c$）、多个约束的交集。\n\n' +
          '**非凸集举例**：月牙形区域、带洞的集合——在这些集合里，两端点连线会穿越集合外部。\n\n' +
          '**重要性质**：任意多个凸集的**交集**仍为凸集。因此，多个线性不等式约束 $g_i(x) \\leq 0$ 同时定义的可行域，一定是凸集。',
        formula: '\\lambda x + (1-\\lambda)y \\in S \\quad \\forall x,y \\in S,\\; \\lambda \\in [0,1]',
        tip: '验证凸集的方法：任取两点，写出中点 $\\frac{x+y}{2}$，检查是否还在集合内。若对所有两点成立，集合为凸。',
      },
      // 卡 3：凸函数的定义与几何直觉
      {
        title: '凸函数：弦线始终在图像上方',
        body:
          '**定义**：函数 $f: \\mathbb{R}^n \\to \\mathbb{R}$ 称为凸函数，若对任意 $x, y$ 与 $\\lambda \\in [0,1]$：\n\n' +
          '$f(\\lambda x + (1-\\lambda)y) \\leq \\lambda f(x) + (1-\\lambda)f(y)$。\n\n' +
          '**几何含义**：函数图像上任意两点连成的弦线，始终位于函数图像的**上方或与之相切**。直觉上，图像是"向下凹的碗形"。\n\n' +
          '**严格凸函数**：若上式在 $x \\neq y$ 且 $\\lambda \\in (0,1)$ 时严格成立（不等号为 $<$），则称严格凸——碗形无平坦底部，有唯一最低点。\n\n' +
          '**凸函数举例**：$x^2$（抛物线）、$e^x$、$|x|$、$\\|w\\|_2^2$（参数的 $L_2$ 范数平方）。\n\n' +
          '**非凸函数举例**：$\\sin x$、$x^3$、神经网络的整体损失函数。',
        formula: 'f(\\lambda x + (1-\\lambda)y) \\leq \\lambda f(x) + (1-\\lambda)f(y)',
        tip: '记忆口诀：凸函数的弦线在上，图像在下——像一个碗底。不要和"凸面向下（concave，中文通常叫上凸）"混淆。',
      },
      // 卡 4：一阶判别条件——切平面是全局下界
      {
        title: '一阶判别：切平面是全局下界',
        body:
          '若 $f$ 可微，则 $f$ 是凸函数当且仅当对任意 $x, y$：\n\n' +
          '$f(y) \\geq f(x) + \\nabla f(x)^\\top (y - x)$。\n\n' +
          '**几何含义**：在任意一点 $x$ 处，$f$ 的切平面（一阶 Taylor 近似）始终位于函数图像**下方**，是全局下界。\n\n' +
          '**重大推论**：若 $\\nabla f(x^\\star) = 0$（驻点），则对任意 $y$：\n\n' +
          '$f(y) \\geq f(x^\\star) + \\nabla f(x^\\star)^\\top (y - x^\\star) = f(x^\\star)$。\n\n' +
          '也就是说，**驻点就是全局最优点**！这是凸函数最核心的性质，也是梯度下降对凸问题奏效的数学理由。\n\n' +
          '**对比非凸情形**：非凸函数的驻点可能只是局部极小值或鞍点，梯度为零不保证全局最优。',
        formula: 'f(y) \\geq f(x) + \\nabla f(x)^\\top (y-x) \\quad \\forall x,y',
        tip: '关键词：切平面=全局下界=驻点是全局最优。这三者在凸函数中是等价的。',
      },
      // 卡 5：二阶判别——Hessian 半正定
      {
        title: '二阶判别：Hessian 矩阵半正定',
        body:
          '若 $f$ 二阶可导，则 $f$ 是凸函数的充分条件：**Hessian 矩阵处处半正定**（$\\nabla^2 f(x) \\succeq 0$）。\n\n' +
          '**什么是 Hessian？** $\\nabla^2 f(x)$ 是由所有二阶偏导数组成的矩阵：\n\n' +
          '对二元函数 $f(x_1, x_2)$，Hessian $H = \\begin{pmatrix} f_{x_1x_1} & f_{x_1x_2} \\\\ f_{x_2x_1} & f_{x_2x_2} \\end{pmatrix}$。\n\n' +
          '**半正定的含义**：对任意向量 $v \\neq 0$，$v^\\top H v \\geq 0$。等价地，Hessian 的所有特征值 $\\geq 0$。\n\n' +
          '**正定与严格凸**：若 Hessian 处处正定（$H \\succ 0$，所有特征值 $> 0$），则 $f$ 是严格凸函数，有唯一全局最优点。\n\n' +
          '**一维退化**：一维时，$\\nabla^2 f = f\'\'$，半正定退化为 $f\'\' \\geq 0$，正定退化为 $f\'\' > 0$——这是高中就学过的凸性判据。',
        formula: '\\nabla^2 f(x) \\succeq 0 \\Rightarrow f \\text{ 是凸函数};\\quad \\nabla^2 f(x) \\succ 0 \\Rightarrow f \\text{ 是严格凸函数}',
        tip: '验证 $2\\times2$ 矩阵正定的两种方法：（1）顺序主子式均正：$a_{11}>0$ 且 $\\det H > 0$；（2）两个特征值均正。',
      },
      // 卡 6：例题——判断二元函数的凸性（完整推导）
      {
        title: '例题：$f(x,y)=x^2+2y^2-xy$ 是否为凸函数？',
        body:
          '这是本章的"引入题"，我们按 Hessian 四步法完整推导。',
        steps: [
          '第一步（求二阶偏导）：$f_{xx} = 2$，$f_{yy} = 4$，$f_{xy} = f_{yx} = -1$。',
          '第二步（写出 Hessian）：$H = \\nabla^2 f = \\begin{pmatrix} 2 & -1 \\\\ -1 & 4 \\end{pmatrix}$。',
          '第三步（验证正定）：顺序主子式：$H_{11} = 2 > 0$；$\\det H = 2 \\times 4 - (-1)^2 = 8 - 1 = 7 > 0$。两个主子式均正，$H$ 正定。',
          '第四步（得出结论）：Hessian 处处正定，$f$ 是严格凸函数，存在唯一全局最小值。',
          '机器学习意义：若 $f(x,y)$ 是某个二维参数的损失函数，梯度下降可稳定收敛到唯一全局最优 $(0,0)$。Hessian 特征值比 $\\approx 4.41/1.59 \\approx 2.77$，较大，说明等高线呈较扁椭圆，需用较小学习率。',
        ],
        formula: 'H = \\begin{pmatrix} 2 & -1 \\\\ -1 & 4 \\end{pmatrix},\\quad \\det H = 7 > 0 \\Rightarrow H \\succ 0',
      },
      // 卡 7：常见凸函数与保凸运算
      {
        title: '常见凸函数与保凸运算',
        body:
          '**机器学习中的常见凸函数**（直接记住，无需每次验证）：\n\n' +
          '线性/仿射函数（如 $w^\\top x + b$）；$\\|w\\|_2^2$（参数范数平方）；$\\|w\\|_1$（$L_1$ 范数）；$e^x$；$-\\ln x$（$x>0$）；log-sum-exp（softmax 的底层）。\n\n' +
          '**保凸运算**——若已知某些函数是凸的，以下操作保持凸性：\n\n' +
          '（1）**非负加权和**：$f(x) = \\lambda_1 f_1(x) + \\lambda_2 f_2(x)$，$\\lambda_i \\geq 0$。例：Ridge 损失 $= $ 均方误差 $+$ $L_2$ 惩罚，两者均凸，权重非负，结果仍凸。\n\n' +
          '（2）**与仿射变换复合**：$g(x) = f(Ax + b)$，$f$ 凸则 $g$ 也凸。\n\n' +
          '（3）**逐点最大值**：$f(x) = \\max(f_1(x), f_2(x))$，两者凸则取最大值也凸（SVM 的 hinge loss 即此）。',
        tip: '快速判断：看函数是否由已知凸函数经过以上操作构成。若是，直接结论为凸，无需重算 Hessian。',
      },
      // 卡 8：易错点 + reveal
      {
        title: '易错点与自测',
        body:
          '**易错一（正定 vs 半正定）**：$H \\succ 0$（正定）保证严格凸；$H \\succeq 0$（半正定）只保证凸，可能有平坦方向（无唯一最优点）。例如 $f(x_1, x_2) = x_1^2$，Hessian $= \\mathrm{diag}(2,0)$ 半正定，$x_2$ 方向任意值都是全局最优。加 $L_2$ 正则化就解决这个问题。\n\n' +
          '**易错二（梯度为零 $\\ne$ 全局最优）**：对非凸函数，$\\nabla f = 0$ 可能是局部极大值、鞍点，只有凸函数才能从驻点推出全局最优。\n\n' +
          '**易错三（凸函数 vs 上凸函数）**：数学分析中"凸函数"通常指图像呈碗状（弦线在上），但有些教材称此为"下凸"，称 $-f$ 为"上凸"。本课程统一：凸函数 $=$ 弦线在上 $=$ 碗形 $=$ $f\'\' \\geq 0$（一维）。',
        reveal: {
          q: '判断 $f(x) = e^x - x$ 的凸性，并求其最小值（提示：先求二阶导）。',
          a: '$f\'\'(x) = e^x > 0$ 对所有 $x$ 成立，故 $f$ 严格凸。令 $f\'(x) = e^x - 1 = 0$，得 $x^\\star = 0$，最小值 $f(0) = 1$。',
        },
      },
    ],
    questions: [
      // 题 1：judge — 凸集的线段性
      {
        id: 'c28-a-q1',
        type: 'judge',
        prompt: '集合 $S = \\{x \\in \\mathbb{R}^2 \\mid x_1 + x_2 \\leq 1,\\; x_1 \\geq 0,\\; x_2 \\geq 0\\}$ 是凸集。',
        answer: true,
        explain:
          '$S$ 是三个半空间的交集：$x_1 + x_2 \\leq 1$、$x_1 \\geq 0$、$x_2 \\geq 0$。每个半空间都是凸集，有限个凸集的交集仍为凸集，因此 $S$ 是凸集（实际上是一个三角形区域）。',
      },
      // 题 2：choice — 凸函数定义
      {
        id: 'c28-a-q2',
        type: 'choice',
        prompt: '下列哪个函数在整个实数轴上是凸函数？',
        options: ['$\\sin x$', '$x^3$', '$e^x$', '$-x^2$'],
        answer: 2,
        explain:
          '$e^x$ 的二阶导数 $(e^x)\'\' = e^x > 0$ 对所有 $x$ 成立，故严格凸。$\\sin x$ 在不同区间凹凸交替；$x^3$ 的 $f\'\' = 6x$，$x < 0$ 时为负，非凸；$-x^2$ 的 $f\'\' = -2 < 0$，是凹函数（上凸）。',
      },
      // 题 3：input — Hessian 行列式
      {
        id: 'c28-a-q3',
        type: 'input',
        prompt: '计算函数 $f(x,y) = x^2 + 2y^2 - xy$ 的 Hessian 矩阵行列式 $\\det H$，结果等于多少？',
        accept: ['7'],
        placeholder: '输入整数',
        explain:
          '$f_{xx} = 2$，$f_{yy} = 4$，$f_{xy} = -1$，所以 $H = \\begin{pmatrix} 2 & -1 \\\\ -1 & 4 \\end{pmatrix}$，$\\det H = 2 \\times 4 - (-1)^2 = 8 - 1 = 7$。',
      },
      // 题 4：choice — 凸函数的驻点
      {
        id: 'c28-a-q4',
        type: 'choice',
        prompt: '对严格凸函数 $f$，若 $\\nabla f(x^\\star) = 0$，则 $x^\\star$ 是：',
        options: [
          '可能是局部极小值，也可能是鞍点',
          '唯一的全局最小值点',
          '全局最大值点',
          '无法判断，需要更多信息',
        ],
        answer: 1,
        explain:
          '对凸函数，一阶条件 $\\nabla f(x^\\star) = 0$ 保证 $x^\\star$ 是全局最小值点（由一阶条件 $f(y) \\geq f(x^\\star) + 0 = f(x^\\star)$ 对所有 $y$ 成立）。严格凸函数进一步保证该点是唯一的全局最小值点。',
      },
      // 题 5：judge — 保凸运算
      {
        id: 'c28-a-q5',
        type: 'judge',
        prompt: 'Ridge 回归目标 $f(w) = \\frac{1}{2}\\|Xw-y\\|_2^2 + \\frac{\\lambda}{2}\\|w\\|_2^2$（$\\lambda > 0$）是严格凸函数。',
        answer: true,
        explain:
          '第一项 $\\frac{1}{2}\\|Xw-y\\|_2^2$ 的 Hessian 是 $X^\\top X \\succeq 0$（半正定）；第二项的 Hessian 是 $\\lambda I \\succ 0$（$\\lambda > 0$ 时正定）。两项之和的 Hessian 为 $X^\\top X + \\lambda I$，因 $\\lambda > 0$，该矩阵正定，故 $f$ 严格凸，有唯一最优解。',
      },
      // 题 6：match — 凸性判据
      {
        id: 'c28-a-q6',
        type: 'match',
        prompt: '将凸性条件与对应的结论配对。',
        left: [
          'Hessian 处处半正定（$H \\succeq 0$）',
          'Hessian 处处正定（$H \\succ 0$）',
          '一维时 $f\'\'(x) = 0$（处处成立）',
        ],
        right: ['函数是凸函数（可能非严格）', '函数是严格凸函数（唯一极小值）', '函数是线性函数（仿射）'],
      },
      // 题 7：choice — 非凸反例
      {
        id: 'c28-a-q7',
        type: 'choice',
        prompt: '以下哪个说法关于凸函数是错误的？',
        options: [
          '凸函数的任意局部极小值都是全局极小值',
          '深度神经网络的整体损失函数通常是凸的',
          '两个凸函数的非负加权和仍是凸函数',
          '$|x|$ 是凸函数',
        ],
        answer: 1,
        explain:
          '深度神经网络由多层非线性变换组成，其整体损失函数通常是**非凸**的，存在大量局部极小值和鞍点。这正是深度学习优化难于理论分析的原因。其余三项均正确：局部极小值是全局极小值（凸函数核心性质）、非负加权和保凸、$|x|$ 是凸函数（$f(x) = \\max(x,-x)$ 是两个线性函数的逐点最大值）。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c28-b  KKT 条件与机器学习中的凸优化
  // ─────────────────────────────────────────
  {
    id: 'c28-b',
    title: 'KKT 条件与机器学习中的凸优化',
    subtitle: 'Lagrange 乘子·KKT 四条件·例题推导·SVM·正则化几何',
    intro: [
      // 卡 1：动机——约束优化问题在 ML 中的地位
      {
        title: '约束优化：不是所有参数都能自由取值',
        body:
          '机器学习中的许多问题都带有约束。例如：\n\n' +
          '支持向量机（SVM）要求分类超平面满足间隔约束；正则化可以理解为限制参数范数不超过某个半径；概率模型要求参数满足归一化约束 $\\sum_i p_i = 1$。\n\n' +
          '约束优化问题的标准形式：\n\n' +
          '$\\min_x f(x)$，满足 $g_i(x) \\leq 0$（不等式约束，$i=1,\\dots,m$）和 $h_j(x) = 0$（等式约束，$j=1,\\dots,p$）。\n\n' +
          '若 $f$ 和所有 $g_i$ 均凸，而 $h_j$ 是仿射函数，则这是一个**凸优化问题**，局部最优就是全局最优。\n\n' +
          '**关键问题**：如何在有约束的情况下，系统地找到最优点？答案是 **KKT 条件**。',
        tip: '把约束想象成"围栏"：无约束时最优点在围栏外，有约束时最优点被迫移到围栏边界上。KKT 条件就是描述这个"被边界推到平衡"的数学语言。',
      },
      // 卡 2：Lagrange 函数——将约束"纳入"目标
      {
        title: 'Lagrange 函数：把约束打包进目标',
        body:
          '处理约束优化的经典方法是引入 **Lagrange 函数**，将约束以惩罚的形式加入目标：\n\n' +
          '$L(x, \\lambda, \\nu) = f(x) + \\sum_{i=1}^m \\lambda_i g_i(x) + \\sum_{j=1}^p \\nu_j h_j(x)$，\n\n' +
          '其中 $\\lambda_i \\geq 0$ 称为不等式约束的 **Lagrange 乘子**（对偶变量），$\\nu_j$ 是等式约束的乘子。\n\n' +
          '**直觉**：$\\lambda_i$ 衡量约束 $g_i(x) \\leq 0$ 对最优值的"影子价格"——约束越紧，$\\lambda_i$ 越大，代表放松此约束能带来的收益越高。\n\n' +
          '**弱对偶性**：对偶函数 $g(\\lambda, \\nu) = \\inf_x L(x, \\lambda, \\nu)$ 始终是原问题最优值 $p^\\star$ 的下界：$g(\\lambda, \\nu) \\leq p^\\star$。\n\n' +
          '当满足 Slater 条件时，凸问题的最优原始值 $=$ 对偶值（强对偶性），这是 SVM 利用核技巧的数学基础。',
        formula: 'L(x,\\lambda,\\nu) = f(x) + \\sum_{i=1}^m \\lambda_i g_i(x) + \\sum_{j=1}^p \\nu_j h_j(x),\\quad \\lambda_i \\geq 0',
      },
      // 卡 3：KKT 四条件——约束优化的最优性方程组
      {
        title: 'KKT 四条件：约束最优解必须满足的方程组',
        body:
          '在凸问题满足适当正则性条件（如 Slater 条件）时，最优解 $x^\\star$ 与对偶变量 $\\lambda^\\star, \\nu^\\star$ 必须同时满足以下四条件：\n\n' +
          '**条件一（驻点）**：$\\nabla_x L(x^\\star, \\lambda^\\star, \\nu^\\star) = 0$——对 $x$ 求梯度等于零。\n\n' +
          '**条件二（原始可行）**：$g_i(x^\\star) \\leq 0$，$h_j(x^\\star) = 0$——所有约束在最优点处满足。\n\n' +
          '**条件三（对偶可行）**：$\\lambda_i^\\star \\geq 0$——乘子非负（只惩罚约束违反，不奖励）。\n\n' +
          '**条件四（互补松弛）**：$\\lambda_i^\\star g_i(x^\\star) = 0$——约束要么"活跃"（等号成立且 $\\lambda_i > 0$），要么"休眠"（严格不等且 $\\lambda_i = 0$）。\n\n' +
          '互补松弛是最有意思的条件：它告诉我们哪些约束在最优点处"真正起作用"，哪些约束其实是多余的。',
        formula: '\\nabla_x L = 0,\\quad g_i(x^\\star) \\leq 0,\\quad \\lambda_i^\\star \\geq 0,\\quad \\lambda_i^\\star g_i(x^\\star) = 0',
        tip: 'KKT 标准四步法：（1）写 Lagrange 函数；（2）令 $\\nabla_x L = 0$；（3）用互补松弛分情况讨论（约束活跃/休眠）；（4）检验所有条件。',
      },
      // 卡 4：例题一——不等式约束（分步详解）
      {
        title: '例题一：$\\min x^2$，约束 $x \\geq 1$',
        body:
          '直觉上答案是 $x^\\star = 1$，但我们用 KKT 系统推导，练习标准流程。',
        steps: [
          '第一步（写 Lagrange 函数）：把约束 $x \\geq 1$ 改写为 $g(x) = 1 - x \\leq 0$，构造 $L(x, \\lambda) = x^2 + \\lambda(1-x)$，$\\lambda \\geq 0$。',
          '第二步（驻点条件）：$\\frac{\\partial L}{\\partial x} = 2x - \\lambda = 0$，得 $\\lambda = 2x$。',
          '第三步（互补松弛）：$\\lambda(1-x) = 0$。情形 A：若 $\\lambda = 0$，则 $x = 0$，但 $x < 1$ 违反原始可行性，矛盾。情形 B：若 $1-x = 0$，则 $x = 1$，$\\lambda = 2 > 0$ 满足对偶可行性。',
          '第四步（验证所有 KKT 条件）：$x^\\star = 1$ 满足原始可行 $x \\geq 1$；$\\lambda^\\star = 2 \\geq 0$；$\\lambda^\\star g(x^\\star) = 2 \\times 0 = 0$；驻点 $\\nabla_x L|_{x^\\star} = 0$。全部满足。',
          '结论：$x^\\star = 1$，最优值为 $1^2 = 1$。',
        ],
      },
      // 卡 5：例题二——等式约束（$x+y=1$ 最小化 $x^2+y^2$）
      {
        title: '例题二：在 $x+y=1$ 约束下最小化 $x^2+y^2$',
        body:
          '这是一个纯等式约束问题，经典的 Lagrange 乘子法例题。',
        steps: [
          '第一步（构造 Lagrange 函数）：$L(x, y, \\nu) = x^2 + y^2 + \\nu(x + y - 1)$。',
          '第二步（驻点条件）：$\\partial L / \\partial x = 2x + \\nu = 0$；$\\partial L / \\partial y = 2y + \\nu = 0$，联立得 $x = y$。',
          '第三步（代入约束）：$x + y = 1$ 且 $x = y$，解得 $x = y = \\tfrac{1}{2}$，乘子 $\\nu = -1$。',
          '第四步（计算最优值）：$f(\\tfrac{1}{2}, \\tfrac{1}{2}) = (\\tfrac{1}{2})^2 + (\\tfrac{1}{2})^2 = \\tfrac{1}{2}$。',
          '几何直觉：等高线 $x^2+y^2=c$ 是圆，约束是直线 $x+y=1$，最优点在圆与直线相切处，即最近点 $(\\tfrac{1}{2}, \\tfrac{1}{2})$。',
        ],
        formula: 'x^\\star = y^\\star = \\frac{1}{2},\\quad f^\\star = \\frac{1}{2}',
      },
      // 卡 6：SVM 与 KKT——支持向量的来源
      {
        title: 'SVM：KKT 条件揭示"支持向量"',
        body:
          '软间隔 SVM 的原始问题是一个凸二次规划：\n\n' +
          '$\\min_{w,b,\\xi} \\frac{1}{2}\\|w\\|_2^2 + C\\sum_i \\xi_i$，满足 $y_i(w^\\top x_i + b) \\geq 1 - \\xi_i$，$\\xi_i \\geq 0$。\n\n' +
          '对每个训练样本 $i$，KKT 互补松弛条件为：\n\n' +
          '$\\alpha_i[y_i(w^\\top x_i + b) - 1 + \\xi_i] = 0$（$\\alpha_i$ 是对应不等式的乘子）。\n\n' +
          '**推论**：只有使约束"活跃"（等号成立，即 $\\alpha_i > 0$）的样本才对模型有影响——这些就是**支持向量**。远离分类边界的样本 $\\alpha_i = 0$，对 $w$ 的贡献为零。\n\n' +
          '**核技巧**：SVM 对偶问题中，$w$ 可以用支持向量表示为 $w = \\sum_i \\alpha_i y_i x_i$，目标函数只涉及内积 $x_i^\\top x_j$，因此可以用核函数替换，将 SVM 推广到非线性分类。',
        tip: '记住：支持向量 = KKT 互补松弛中约束"活跃"的样本 = 对偶变量 $\\alpha_i > 0$ 的样本。大多数样本 $\\alpha_i = 0$，对模型毫无贡献。',
      },
      // 卡 7：正则化的凸优化几何直觉 + reveal
      {
        title: '正则化的凸优化几何：惩罚形式 vs 约束形式',
        body:
          '正则化（Ridge、Lasso）可以用两种等价形式理解：\n\n' +
          '**惩罚形式**：$\\min_w L(w) + \\lambda \\|w\\|_2^2$——在损失函数上加惩罚项，$\\lambda$ 控制惩罚强度。\n\n' +
          '**约束形式**：$\\min_w L(w)$，满足 $\\|w\\|_2^2 \\leq r$——限制参数范数不超过 $r$。\n\n' +
          '由 KKT 条件，两者在对应的 $\\lambda$ 和 $r$ 下完全等价。\n\n' +
          '**$L_1$ vs $L_2$ 的几何差异**：$L_2$ 约束区域是球（光滑，无尖角），等高线一般不在坐标轴上与约束相切，解通常不稀疏。$L_1$ 约束区域是菱形（顶点在坐标轴上），等高线容易先碰到顶点，使某些参数恰好等于零——这是 Lasso 产生稀疏解的几何原因。',
        reveal: {
          q: '在约束 $x + 2y = 3$ 下最小化 $f(x,y) = x^2 + y^2 + xy$，用 Lagrange 法求最优解（提示：令 $\\partial L/\\partial x = 0$，$\\partial L/\\partial y = 0$，再代入约束）。',
          a: '$L = x^2 + y^2 + xy + \\nu(x+2y-3)$。驻点：$2x+y+\\nu=0$，$x+2y+2\\nu=0$。第二式减去第一式：$-x+y+\\nu=0$。联立 $2x+y+\\nu=0$ 解得 $x=1, y=1, \\nu=-3$。最优值 $f(1,1)=1+1+1=3$。',
        },
      },
    ],
    questions: [
      // 题 1：choice — KKT 条件个数
      {
        id: 'c28-b-q1',
        type: 'choice',
        prompt: 'KKT 条件中，以下哪条描述的是"互补松弛"？',
        options: [
          '$\\lambda_i^\\star \\geq 0$',
          '$\\nabla_x L = 0$',
          '$\\lambda_i^\\star g_i(x^\\star) = 0$',
          '$g_i(x^\\star) \\leq 0$',
        ],
        answer: 2,
        explain:
          '互补松弛条件为 $\\lambda_i^\\star g_i(x^\\star) = 0$，意思是：对每个不等式约束，要么约束活跃（$g_i = 0$），要么对偶变量为零（$\\lambda_i = 0$），两者之积恒为零。选项 A 是对偶可行性，B 是驻点条件，D 是原始可行性。',
      },
      // 题 2：input — KKT 求解不等式约束
      {
        id: 'c28-b-q2',
        type: 'input',
        prompt: '用 KKT 条件求解 $\\min_{x} (x-2)^2$，约束 $x \\geq 3$。最优点 $x^\\star$ 等于多少？',
        accept: ['3'],
        placeholder: '输入整数',
        explain:
          '无约束最优点在 $x=2$，不满足 $x \\geq 3$。令 $g(x) = 3-x \\leq 0$，构造 $L = (x-2)^2 + \\lambda(3-x)$。驻点：$2(x-2) - \\lambda = 0$。互补松弛：若约束活跃 $x=3$，则 $\\lambda = 2(3-2) = 2 > 0$，合法。故 $x^\\star = 3$，最优值 $(3-2)^2 = 1$。',
      },
      // 题 3：judge — 对偶可行性
      {
        id: 'c28-b-q3',
        type: 'judge',
        prompt: '在 KKT 条件中，等式约束的 Lagrange 乘子 $\\nu_j$ 必须满足 $\\nu_j \\geq 0$。',
        answer: false,
        explain:
          '只有**不等式约束**的乘子 $\\lambda_i$ 才必须 $\\geq 0$（因为不等式约束是单向的，只惩罚违反方向）。等式约束的乘子 $\\nu_j$ 可以是任意实数，因为等式约束在两个方向上都有约束力。',
      },
      // 题 4：choice — SVM 支持向量
      {
        id: 'c28-b-q4',
        type: 'choice',
        prompt: '在 SVM 中，KKT 互补松弛条件说明：',
        options: [
          '所有训练样本都是支持向量',
          '只有对偶变量 $\\alpha_i > 0$ 的样本才对模型参数 $w$ 有贡献',
          '远离分类边界的样本的对偶变量 $\\alpha_i > 0$',
          '$w$ 与所有样本的内积均相等',
        ],
        answer: 1,
        explain:
          '由 KKT 互补松弛，$\\alpha_i[y_i(w^\\top x_i + b) - 1 + \\xi_i] = 0$。若样本远离分类边界（约束严格不活跃），则 $\\alpha_i = 0$，该样本对 $w = \\sum_i \\alpha_i y_i x_i$ 贡献为零。只有在边界上或违反间隔的样本（支持向量）才有 $\\alpha_i > 0$。',
      },
      // 题 5：input — 等式约束 KKT
      {
        id: 'c28-b-q5',
        type: 'input',
        prompt: '在约束 $x+y=1$ 下最小化 $x^2+y^2$，最优值 $f^\\star$ 等于多少？（分数请写如 1/2）',
        accept: ['1/2', '0.5'],
        placeholder: '如 1/2',
        explain:
          '由 KKT（Lagrange 法）：驻点条件 $2x+\\nu=0$，$2y+\\nu=0$，得 $x=y$。代入约束 $x+y=1$ 得 $x=y=\\frac{1}{2}$，最优值 $f^\\star = (\\frac{1}{2})^2+(\\frac{1}{2})^2 = \\frac{1}{2}$。',
      },
      // 题 6：match — KKT 四条件
      {
        id: 'c28-b-q6',
        type: 'match',
        prompt: '将 KKT 四条件与对应名称配对。',
        left: [
          '$\\nabla_x L(x^\\star, \\lambda^\\star, \\nu^\\star) = 0$',
          '$g_i(x^\\star) \\leq 0$，$h_j(x^\\star) = 0$',
          '$\\lambda_i^\\star \\geq 0$',
          '$\\lambda_i^\\star g_i(x^\\star) = 0$',
        ],
        right: ['驻点条件', '原始可行性', '对偶可行性', '互补松弛'],
      },
      // 题 7：judge — 强对偶性
      {
        id: 'c28-b-q7',
        type: 'judge',
        prompt: '对所有凸优化问题，强对偶性（原始最优值等于对偶最优值）自动成立，无需任何额外条件。',
        answer: false,
        explain:
          '强对偶性需要满足一定的正则性条件，最常用的是 **Slater 条件**：存在一个严格可行点，使得所有不等式约束严格满足（$g_i(x) < 0$）。不满足 Slater 条件时，即使是凸问题，也可能只有弱对偶（$d^\\star \\leq p^\\star$）而非强对偶（$d^\\star = p^\\star$）。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c28-c  Jensen 不等式与信息论应用
  // ─────────────────────────────────────────
  {
    id: 'c28-c',
    title: 'Jensen 不等式与信息论应用',
    subtitle: 'Jensen 不等式·KL 散度非负·ELBO 下界·AM-GM·熵的凸性',
    intro: [
      // 卡 1：动机——凸性与期望的关系
      {
        title: '凸性与期望：Jensen 不等式的核心思想',
        body:
          '对凸函数 $f$，考虑两种操作顺序：\n\n' +
          '（A）先计算数值的平均，再对平均值施加函数：$f(\\mathbb{E}[X])$。\n\n' +
          '（B）先对每个数值施加函数，再计算函数值的平均：$\\mathbb{E}[f(X)]$。\n\n' +
          '对凸函数来说，**方案 A 始终 $\\leq$ 方案 B**，即：\n\n' +
          '$f(\\mathbb{E}[X]) \\leq \\mathbb{E}[f(X)]$。\n\n' +
          '**直觉**：凸函数图像是碗形，碗底（均值处的函数值）低于沿碗壁取各点函数值的平均。\n\n' +
          '**实际例子**：$f(x) = x^2$ 是凸函数。方差公式 $\\mathrm{Var}(X) = \\mathbb{E}[X^2] - (\\mathbb{E}[X])^2 \\geq 0$，其实就是 Jensen 不等式的直接推论：$\\mathbb{E}[X^2] \\geq (\\mathbb{E}[X])^2$。',
        formula: 'f(\\mathbb{E}[X]) \\leq \\mathbb{E}[f(X)]\\quad (f \\text{ 是凸函数})',
        tip: '凹函数（如 $\\log$）不等号反向：$f(\\mathbb{E}[X]) \\geq \\mathbb{E}[f(X)]$。这个反向版本是推导 ELBO 下界的关键。',
      },
      // 卡 2：Jensen 不等式的离散形式与推导
      {
        title: 'Jensen 不等式：从两点到多点',
        body:
          '凸函数定义中，两点情形：$f(\\lambda x + (1-\\lambda)y) \\leq \\lambda f(x) + (1-\\lambda)f(y)$。\n\n' +
          '把 $\\lambda$ 和 $1-\\lambda$ 看成两个"权重"（它们非负且和为 1），这正是在两点上的 Jensen 不等式。\n\n' +
          '**离散形式（多点）**：若 $\\lambda_i \\geq 0$，$\\sum_i \\lambda_i = 1$，则：\n\n' +
          '$f\\left(\\sum_i \\lambda_i x_i\\right) \\leq \\sum_i \\lambda_i f(x_i)$。\n\n' +
          '**连续形式（随机变量）**：若 $X$ 是随机变量，则 $f(\\mathbb{E}[X]) \\leq \\mathbb{E}[f(X)]$。\n\n' +
          '**数字验证**：$f(x) = e^x$（凸函数），$X$ 取 $0$ 和 $2$ 各以 $\\frac{1}{2}$ 概率。$f(\\mathbb{E}[X]) = e^1 \\approx 2.718$，$\\mathbb{E}[f(X)] = \\frac{e^0 + e^2}{2} \\approx \\frac{1+7.389}{2} \\approx 4.195$。确实 $2.718 < 4.195$ ✓。',
        formula: 'f\\!\\left(\\sum_i \\lambda_i x_i\\right) \\leq \\sum_i \\lambda_i f(x_i),\\quad \\lambda_i \\geq 0,\\; \\sum_i \\lambda_i = 1',
      },
      // 卡 3：AM-GM 不等式——Jensen 的经典推论
      {
        title: '例题：用 Jensen 证明 AM-GM 不等式',
        body:
          '算术平均 $\\geq$ 几何平均（AM-GM）：$\\dfrac{a+b}{2} \\geq \\sqrt{ab}$（$a,b>0$）。用 Jensen 不等式一步推出。',
        steps: [
          '取凸函数 $f(x) = -\\ln x$（可验证：$f\'\'(x) = 1/x^2 > 0$）。',
          '对正数 $a, b$，取 $\\lambda = \\frac{1}{2}$，由 Jensen：$-\\ln\\left(\\frac{a+b}{2}\\right) \\leq \\frac{-\\ln a + (-\\ln b)}{2} = -\\frac{\\ln a + \\ln b}{2} = -\\ln\\sqrt{ab}$。',
          '不等号两边乘以 $-1$（方向反转）：$\\ln\\left(\\frac{a+b}{2}\\right) \\geq \\ln\\sqrt{ab}$。',
          '对两边取指数（$e^x$ 单调递增）：$\\frac{a+b}{2} \\geq \\sqrt{ab}$，AM-GM 得证。',
          '等号成立条件：Jensen 不等式等号当且仅当 $a = b$ 时成立（此时 $\\frac{a+b}{2} = a = b = \\sqrt{ab}$）。',
        ],
        tip: '选 $f(x) = -\\ln x$ 的原因：它是凸函数，且 $-\\ln(ab) = -\\ln a - \\ln b$，能把乘法拆成加法，从而连接算术均值和几何均值。',
      },
      // 卡 4：KL 散度非负性——Jensen 在信息论中的应用
      {
        title: 'KL 散度非负：$\\mathrm{KL}(p \\| q) \\geq 0$',
        body:
          'KL 散度（相对熵）衡量两个分布 $p$ 和 $q$ 的差异：\n\n' +
          '$\\mathrm{KL}(p \\| q) = \\int p(x) \\ln \\dfrac{p(x)}{q(x)} dx$。\n\n' +
          '**直觉**：$\\mathrm{KL}(p \\| q)$ 衡量"用 $q$ 来近似 $p$ 时平均多用了多少比特"。它越小，$q$ 越接近 $p$。\n\n' +
          '**性质（非常重要）**：$\\mathrm{KL}(p \\| q) \\geq 0$，等号当且仅当 $p = q$ 几乎处处成立。\n\n' +
          '**用 Jensen 证明**：$-\\ln t$ 是凸函数。令 $Z = q(X)/p(X)$（在 $X \\sim p$ 下），则 $\\mathbb{E}_p[Z] = \\int p(x) \\cdot \\frac{q(x)}{p(x)} dx = \\int q(x) dx = 1$。\n\n' +
          '由 Jensen：$-\\ln \\mathbb{E}_p[Z] \\leq \\mathbb{E}_p[-\\ln Z]$，即 $0 = -\\ln 1 \\leq \\mathbb{E}_p[-\\ln \\frac{q}{p}] = \\mathbb{E}_p[\\ln \\frac{p}{q}] = \\mathrm{KL}(p \\| q)$。',
        formula: '\\mathrm{KL}(p\\|q) = \\mathbb{E}_p\\!\\left[\\ln\\frac{p(X)}{q(X)}\\right] \\geq 0',
        tip: 'KL 散度非对称：$\\mathrm{KL}(p \\| q) \\neq \\mathrm{KL}(q \\| p)$（通常）。这是它不能作为"真正距离"的原因，但它的非负性已足够用于推导机器学习中的许多不等式。',
      },
      // 卡 5：ELBO——Jensen 在变分推断中的应用
      {
        title: 'ELBO：Jensen 不等式推导变分下界',
        body:
          '变分自编码器（VAE）和贝叶斯推断中，一个核心难题是计算边缘似然 $\\log p(x)$，因为需要对潜变量 $z$ 积分，通常无法解析求解。\n\n' +
          '**解决方案**：引入近似分布 $q(z)$，把积分改写为关于 $q(z)$ 的期望：\n\n' +
          '$\\log p(x) = \\log \\int q(z) \\cdot \\dfrac{p(x,z)}{q(z)} dz = \\log \\mathbb{E}_q\\!\\left[\\dfrac{p(x,z)}{q(z)}\\right]$。\n\n' +
          '因为 $\\log$ 是**凹函数**，Jensen 不等式对凹函数方向反向：$\\log \\mathbb{E}[Y] \\geq \\mathbb{E}[\\log Y]$。\n\n' +
          '因此：$\\log p(x) \\geq \\mathbb{E}_q[\\log p(x,z)] - \\mathbb{E}_q[\\log q(z)] =: \\mathrm{ELBO}$。\n\n' +
          '**$\\mathrm{ELBO}$ 的含义**：它是 $\\log p(x)$ 的一个**可计算下界**，最大化 ELBO 等价于最小化 $\\mathrm{KL}(q(z) \\| p(z|x))$。当 $q = p(z|x)$ 时等号成立。',
        formula: '\\log p(x) \\geq \\underbrace{\\mathbb{E}_q[\\log p(x,z)] - \\mathbb{E}_q[\\log q(z)]}_{\\mathrm{ELBO}}',
        tip: 'ELBO（证据下界）是 VAE 训练目标的数学基础。Jensen 不等式将一个棘手的积分替换为可优化的下界——凸分析的强大之处。',
      },
      // 卡 6：易错点汇总 + reveal
      {
        title: '易错点汇总与自测',
        body:
          '**易错一（方向）**：凸函数 $f(\\mathbb{E}[X]) \\leq \\mathbb{E}[f(X)]$；凹函数（如 $\\log$）方向反向 $f(\\mathbb{E}[X]) \\geq \\mathbb{E}[f(X)]$。混淆方向会导致推导完全相反。\n\n' +
          '**易错二（KL 非对称）**：$\\mathrm{KL}(p \\| q) \\neq \\mathrm{KL}(q \\| p)$，两者不同，分别在正向 KL 和反向 KL 最小化时有不同行为。\n\n' +
          '**易错三（ELBO 等号条件）**：ELBO 等于 $\\log p(x)$ 当且仅当 $q(z) = p(z|x)$（后验分布）。如果 $q$ 选得很差，ELBO 会远低于 $\\log p(x)$，此时 ELBO 只是一个松弛下界。\n\n' +
          '**易错四（Jensen 的适用条件）**：$f$ 必须在整个取值范围内是凸函数（或凹函数）。若只在某段区间凸，只有当随机变量几乎必然取该区间内的值时 Jensen 才能直接用。',
        reveal: {
          q: '用 Jensen 不等式证明 $\\mathbb{E}[X^2] \\geq (\\mathbb{E}[X])^2$（即方差非负），请说明用的是哪个凸函数。',
          a: '取凸函数 $f(t) = t^2$（$f\'\'(t) = 2 > 0$，严格凸）。由 Jensen：$f(\\mathbb{E}[X]) \\leq \\mathbb{E}[f(X)]$，即 $(\\mathbb{E}[X])^2 \\leq \\mathbb{E}[X^2]$。整理得 $\\mathbb{E}[X^2] - (\\mathbb{E}[X])^2 \\geq 0$，这正是 $\\mathrm{Var}(X) \\geq 0$。',
        },
      },
    ],
    questions: [
      // 题 1：judge — Jensen 不等式方向
      {
        id: 'c28-c-q1',
        type: 'judge',
        prompt: '对凸函数 $f$，Jensen 不等式为 $f(\\mathbb{E}[X]) \\leq \\mathbb{E}[f(X)]$（先平均再取函数 $\\leq$ 先取函数再平均）。',
        answer: true,
        explain:
          '这是 Jensen 不等式的标准形式，对凸函数成立。几何直觉：凸函数的碗底（均值处函数值）低于沿碗壁各点函数值的加权平均。对凹函数（如 $\\log$）方向反向。',
      },
      // 题 2：choice — AM-GM 证明中的凸函数
      {
        id: 'c28-c-q2',
        type: 'choice',
        prompt: '用 Jensen 不等式证明 AM-GM 不等式 $\\frac{a+b}{2} \\geq \\sqrt{ab}$ 时，选用的凸函数是：',
        options: ['$f(x) = \\ln x$', '$f(x) = -\\ln x$', '$f(x) = x^2$', '$f(x) = e^x$'],
        answer: 1,
        explain:
          '选 $f(x) = -\\ln x$，它是凸函数（$f\'\'(x) = 1/x^2 > 0$）。对 $a, b > 0$，$\\lambda = 1/2$，Jensen 给出 $-\\ln\\frac{a+b}{2} \\leq \\frac{-\\ln a - \\ln b}{2} = -\\ln\\sqrt{ab}$，等价于 $\\frac{a+b}{2} \\geq \\sqrt{ab}$。$\\ln x$ 是凹函数，方向相反，不能直接用。',
      },
      // 题 3：input — 验证 Jensen
      {
        id: 'c28-c-q3',
        type: 'input',
        prompt:
          '设 $X$ 等可能取 $0$ 或 $4$（各 $\\frac{1}{2}$ 概率），$f(x) = x^2$。计算 $f(\\mathbb{E}[X])$ 的值等于多少？',
        accept: ['4'],
        placeholder: '输入整数',
        explain:
          '$\\mathbb{E}[X] = \\frac{0+4}{2} = 2$，故 $f(\\mathbb{E}[X]) = 2^2 = 4$。另外 $\\mathbb{E}[f(X)] = \\frac{f(0)+f(4)}{2} = \\frac{0+16}{2} = 8$。由 Jensen，$4 = f(\\mathbb{E}[X]) \\leq \\mathbb{E}[f(X)] = 8$ ✓。',
      },
      // 题 4：choice — KL 散度
      {
        id: 'c28-c-q4',
        type: 'choice',
        prompt: '关于 KL 散度 $\\mathrm{KL}(p \\| q)$，下列哪条是正确的？',
        options: [
          '$\\mathrm{KL}(p\\|q) = \\mathrm{KL}(q\\|p)$',
          '$\\mathrm{KL}(p\\|q) \\geq 0$，等号当且仅当 $p = q$ 几乎处处成立',
          '$\\mathrm{KL}(p\\|q)$ 可能取负值',
          '$\\mathrm{KL}(p\\|q)$ 满足三角不等式，是一种距离',
        ],
        answer: 1,
        explain:
          '$\\mathrm{KL}(p\\|q) \\geq 0$ 由 Jensen 不等式（对凸函数 $-\\ln t$）推出，等号当且仅当 $p = q$ 几乎处处成立。KL 散度是非对称的（$\\mathrm{KL}(p\\|q) \\neq \\mathrm{KL}(q\\|p)$ 一般），也不满足三角不等式，因此不是严格意义上的距离。',
      },
      // 题 5：judge — ELBO 与 log p(x)
      {
        id: 'c28-c-q5',
        type: 'judge',
        prompt: 'ELBO（证据下界）始终满足 $\\mathrm{ELBO} \\leq \\log p(x)$，等号在 $q(z) = p(z|x)$ 时成立。',
        answer: true,
        explain:
          '由 Jensen 不等式（对凹函数 $\\log$）推导：$\\log p(x) \\geq \\mathbb{E}_q[\\log p(x,z)] - \\mathbb{E}_q[\\log q(z)] = \\mathrm{ELBO}$。等号成立当且仅当 $q(z) = p(z|x)$（后验分布），此时近似后验恰好等于真实后验，KL 散度为零。',
      },
      // 题 6：match — Jensen 不等式应用
      {
        id: 'c28-c-q6',
        type: 'match',
        prompt: '将 Jensen 不等式的应用场景与对应的凸/凹函数配对。',
        left: [
          '证明 $\\mathrm{KL}(p\\|q) \\geq 0$',
          '推导 ELBO 下界',
          '证明 $\\mathbb{E}[X^2] \\geq (\\mathbb{E}[X])^2$',
        ],
        right: ['凸函数 $-\\ln t$，正向 Jensen', '凹函数 $\\log$，反向 Jensen', '凸函数 $t^2$，正向 Jensen'],
      },
    ],
  },
]
