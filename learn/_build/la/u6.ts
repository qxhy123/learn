import type { Unit } from '../../src/types'

export const UNIT: Unit = {
  id: 'u6',
  title: '内积与正交',
  color: '#2b70c9',
  icon: '⟂',
  blurb: '用内积度量长度与夹角，用 Gram-Schmidt 建正交基，用最小二乘把 $\\mathbf{b}$ 投影到列空间——三步让线性代数从抽象到可算。',
  lessons: [
    // =====================================================================
    // 关 1：内积·范数·夹角
    // =====================================================================
    {
      id: 'u6-l1',
      title: '内积·范数·夹角',
      subtitle: '给向量空间装上刻度尺与量角器',
      intro: [
        {
          title: '为什么需要内积？',
          body: '向量空间只告诉我们怎么"移动"向量：加法与标量乘。但它什么都没说关于"长度"和"角度"——如果你问"$(1,2)^T$ 有多长？"或者"这两个向量垂直吗？"，单凭加法无法回答。\n\n**内积**正是为此而生：它给向量空间装上一把"量角器+刻度尺"，使我们能谈论距离、方向、垂直。一旦有了内积，向量空间就升级为**内积空间**。',
          tip: '标准内积（点积）$\\langle \\mathbf{u},\\mathbf{v}\\rangle=\\mathbf{u}^T\\mathbf{v}=\\sum_i u_i v_i$ 是最常用的内积。'
        },
        {
          title: '内积的四条公理',
          body: '在实向量空间 $V$ 上，**内积** $\\langle \\cdot,\\cdot\\rangle:V\\times V\\to\\mathbb{R}$ 需满足（对所有 $\\mathbf{u},\\mathbf{v},\\mathbf{w}\\in V$，$c\\in\\mathbb{R}$）：\n\n**① 正性**：$\\langle \\mathbf{v},\\mathbf{v}\\rangle\\geq 0$\n\n**② 正定性**：$\\langle \\mathbf{v},\\mathbf{v}\\rangle=0\\Leftrightarrow\\mathbf{v}=\\mathbf{0}$\n\n**③ 对称性**：$\\langle \\mathbf{u},\\mathbf{v}\\rangle=\\langle \\mathbf{v},\\mathbf{u}\\rangle$\n\n**④ 第一变元线性性**：$\\langle c\\mathbf{u}+\\mathbf{w},\\mathbf{v}\\rangle=c\\langle \\mathbf{u},\\mathbf{v}\\rangle+\\langle \\mathbf{w},\\mathbf{v}\\rangle$',
          formula: '\\langle \\mathbf{u},\\mathbf{v}\\rangle = \\sum_{i=1}^n u_i v_i = \\mathbf{u}^T\\mathbf{v}',
          tip: '正性（$\\geq0$）和正定性（$=0\\Rightarrow\\mathbf{0}$）是两条独立公理，缺一不可——只验正性不够！'
        },
        {
          title: '范数：内积诱导的长度',
          body: '有了内积，立刻可以定义向量的**范数**（长度）：',
          formula: '\\lVert\\mathbf{v}\\rVert = \\sqrt{\\langle \\mathbf{v},\\mathbf{v}\\rangle}',
          steps: [
            '**例**：$\\mathbf{v}=(3,4)^T$，$\\langle \\mathbf{v},\\mathbf{v}\\rangle=9+16=25$，所以 $\\lVert\\mathbf{v}\\rVert=5$',
            '**归一化**：把任意非零向量除以范数，得同方向的单位向量 $\\hat{\\mathbf{v}}=\\mathbf{v}/\\lVert\\mathbf{v}\\rVert$',
            '对 $\\mathbf{v}=(3,4)^T$：$\\hat{\\mathbf{v}}=(3/5,4/5)^T$，验证 $\\lVert\\hat{\\mathbf{v}}\\rVert=\\sqrt{9/25+16/25}=1$ ✓',
            '**距离**：两向量之间的距离 $d(\\mathbf{u},\\mathbf{v})=\\lVert\\mathbf{u}-\\mathbf{v}\\rVert$'
          ],
          tip: '注意 $\\lVert\\mathbf{v}\\rVert^2=\\langle \\mathbf{v},\\mathbf{v}\\rangle$（平方内积），计算时先算内积再开根。'
        },
        {
          title: '夹角的定义',
          body: '两非零向量 $\\mathbf{u},\\mathbf{v}$ 的**夹角** $\\theta\\in[0,\\pi]$ 由下式定义：',
          formula: '\\cos\\theta = \\frac{\\langle \\mathbf{u},\\mathbf{v}\\rangle}{\\lVert\\mathbf{u}\\rVert\\,\\lVert\\mathbf{v}\\rVert}',
          steps: [
            '**例 1**：$\\mathbf{u}=(1,1)^T$，$\\mathbf{v}=(1,-1)^T$，$\\langle \\mathbf{u},\\mathbf{v}\\rangle=1-1=0$，$\\theta=90°$',
            '**例 2**：$\\mathbf{u}=(1,1)^T$，$\\mathbf{v}=(2,0)^T$，$\\langle \\mathbf{u},\\mathbf{v}\\rangle=2$，$\\lVert\\mathbf{u}\\rVert=\\sqrt{2}$，$\\lVert\\mathbf{v}\\rVert=2$',
            '$\\cos\\theta=2/(\\sqrt{2}\\times2)=1/\\sqrt{2}$，故 $\\theta=45°$'
          ],
          tip: '当 $\\langle \\mathbf{u},\\mathbf{v}\\rangle=0$ 时 $\\theta=90°$，称两向量**正交**，记 $\\mathbf{u}\\perp\\mathbf{v}$。'
        },
        {
          title: '加权内积与函数内积',
          body: '标准内积并非唯一选择。给定正权重 $w_i>0$，可定义**加权内积** $\\langle \\mathbf{u},\\mathbf{v}\\rangle_W=\\sum_i w_i u_i v_i$，权重体现各分量的"重要程度"。\n\n在连续函数空间 $C([a,b])$ 上，$\\langle f,g\\rangle=\\int_a^b f(x)g(x)\\,dx$ 也是合法的内积。Fourier 级数的正交性正基于此内积。\n\n**关键**：换了内积，范数、夹角、正交的判定都要重算。',
          tip: '加权内积下"正交"与标准内积下"正交"是不同的条件，切勿混用。'
        },
        {
          title: '勾股定理的内积版',
          body: '若 $\\mathbf{u}\\perp\\mathbf{v}$（即 $\\langle \\mathbf{u},\\mathbf{v}\\rangle=0$），则',
          formula: '\\lVert\\mathbf{u}+\\mathbf{v}\\rVert^2 = \\lVert\\mathbf{u}\\rVert^2 + \\lVert\\mathbf{v}\\rVert^2',
          steps: [
            '**证明**：$\\lVert\\mathbf{u}+\\mathbf{v}\\rVert^2=\\langle \\mathbf{u}+\\mathbf{v},\\mathbf{u}+\\mathbf{v}\\rangle=\\lVert\\mathbf{u}\\rVert^2+2\\langle \\mathbf{u},\\mathbf{v}\\rangle+\\lVert\\mathbf{v}\\rVert^2=\\lVert\\mathbf{u}\\rVert^2+\\lVert\\mathbf{v}\\rVert^2$',
            '**含义**：正交就是内积版勾股定理——垂直的两边，斜边的平方等于两边的平方和'
          ],
          reveal: {
            q: '$\\mathbf{u}=(1,2,2)^T$，$\\mathbf{v}=(2,-1,0)^T$，两者是否正交？$\\lVert\\mathbf{u}+\\mathbf{v}\\rVert^2$ 等于多少？',
            a: '$\\langle \\mathbf{u},\\mathbf{v}\\rangle=2-2+0=0$，正交。由勾股定理：$\\lVert\\mathbf{u}\\rVert^2=9$，$\\lVert\\mathbf{v}\\rVert^2=5$，$\\lVert\\mathbf{u}+\\mathbf{v}\\rVert^2=14$。'
          }
        },
        {
          title: '易错点小结',
          body: '**易错点**：\n\n① 验证内积时"正性"和"正定性"是两条，缺一不可。\n\n② $\\lVert\\mathbf{v}\\rVert=\\sqrt{\\langle \\mathbf{v},\\mathbf{v}\\rangle}$（要开根号），而 $\\lVert\\mathbf{v}\\rVert^2=\\langle \\mathbf{v},\\mathbf{v}\\rangle$（平方）。\n\n③ 夹角公式能成立，依赖 Cauchy-Schwarz 不等式保证 $|\\cos\\theta|\\leq1$。\n\n**推导链**：内积 $\\to$ 范数（长度）$\\to$ 夹角 $\\to$ 正交（内积为零）。',
          tip: '零向量与任何向量的内积为零——零向量正交于所有向量，是特殊规定。'
        }
      ],
      questions: [
        {
          id: 'u6-l1-q1',
          type: 'choice',
          prompt: '在 $\\mathbb{R}^n$ 的标准内积下，$\\langle \\mathbf{u},\\mathbf{v}\\rangle$ 等于？',
          options: [
            '$\\lVert\\mathbf{u}\\rVert+\\lVert\\mathbf{v}\\rVert$',
            '$\\mathbf{u}^T\\mathbf{v}=\\sum_{i=1}^n u_i v_i$',
            '$\\lVert\\mathbf{u}-\\mathbf{v}\\rVert^2$',
            '$\\mathbf{u}\\mathbf{v}^T$（外积矩阵）'
          ],
          answer: 1,
          explain: '标准内积就是点积：$\\langle \\mathbf{u},\\mathbf{v}\\rangle=\\mathbf{u}^T\\mathbf{v}=\\sum_i u_i v_i$。外积 $\\mathbf{u}\\mathbf{v}^T$ 是矩阵而非实数。'
        },
        {
          id: 'u6-l1-q2',
          type: 'input',
          prompt: '计算 $\\mathbf{v}=(1,2,2)^T$ 的欧几里得范数 $\\lVert\\mathbf{v}\\rVert$（写整数）。',
          accept: ['3'],
          placeholder: '输入整数',
          explain: '$\\lVert\\mathbf{v}\\rVert=\\sqrt{1^2+2^2+2^2}=\\sqrt{9}=3$。'
        },
        {
          id: 'u6-l1-q3',
          type: 'judge',
          prompt: '向量 $\\mathbf{u}=(3,0)^T$ 与 $\\mathbf{v}=(0,-5)^T$ 正交。',
          answer: true,
          explain: '$\\langle \\mathbf{u},\\mathbf{v}\\rangle=3\\times0+0\\times(-5)=0$，内积为零，两向量正交。'
        },
        {
          id: 'u6-l1-q4',
          type: 'choice',
          prompt: '$\\mathbf{u}=(1,1)^T$，$\\mathbf{v}=(1,-1)^T$，两向量夹角为？',
          options: ['$0°$（同向）', '$45°$', '$90°$（正交）', '$180°$（反向）'],
          answer: 2,
          explain: '$\\langle \\mathbf{u},\\mathbf{v}\\rangle=1\\times1+1\\times(-1)=0$，内积为 $0$，夹角恰好是 $90°$。'
        },
        {
          id: 'u6-l1-q5',
          type: 'input',
          prompt: '$\\mathbf{u}=(3,4)^T$，归一化后得 $\\hat{\\mathbf{u}}=(3/5,\\,?)^T$，第二分量为多少？写分数。',
          accept: ['4/5'],
          placeholder: '如 3/5',
          explain: '$\\lVert\\mathbf{u}\\rVert=\\sqrt{9+16}=5$，$\\hat{\\mathbf{u}}=(3/5,4/5)^T$，第二分量为 $4/5$。'
        },
        {
          id: 'u6-l1-q6',
          type: 'judge',
          prompt: '验证内积公理时，只需证明正性 $\\langle \\mathbf{v},\\mathbf{v}\\rangle\\geq0$ 即可，不需要单独验证正定性。',
          answer: false,
          explain: '正性（$\\geq0$）和正定性（$=0\\Rightarrow\\mathbf{v}=\\mathbf{0}$）是独立的两条公理，必须分别验证。'
        },
        {
          id: 'u6-l1-q7',
          type: 'match',
          prompt: '将下列概念与其定义配对。',
          left: [
            '内积为零',
            '范数为 $1$',
            '$\\sqrt{\\langle \\mathbf{v},\\mathbf{v}\\rangle}$',
            '$\\langle \\mathbf{v},\\mathbf{v}\\rangle=0\\Rightarrow\\mathbf{v}=\\mathbf{0}$'
          ],
          right: [
            '两向量正交',
            '单位向量',
            '向量的范数',
            '正定性公理'
          ],
          explain: '内积为零 ↔ 正交；范数为 1 ↔ 单位向量；范数定义是内积开根；正定性排除"非零但自内积为零"的情形。'
        },
        {
          id: 'u6-l1-q8',
          type: 'choice',
          prompt: '$\\mathbf{u}=(0,1)^T$，$\\mathbf{v}=(1,0)^T$，使用加权内积 $\\langle \\mathbf{a},\\mathbf{b}\\rangle_W=2a_1b_1+3a_2b_2$，这两个向量是否正交？',
          options: [
            '正交，因为 $\\langle \\mathbf{u},\\mathbf{v}\\rangle_W=0$',
            '不正交，因为 $\\langle \\mathbf{u},\\mathbf{v}\\rangle_W\\neq0$',
            '正交，因为标准内积为零',
            '无法判断，需要更多信息'
          ],
          answer: 0,
          explain: '$\\langle \\mathbf{u},\\mathbf{v}\\rangle_W=2\\times0\\times1+3\\times1\\times0=0$，在此加权内积下正交。（与标准内积结论一致，但需用所给内积判断。）'
        },
        {
          id: 'u6-l1-q9',
          type: 'input',
          prompt: '$\\mathbf{u}=(1,0)^T$，$\\mathbf{v}=(1,1)^T$，计算 $\\cos\\theta$（写分数或小数，化简）。',
          accept: ['1/sqrt(2)', '\\sqrt{2}/2', 'sqrt(2)/2', '0.707'],
          placeholder: '如 1/2',
          explain: '$\\langle \\mathbf{u},\\mathbf{v}\\rangle=1$，$\\lVert\\mathbf{u}\\rVert=1$，$\\lVert\\mathbf{v}\\rVert=\\sqrt{2}$，$\\cos\\theta=1/\\sqrt{2}=\\sqrt{2}/2$，夹角 $45°$。'
        }
      ]
    },

    // =====================================================================
    // 关 2：正交与 Cauchy-Schwarz
    // =====================================================================
    {
      id: 'u6-l2',
      title: '正交与 Cauchy-Schwarz',
      subtitle: '正交集线性无关；不等式保证夹角合法',
      intro: [
        {
          title: '正交集与标准正交集',
          body: '**正交集**：集合中任意两不同向量的内积均为零，即 $\\langle \\mathbf{v}_i,\\mathbf{v}_j\\rangle=0$（$i\\neq j$）。\n\n**标准正交集**（ONS）：在正交集的基础上，每个向量都是单位向量。用 Kronecker delta 写作 $\\langle \\mathbf{v}_i,\\mathbf{v}_j\\rangle=\\delta_{ij}$。\n\n**标准正交基**（ONB）：同时是所在空间的基——最"舒适"的坐标系，坐标计算化为内积计算。',
          formula: '\\langle \\mathbf{v}_i,\\mathbf{v}_j\\rangle = \\delta_{ij} = \\begin{cases}1 & i=j \\\\ 0 & i\\neq j\\end{cases}',
          tip: '标准正交基下，任意向量 $\\mathbf{v}=\\sum_i\\langle \\mathbf{v},\\mathbf{e}_i\\rangle\\mathbf{e}_i$——坐标直接用内积读出，无需解方程组！'
        },
        {
          title: '正交集中的非零向量线性无关',
          body: '**命题**：正交集中的非零向量必线性无关。\n\n**证明**：设 $c_1\\mathbf{v}_1+\\cdots+c_k\\mathbf{v}_k=\\mathbf{0}$，对两边取与 $\\mathbf{v}_i$ 的内积：\n\n由正交性，$j\\neq i$ 时 $\\langle \\mathbf{v}_j,\\mathbf{v}_i\\rangle=0$，只剩 $c_i\\langle \\mathbf{v}_i,\\mathbf{v}_i\\rangle=0$。\n\n因为 $\\mathbf{v}_i\\neq\\mathbf{0}$，所以 $c_i=0$。对所有 $i$ 成立，故线性无关。',
          tip: '这正是正交基"好用"的根本原因：正交方向彼此独立，互无重叠。'
        },
        {
          title: '正交补',
          body: '设 $W$ 是内积空间 $V$ 的子空间，**正交补** $W^\\perp$ 是 $V$ 中所有与 $W$ 的每个向量都正交的向量的集合：\n\n$W^\\perp=\\{\\mathbf{v}\\in V\\mid\\langle \\mathbf{v},\\mathbf{w}\\rangle=0\\text{ 对所有 }\\mathbf{w}\\in W\\}$\n\n**关键性质**（$V=\\mathbb{R}^n$）：\n\n① $V=W\\oplus W^\\perp$（直和分解）：每个向量唯一分解为 $W$ 内分量与 $W^\\perp$ 内分量之和。\n\n② $\\dim(W)+\\dim(W^\\perp)=n$。\n\n③ $(W^\\perp)^\\perp=W$，$W\\cap W^\\perp=\\{\\mathbf{0}\\}$。',
          steps: [
            '**例**：$W=\\{(x,y,0)^T\\mid x,y\\in\\mathbb{R}\\}$（$xy$ 平面），则 $W^\\perp=\\{(0,0,z)^T\\mid z\\in\\mathbb{R}\\}$（$z$ 轴）',
            '矩阵四子空间：$(\\text{Col}(A))^\\perp=\\text{Null}(A^T)$，$(\\text{Row}(A))^\\perp=\\text{Null}(A)$'
          ]
        },
        {
          title: 'Cauchy-Schwarz 不等式',
          body: '**定理**：设 $V$ 是实内积空间，对所有 $\\mathbf{u},\\mathbf{v}\\in V$：',
          formula: '|\\langle \\mathbf{u},\\mathbf{v}\\rangle| \\leq \\lVert\\mathbf{u}\\rVert\\,\\lVert\\mathbf{v}\\rVert',
          steps: [
            '**等号成立**当且仅当 $\\mathbf{u}$ 与 $\\mathbf{v}$ 线性相关（一个是另一个的实数倍）',
            '**重要推论**：夹角公式 $\\cos\\theta=\\langle \\mathbf{u},\\mathbf{v}\\rangle/(\\lVert\\mathbf{u}\\rVert\\lVert\\mathbf{v}\\rVert)$ 因为 $|\\cos\\theta|\\leq1$ 而合法',
            '**数值验证**：$\\mathbf{u}=(3,4)^T$，$\\mathbf{v}=(4,3)^T$，$|\\langle \\mathbf{u},\\mathbf{v}\\rangle|=24\\leq5\\times5=25$ ✓'
          ],
          tip: '余弦相似度 $=\\langle\\hat{\\mathbf{u}},\\hat{\\mathbf{v}}\\rangle$，值域 $[-1,1]$，$=1$ 同向，$=0$ 不相关，$=-1$ 反向——这正是 Cauchy-Schwarz 的保证。'
        },
        {
          title: '三角不等式',
          body: '由 Cauchy-Schwarz 立即推出**三角不等式**：',
          formula: '\\lVert\\mathbf{u}+\\mathbf{v}\\rVert \\leq \\lVert\\mathbf{u}\\rVert+\\lVert\\mathbf{v}\\rVert',
          steps: [
            '展开：$\\lVert\\mathbf{u}+\\mathbf{v}\\rVert^2=\\lVert\\mathbf{u}\\rVert^2+2\\langle \\mathbf{u},\\mathbf{v}\\rangle+\\lVert\\mathbf{v}\\rVert^2\\leq\\lVert\\mathbf{u}\\rVert^2+2\\lVert\\mathbf{u}\\rVert\\lVert\\mathbf{v}\\rVert+\\lVert\\mathbf{v}\\rVert^2=(\\lVert\\mathbf{u}\\rVert+\\lVert\\mathbf{v}\\rVert)^2$',
            '取平方根即得。几何意义：三角形任意一边不超过另外两边之和。'
          ]
        },
        {
          title: '正交矩阵',
          body: '**正交矩阵** $Q$：实方阵，满足 $Q^TQ=QQ^T=I$，等价于 $Q^{-1}=Q^T$。\n\n**关键性质**：\n\n① **保持内积**：$\\langle Q\\mathbf{u},Q\\mathbf{v}\\rangle=\\mathbf{u}^TQ^TQ\\mathbf{v}=\\langle \\mathbf{u},\\mathbf{v}\\rangle$\n\n② **保持范数（等距）**：$\\lVert Q\\mathbf{v}\\rVert=\\lVert\\mathbf{v}\\rVert$\n\n③ $\\det(Q)=\\pm1$（$+1$ 为旋转，$-1$ 为反射）\n\n典型例子：旋转矩阵、置换矩阵、Householder 反射矩阵。',
          tip: '正交矩阵的条件数 $\\kappa(Q)=1$——数值计算中最"优良"的矩阵，不放大误差。'
        },
        {
          title: '易错点与小结',
          body: '**易错点**：\n\n① 正交集中向量两两正交且非零 $\\Rightarrow$ 线性无关；但反过来"线性无关"不能推出"两两正交"。\n\n② $(W^\\perp)^\\perp=W$，但要求 $V=\\mathbb{R}^n$（有限维），无限维需额外条件。\n\n③ Cauchy-Schwarz 等号成立 $\\Leftrightarrow$ 线性相关，不是"等范数"。\n\n**小结**：正交性 $\\to$ 线性无关 $\\to$ 正交基（坐标计算简化）；Cauchy-Schwarz 保证夹角合法且给出余弦相似度的值域。',
          reveal: {
            q: '$\\mathbf{u}=(1,1,0)^T/\\sqrt{2}$ 和 $\\mathbf{v}=(0,0,1)^T$，验证两者是标准正交集。',
            a: '$\\langle \\mathbf{u},\\mathbf{v}\\rangle=0/\\sqrt{2}+0/\\sqrt{2}+0=0$，正交。$\\lVert\\mathbf{u}\\rVert=1$，$\\lVert\\mathbf{v}\\rVert=1$，单位向量。满足标准正交集定义。'
          }
        }
      ],
      questions: [
        {
          id: 'u6-l2-q1',
          type: 'judge',
          prompt: '正交集中的非零向量一定线性无关。',
          answer: true,
          explain: '设 $c_1\\mathbf{v}_1+\\cdots+c_k\\mathbf{v}_k=\\mathbf{0}$，对两边取与 $\\mathbf{v}_i$ 的内积，由正交性只剩 $c_i\\lVert\\mathbf{v}_i\\rVert^2=0$，因为 $\\mathbf{v}_i\\neq\\mathbf{0}$，故 $c_i=0$，线性无关。'
        },
        {
          id: 'u6-l2-q2',
          type: 'choice',
          prompt: 'Cauchy-Schwarz 不等式 $|\\langle \\mathbf{u},\\mathbf{v}\\rangle|\\leq\\lVert\\mathbf{u}\\rVert\\lVert\\mathbf{v}\\rVert$ 等号成立的条件是？',
          options: [
            '$\\mathbf{u}=\\mathbf{v}$',
            '$\\mathbf{u}$ 与 $\\mathbf{v}$ 线性相关（一个是另一个的实数倍）',
            '$\\mathbf{u}\\perp\\mathbf{v}$',
            '$\\lVert\\mathbf{u}\\rVert=\\lVert\\mathbf{v}\\rVert$'
          ],
          answer: 1,
          explain: '等号成立当且仅当两向量共线（线性相关），此时 $\\cos\\theta=\\pm1$，方向完全对齐或完全反向。'
        },
        {
          id: 'u6-l2-q3',
          type: 'input',
          prompt: '$\\mathbf{u}=(3,4)^T$，$\\mathbf{v}=(4,3)^T$，计算余弦相似度（写分数）。',
          accept: ['24/25'],
          placeholder: '如 3/5',
          explain: '$\\langle \\mathbf{u},\\mathbf{v}\\rangle=12+12=24$，$\\lVert\\mathbf{u}\\rVert=\\lVert\\mathbf{v}\\rVert=5$，$\\cos\\theta=24/25$。'
        },
        {
          id: 'u6-l2-q4',
          type: 'choice',
          prompt: '$W=\\text{span}\\{(1,0,0)^T,(0,1,0)^T\\}$（$xy$ 平面），$W^\\perp$ 是？',
          options: [
            '$\\text{span}\\{(1,1,0)^T\\}$',
            '$\\text{span}\\{(0,0,1)^T\\}$（$z$ 轴）',
            '$W$ 本身',
            '$\\mathbb{R}^3$ 全空间'
          ],
          answer: 1,
          explain: '$W^\\perp$ 包含所有与 $(1,0,0)^T$ 和 $(0,1,0)^T$ 都正交的向量，即第三分量任意、前两分量为零，即 $z$ 轴。'
        },
        {
          id: 'u6-l2-q5',
          type: 'judge',
          prompt: '对正交矩阵 $Q$，有 $\\lVert Q\\mathbf{x}\\rVert=\\lVert\\mathbf{x}\\rVert$ 对所有 $\\mathbf{x}$ 成立。',
          answer: true,
          explain: '$\\lVert Q\\mathbf{x}\\rVert^2=(Q\\mathbf{x})^T(Q\\mathbf{x})=\\mathbf{x}^TQ^TQ\\mathbf{x}=\\mathbf{x}^TI\\mathbf{x}=\\lVert\\mathbf{x}\\rVert^2$，正交矩阵是等距变换。'
        },
        {
          id: 'u6-l2-q6',
          type: 'choice',
          prompt: '在标准正交基 $\\{\\mathbf{e}_1,\\ldots,\\mathbf{e}_n\\}$ 下，向量 $\\mathbf{v}$ 在第 $i$ 个基向量方向的坐标等于？',
          options: [
            '$\\lVert\\mathbf{v}\\rVert/\\lVert\\mathbf{e}_i\\rVert$',
            '$\\langle \\mathbf{v},\\mathbf{e}_i\\rangle$',
            '$\\langle \\mathbf{e}_i,\\mathbf{e}_i\\rangle$',
            '需要解线性方程组才能求出'
          ],
          answer: 1,
          explain: '标准正交基的最大优势：坐标 $c_i=\\langle \\mathbf{v},\\mathbf{e}_i\\rangle$，直接用内积读出，无需解方程组。'
        },
        {
          id: 'u6-l2-q7',
          type: 'match',
          prompt: '将概念与其性质或含义配对。',
          left: [
            'Cauchy-Schwarz',
            '正交补 $W^\\perp$',
            '正交矩阵',
            '三角不等式'
          ],
          right: [
            '保证 $|\\cos\\theta|\\leq1$，夹角定义合法',
            '$V=W\\oplus W^\\perp$，维数互补',
            '$Q^TQ=I$，等距变换',
            '$\\lVert\\mathbf{u}+\\mathbf{v}\\rVert\\leq\\lVert\\mathbf{u}\\rVert+\\lVert\\mathbf{v}\\rVert$'
          ],
          explain: 'Cauchy-Schwarz 是夹角合法的保证；正交补给出直和分解；正交矩阵保持范数；三角不等式是范数的基本性质。'
        },
        {
          id: 'u6-l2-q8',
          type: 'judge',
          prompt: '$W\\cap W^\\perp=\\{\\mathbf{0}\\}$——子空间与其正交补的交集只有零向量。',
          answer: true,
          explain: '若 $\\mathbf{v}\\in W\\cap W^\\perp$，则 $\\mathbf{v}\\in W$ 且 $\\mathbf{v}\\in W^\\perp$，故 $\\langle \\mathbf{v},\\mathbf{v}\\rangle=0$，由正定性 $\\mathbf{v}=\\mathbf{0}$。'
        }
      ]
    },

    // =====================================================================
    // 关 3：正交投影
    // =====================================================================
    {
      id: 'u6-l3',
      title: '正交投影',
      subtitle: '向量的"影子"：从点到直线或子空间的最短距离',
      intro: [
        {
          title: '向量在向量上的投影',
          body: '**问题**：给定向量 $\\mathbf{a}$ 和方向 $\\mathbf{b}$，找与 $\\mathbf{b}$ 共线、且使差 $\\mathbf{a}-c\\mathbf{b}$ 垂直于 $\\mathbf{b}$ 的向量。\n\n**推导**：设投影为 $c\\mathbf{b}$，垂直条件 $(\\mathbf{a}-c\\mathbf{b})\\cdot\\mathbf{b}=0$ 给出 $c=\\mathbf{a}\\cdot\\mathbf{b}/(\\mathbf{b}\\cdot\\mathbf{b})$。',
          formula: '\\text{proj}_{\\mathbf{b}}\\,\\mathbf{a} = \\frac{\\langle \\mathbf{a},\\mathbf{b}\\rangle}{\\lVert\\mathbf{b}\\rVert^2}\\,\\mathbf{b}',
          steps: [
            '若 $\\mathbf{b}$ 已是单位向量 $\\hat{\\mathbf{b}}$，公式化简为 $\\text{proj}_{\\hat{\\mathbf{b}}}\\,\\mathbf{a}=(\\mathbf{a}\\cdot\\hat{\\mathbf{b}})\\hat{\\mathbf{b}}$',
            '**例**：$\\mathbf{a}=(3,1)^T$，$\\mathbf{b}=(1,1)^T$，$c=4/2=2$，投影为 $(2,2)^T$',
            '残差 $\\mathbf{a}^\\perp=(3,1)^T-(2,2)^T=(1,-1)^T$，验证：$(1,-1)^T\\cdot(1,1)^T=0$ ✓'
          ],
          tip: '**正交分解**：$\\mathbf{a}=\\text{proj}_{\\mathbf{b}}\\mathbf{a}+\\mathbf{a}^\\perp$，即"沿 $\\mathbf{b}$ 的影子"加"垂直于 $\\mathbf{b}$ 的纯新分量"。'
        },
        {
          title: '投影矩阵（到直线）',
          body: '投影 $\\text{proj}_{\\mathbf{b}}\\mathbf{a}$ 对 $\\mathbf{a}$ 是线性的，因此可以写成矩阵形式 $P\\mathbf{a}$，其中',
          formula: 'P = \\frac{\\mathbf{b}\\mathbf{b}^T}{\\mathbf{b}^T\\mathbf{b}}',
          steps: [
            '**对称性**：$P^T=P$（正交投影的标志）',
            '**幂等性**：$P^2=P$（投影两次等于投影一次——已在直线上了，再投影不变）',
            '**例**：$\\mathbf{b}=(1,1)^T$，$P=\\frac{1}{2}\\begin{pmatrix}1&1\\\\1&1\\end{pmatrix}$，$P^2=\\frac{1}{4}\\begin{pmatrix}2&2\\\\2&2\\end{pmatrix}=P$ ✓'
          ]
        },
        {
          title: '向量在子空间上的投影',
          body: '设子空间 $W$ 有标准正交基 $\\{q_1,\\ldots,q_k\\}$（两两正交且单位化），向量 $\\mathbf{a}$ 在 $W$ 上的**正交投影**为：',
          formula: '\\text{proj}_{W}\\,\\mathbf{a} = \\sum_{i=1}^{k}\\langle \\mathbf{a},q_i\\rangle\\,q_i = QQ^T\\mathbf{a}',
          steps: [
            '其中 $Q=[q_1|\\cdots|q_k]$（$n\\times k$ 矩阵），$P_W=QQ^T$ 是投影矩阵',
            '**关键性质**：残差 $\\mathbf{a}-\\text{proj}_W\\mathbf{a}$ 与 $W$ 的**每一个向量**正交',
            '若 $W$ 的基不正交（非 ONB），需用 $P_W=A(A^TA)^{-1}A^T$（$A$ 的列是任意基）'
          ],
          tip: '子空间投影矩阵满足 $P_W^2=P_W$（幂等）和 $P_W^T=P_W$（对称）——这是正交投影矩阵的特征性质。'
        },
        {
          title: '例题：投影到二维子空间',
          body: '设 $W=\\text{span}\\{q_1,q_2\\}$，$q_1=(1,0,0)^T$，$q_2=(0,1,0)^T$，$\\mathbf{a}=(2,3,5)^T$，求 $\\mathbf{a}$ 在 $W$ 上的投影。',
          steps: [
            '$\\langle \\mathbf{a},q_1\\rangle=2$，$\\langle \\mathbf{a},q_2\\rangle=3$',
            '$\\text{proj}_W\\mathbf{a}=2q_1+3q_2=(2,0,0)^T+(0,3,0)^T=(2,3,0)^T$',
            '残差 $\\mathbf{a}^\\perp=(2,3,5)^T-(2,3,0)^T=(0,0,5)^T$',
            '验证：$(0,0,5)^T\\perp q_1$ ✓，$(0,0,5)^T\\perp q_2$ ✓'
          ]
        },
        {
          title: '正交投影的最短距离性质',
          body: '**定理**：$\\text{proj}_W\\mathbf{a}$ 是 $W$ 中离 $\\mathbf{a}$ **最近**的向量。\n\n**直觉**：从空间中一点到子空间的最短连线必须垂直于该子空间——这就是"正交"投影的"正交"的来源。\n\n设 $\\mathbf{w}$ 是 $W$ 中任意一点，$\\hat{\\mathbf{a}}=\\text{proj}_W\\mathbf{a}$，则：\n\n$\\lVert\\mathbf{a}-\\mathbf{w}\\rVert^2=\\lVert\\mathbf{a}-\\hat{\\mathbf{a}}+\\hat{\\mathbf{a}}-\\mathbf{w}\\rVert^2=\\lVert\\mathbf{a}-\\hat{\\mathbf{a}}\\rVert^2+\\lVert\\hat{\\mathbf{a}}-\\mathbf{w}\\rVert^2\\geq\\lVert\\mathbf{a}-\\hat{\\mathbf{a}}\\rVert^2$',
          tip: '这个最短距离性质正是最小二乘法的几何基础——下一章将直接用到它。'
        },
        {
          title: '易错点与小结',
          body: '**易错点**：\n\n① 投影系数：$\\text{proj}_{\\mathbf{b}}\\mathbf{a}=\\dfrac{\\langle \\mathbf{a},\\mathbf{b}\\rangle}{\\lVert\\mathbf{b}\\rVert^2}\\mathbf{b}$，分母是 $\\lVert\\mathbf{b}\\rVert^2$ 而不是 $\\lVert\\mathbf{b}\\rVert$。\n\n② $QQ^T\\neq I$（非方形时）：$Q$ 是 $n\\times k$ 且 $k<n$ 时，$Q^TQ=I_k$ 但 $QQ^T$ 是秩 $k$ 的投影矩阵，不是单位阵。\n\n③ 子空间投影必须用正交基：用非正交基时公式要改为 $A(A^TA)^{-1}A^T$。',
          reveal: {
            q: '$\\mathbf{a}=(4,2)^T$，$\\mathbf{b}=(1,0)^T$，计算 $\\text{proj}_{\\mathbf{b}}\\mathbf{a}$。',
            a: '$c=\\langle \\mathbf{a},\\mathbf{b}\\rangle/\\lVert\\mathbf{b}\\rVert^2=4/1=4$，投影为 $4\\cdot(1,0)^T=(4,0)^T$。残差 $(0,2)^T$ 与 $\\mathbf{b}$ 正交 ✓。'
          }
        }
      ],
      questions: [
        {
          id: 'u6-l3-q1',
          type: 'choice',
          prompt: '向量 $\\mathbf{a}$ 在非零向量 $\\mathbf{b}$ 上的正交投影公式是？',
          options: [
            '$\\dfrac{\\langle \\mathbf{a},\\mathbf{b}\\rangle}{\\lVert\\mathbf{b}\\rVert}\\mathbf{b}$',
            '$\\dfrac{\\langle \\mathbf{a},\\mathbf{b}\\rangle}{\\lVert\\mathbf{b}\\rVert^2}\\mathbf{b}$',
            '$\\langle \\mathbf{a},\\mathbf{b}\\rangle$',
            '$\\dfrac{\\mathbf{a}}{\\lVert\\mathbf{a}\\rVert}$'
          ],
          answer: 1,
          explain: '投影系数 $c=\\langle \\mathbf{a},\\mathbf{b}\\rangle/\\lVert\\mathbf{b}\\rVert^2$，投影向量 $c\\mathbf{b}$。分母是 $\\lVert\\mathbf{b}\\rVert^2$（内积），不是 $\\lVert\\mathbf{b}\\rVert$（范数）。'
        },
        {
          id: 'u6-l3-q2',
          type: 'input',
          prompt: '$\\mathbf{a}=(3,1)^T$，$\\mathbf{b}=(1,1)^T$，投影系数 $c=\\langle \\mathbf{a},\\mathbf{b}\\rangle/\\lVert\\mathbf{b}\\rVert^2$ 等于多少（整数）？',
          accept: ['2'],
          placeholder: '输入整数',
          explain: '$\\langle \\mathbf{a},\\mathbf{b}\\rangle=3+1=4$，$\\lVert\\mathbf{b}\\rVert^2=1+1=2$，$c=4/2=2$。'
        },
        {
          id: 'u6-l3-q3',
          type: 'judge',
          prompt: '投影矩阵 $P=\\mathbf{b}\\mathbf{b}^T/(\\mathbf{b}^T\\mathbf{b})$ 满足 $P^2=P$（幂等性）。',
          answer: true,
          explain: '$P^2=\\frac{\\mathbf{b}\\mathbf{b}^T}{\\mathbf{b}^T\\mathbf{b}}\\cdot\\frac{\\mathbf{b}\\mathbf{b}^T}{\\mathbf{b}^T\\mathbf{b}}=\\frac{\\mathbf{b}(\\mathbf{b}^T\\mathbf{b})\\mathbf{b}^T}{(\\mathbf{b}^T\\mathbf{b})^2}=\\frac{\\mathbf{b}\\mathbf{b}^T}{\\mathbf{b}^T\\mathbf{b}}=P$ ✓。'
        },
        {
          id: 'u6-l3-q4',
          type: 'choice',
          prompt: '设 $W$ 有标准正交基 $Q=[q_1|\\cdots|q_k]$，投影到 $W$ 的矩阵是？',
          options: ['$Q^TQ$', '$QQ^T$', '$Q^{-1}$', '$Q^TQ^{-1}$'],
          answer: 1,
          explain: '投影矩阵 $P_W=QQ^T$（$n\\times n$），满足 $P_W^2=P_W$ 和 $P_W^T=P_W$。注意 $Q^TQ=I_k$ 是 $k\\times k$ 单位阵，不是投影矩阵。'
        },
        {
          id: 'u6-l3-q5',
          type: 'judge',
          prompt: '若 $Q$ 是 $3\\times2$ 列正交矩阵（即 $Q^TQ=I_2$），则 $QQ^T=I_3$。',
          answer: false,
          explain: '$QQ^T$ 是 $3\\times3$ 矩阵，秩为 $2$（列空间维数），不是单位阵。$Q^TQ=I_2$ 仅说明两列正交且单位化。'
        },
        {
          id: 'u6-l3-q6',
          type: 'input',
          prompt: '$q_1=(1,0,0)^T$，$q_2=(0,1,0)^T$，$\\mathbf{a}=(2,3,5)^T$，投影 $\\text{proj}_W\\mathbf{a}$ 的第三个分量是多少（整数）？',
          accept: ['0'],
          placeholder: '输入整数',
          explain: '$\\text{proj}_W\\mathbf{a}=\\langle \\mathbf{a},q_1\\rangle q_1+\\langle \\mathbf{a},q_2\\rangle q_2=2(1,0,0)^T+3(0,1,0)^T=(2,3,0)^T$，第三分量为 $0$。'
        },
        {
          id: 'u6-l3-q7',
          type: 'choice',
          prompt: '正交投影 $\\hat{\\mathbf{a}}=\\text{proj}_W\\mathbf{a}$ 的最短距离性质是指？',
          options: [
            '$\\hat{\\mathbf{a}}$ 的范数最小',
            '$\\hat{\\mathbf{a}}$ 是 $W$ 中离 $\\mathbf{a}$ 最近的向量',
            '$\\mathbf{a}-\\hat{\\mathbf{a}}$ 的范数最大',
            '$\\hat{\\mathbf{a}}$ 与 $\\mathbf{a}$ 方向相同'
          ],
          answer: 1,
          explain: '对 $W$ 中任意 $\\mathbf{w}$，$\\lVert\\mathbf{a}-\\mathbf{w}\\rVert\\geq\\lVert\\mathbf{a}-\\hat{\\mathbf{a}}\\rVert$，等号在 $\\mathbf{w}=\\hat{\\mathbf{a}}$ 时取到——投影点是最近点。'
        },
        {
          id: 'u6-l3-q8',
          type: 'match',
          prompt: '将投影相关的量与其描述配对。',
          left: [
            '$\\text{proj}_{\\mathbf{b}}\\mathbf{a}$',
            '$\\mathbf{a}-\\text{proj}_{\\mathbf{b}}\\mathbf{a}$',
            '$P_W=QQ^T$',
            '$P^2=P$'
          ],
          right: [
            '$\\mathbf{a}$ 在 $\\mathbf{b}$ 方向的"影子"',
            '与 $\\mathbf{b}$ 正交的残差分量',
            '投影到子空间 $W$ 的矩阵',
            '幂等性：投影两次等于投影一次'
          ],
          explain: '投影 = 影子；残差 = 垂直分量；$P_W=QQ^T$ 是子空间投影矩阵；幂等性是所有正交投影矩阵的共同性质。'
        }
      ]
    },

    // =====================================================================
    // 关 4：Gram-Schmidt 与 QR
    // =====================================================================
    {
      id: 'u6-l4',
      title: 'Gram-Schmidt 与 QR 分解',
      subtitle: '逐步消除旧方向的影子，建正交基',
      intro: [
        {
          title: '问题：如何把任意基变成正交基？',
          body: '给定 $\\mathbb{R}^2$ 中线性无关的 $a_1=(1,1)^T$，$a_2=(3,1)^T$，它们张成整个平面，但不垂直。\n\n**目标**：找到同一平面的另一组基 $\\{q_1,q_2\\}$，使两向量**互相垂直且单位长度**（标准正交基）。\n\nGram-Schmidt 算法的核心思想：**每次处理一个新向量时，先减去它在已有正交向量上的全部投影，剩余部分即为"纯新方向"，再归一化。**',
          tip: '正交基让坐标计算变成内积计算，而不需要解线性方程组——这是它的最大优势。'
        },
        {
          title: 'Gram-Schmidt 算法（通用步骤）',
          body: '给定线性无关组 $\\{a_1,a_2,\\ldots,a_k\\}$，第 $j$ 步：',
          formula: '\\mathbf{v}_j = a_j - \\sum_{i=1}^{j-1}(a_j\\cdot q_i)\\,q_i,\\quad q_j=\\frac{\\mathbf{v}_j}{\\lVert\\mathbf{v}_j\\rVert}',
          steps: [
            '**第 1 步**：$\\mathbf{v}_1=a_1$，归一化得 $q_1$',
            '**第 2 步**：从 $a_2$ 中减去其在 $q_1$ 上的投影，归一化得 $q_2$',
            '**第 3 步**：从 $a_3$ 中减去其在 $q_1$、$q_2$ 上的投影，归一化得 $q_3$',
            '若 $\\mathbf{v}_j=\\mathbf{0}$，说明 $a_j$ 已被前面向量线性表示，原始向量组**线性相关**'
          ],
          tip: '注意：$q_i$ 必须是归一化向量，才能用 $a_j\\cdot q_i$ 作为投影系数。'
        },
        {
          title: '例题：二维 Gram-Schmidt',
          body: '设 $a_1=(1,1)^T$，$a_2=(3,1)^T$，求标准正交基。',
          steps: [
            '**第 1 步**：$\\mathbf{v}_1=(1,1)^T$，$\\lVert\\mathbf{v}_1\\rVert=\\sqrt{2}$，$q_1=(1,1)^T/\\sqrt{2}$',
            '**第 2 步**：投影系数 $a_2\\cdot q_1=(3+1)/\\sqrt{2}=4/\\sqrt{2}=2\\sqrt{2}$',
            '$\\mathbf{v}_2=(3,1)^T-2\\sqrt{2}\\cdot(1,1)^T/\\sqrt{2}=(3,1)^T-(2,2)^T=(1,-1)^T$',
            '$\\lVert\\mathbf{v}_2\\rVert=\\sqrt{2}$，$q_2=(1,-1)^T/\\sqrt{2}$',
            '**验证**：$q_1\\cdot q_2=(1-1)/2=0$ ✓，$\\lVert q_1\\rVert=\\lVert q_2\\rVert=1$ ✓'
          ]
        },
        {
          title: '例题：三维 Gram-Schmidt',
          body: '设 $a_1=(1,0,0)^T$，$a_2=(1,1,0)^T$，$a_3=(1,1,1)^T$：',
          steps: [
            '$q_1=(1,0,0)^T$',
            '$\\mathbf{v}_2=a_2-(a_2\\cdot q_1)q_1=(1,1,0)^T-1\\cdot(1,0,0)^T=(0,1,0)^T$，$q_2=(0,1,0)^T$',
            '$\\mathbf{v}_3=a_3-(a_3\\cdot q_1)q_1-(a_3\\cdot q_2)q_2=(1,1,1)^T-(1,0,0)^T-(0,1,0)^T=(0,0,1)^T$，$q_3=(0,0,1)^T$',
            '结果恰好是标准基——因为原始向量恰好是"下三角"形式，Gram-Schmidt 还原出坐标轴。'
          ]
        },
        {
          title: 'QR 分解',
          body: 'Gram-Schmidt 过程可以写成矩阵形式。若 $A=[a_1|a_2|\\cdots|a_k]$（$m\\times k$ 列满秩），则',
          formula: 'A = QR',
          steps: [
            '$Q=[q_1|q_2|\\cdots|q_k]$（$m\\times k$），满足 $Q^TQ=I_k$',
            '$R$ 是 $k\\times k$ **上三角矩阵**，对角元 $r_{jj}=\\lVert\\mathbf{v}_j\\rVert>0$，超对角元 $r_{ij}=a_j\\cdot q_i$（$i<j$）',
            '快速计算：$R=Q^TA$（两边左乘 $Q^T$，利用 $Q^TQ=I$）',
            '**应用**：解方程组 $A\\mathbf{x}=\\mathbf{b}$ 等价于 $R\\mathbf{x}=Q^T\\mathbf{b}$（上三角回代）',
            '**最小二乘**：$\\min\\lVert A\\mathbf{x}-\\mathbf{b}\\rVert$ 同样归结为 $R\\hat{\\mathbf{x}}=Q^T\\mathbf{b}$，数值比法方程更稳定（条件数 $\\kappa(A)$ 而非 $\\kappa(A)^2$）'
          ]
        },
        {
          title: '数值稳定性',
          body: '经典 Gram-Schmidt（CGS）在有限精度下会累积误差，正交性误差 $O(\\epsilon\\kappa^2)$（$\\kappa$ 为条件数）。\n\n**修正 Gram-Schmidt**（MGS）：每步减去投影后立即更新向量，误差降至 $O(\\epsilon\\kappa)$。\n\n**Householder 反射**：误差仅 $O(\\epsilon)$，是 LAPACK/NumPy 的默认实现方法，实践首选。',
          tip: 'QR 分解求最小二乘的数值稳定性优于直接计算法方程 $(A^TA)^{-1}A^T\\mathbf{b}$——法方程的条件数是 $\\kappa(A)^2$，QR 的是 $\\kappa(A)$。'
        },
        {
          title: '易错点与小结',
          body: '**易错点**：\n\n① **先归一化再投影**：Gram-Schmidt 每步用已归一化的 $q_i$ 算投影系数 $a_j\\cdot q_i$。若用未归一化向量，系数公式要改成 $\\langle a_j,\\mathbf{v}_i\\rangle/\\lVert\\mathbf{v}_i\\rVert^2$，容易出错。\n\n② **$Q^TQ=I_k$ 不等于 $QQ^T=I_m$**：$Q$ 是 $m\\times k$（$m>k$）时，$Q^TQ=I_k$ 成立，但 $QQ^T$ 是 $m\\times m$ 的秩 $k$ 投影矩阵。',
          reveal: {
            q: '$a_1=(1,1,0)^T$，$a_2=(2,0,1)^T$，做 Gram-Schmidt 第 2 步，求 $q_2$。',
            a: '$q_1=(1,1,0)^T/\\sqrt{2}$；$a_2\\cdot q_1=2/\\sqrt{2}=\\sqrt{2}$；$\\mathbf{v}_2=(2,0,1)^T-\\sqrt{2}\\cdot(1,1,0)^T/\\sqrt{2}=(2,0,1)^T-(1,1,0)^T=(1,-1,1)^T$；$q_2=(1,-1,1)^T/\\sqrt{3}$。'
          }
        }
      ],
      questions: [
        {
          id: 'u6-l4-q1',
          type: 'choice',
          prompt: 'Gram-Schmidt 第 2 步中，$\\mathbf{v}_2=a_2-(a_2\\cdot q_1)q_1$ 的几何含义是？',
          options: [
            '将 $a_2$ 放大为单位向量',
            '从 $a_2$ 中减去其在 $q_1$ 方向的"影子"，剩下与 $q_1$ 垂直的分量',
            '求 $a_2$ 与 $q_1$ 的外积',
            '将 $a_2$ 旋转 $90°$'
          ],
          answer: 1,
          explain: '减去投影 = 去除"已知方向的成分"，残差 $\\mathbf{v}_2$ 与 $q_1$ 正交，代表 $a_2$ 对正交基的"纯贡献"。'
        },
        {
          id: 'u6-l4-q2',
          type: 'judge',
          prompt: 'Gram-Schmidt 某步得到 $\\mathbf{v}_j=\\mathbf{0}$，这意味着原始向量组线性相关。',
          answer: true,
          explain: '$\\mathbf{v}_j=\\mathbf{0}$ 说明 $a_j$ 已完全被 $q_1,\\ldots,q_{j-1}$ 的线性组合表示，即 $a_j$ 在前面向量的张成空间内。'
        },
        {
          id: 'u6-l4-q3',
          type: 'input',
          prompt: '$a_1=(0,1)^T$，$a_2=(1,1)^T$，第 1 步得 $q_1=(0,1)^T$，第 2 步投影系数 $a_2\\cdot q_1$ 等于多少（整数）？',
          accept: ['1'],
          placeholder: '输入整数',
          explain: '$a_2\\cdot q_1=0\\times0+1\\times1=1$。接着 $\\mathbf{v}_2=(1,1)^T-1\\cdot(0,1)^T=(1,0)^T$，$q_2=(1,0)^T$。'
        },
        {
          id: 'u6-l4-q4',
          type: 'choice',
          prompt: 'QR 分解 $A=QR$ 中，$R$ 矩阵是？',
          options: [
            '下三角矩阵',
            '对角元为正的上三角矩阵',
            '正交矩阵',
            '对称矩阵'
          ],
          answer: 1,
          explain: '$R$ 是上三角矩阵，对角元 $r_{jj}=\\lVert\\mathbf{v}_j\\rVert>0$（原向量线性无关时 $\\mathbf{v}_j\\neq\\mathbf{0}$）。'
        },
        {
          id: 'u6-l4-q5',
          type: 'input',
          prompt: '$a_1=(3,4)^T$，Gram-Schmidt 第 1 步得 $q_1=(3/5,4/5)^T$。对 $a_2=(0,5)^T$，投影系数 $a_2\\cdot q_1$ 等于多少（整数）？',
          accept: ['4'],
          placeholder: '输入整数',
          explain: '$a_2\\cdot q_1=0\\times(3/5)+5\\times(4/5)=4$。'
        },
        {
          id: 'u6-l4-q6',
          type: 'choice',
          prompt: '用 QR 分解解方程组 $A\\mathbf{x}=\\mathbf{b}$（$A$ 可逆方阵）时，等价于解？',
          options: [
            '$Q\\mathbf{x}=\\mathbf{b}$',
            '$R\\mathbf{x}=Q^T\\mathbf{b}$',
            '$Q^TR\\mathbf{x}=\\mathbf{b}$',
            '$R^T\\mathbf{x}=Q\\mathbf{b}$'
          ],
          answer: 1,
          explain: '$A\\mathbf{x}=\\mathbf{b}\\Rightarrow QR\\mathbf{x}=\\mathbf{b}$，左乘 $Q^T$（$Q^{-1}=Q^T$）得 $R\\mathbf{x}=Q^T\\mathbf{b}$，这是上三角方程组，回代即解。'
        },
        {
          id: 'u6-l4-q7',
          type: 'judge',
          prompt: '与法方程 $(A^TA)^{-1}A^T\\mathbf{b}$ 相比，用 QR 分解求最小二乘解数值更稳定，因为条件数从 $\\kappa(A)^2$ 降为 $\\kappa(A)$。',
          answer: true,
          explain: '法方程把条件数平方（$\\kappa(A^TA)=\\kappa(A)^2$），对病态矩阵会严重放大误差；QR 回代只有 $\\kappa(A)$，数值稳定性更好。'
        },
        {
          id: 'u6-l4-q8',
          type: 'match',
          prompt: '将 Gram-Schmidt 的步骤与其描述配对。',
          left: [
            '计算 $a_j\\cdot q_i$',
            '$\\mathbf{v}_j=a_j-\\sum_{i<j}(a_j\\cdot q_i)q_i$',
            '$q_j=\\mathbf{v}_j/\\lVert\\mathbf{v}_j\\rVert$',
            '$R=Q^TA$'
          ],
          right: [
            '$a_j$ 在 $q_i$ 方向的投影长度',
            '去掉已有方向的影子',
            '归一化得单位正交向量',
            'QR 分解中 R 的快速计算'
          ],
          explain: '每步物理意义：量影子长度 → 减影子 → 归一化 → 矩阵编码全过程。'
        }
      ]
    },

    // =====================================================================
    // 关 5：正规方程
    // =====================================================================
    {
      id: 'u6-l5',
      title: '正规方程',
      subtitle: '超定方程的最优近似——残差平方和最小',
      intro: [
        {
          title: '为什么会有无解的方程组？',
          body: '三个数据点 $(1,1),(2,3),(3,4)$，要用直线 $y=ax+b$ 精确穿过三点，需同时满足 $a+b=1$，$2a+b=3$，$3a+b=4$。\n\n三个方程，两个未知数——**超定方程组**（$m>n$）。验算：若 $a=3/2,b=-1/3$ 满足前两个，则第三个 $3\\times3/2-1/3=25/6\\neq4$，矛盾。\n\n**真实数据总有测量噪声**，精确解几乎不存在。我们转而问：哪个 $\\hat{\\mathbf{x}}$ 使残差 $\\mathbf{b}-A\\hat{\\mathbf{x}}$ "尽可能小"？',
          tip: '最小二乘的名字来自"残差平方和最小"（least squares）。'
        },
        {
          title: '几何本质：投影到列空间',
          body: '$A\\mathbf{x}$ 当 $\\mathbf{x}$ 遍历 $\\mathbb{R}^n$ 时，恰好遍历 $\\text{Col}(A)$（$\\mathbb{R}^m$ 中的 $n$ 维子空间）。\n\n当 $\\mathbf{b}\\notin\\text{Col}(A)$ 时，不存在 $\\mathbf{x}$ 使 $A\\mathbf{x}=\\mathbf{b}$ 精确成立。但可以找 $\\text{Col}(A)$ 中**离 $\\mathbf{b}$ 最近的点**——即 $\\mathbf{b}$ 在 $\\text{Col}(A)$ 上的**正交投影** $\\hat{\\mathbf{b}}$，对应的 $\\hat{\\mathbf{x}}$ 即最小二乘解。',
          formula: '\\hat{\\mathbf{x}} = \\arg\\min_{\\mathbf{x}}\\lVert\\mathbf{b}-A\\mathbf{x}\\rVert^2'
        },
        {
          title: '正规方程的推导',
          body: '最小化 $f(\\mathbf{x})=\\lVert\\mathbf{b}-A\\mathbf{x}\\rVert^2$ 的充要条件：**残差** $\\mathbf{r}=\\mathbf{b}-A\\hat{\\mathbf{x}}$ 与 $\\text{Col}(A)$ 正交（即 $A^T\\mathbf{r}=\\mathbf{0}$），展开即得**正规方程**：',
          formula: 'A^TA\\hat{\\mathbf{x}} = A^T\\mathbf{b}',
          steps: [
            '当 $A$ 列满秩时，$A^TA$ 是 $n\\times n$ **正定可逆**矩阵，唯一解：$\\hat{\\mathbf{x}}=(A^TA)^{-1}A^T\\mathbf{b}$',
            '**证明 $A^TA$ 可逆**：若 $A^TA\\mathbf{v}=\\mathbf{0}$，则 $\\lVert A\\mathbf{v}\\rVert^2=\\mathbf{v}^TA^TA\\mathbf{v}=0$，故 $A\\mathbf{v}=\\mathbf{0}$，列满秩时 $\\mathbf{v}=\\mathbf{0}$',
            '**验证用**：求解后算 $A^T(\\mathbf{b}-A\\hat{\\mathbf{x}})$，若不为零则计算有误'
          ]
        },
        {
          title: '具体数值推导：三点拟合直线',
          body: '数据点 $(1,1),(2,3),(3,4)$，设计矩阵与正规方程求解：',
          steps: [
            '$A=\\begin{pmatrix}1&1\\\\2&1\\\\3&1\\end{pmatrix}$，$\\mathbf{b}=(1,3,4)^T$',
            '$A^TA=\\begin{pmatrix}14&6\\\\6&3\\end{pmatrix}$，$A^T\\mathbf{b}=\\begin{pmatrix}19\\\\8\\end{pmatrix}$',
            '$\\det(A^TA)=42-36=6$，$(A^TA)^{-1}=\\dfrac{1}{6}\\begin{pmatrix}3&-6\\\\-6&14\\end{pmatrix}$',
            '$\\hat{\\mathbf{x}}=\\dfrac{1}{6}\\begin{pmatrix}3&-6\\\\-6&14\\end{pmatrix}\\begin{pmatrix}19\\\\8\\end{pmatrix}=\\dfrac{1}{6}\\begin{pmatrix}9\\\\-2\\end{pmatrix}=\\begin{pmatrix}3/2\\\\-1/3\\end{pmatrix}$',
            '拟合直线：$y=\\dfrac{3}{2}x-\\dfrac{1}{3}$'
          ]
        },
        {
          title: '验证残差正交性',
          body: '接上例，验证 $A^T\\mathbf{r}=\\mathbf{0}$：',
          steps: [
            '$A\\hat{\\mathbf{x}}=(7/6,8/3,25/6)^T$',
            '$\\mathbf{r}=\\mathbf{b}-A\\hat{\\mathbf{x}}=(-1/6,1/3,-1/6)^T$',
            '$A^T\\mathbf{r}$ 第一分量 $=1\\times(-1/6)+2\\times(1/3)+3\\times(-1/6)=-1/6+2/3-1/2=0$ ✓',
            '$A^T\\mathbf{r}$ 第二分量 $=1\\times(-1/6)+1\\times(1/3)+1\\times(-1/6)=0$ ✓'
          ],
          tip: '残差正交性 $A^T\\mathbf{r}=\\mathbf{0}$ 是最小二乘解的充要条件——求解后必须验证这一点。'
        },
        {
          title: '简单例题：一维列空间',
          body: '**例**：$A=(1,1,1)^T$（$3\\times1$），$\\mathbf{b}=(1,2,3)^T$，求最小二乘解。',
          steps: [
            '$A^TA=1+1+1=3$（标量），$(A^TA)^{-1}=1/3$',
            '$A^T\\mathbf{b}=1+2+3=6$，$\\hat{x}=6/3=2$',
            '$\\hat{\\mathbf{b}}=2\\cdot(1,1,1)^T=(2,2,2)^T$（三个 $y$ 值的均值）',
            '残差 $\\mathbf{r}=(-1,0,1)^T$，验证：$A^T\\mathbf{r}=-1+0+1=0$ ✓'
          ]
        },
        {
          title: '易错点与小结',
          body: '**易错点**：\n\n① **必须两边乘 $A^T$**：$A\\mathbf{x}=\\mathbf{b}$ 两边左乘 $A^T$，才能得到方阵 $A^TA$；直接写 $A^{-1}\\mathbf{b}$ 错（$A$ 非方阵）。\n\n② **条件数翻倍**：法方程 $A^TA$ 的条件数是 $\\kappa(A)^2$，数值不稳定；实践中用 QR 分解 $R\\hat{\\mathbf{x}}=Q^T\\mathbf{b}$，条件数仅 $\\kappa(A)$。\n\n③ **不要忘记验证残差**：求出 $\\hat{\\mathbf{x}}$ 后，算 $A^T(\\mathbf{b}-A\\hat{\\mathbf{x}})$，若不为零则出错。',
          reveal: {
            q: '数据 $(0,1),(1,2),(2,2)$，拟合 $y=ax+b$，写出 $A$ 和 $A^TA$。',
            a: '$A=\\begin{pmatrix}0&1\\\\1&1\\\\2&1\\end{pmatrix}$，$A^TA=\\begin{pmatrix}0+1+4&0+1+2\\\\0+1+2&1+1+1\\end{pmatrix}=\\begin{pmatrix}5&3\\\\3&3\\end{pmatrix}$。'
          }
        }
      ],
      questions: [
        {
          id: 'u6-l5-q1',
          type: 'choice',
          prompt: '最小二乘问题 $\\min_{\\mathbf{x}}\\lVert\\mathbf{b}-A\\mathbf{x}\\rVert^2$ 的几何含义是？',
          options: [
            '在 $\\mathbf{b}$ 的方向上找最大模向量',
            '在列空间 $\\text{Col}(A)$ 中找离 $\\mathbf{b}$ 最近的向量',
            '最小化 $A$ 的行列式',
            '找范数最小的 $\\mathbf{x}$'
          ],
          answer: 1,
          explain: '$A\\mathbf{x}$ 遍历 $\\text{Col}(A)$，最小化 $\\lVert\\mathbf{b}-A\\mathbf{x}\\rVert$ 等价于在列空间中找离 $\\mathbf{b}$ 最近的点，即 $\\mathbf{b}$ 在列空间上的正交投影。'
        },
        {
          id: 'u6-l5-q2',
          type: 'choice',
          prompt: '正规方程是？',
          options: [
            '$A\\hat{\\mathbf{x}}=\\mathbf{b}$',
            '$A^T\\hat{\\mathbf{x}}=\\mathbf{b}$',
            '$A^TA\\hat{\\mathbf{x}}=A^T\\mathbf{b}$',
            '$AA^T\\hat{\\mathbf{x}}=\\mathbf{b}$'
          ],
          answer: 2,
          explain: '由残差正交于列空间（$A^T\\mathbf{r}=\\mathbf{0}$）展开：$A^T(\\mathbf{b}-A\\hat{\\mathbf{x}})=\\mathbf{0}$，即 $A^TA\\hat{\\mathbf{x}}=A^T\\mathbf{b}$。'
        },
        {
          id: 'u6-l5-q3',
          type: 'judge',
          prompt: '最小二乘解的残差 $\\mathbf{r}=\\mathbf{b}-A\\hat{\\mathbf{x}}$ 与 $A$ 的列空间正交，即 $A^T\\mathbf{r}=\\mathbf{0}$。',
          answer: true,
          explain: '这是最小二乘最优性的充要条件：$\\hat{\\mathbf{x}}$ 使 $\\lVert\\mathbf{b}-A\\mathbf{x}\\rVert^2$ 取最小值，当且仅当残差与列空间正交。'
        },
        {
          id: 'u6-l5-q4',
          type: 'input',
          prompt: '$A=\\begin{pmatrix}1\\\\2\\\\2\\end{pmatrix}$，$\\mathbf{b}=(3,6,6)^T$，计算 $A^TA$（标量）。',
          accept: ['9'],
          placeholder: '输入整数',
          explain: '$A^TA=1^2+2^2+2^2=1+4+4=9$。'
        },
        {
          id: 'u6-l5-q5',
          type: 'input',
          prompt: '接上题：$A^T\\mathbf{b}$ 等于多少（标量）？',
          accept: ['27'],
          placeholder: '输入整数',
          explain: '$A^T\\mathbf{b}=1\\times3+2\\times6+2\\times6=3+12+12=27$。'
        },
        {
          id: 'u6-l5-q6',
          type: 'choice',
          prompt: '接上两题：最小二乘解 $\\hat{x}=(A^TA)^{-1}A^T\\mathbf{b}$ 等于？',
          options: ['$2$', '$3$', '$6$', '$9$'],
          answer: 1,
          explain: '$\\hat{x}=27/9=3$。投影 $\\hat{\\mathbf{b}}=3A=(3,6,6)^T=\\mathbf{b}$，说明 $\\mathbf{b}$ 恰好在列空间中（精确解存在，残差为零）。'
        },
        {
          id: 'u6-l5-q7',
          type: 'judge',
          prompt: '当 $A$ 列满秩时，$A^TA$ 是可逆的正定矩阵。',
          answer: true,
          explain: '若 $A^TA\\mathbf{v}=\\mathbf{0}$，则 $\\lVert A\\mathbf{v}\\rVert^2=0$，$A\\mathbf{v}=\\mathbf{0}$，列满秩时 $\\mathbf{v}=\\mathbf{0}$，故可逆。正定性：$\\mathbf{v}^TA^TA\\mathbf{v}=\\lVert A\\mathbf{v}\\rVert^2\\geq0$，仅当 $\\mathbf{v}=\\mathbf{0}$ 时等号成立。'
        },
        {
          id: 'u6-l5-q8',
          type: 'choice',
          prompt: '三点 $(1,1),(2,3),(3,4)$ 拟合直线 $y=ax+b$，正规方程的系数矩阵 $A^TA$ 是？',
          options: [
            '$\\begin{pmatrix}14&6\\\\6&3\\end{pmatrix}$',
            '$\\begin{pmatrix}14&3\\\\6&3\\end{pmatrix}$',
            '$\\begin{pmatrix}1&2&3\\\\1&1&1\\end{pmatrix}$',
            '$\\begin{pmatrix}6&14\\\\3&6\\end{pmatrix}$'
          ],
          answer: 0,
          explain: '$A=\\begin{pmatrix}1&1\\\\2&1\\\\3&1\\end{pmatrix}$，$A^TA=\\begin{pmatrix}1^2+2^2+3^2&1+2+3\\\\1+2+3&1+1+1\\end{pmatrix}=\\begin{pmatrix}14&6\\\\6&3\\end{pmatrix}$。'
        },
        {
          id: 'u6-l5-q9',
          type: 'match',
          prompt: '将最小二乘的概念与其公式配对。',
          left: [
            '残差向量',
            '正规方程',
            '列满秩时的闭式解',
            '最优性充要条件'
          ],
          right: [
            '$\\mathbf{r}=\\mathbf{b}-A\\hat{\\mathbf{x}}$',
            '$A^TA\\hat{\\mathbf{x}}=A^T\\mathbf{b}$',
            '$\\hat{\\mathbf{x}}=(A^TA)^{-1}A^T\\mathbf{b}$',
            '$A^T\\mathbf{r}=\\mathbf{0}$'
          ],
          explain: '残差衡量误差；正规方程是方程形式；闭式解要求列满秩；$A^T\\mathbf{r}=\\mathbf{0}$ 等价于正规方程成立。'
        }
      ]
    },

    // =====================================================================
    // 关 6：投影矩阵与应用
    // =====================================================================
    {
      id: 'u6-l6',
      title: '投影矩阵与应用',
      subtitle: '$P=A(A^TA)^{-1}A^T$：把任意向量投影到列空间',
      intro: [
        {
          title: '投影矩阵的定义',
          body: '当 $A\\in\\mathbb{R}^{m\\times n}$ 列满秩时，$\\mathbf{b}$ 在 $\\text{Col}(A)$ 上的正交投影为：',
          formula: '\\hat{\\mathbf{b}} = A(A^TA)^{-1}A^T\\mathbf{b} = P\\mathbf{b}',
          steps: [
            '其中 $P=A(A^TA)^{-1}A^T\\in\\mathbb{R}^{m\\times m}$ 是**投影矩阵**',
            '注意：$P$ 是 $m\\times m$ 大矩阵，把 $\\mathbb{R}^m$ 中的任意向量映射到 $\\text{Col}(A)$ 中',
            '最小二乘解满足 $A\\hat{\\mathbf{x}}=P\\mathbf{b}$，即 $\\hat{\\mathbf{b}}=P\\mathbf{b}$'
          ]
        },
        {
          title: '投影矩阵的两个核心性质',
          body: '**性质 1：对称性** $P^T=P$\n\n**性质 2：幂等性** $P^2=P$（投影两次等于投影一次）\n\n**证明幂等性**：',
          steps: [
            '$P^2=A(A^TA)^{-1}A^T\\cdot A(A^TA)^{-1}A^T$',
            '$=A(A^TA)^{-1}(A^TA)(A^TA)^{-1}A^T$',
            '$=A(A^TA)^{-1}A^T=P$ ✓'
          ],
          tip: '幂等性和对称性是正交投影矩阵的充要条件——凡满足 $P^2=P$，$P^T=P$ 的矩阵都是某子空间的正交投影矩阵。'
        },
        {
          title: '四个子空间的视角',
          body: '从子空间角度看最小二乘：\n\n$\\mathbb{R}^m = \\text{Col}(A) \\oplus \\text{Null}(A^T)$\n\n$\\mathbf{b} = \\underbrace{\\hat{\\mathbf{b}}}_{\\text{在}\\,\\text{Col}(A)\\,\\text{中}} + \\underbrace{\\mathbf{r}}_{\\text{在}\\,\\text{Null}(A^T)\\,\\text{中}}$\n\n最小二乘法把 $\\mathbf{b}$ 分解为两个正交分量：列空间中的部分 $\\hat{\\mathbf{b}}$（可被 $A$ 的列表示），以及左零空间中的部分 $\\mathbf{r}$（不可消除的误差）。',
          tip: '残差 $\\mathbf{r}$ 必然落在 $\\text{Null}(A^T)$ 中，这就是为什么 $A^T\\mathbf{r}=\\mathbf{0}$——左零空间的向量与所有列正交。'
        },
        {
          title: '例题：投影矩阵的计算',
          body: '**例 1**：$A=(1,1,1)^T$（$3\\times1$），计算 $P$。',
          steps: [
            '$A^TA=3$（标量），$(A^TA)^{-1}=1/3$',
            '$P=A(A^TA)^{-1}A^T=\\dfrac{1}{3}\\begin{pmatrix}1\\\\1\\\\1\\end{pmatrix}\\begin{pmatrix}1&1&1\\end{pmatrix}=\\dfrac{1}{3}\\begin{pmatrix}1&1&1\\\\1&1&1\\\\1&1&1\\end{pmatrix}$',
            '验证：$P\\mathbf{b}=(1,2,3)^T$ 时，$P\\mathbf{b}=\\dfrac{1}{3}(6,6,6)^T=(2,2,2)^T$（均值向量）',
            '验证幂等性：$P^2=\\dfrac{1}{9}\\begin{pmatrix}3&3&3\\\\3&3&3\\\\3&3&3\\end{pmatrix}=\\dfrac{1}{3}\\begin{pmatrix}1&1&1\\\\1&1&1\\\\1&1&1\\end{pmatrix}=P$ ✓'
          ]
        },
        {
          title: '线性回归中的投影矩阵',
          body: '线性回归 $\\mathbf{y}=X\\boldsymbol{\\theta}+\\boldsymbol{\\epsilon}$ 的**帽矩阵**（Hat Matrix）$H=X(X^TX)^{-1}X^T$：\n\n$\\hat{\\mathbf{y}}=H\\mathbf{y}$（拟合值 = $\\mathbf{y}$ 在列空间的投影）\n\n$(I-H)\\mathbf{y}=\\mathbf{r}$（残差 = 补投影空间的部分）\n\n**性质**：\n\n① $H$ 和 $I-H$ 都是投影矩阵（幂等且对称）\n\n② $H(I-H)=\\mathbf{0}$（帽矩阵与补矩阵正交）\n\n③ $\\text{tr}(H)=\\text{rank}(X)=n$（参数个数）',
          tip: '在统计学中，帽矩阵 $H$ 把观测值 $\\mathbf{y}$ "变成"拟合值 $\\hat{\\mathbf{y}}$，像给向量戴了一顶"帽子"，故名。'
        },
        {
          title: '正则化与伪逆',
          body: '当 $A^TA$ 近似奇异（条件数极大）时，直接求逆数值不稳定。**岭回归**（Ridge Regression）添加正则化：',
          formula: '\\hat{\\mathbf{x}}_{\\text{ridge}} = (A^TA+\\lambda I)^{-1}A^T\\mathbf{b}',
          steps: [
            '$\\lambda>0$ 保证 $A^TA+\\lambda I$ 恒为正定可逆矩阵',
            '**Moore-Penrose 伪逆** $A^+=V\\Sigma^+U^T$（由 SVD 定义），统一表达最小范数最小二乘解 $\\hat{\\mathbf{x}}=A^+\\mathbf{b}$',
            '列满秩时 $A^+=(A^TA)^{-1}A^T$，即法方程的解'
          ]
        },
        {
          title: '易错点与单元总结',
          body: '**易错点**：\n\n① $P=A(A^TA)^{-1}A^T$ 是 $m\\times m$ 矩阵（不是 $n\\times n$），把 $\\mathbb{R}^m$ 投影到 $\\text{Col}(A)$。\n\n② 幂等性 $P^2=P$ 等价于"已在列空间的向量投影后不变"——$P(P\\mathbf{b})=P\\mathbf{b}$。\n\n**单元总结**：内积 $\\to$ 范数/夹角/正交 $\\to$ 正交投影 $\\to$ Gram-Schmidt/QR $\\to$ 最小二乘/投影矩阵，形成一条完整的理论链。',
          reveal: {
            q: '若 $P$ 是投影矩阵（$P^2=P$，$P^T=P$），$I-P$ 是否也是投影矩阵？它投影到哪个子空间？',
            a: '是的。$(I-P)^2=I-2P+P^2=I-P$ ✓；$(I-P)^T=I-P$ ✓。$I-P$ 投影到 $P$ 的零空间（即 $\\text{Col}(A)^\\perp=\\text{Null}(A^T)$），即残差空间。'
          }
        }
      ],
      questions: [
        {
          id: 'u6-l6-q1',
          type: 'choice',
          prompt: '投影矩阵 $P=A(A^TA)^{-1}A^T$ 把 $\\mathbb{R}^m$ 中任意向量映射到？',
          options: [
            '$\\text{Null}(A)$',
            '$\\text{Row}(A)$',
            '$\\text{Col}(A)$',
            '$\\text{Null}(A^T)$'
          ],
          answer: 2,
          explain: '$P\\mathbf{b}=A(A^TA)^{-1}A^T\\mathbf{b}=A\\hat{\\mathbf{x}}\\in\\text{Col}(A)$，投影矩阵把任意向量投影到 $A$ 的列空间。'
        },
        {
          id: 'u6-l6-q2',
          type: 'judge',
          prompt: '投影矩阵满足 $P^2=P$（幂等性）和 $P^T=P$（对称性）。',
          answer: true,
          explain: '幂等性：$P^2=A(A^TA)^{-1}(A^TA)(A^TA)^{-1}A^T=P$ ✓；对称性：$(A(A^TA)^{-1}A^T)^T=A((A^TA)^{-1})^TA^T=A(A^TA)^{-1}A^T=P$ ✓。'
        },
        {
          id: 'u6-l6-q3',
          type: 'input',
          prompt: '$A=(1,1,1)^T$，$\\mathbf{b}=(1,2,3)^T$，投影 $P\\mathbf{b}$ 的第一个分量是多少（整数）？',
          accept: ['2'],
          placeholder: '输入整数',
          explain: '$P=\\dfrac{1}{3}\\mathbf{1}\\mathbf{1}^T$，$P\\mathbf{b}=\\dfrac{1}{3}(1+2+3)(1,1,1)^T=(2,2,2)^T$，第一分量为 $2$。'
        },
        {
          id: 'u6-l6-q4',
          type: 'choice',
          prompt: '若 $P$ 是正交投影矩阵，则 $I-P$ 是？',
          options: [
            '不是投影矩阵',
            '投影到 $\\text{Col}(P)^\\perp$ 的正交投影矩阵',
            '等于 $P^{-1}$',
            '等于 $-P$'
          ],
          answer: 1,
          explain: '$(I-P)^2=I-2P+P^2=I-P$ ✓（幂等）；$(I-P)^T=I-P$ ✓（对称）。$I-P$ 是正交投影矩阵，投影到 $\\text{Col}(P)$ 的正交补（即残差空间）。'
        },
        {
          id: 'u6-l6-q5',
          type: 'judge',
          prompt: '残差向量 $\\mathbf{r}=\\mathbf{b}-P\\mathbf{b}$ 满足 $P\\mathbf{r}=\\mathbf{0}$（残差被 $P$ 湮灭）。',
          answer: true,
          explain: '$P\\mathbf{r}=P(\\mathbf{b}-P\\mathbf{b})=P\\mathbf{b}-P^2\\mathbf{b}=P\\mathbf{b}-P\\mathbf{b}=\\mathbf{0}$，因为 $P^2=P$。几何上：残差在列空间的正交补中，投影到列空间结果为零。'
        },
        {
          id: 'u6-l6-q6',
          type: 'choice',
          prompt: '岭回归 $\\hat{\\mathbf{x}}=(A^TA+\\lambda I)^{-1}A^T\\mathbf{b}$（$\\lambda>0$）的主要作用是？',
          options: [
            '使 $A^TA$ 的条件数降低，提高数值稳定性',
            '保证解的范数最大',
            '让残差为零',
            '使解与普通最小二乘解完全相同'
          ],
          answer: 0,
          explain: '$\\lambda I$ 把所有奇异值从 $\\sigma_j$ 改为 $\\sigma_j+\\lambda$，对小奇异值起到正则化作用，$A^TA+\\lambda I$ 恒为正定可逆矩阵，条件数降低，数值稳定。'
        },
        {
          id: 'u6-l6-q7',
          type: 'match',
          prompt: '将投影矩阵相关的量与其描述配对。',
          left: [
            '$P=A(A^TA)^{-1}A^T$',
            '$I-P$',
            '$P^2=P$',
            '$\\text{tr}(P)$'
          ],
          right: [
            '投影到 $\\text{Col}(A)$ 的矩阵',
            '投影到残差空间（左零空间）的矩阵',
            '幂等性：二次投影等于一次投影',
            '等于 $\\text{rank}(A)$（列空间维数）'
          ],
          explain: '$P$ 投影到列空间；$I-P$ 投影到补空间；幂等性是正交投影的标志；迹等于投影子空间的维数。'
        },
        {
          id: 'u6-l6-q8',
          type: 'choice',
          prompt: '用 QR 分解（$A=QR$）表示投影矩阵 $P$，等于？',
          options: [
            '$RR^T$',
            '$QQ^T$',
            '$R^TQ^T$',
            '$QR(R^TQ^T)$'
          ],
          answer: 1,
          explain: '$P=A(A^TA)^{-1}A^T=(QR)(R^TQ^TQR)^{-1}(QR)^T=QR(R^TR)^{-1}R^TQ^T=Q(RR^{-1})(R^{-T}R^T)Q^T=QQ^T$。正交基直接给出投影矩阵。'
        },
        {
          id: 'u6-l6-q9',
          type: 'judge',
          prompt: '若 $\\mathbf{b}$ 本来就在 $\\text{Col}(A)$ 中，则最小二乘解 $\\hat{\\mathbf{x}}$ 满足 $A\\hat{\\mathbf{x}}=\\mathbf{b}$（精确成立，残差为零）。',
          answer: true,
          explain: '若 $\\mathbf{b}\\in\\text{Col}(A)$，则 $P\\mathbf{b}=\\mathbf{b}$（已在列空间中的向量投影后不变），故 $\\hat{\\mathbf{b}}=\\mathbf{b}$，残差 $\\mathbf{r}=\\mathbf{0}$，方程精确成立。'
        }
      ]
    }
  ]
}
