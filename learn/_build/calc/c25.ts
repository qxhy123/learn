// 第22章  向量分析
// 关 c25-a：梯度、散度与旋度（8 卡 8 题）
// 关 c25-b：Green 定理与 Gauss 定理（7 卡 7 题）
// 关 c25-c：Stokes 定理与三大定理统一（7 卡 7 题）

export const LESSONS = [
  // ─────────────────────────────────────────
  // 关 c25-a  梯度、散度与旋度：场的三把解剖刀
  // ─────────────────────────────────────────
  {
    id: 'c25-a',
    title: '梯度、散度与旋度',
    subtitle: '动机·Nabla算子·三算子定义·物理意义·恒等式·易错点',
    intro: [
      // 卡 1：动机——为什么要分析"场"？
      {
        title: '为什么要分析"场"？',
        body:
          '一元微积分处理的是单变量函数 $f(x)$，导数描述函数沿数轴的变化率。\n\n' +
          '但物理世界是三维的：温度分布、风速、电场……每一点都有一个值（或向量），这就是**场**。\n\n' +
          '**标量场**：每点对应一个数，如温度 $T(x,y,z)$、电势 $\\varphi(x,y,z)$。\n\n' +
          '**向量场**：每点对应一个向量，如速度场 $\\mathbf{v}(x,y,z)$、电场 $\\mathbf{E}(x,y,z)$。\n\n' +
          '分析场，需要三把新的微分"解剖刀"：**梯度**（方向变化率）、**散度**（源的强度）、**旋度**（涡旋强度）。它们统统由一个叫 $\\nabla$ 的算子给出。',
        tip: '场论的核心问题：在某点处，这个物理量怎么变化？往哪个方向变化最快？有没有源？有没有旋转？',
      },
      // 卡 2：Nabla 算子
      {
        title: 'Nabla 算子 $\\nabla$：统一的微分工具',
        body:
          '引入**哈密顿算子**（Nabla 算子）$\\nabla$，它是一个向量型微分算子：\n\n' +
          '把 $\\nabla$ 想象成"向量版的求导机器"——它有三个分量，分别对 $x,y,z$ 偏导。\n\n' +
          '用它作用于不同类型的场，就得到三种算子：\n\n' +
          '作用于**标量场**（点乘风格）：$\\nabla f$ → 梯度，结果是向量场；\n\n' +
          '与**向量场点积**：$\\nabla\\cdot\\mathbf{F}$ → 散度，结果是标量场；\n\n' +
          '与**向量场叉积**：$\\nabla\\times\\mathbf{F}$ → 旋度，结果是向量场。',
        formula:
          '\\nabla = \\frac{\\partial}{\\partial x}\\mathbf{i}+\\frac{\\partial}{\\partial y}\\mathbf{j}+\\frac{\\partial}{\\partial z}\\mathbf{k}',
        tip: '只需记住 $\\nabla$ 是"偏导向量"，三种算子的区别仅在于怎么和场"相乘"：无符号是梯度，点积是散度，叉积是旋度。',
      },
      // 卡 3：梯度——最速上升方向
      {
        title: '梯度：标量场增长最快的方向',
        body:
          '设标量场 $f(x,y,z)$ 有连续偏导数，其**梯度**定义为：\n\n' +
          '$\\nabla f=(f_x,\\,f_y,\\,f_z)$\n\n' +
          '**三个核心性质**：\n\n' +
          '（1）梯度的**方向**是 $f$ 增长最快的方向；\n\n' +
          '（2）梯度的**模** $|\\nabla f|$ 等于该点沿最速方向的方向导数（即最大变化率）；\n\n' +
          '（3）梯度与**等值面**（$f=c$ 的曲面）正交。\n\n' +
          '**具体数字例**：设 $f=x^2+y^2+z^2$，在点 $(1,2,2)$ 处：\n\n' +
          '$\\nabla f=(2x,2y,2z)\\big|_{(1,2,2)}=(2,4,4)$，模为 $\\sqrt{4+16+16}=6$。\n\n' +
          '这说明从 $(1,2,2)$ 出发，沿方向 $(2,4,4)$（即径向向外）温度上升最快，最大变化率为 $6$。',
        formula: '\\nabla f = \\frac{\\partial f}{\\partial x}\\mathbf{i}+\\frac{\\partial f}{\\partial y}\\mathbf{j}+\\frac{\\partial f}{\\partial z}\\mathbf{k}',
        tip: '爬山类比：梯度是"最陡上坡"的方向；负梯度 $-\\nabla f$ 是"最速下坡"的方向（梯度下降法就来自这里）。',
      },
      // 卡 4：散度——源的强度
      {
        title: '散度：向量场在某点"涌出"多少？',
        body:
          '设向量场 $\\mathbf{F}=(P,Q,R)$ 有连续偏导数，其**散度**定义为：\n\n' +
          '$\\nabla\\cdot\\mathbf{F}=P_x+Q_y+R_z$\n\n' +
          '**物理意义**：散度描述该点单位体积内净流出量（"源"的强度）。\n\n' +
          '散度 $>0$：该点是**源**，流体从这里向外涌出（如正电荷处电场线向外发散）；\n\n' +
          '散度 $<0$：该点是**汇**，流体向这里汇聚（如负电荷处电场线向内汇聚）；\n\n' +
          '散度 $=0$：无源无汇，该场叫**无源场**（不可压缩流体满足此条件）。\n\n' +
          '**数字例**：$\\mathbf{v}=x\\mathbf{i}+y\\mathbf{j}+z\\mathbf{k}$，$\\nabla\\cdot\\mathbf{v}=1+1+1=3>0$，说明流体在每点均匀向外膨胀，膨胀率为 $3$（每单位体积每秒涌出 $3$ 个单位流量）。',
        formula: '\\nabla\\cdot\\mathbf{F}=\\frac{\\partial P}{\\partial x}+\\frac{\\partial Q}{\\partial y}+\\frac{\\partial R}{\\partial z}',
        tip: '散度是**标量**，结果只有大小没有方向。检查计算：散度就是"把三个偏导加起来"，最容易算错的是漏掉某一项。',
      },
      // 卡 5：旋度——涡旋的轴与强度
      {
        title: '旋度：向量场在某点如何"旋转"？',
        body:
          '设向量场 $\\mathbf{F}=(P,Q,R)$ 有连续偏导数，其**旋度**定义为行列式展开：\n\n' +
          '旋度是**向量**，方向（按右手定则）是旋转轴，模是旋转角速度大小。\n\n' +
          '**物理意义**：$\\nabla\\times\\mathbf{F}=\\mathbf{0}$ 称为**无旋场**（保守场）；保守场路径无关，存在势函数。\n\n' +
          '**数字例**：$\\mathbf{F}=(-y,x,0)$ 描述绕 $z$ 轴的逆时针旋转，$P=-y,Q=x,R=0$：\n\n' +
          '$k$ 分量：$Q_x-P_y=1-(-1)=2$；$i,j$ 分量均为 $0$。\n\n' +
          '故 $\\nabla\\times\\mathbf{F}=2\\mathbf{k}$，方向沿 $z$ 轴正方向，强度为 $2$——与直觉一致：顺时针看旋转是均匀的，角速度为 $1$，旋度模为 $2\\times1=2$。',
        formula:
          '\\nabla\\times\\mathbf{F}=\\left(\\frac{\\partial R}{\\partial y}-\\frac{\\partial Q}{\\partial z}\\right)\\mathbf{i}+\\left(\\frac{\\partial P}{\\partial z}-\\frac{\\partial R}{\\partial x}\\right)\\mathbf{j}+\\left(\\frac{\\partial Q}{\\partial x}-\\frac{\\partial P}{\\partial y}\\right)\\mathbf{k}',
        tip: '记忆行列式顺序：$i$ 分量看 $R_y-Q_z$，$j$ 分量看 $P_z-R_x$，$k$ 分量看 $Q_x-P_y$。旋度只在三维有意义，在二维中退化为标量 $Q_x-P_y$。',
      },
      // 卡 6：两条恒等式——梯度无旋、旋度无散
      {
        title: '两条黄金恒等式',
        body:
          '向量分析有两条最重要的恒等式，它们是许多物理定律的数学基础：\n\n' +
          '**恒等式一：梯度场必无旋**\n\n' +
          '$\\nabla\\times(\\nabla f)=\\mathbf{0}$\n\n' +
          '直觉：梯度场是"上坡方向"，绕任何小环路一圈做的功为零——路径无关 $\\Leftrightarrow$ 无旋。\n\n' +
          '证明关键：$k$ 分量 $= f_{yx}-f_{xy}=0$（混合偏导相等，Clairaut 定理）。\n\n' +
          '**恒等式二：旋度场必无散**\n\n' +
          '$\\nabla\\cdot(\\nabla\\times\\mathbf{F})=0$\n\n' +
          '直觉：旋度场是"涡旋"，涡旋只转动不发散——有源无旋，或无源有旋。\n\n' +
          '物理应用：磁场 $\\mathbf{B}=\\nabla\\times\\mathbf{A}$ 满足 $\\nabla\\cdot\\mathbf{B}=0$（无磁单极子，Maxwell 方程之一）。',
        formula:
          '\\nabla\\times(\\nabla f)=\\mathbf{0},\\qquad \\nabla\\cdot(\\nabla\\times\\mathbf{F})=0',
        tip: '这两条恒等式的记忆口诀：**梯度无旋，旋度无散**。在考试中遇到"证明某场无源/无旋"，先想这两条——往往可以直接套用。',
      },
      // 卡 7：Laplace 算子与调和函数
      {
        title: 'Laplace 算子：梯度的散度',
        body:
          '对标量场 $f$ 先取梯度再取散度，得到 **Laplace 算子**（也记作 $\\Delta f$）：\n\n' +
          '$\\nabla^2 f = \\nabla\\cdot(\\nabla f)=f_{xx}+f_{yy}+f_{zz}$\n\n' +
          '满足 $\\nabla^2 f=0$ 的函数称为**调和函数**，对应物理中的稳定态：静电势、稳定热场、理想流体压力场。\n\n' +
          '**验证例**：设 $f=\\dfrac{1}{r}$，其中 $r=\\sqrt{x^2+y^2+z^2}$（$r\\ne0$）。\n\n' +
          '计算可得 $f_{xx}=\\dfrac{2x^2-y^2-z^2}{r^5}$（及轮换），三项相加恰好为 $0$。\n\n' +
          '这说明**点源电势** $\\varphi=\\dfrac{1}{r}$ 在源外（$r>0$）满足 Laplace 方程——是静电学的核心。',
        formula: '\\nabla^2 f = f_{xx}+f_{yy}+f_{zz}=0\\quad\\text{（调和函数）}',
        tip: '调和函数满足最大值原理：在有界闭区域上，调和函数的最大值和最小值都在边界取到，内部无极值。',
      },
      // 卡 8：易错点与自测
      {
        title: '易错点与自测',
        body:
          '**易错一**：旋度仅在三维有意义。在二维中，$\\nabla\\times\\mathbf{F}$ 退化为标量 $Q_x-P_y$，它正好出现在 Green 定理中。\n\n' +
          '**易错二**：散度是**标量**（数），旋度是**向量**。混淆会导致维度错误。\n\n' +
          '**易错三**：$\\nabla\\times(\\nabla f)=\\mathbf{0}$ 是恒等式，但逆命题（无旋 $\\Rightarrow$ 梯度场）仅在单连通区域成立；若区域有"洞"，无旋场未必有势函数。\n\n' +
          '**易错四**：计算旋度时 $j$ 分量的符号。行列式展开中 $j$ 分量有负号：$j$ 分量 $=-(P_z-R_x)=R_x-P_z$，容易漏掉这个负号。',
        reveal: {
          q: '向量场 $\\mathbf{F}=(yz,xz,xy)$，直觉上它是否有势函数（保守场）？请先猜一猜，再算旋度验证。',
          a: '计算：$i$ 分量 $=x_y-z_z\\cdot x=(xy)_y-(xz)_z=x-x=0$；$j$ 分量 $=(yz)_z-(xy)_x=y-y=0$；$k$ 分量 $=(xz)_x-(yz)_y=z-z=0$。故 $\\nabla\\times\\mathbf{F}=\\mathbf{0}$，确实是保守场，势函数为 $\\varphi=xyz+C$。',
        },
      },
    ],
    questions: [
      // 题 1：choice — 梯度定义
      {
        id: 'c25-a-q1',
        type: 'choice',
        prompt: '设 $f(x,y,z)=x^2y+yz^2$，在点 $(1,1,1)$ 处，$\\nabla f$ 等于哪个向量？',
        options: [
          '$(2,\\,1,\\,2)$',
          '$(2,\\,2,\\,2)$',
          '$(1,\\,2,\\,2)$',
          '$(2,\\,3,\\,2)$',
        ],
        answer: 1,
        explain:
          '$f_x=\\frac{\\partial(x^2y)}{\\partial x}=2xy$，在 $(1,1,1)$ 处为 $2$；$f_y=\\frac{\\partial(x^2y+yz^2)}{\\partial y}=x^2+z^2$，在 $(1,1,1)$ 处为 $1+1=2$；$f_z=\\frac{\\partial(yz^2)}{\\partial z}=2yz$，在 $(1,1,1)$ 处为 $2$。故 $\\nabla f=(2,2,2)$，对应选项第二项（下标 $1$）。',
      },
      // 题 2：input — 散度计算
      {
        id: 'c25-a-q2',
        type: 'input',
        prompt: '向量场 $\\mathbf{v}=(x^2,y^2,z^2)$，求 $\\nabla\\cdot\\mathbf{v}$ 在点 $(1,2,3)$ 处的值。',
        accept: ['12', '12.0'],
        placeholder: '输入整数',
        explain:
          '$\\nabla\\cdot\\mathbf{v}=2x+2y+2z$。在点 $(1,2,3)$ 处：$2\\times1+2\\times2+2\\times3=2+4+6=12$。',
      },
      // 题 3：judge — 旋度恒等式
      {
        id: 'c25-a-q3',
        type: 'judge',
        prompt: '对任意具有二阶连续偏导数的标量场 $f$，$\\nabla\\times(\\nabla f)$ 恒等于零向量。',
        answer: true,
        explain:
          '这是向量分析的基本恒等式：梯度的旋度恒为零。证明核心是混合偏导数相等（Clairaut 定理）：$k$ 分量 $=f_{yx}-f_{xy}=0$，其余分量类似。物理含义：保守场（梯度场）必无旋。',
      },
      // 题 4：choice — 旋度计算
      {
        id: 'c25-a-q4',
        type: 'choice',
        prompt: '向量场 $\\mathbf{F}=(-y,x,0)$ 的旋度 $\\nabla\\times\\mathbf{F}$ 等于哪个向量？',
        options: [
          '$\\mathbf{0}$',
          '$\\mathbf{k}$',
          '$2\\mathbf{k}$',
          '$-2\\mathbf{k}$',
        ],
        answer: 2,
        explain:
          '$P=-y,Q=x,R=0$。$i$ 分量：$R_y-Q_z=0-0=0$；$j$ 分量：$P_z-R_x=0-0=0$；$k$ 分量：$Q_x-P_y=1-(-1)=2$。故 $\\nabla\\times\\mathbf{F}=2\\mathbf{k}$，描述绕 $z$ 轴均匀旋转，角速度为 $1$（旋度模 $=2\\omega$，$\\omega=1$）。',
      },
      // 题 5：judge — 散度恒等式
      {
        id: 'c25-a-q5',
        type: 'judge',
        prompt: '对任意向量场 $\\mathbf{F}$，$\\nabla\\cdot(\\nabla\\times\\mathbf{F})$ 恒等于 $0$。',
        answer: true,
        explain:
          '这是第二条黄金恒等式：旋度的散度恒为零。展开后每对混合偏导数互相抵消。物理含义：磁场 $\\mathbf{B}=\\nabla\\times\\mathbf{A}$ 满足 $\\nabla\\cdot\\mathbf{B}=0$（无磁单极子）。',
      },
      // 题 6：match — 算子与类型
      {
        id: 'c25-a-q6',
        type: 'match',
        prompt: '将算子与其输入输出类型配对。',
        left: ['梯度 $\\nabla f$', '散度 $\\nabla\\cdot\\mathbf{F}$', '旋度 $\\nabla\\times\\mathbf{F}$'],
        right: ['标量场变向量场', '向量场变标量场', '向量场变向量场'],
      },
      // 题 7：input — Laplace 算子
      {
        id: 'c25-a-q7',
        type: 'input',
        prompt: '求 $f(x,y,z)=x^2+y^2+z^2$ 的 Laplace 算子 $\\nabla^2 f$ 的值（常数）。',
        accept: ['6', '6.0'],
        placeholder: '输入整数',
        explain:
          '$f_{xx}=2$，$f_{yy}=2$，$f_{zz}=2$，故 $\\nabla^2 f=f_{xx}+f_{yy}+f_{zz}=2+2+2=6$。这是常数，说明该场在每点的"拉普拉斯量"相同。',
      },
      // 题 8：choice — 无旋场与势函数
      {
        id: 'c25-a-q8',
        type: 'choice',
        prompt: '下列哪个说法关于无旋场（$\\nabla\\times\\mathbf{F}=\\mathbf{0}$）是正确的？',
        options: [
          '无旋场一定无散',
          '在单连通区域内，无旋场存在势函数 $\\varphi$ 使得 $\\mathbf{F}=\\nabla\\varphi$',
          '无旋场的旋度等于梯度',
          '无旋场只在二维存在',
        ],
        answer: 1,
        explain:
          '在**单连通区域**（无洞的区域）内，无旋场等价于保守场等价于存在势函数 $\\varphi$ 使 $\\mathbf{F}=\\nabla\\varphi$。若区域有洞（如环形区域），无旋场未必有势函数。无旋与无散是独立条件，不能互推。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c25-b  Green 定理与 Gauss 定理
  // ─────────────────────────────────────────
  {
    id: 'c25-b',
    title: 'Green 定理与 Gauss 定理',
    subtitle: '动机·Green公式推导·Gauss定理·边界积分化内部积分·典型例题',
    intro: [
      // 卡 1：动机——封闭曲线/曲面积分太难，能不能换个区域？
      {
        title: '动机：封闭积分能转化成区域积分吗？',
        body:
          '计算 $\\oint_C P\\,dx+Q\\,dy$（沿封闭曲线的线积分），直接参数化往往很繁。\n\n' +
          '思路：曲线 $C$ 围住一个区域 $D$，能不能把曲线积分变成 $D$ 上的二重积分？\n\n' +
          '这就是 **Green 定理**要回答的问题：**边界上的积分 = 区域内的积分**。\n\n' +
          '类比一维微积分基本定理：$\\displaystyle\\int_a^b f\'(x)\\,dx=f(b)-f(a)$，即"区间内部的导数积分等于端点值之差（边界值）"。Green 定理是这一思想在二维的推广。',
        tip: '遇到封闭曲线 + 二维向量场，第一反应是 Green 定理。遇到封闭曲面 + 三维向量场，第一反应是 Gauss（散度）定理。',
      },
      // 卡 2：Green 定理的陈述与符号
      {
        title: 'Green 定理：精确陈述',
        body:
          '设 $D$ 是平面上的有界单连通闭区域，边界 $\\partial D$ 是分段光滑曲线（正方向：逆时针，即区域在左侧），$P,Q$ 在 $D$ 上有连续一阶偏导数，则：\n\n' +
          '$\\oint_{\\partial D}(P\\,dx+Q\\,dy)=\\iint_D\\left(\\frac{\\partial Q}{\\partial x}-\\frac{\\partial P}{\\partial y}\\right)dA$\n\n' +
          '**向量场语言**：$P\\,dx+Q\\,dy$ 是向量场 $\\mathbf{F}=(P,Q)$ 的线积分；右侧 $Q_x-P_y$ 是 $\\mathbf{F}$ 的二维旋度。\n\n' +
          '**直白含义**：沿边界一圈的"环量"等于区域内每点"旋转密度"的面积积分。',
        formula:
          '\\oint_{\\partial D}(P\\,dx+Q\\,dy)=\\iint_D\\left(\\frac{\\partial Q}{\\partial x}-\\frac{\\partial P}{\\partial y}\\right)dA',
        tip: '注意被积函数顺序：$Q_x-P_y$，先 $Q$ 后 $P$，不是 $P_x-Q_y$。方向：边界**逆时针**对应区域在左，正方向。',
      },
      // 卡 3：Green 定理应用——思维路径（章首引入题）
      {
        title: '例题：用 Green 定理算单位圆上的线积分',
        body:
          '计算 $\\oint_C(x^2y\\,dx+xy^2\\,dy)$，$C$ 为单位圆 $x^2+y^2=1$ 逆时针方向。\n\n' +
          '这是本章的引入题，我们用思维路径还原来演示 Green 定理的用法。',
        steps: [
          '第一步（判断条件）：$C$ 是封闭曲线，$D$ 是单位圆盘，$P=x^2y,Q=xy^2$ 在 $D$ 内连续可导。Green 定理可用。',
          '第二步（算旋度密度）：$Q_x=\\frac{\\partial(xy^2)}{\\partial x}=y^2$，$P_y=\\frac{\\partial(x^2y)}{\\partial y}=x^2$，故 $Q_x-P_y=y^2-x^2$。',
          '第三步（转化积分）：$\\oint_C=\\iint_D(y^2-x^2)\\,dA$。',
          '第四步（对称性）：$D$ 关于 $x,y$ 轴对称，$\\iint_D y^2\\,dA=\\iint_D x^2\\,dA$（由对称性），故差值为 $0$。',
          '结论：$\\oint_C(x^2y\\,dx+xy^2\\,dy)=0$。',
        ],
        tip: '对称性是化简二重积分的利器：$D$ 关于 $y$ 轴对称且被积函数关于 $x$ 为奇函数，则积分为 $0$；偶函数则可翻倍。',
      },
      // 卡 4：Green 定理的应用——求面积
      {
        title: '巧用 Green 定理求平面面积',
        body:
          '取 $P=-y/2$，$Q=x/2$，则 $Q_x-P_y=1/2-(-1/2)=1$。\n\n' +
          '代入 Green 定理：$\\oint_C(-\\frac{y}{2}\\,dx+\\frac{x}{2}\\,dy)=\\iint_D 1\\,dA=A$（面积）。\n\n' +
          '因此，面积公式为 $A=\\dfrac{1}{2}\\oint_C(x\\,dy-y\\,dx)$。\n\n' +
          '**应用**：椭圆 $x=a\\cos t,y=b\\sin t$，$t\\in[0,2\\pi]$：\n\n' +
          '$A=\\dfrac{1}{2}\\int_0^{2\\pi}[(a\\cos t)(b\\cos t)-(-b\\sin t)(-a\\sin t)]\\,dt=\\dfrac{1}{2}\\int_0^{2\\pi}ab(\\cos^2 t+\\sin^2 t)\\,dt=\\dfrac{ab}{2}\\cdot2\\pi=\\pi ab$',
        formula: 'A=\\frac{1}{2}\\oint_{\\partial D}(x\\,dy-y\\,dx)',
        tip: '这个面积公式在参数化边界已知时非常高效——不需要算二重积分，直接沿边界线积分即可。',
      },
      // 卡 5：Gauss 散度定理
      {
        title: 'Gauss 散度定理：封闭曲面通量 = 内部散度积分',
        body:
          '**Gauss 定理**是 Green 定理向三维体积的推广：\n\n' +
          '设 $\\Omega$ 是空间中的有界闭区域，$\\partial\\Omega$ 是其分片光滑封闭边界曲面（外法向），$\\mathbf{F}=(P,Q,R)$ 在 $\\Omega$ 上有连续一阶偏导数，则：\n\n' +
          '$\\oiint_{\\partial\\Omega}\\mathbf{F}\\cdot d\\mathbf{S}=\\iiint_\\Omega(\\nabla\\cdot\\mathbf{F})\\,dV$\n\n' +
          '**物理含义**：穿过封闭曲面向外的**总通量**等于内部**源强度**（散度）的体积积分。\n\n' +
          '例：水管无渗漏时，进出封闭面的净流量（通量）等于内部净涌出量（散度积分）。',
        formula:
          '\\oiint_{\\partial\\Omega}\\mathbf{F}\\cdot d\\mathbf{S}=\\iiint_\\Omega\\nabla\\cdot\\mathbf{F}\\,dV',
        tip: '使用 Gauss 定理的三步：① 确认是封闭曲面（外法向）；② 计算散度 $\\nabla\\cdot\\mathbf{F}$（通常很简单）；③ 算体积积分（利用几何形状）。',
      },
      // 卡 6：Gauss 定理例题——球面通量
      {
        title: '例题：用 Gauss 定理算球面通量',
        body:
          '计算 $\\oiint_S\\mathbf{r}\\cdot d\\mathbf{S}$，其中 $\\mathbf{r}=(x,y,z)$，$S$ 是球面 $x^2+y^2+z^2=R^2$ 的外侧。\n\n' +
          '直接参数化球面计算通量很繁，用 Gauss 定理只需三步：',
        steps: [
          '第一步（散度）：$\\mathbf{F}=\\mathbf{r}=(x,y,z)$，$\\nabla\\cdot\\mathbf{F}=\\frac{\\partial x}{\\partial x}+\\frac{\\partial y}{\\partial y}+\\frac{\\partial z}{\\partial z}=1+1+1=3$。',
          '第二步（Gauss 定理）：$\\oiint_S=\\iiint_\\Omega 3\\,dV=3\\cdot V(\\text{球})$。',
          '第三步（体积）：半径 $R$ 的球体积为 $\\frac{4}{3}\\pi R^3$，故 $\\oiint_S=3\\times\\frac{4}{3}\\pi R^3=4\\pi R^3$。',
        ],
        tip: '球面的通量问题几乎都可以用 Gauss 定理：先求散度，再算体积。关键是散度往往比通量积分简单得多。',
      },
      // 卡 7：易错点与注意事项
      {
        title: '易错点与使用条件',
        body:
          '**Green 定理易错点**：\n\n' +
          '易错一：边界方向——边界必须是**逆时针**（正方向，区域在左侧）；若给出顺时针，需加负号。\n\n' +
          '易错二：$Q_x-P_y$ 不是 $P_x-Q_y$，顺序不要弄反。\n\n' +
          '**Gauss 定理易错点**：\n\n' +
          '易错三：边界曲面必须是**封闭的**（有外法向）。若题目给出非封闭曲面，需补"盖子"，再减去盖子上的积分。\n\n' +
          '易错四：方向约定——外法向（向外），若用内法向结果要取反。\n\n' +
          '**共同陷阱**：两个定理都要求被积函数在整个区域内有连续偏导数——若区域内有奇点（如原点），需要先"挖去"奇点所在的小区域再应用。',
        reveal: {
          q: '用 Gauss 定理计算 $\\oiint_S(x^2\\,dy\\,dz+y^2\\,dz\\,dx+z^2\\,dx\\,dy)$，其中 $S$ 为单位立方体 $[0,1]^3$ 的外侧。',
          a: '$\\mathbf{F}=(x^2,y^2,z^2)$，散度 $=2x+2y+2z$。Gauss 定理：$\\iiint_{[0,1]^3}2(x+y+z)\\,dV=2\\times3\\times\\int_0^1 x\\,dx\\cdot1\\cdot1=2\\times3\\times\\frac{1}{2}=3$。（三个变量对称，各贡献 $\\int_0^1 2x\\,dx=1$，共 $3$。）',
        },
      },
    ],
    questions: [
      // 题 1：choice — Green 定理核心被积函数
      {
        id: 'c25-b-q1',
        type: 'choice',
        prompt: 'Green 定理将 $\\oint_C(P\\,dx+Q\\,dy)$ 转化为二重积分，被积函数是什么？',
        options: [
          '$P_x+Q_y$',
          '$Q_x-P_y$',
          '$P_y-Q_x$',
          '$P_x-Q_y$',
        ],
        answer: 1,
        explain:
          'Green 定理：$\\oint_{\\partial D}(P\\,dx+Q\\,dy)=\\iint_D(Q_x-P_y)\\,dA$。被积函数是 $Q_x-P_y$（先 $Q$ 后 $P$，注意顺序）。这正是向量场 $(P,Q)$ 的二维旋度。',
      },
      // 题 2：input — Green 定理计算
      {
        id: 'c25-b-q2',
        type: 'input',
        prompt: '用 Green 定理计算 $\\oint_C(x^2y\\,dx+xy^2\\,dy)$，其中 $C$ 是单位圆逆时针方向，结果等于多少？',
        accept: ['0', '0.0'],
        placeholder: '输入数值',
        explain:
          '$Q_x-P_y=y^2-x^2$。转化为 $\\iint_D(y^2-x^2)\\,dA$，$D$ 为单位圆盘。由圆盘关于两轴对称，$\\iint_D y^2\\,dA=\\iint_D x^2\\,dA$，差为 $0$。',
      },
      // 题 3：judge — Gauss 定理条件
      {
        id: 'c25-b-q3',
        type: 'judge',
        prompt: 'Gauss 定理可以直接用于非封闭曲面（如半球面，没有底面）计算通量。',
        answer: false,
        explain:
          'Gauss 定理要求边界曲面必须是**封闭的**。对非封闭曲面，需要先补充"盖子"使之封闭，应用 Gauss 定理后，再减去盖子上的通量积分。',
      },
      // 题 4：input — Gauss 定理，球面通量
      {
        id: 'c25-b-q4',
        type: 'input',
        prompt: '用 Gauss 定理计算 $\\oiint_S\\mathbf{r}\\cdot d\\mathbf{S}$，$\\mathbf{r}=(x,y,z)$，$S$ 为半径 $R=2$ 的球面外侧，结果等于多少？（结果含 $\\pi$，如 $32\\pi$）',
        accept: ['32pi', '32\\pi', '32π'],
        placeholder: '如 32pi',
        explain:
          '散度 $\\nabla\\cdot\\mathbf{r}=3$。Gauss 定理：$\\iiint_\\Omega 3\\,dV=3\\times\\frac{4}{3}\\pi R^3=3\\times\\frac{4}{3}\\pi\\times8=32\\pi$。',
      },
      // 题 5：choice — 物理意义
      {
        id: 'c25-b-q5',
        type: 'choice',
        prompt: 'Gauss 定理中，若某区域内 $\\nabla\\cdot\\mathbf{F}>0$，则穿过包围该区域的封闭曲面的净通量是？',
        options: [
          '净通量为负（向内净流入）',
          '净通量为零（进出相等）',
          '净通量为正（向外净流出）',
          '无法判断',
        ],
        answer: 2,
        explain:
          '$\\nabla\\cdot\\mathbf{F}>0$ 说明区域内有"源"（流体向外涌出），由 Gauss 定理，封闭曲面上的净通量 $\\oiint=\\iiint\\nabla\\cdot\\mathbf{F}\\,dV>0$，即净通量为正（向外净流出）。',
      },
      // 题 6：judge — Green 定理与面积
      {
        id: 'c25-b-q6',
        type: 'judge',
        prompt: '利用 Green 定理，平面区域面积可以表示为 $A=\\dfrac{1}{2}\\oint_C(x\\,dy-y\\,dx)$。',
        answer: true,
        explain:
          '取 $P=-y/2,Q=x/2$，则 $Q_x-P_y=1/2+1/2=1$。Green 定理给出 $\\oint_C(-y/2\\,dx+x/2\\,dy)=\\iint_D 1\\,dA=A$，即 $A=\\frac{1}{2}\\oint_C(x\\,dy-y\\,dx)$。正确。',
      },
      // 题 7：match — Green/Gauss 定理配对
      {
        id: 'c25-b-q7',
        type: 'match',
        prompt: '将定理与其"边界积分→区域积分"的微分算子配对。',
        left: ['Green 定理（二维）', 'Gauss 定理（三维）'],
        right: ['$Q_x-P_y$（二维旋度）', '$\\nabla\\cdot\\mathbf{F}$（三维散度）'],
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c25-c  Stokes 定理与三大定理的统一
  // ─────────────────────────────────────────
  {
    id: 'c25-c',
    title: 'Stokes 定理与三大定理统一',
    subtitle: 'Stokes公式·三大定理对比·广义Stokes定理·典型例题·选定理流程',
    intro: [
      // 卡 1：Stokes 定理的动机
      {
        title: 'Stokes 定理：三维曲面上的 Green 定理',
        body:
          'Green 定理处理**平面区域**和其**平面边界曲线**的关系。\n\n' +
          '但物理世界中的曲面（如抛物面、球面一片）并不是平面，其边界是空间曲线。\n\n' +
          'Stokes 定理将这一关系推广到**任意光滑曲面** $S$ 和其**边界曲线** $\\partial S$：\n\n' +
          '沿边界 $\\partial S$ 的**环量**（线积分）= 曲面 $S$ 上**旋度通量**（面积分）。\n\n' +
          '直觉：旋度描述场的"旋转密度"，把曲面上每一点的旋转密度加起来，应该等于沿边界绕一圈的总旋转量。',
        tip: '判断用 Stokes 定理的信号：① 空间中的封闭曲线积分；② 问题涉及旋度；③ 曲线可以被某个简单曲面"盖住"。',
      },
      // 卡 2：Stokes 定理的陈述
      {
        title: 'Stokes 定理：精确陈述与方向约定',
        body:
          '设 $S$ 是空间中的分片光滑有向曲面，$\\partial S$ 是其边界曲线，方向与 $S$ 的法向量满足**右手法则**（右手四指沿 $\\partial S$ 方向，大拇指朝法向量方向），$\\mathbf{F}=(P,Q,R)$ 在 $S$ 上有连续一阶偏导数，则：\n\n' +
          '$\\oint_{\\partial S}\\mathbf{F}\\cdot d\\mathbf{r}=\\iint_S(\\nabla\\times\\mathbf{F})\\cdot d\\mathbf{S}$\n\n' +
          '右侧展开：$d\\mathbf{S}=\\mathbf{n}\\,dS$，$\\mathbf{n}$ 是曲面的单位法向量。\n\n' +
          '**特例**：取 $S$ 为平面区域 $D$，$\\mathbf{n}=\\mathbf{k}$（$z$ 轴正向），$\\nabla\\times\\mathbf{F}$ 的 $z$ 分量就是 $Q_x-P_y$。代入即得 Green 定理——**Green 定理是 Stokes 定理的平面特例**。',
        formula:
          '\\oint_{\\partial S}\\mathbf{F}\\cdot d\\mathbf{r}=\\iint_S(\\nabla\\times\\mathbf{F})\\cdot d\\mathbf{S}',
        tip: '方向约定是关键：右手法则——四指弯向边界方向，大拇指指向曲面法向量方向。选错方向，结果反号。',
      },
      // 卡 3：Stokes 定理例题
      {
        title: '例题：用 Stokes 定理化简空间曲线积分',
        body:
          '计算 $\\oint_C\\mathbf{F}\\cdot d\\mathbf{r}$，$\\mathbf{F}=(y,z,x)$，$C$ 是平面 $x+y+z=1$ 与三个坐标平面围成三角形的边界（从 $z$ 轴正向看为逆时针）。',
        steps: [
          '第一步（算旋度）：$P=y,Q=z,R=x$。$i$ 分量：$R_y-Q_z=0-1=-1$；$j$ 分量：$P_z-R_x=0-1=-1$；$k$ 分量：$Q_x-P_y=0-1=-1$。故 $\\nabla\\times\\mathbf{F}=(-1,-1,-1)$。',
          '第二步（确定曲面）：取 $S$ 为平面 $x+y+z=1$ 上的三角形（三顶点：$(1,0,0),(0,1,0),(0,0,1)$）。',
          '第三步（法向量）：平面 $x+y+z=1$ 的法向量为 $(1,1,1)$，单位化 $\\mathbf{n}=\\frac{1}{\\sqrt{3}}(1,1,1)$；与边界逆时针方向成右手系。',
          '第四步（面积）：三角形面积 $A=\\frac{\\sqrt{3}}{2}$（边长为 $\\sqrt{2}$ 的等边三角形）。',
          '第五步（计算）：$\\iint_S(\\nabla\\times\\mathbf{F})\\cdot\\mathbf{n}\\,dS=(-1,-1,-1)\\cdot\\frac{1}{\\sqrt{3}}(1,1,1)\\times\\frac{\\sqrt{3}}{2}=(-3)\\times\\frac{1}{2}=-\\frac{3}{2}$。',
        ],
        tip: '选择曲面的技巧：优先选最简单的（如平面三角形、圆盘），而不是沿着题目给的空间曲线硬算。Stokes 定理允许替换曲面，只要边界相同。',
      },
      // 卡 4：三大定理的统一对比
      {
        title: '三大定理：统一框架',
        body:
          '三大定理的本质都是同一句话：**区域内部微分算子的积分 = 边界上场量的积分**。\n\n' +
          '一维微积分基本定理：$\\displaystyle\\int_a^b f\'\\,dx=f(b)-f(a)$\n\n' +
          'Green 定理（2D 区域 → 1D 边界）：$\\iint_D(Q_x-P_y)\\,dA=\\oint_{\\partial D}(P\\,dx+Q\\,dy)$\n\n' +
          'Gauss 定理（3D 体积 → 2D 边界曲面）：$\\iiint_\\Omega\\nabla\\cdot\\mathbf{F}\\,dV=\\oiint_{\\partial\\Omega}\\mathbf{F}\\cdot d\\mathbf{S}$\n\n' +
          'Stokes 定理（3D 曲面 → 1D 边界曲线）：$\\iint_S(\\nabla\\times\\mathbf{F})\\cdot d\\mathbf{S}=\\oint_{\\partial S}\\mathbf{F}\\cdot d\\mathbf{r}$\n\n' +
          '所有这些都是**广义 Stokes 定理** $\\displaystyle\\int_\\Omega d\\omega=\\int_{\\partial\\Omega}\\omega$（微分形式语言）的特殊情形。',
        tip: '记忆框架：维度降一维，算子换边界。Green 是 Stokes 的平面特例；三者共同来源是微积分基本定理的高维推广。',
      },
      // 卡 5：选定理的流程图
      {
        title: '解题时如何选定理？',
        body:
          '遇到向量分析积分题，按以下流程快速判断：\n\n' +
          '**路线一：线积分**\n\n' +
          '线积分 $\\int_C\\mathbf{F}\\cdot d\\mathbf{r}$，$C$ 封闭？\n' +
          '→ 是，2D：用 Green 定理化为二重积分；\n' +
          '→ 是，3D：用 Stokes 定理化为曲面积分（选最简单的曲面）；\n' +
          '→ 否：看是否无旋，无旋则找势函数直接用 $\\varphi(B)-\\varphi(A)$。\n\n' +
          '**路线二：面积分（通量）**\n\n' +
          '面积分 $\\iint_S\\mathbf{F}\\cdot d\\mathbf{S}$，$S$ 封闭？\n' +
          '→ 是：用 Gauss 定理化为三重积分（算散度）；\n' +
          '→ 否：若能补全为封闭，补盖子用 Gauss 再减去盖子积分。',
        tip: '优先用定理！直接参数化计算通常是最后的退路。能用定理就用定理，把复杂积分降维或换形。',
      },
      // 卡 6：恒等式汇总与连接
      {
        title: '场论恒等式速查表',
        body:
          '以下恒等式是向量分析的骨架，需要熟记：\n\n' +
          '（1）$\\nabla\\times(\\nabla f)=\\mathbf{0}$——梯度场无旋（保守场必无旋）\n\n' +
          '（2）$\\nabla\\cdot(\\nabla\\times\\mathbf{F})=0$——旋度场无散（磁场无源）\n\n' +
          '（3）$\\nabla^2 f=\\nabla\\cdot(\\nabla f)=f_{xx}+f_{yy}+f_{zz}$——Laplace 算子\n\n' +
          '（4）$\\nabla\\cdot(f\\mathbf{F})=f(\\nabla\\cdot\\mathbf{F})+\\mathbf{F}\\cdot\\nabla f$——乘积散度法则\n\n' +
          '（5）$\\nabla\\times(f\\mathbf{F})=f(\\nabla\\times\\mathbf{F})+(\\nabla f)\\times\\mathbf{F}$——乘积旋度法则\n\n' +
          'Maxwell 方程组正是用这些符号优雅写出：$\\nabla\\cdot\\mathbf{E}=\\rho/\\varepsilon_0$，$\\nabla\\cdot\\mathbf{B}=0$，$\\nabla\\times\\mathbf{E}=-\\partial\\mathbf{B}/\\partial t$，$\\nabla\\times\\mathbf{B}=\\mu_0\\mathbf{J}+\\mu_0\\varepsilon_0\\partial\\mathbf{E}/\\partial t$。',
        tip: '恒等式（1）（2）是最常考的，可以用于证明某场无旋/无散，或者化简旋度/散度运算，不需要展开所有分量。',
      },
      // 卡 7：易错点总结
      {
        title: '易错点汇总',
        body:
          '**Stokes 定理易错点**：\n\n' +
          '易错一：右手法则方向——边界与法向量方向必须匹配；选错方向结果反号。\n\n' +
          '易错二：旋度 $j$ 分量有额外负号：$j$ 分量 $=P_z-R_x$（展开行列式时 $j$ 项系数为 $-1$），容易算成 $R_x-P_z$。\n\n' +
          '**三大定理共同易错点**：\n\n' +
          '易错三：条件——被积函数必须在**整个区域**连续可导。若有奇点（如原点），需挖去后应用。\n\n' +
          '易错四：Green 是 Stokes 在平面的特例，不是独立的定理——但在考试中仍需能独立陈述和使用 Green 定理。\n\n' +
          '易错五：**散度定理要封闭曲面，Stokes 定理要曲面有边界**——不要混用适用范围。',
        reveal: {
          q: '用 Stokes 定理，计算 $\\oint_C(z-y)\\,dx+(x-z)\\,dy+(y-x)\\,dz$，$C$ 是圆周 $x^2+y^2=1$，$z=0$，逆时针方向（从 $z$ 轴正向看）。',
          a: '$\\mathbf{F}=(z-y,x-z,y-x)$，旋度 $i$ 分量：$(y-x)_y-(x-z)_z=1-(-1)=2$；$j$ 分量：$(z-y)_z-(y-x)_x=1-(-1)=2$；$k$ 分量：$(x-z)_x-(z-y)_y=1-(-1)=2$。故 $\\nabla\\times\\mathbf{F}=(2,2,2)$。取 $S$ 为圆盘 $x^2+y^2\\le1,z=0$，$\\mathbf{n}=\\mathbf{k}$（与逆时针成右手系）。$\\iint_S(2,2,2)\\cdot(0,0,1)\\,dA=2\\times\\pi\\times1^2=2\\pi$。',
        },
      },
    ],
    questions: [
      // 题 1：choice — Stokes 定理核心
      {
        id: 'c25-c-q1',
        type: 'choice',
        prompt: 'Stokes 定理将空间曲线积分 $\\oint_{\\partial S}\\mathbf{F}\\cdot d\\mathbf{r}$ 转化为曲面积分，被积向量场是什么？',
        options: [
          '$\\nabla\\cdot\\mathbf{F}$',
          '$\\nabla f$',
          '$\\nabla\\times\\mathbf{F}$',
          '$\\nabla^2\\mathbf{F}$',
        ],
        answer: 2,
        explain:
          'Stokes 定理：$\\oint_{\\partial S}\\mathbf{F}\\cdot d\\mathbf{r}=\\iint_S(\\nabla\\times\\mathbf{F})\\cdot d\\mathbf{S}$。左边是曲线积分（环量），右边是**旋度**的曲面通量。旋度 $\\nabla\\times\\mathbf{F}$ 描述场的涡旋强度，沿边界的总环量等于曲面上旋度的积分。',
      },
      // 题 2：judge — Green 是 Stokes 特例
      {
        id: 'c25-c-q2',
        type: 'judge',
        prompt: 'Green 定理是 Stokes 定理取曲面为平面区域（法向量取 $\\mathbf{k}$）时的特殊情形。',
        answer: true,
        explain:
          '对平面区域 $D$（$z=0$），取法向量 $\\mathbf{n}=\\mathbf{k}$。此时 $\\nabla\\times\\mathbf{F}$ 的 $k$ 分量是 $Q_x-P_y$，$(\\nabla\\times\\mathbf{F})\\cdot\\mathbf{k}=Q_x-P_y$。代入 Stokes 定理，$\\iint_D(Q_x-P_y)\\,dA=\\oint_{\\partial D}(P\\,dx+Q\\,dy)$，正好是 Green 定理。',
      },
      // 题 3：input — Stokes 定理计算
      {
        id: 'c25-c-q3',
        type: 'input',
        prompt: '用 Stokes 定理计算 $\\oint_C(z-y)\\,dx+(x-z)\\,dy+(y-x)\\,dz$，$C$ 是圆周 $x^2+y^2=1,z=0$，逆时针方向。结果等于多少？（结果含 $\\pi$，如 $2\\pi$）',
        accept: ['2pi', '2\\pi', '2π'],
        placeholder: '如 2pi',
        explain:
          '$\\mathbf{F}=(z-y,x-z,y-x)$，旋度各分量均为 $2$，即 $\\nabla\\times\\mathbf{F}=(2,2,2)$。取 $S$ 为圆盘（$z=0$，$\\mathbf{n}=\\mathbf{k}$）。$(\\nabla\\times\\mathbf{F})\\cdot\\mathbf{k}=2$，$\\iint_S 2\\,dA=2\\pi$。',
      },
      // 题 4：match — 三大定理配对
      {
        id: 'c25-c-q4',
        type: 'match',
        prompt: '将三大定理与其"积分类型转化"配对。',
        left: ['Green 定理', 'Gauss 定理', 'Stokes 定理'],
        right: [
          '平面线积分 $\\to$ 二重积分',
          '曲面积分 $\\to$ 三重积分',
          '空间曲线积分 $\\to$ 曲面积分',
        ],
      },
      // 题 5：judge — 广义 Stokes
      {
        id: 'c25-c-q5',
        type: 'judge',
        prompt: 'Green、Gauss、Stokes 三大定理以及一维微积分基本定理，都是广义 Stokes 定理 $\\int_\\Omega d\\omega=\\int_{\\partial\\Omega}\\omega$ 在不同维度下的特例。',
        answer: true,
        explain:
          '广义 Stokes 定理（微分形式语言）统一了所有这些定理：$d\\omega$ 对应各种微分算子（导数、二维旋度、散度、三维旋度），$\\partial\\Omega$ 是区域的边界。一维时是端点差，二维是 Green 定理，三维散度是 Gauss，三维旋度是 Stokes。',
      },
      // 题 6：choice — 选哪个定理
      {
        id: 'c25-c-q6',
        type: 'choice',
        prompt: '要计算向量场穿过封闭球面的**通量**，应首先考虑哪个定理？',
        options: [
          'Green 定理',
          'Stokes 定理',
          'Gauss（散度）定理',
          '微积分基本定理',
        ],
        answer: 2,
        explain:
          '封闭曲面上的通量积分 $\\oiint_S\\mathbf{F}\\cdot d\\mathbf{S}$，首选 Gauss（散度）定理：$\\oiint=\\iiint\\nabla\\cdot\\mathbf{F}\\,dV$。计算散度（通常是简单的加法）远比直接参数化球面简单。',
      },
      // 题 7：input — 旋度计算后用 Stokes
      {
        id: 'c25-c-q7',
        type: 'input',
        prompt: '向量场 $\\mathbf{F}=(y,z,x)$，计算 $\\oint_C\\mathbf{F}\\cdot d\\mathbf{r}$，其中 $C$ 是平面 $x+y+z=1$ 上三角形的边界（从 $(1,1,1)$ 方向看为逆时针）。结果等于多少？填写分数如 $-3/2$。',
        accept: ['-3/2', '-1.5'],
        placeholder: '如 -3/2',
        explain:
          '$P=y,Q=z,R=x$。旋度：$i$ 分量 $=R_y-Q_z=0-1=-1$，$j$ 分量 $=P_z-R_x=0-1=-1$，$k$ 分量 $=Q_x-P_y=0-1=-1$。故 $\\nabla\\times\\mathbf{F}=(-1,-1,-1)$。曲面法向量 $\\mathbf{n}=\\frac{1}{\\sqrt{3}}(1,1,1)$，面积 $A=\\frac{\\sqrt{3}}{2}$。$\\iint(-1,-1,-1)\\cdot\\frac{(1,1,1)}{\\sqrt{3}}\\,dS=\\frac{-3}{\\sqrt{3}}\\times\\frac{\\sqrt{3}}{2}=-\\frac{3}{2}$。',
      },
    ],
  },
]
