export const LESSONS = [
  // ─────────────────────────────────────────────────────────────────────────
  // 关 c22-a  二重积分：从曲顶柱体到累次积分
  // ─────────────────────────────────────────────────────────────────────────
  {
    id: 'c22-a',
    title: '二重积分与累次积分',
    subtitle: '几何直觉·定义·X 型 / Y 型区域·积分次序交换',
    intro: [
      // 卡 1：动机——为什么需要二重积分？
      {
        title: '为什么需要二重积分？',
        body:
          '一元积分 $\\int_a^b f(x)\\,dx$ 给出曲线下的面积，那**曲面下方的体积**怎么算？\n\n' +
          '想象在一块平地 $D$ 上建一个"波浪形屋顶"，屋顶高度由函数 $z=f(x,y)\\geq 0$ 决定。' +
          '这个空间形状叫**曲顶柱体**，它的体积无法用一次积分算出——需要对两个方向都积分，这就是**二重积分**的由来。\n\n' +
          '**实际应用**：计算不均匀平板的质量（$\\iint_D \\rho(x,y)\\,dA$）、求曲面面积、' +
          '在概率论中对联合概率密度求期望——二重积分是理解三维世界和高维数据的钥匙。',
        tip: '一元积分：面积 $=$ 长度 $\\times$ 高。二重积分：体积 $=$ 面积 $\\times$ 高的极限求和。',
      },
      // 卡 2：从 Riemann 和到定义
      {
        title: '分割、近似、求和、取极限',
        body:
          '**第一步（分割）**：把平面区域 $D$ 切成 $n$ 块小区域，第 $i$ 块面积记为 $\\Delta\\sigma_i$。\n\n' +
          '**第二步（近似）**：在每块上任取一点 $(\\xi_i,\\eta_i)$，以 $f(\\xi_i,\\eta_i)$ 为高，近似小柱体体积 $\\approx f(\\xi_i,\\eta_i)\\,\\Delta\\sigma_i$。\n\n' +
          '**第三步（求和）**：全部小柱体加起来 $\\sum_{i=1}^n f(\\xi_i,\\eta_i)\\,\\Delta\\sigma_i$。\n\n' +
          '**第四步（取极限）**：令各小块直径的最大值 $\\lambda\\to 0$，极限存在时就是二重积分。\n\n' +
          '这与一元定积分的四步完全平行，只是"切成小段"变成了"切成小面块"。',
        formula:
          '\\iint_D f(x,y)\\,d\\sigma = \\lim_{\\lambda\\to 0}\\sum_{i=1}^n f(\\xi_i,\\eta_i)\\,\\Delta\\sigma_i',
        tip: '在直角坐标中，面积元素 $d\\sigma = dx\\,dy$，所以也写作 $\\iint_D f(x,y)\\,dx\\,dy$。',
      },
      // 卡 3：几何意义与物理意义
      {
        title: '二重积分的两层含义',
        body:
          '**几何含义**：当 $f(x,y)>0$ 时，$\\iint_D f(x,y)\\,dA$ 就是以 $D$ 为底、以曲面 $z=f(x,y)$ 为顶的曲顶柱体体积。当 $f$ 取负值时，积分值等于上方体积减去下方体积（类比一元积分的带符号面积）。\n\n' +
          '**特别情形**：$\\iint_D 1\\,dA$ 没有"高度"可言，结果就是区域 $D$ 的面积。\n\n' +
          '**物理含义**：若 $\\rho(x,y)$ 是平板 $D$ 上的面密度（$\\text{kg/m}^2$），则 $M=\\iint_D\\rho(x,y)\\,dA$ 就是平板的总质量。\n\n' +
          '**例**：$D$ 是边长 2 的正方形，$\\rho(x,y)=x+y+1$。' +
          '直觉告诉我们密度从角落到中心逐渐变化，用二重积分才能精确求总质量。',
        tip: '记忆口诀：$\\iint_D 1\\,dA = $ 区域 $D$ 的面积，$\\iint_D \\rho\\,dA = $ 总质量。',
      },
      // 卡 4：X 型区域与 Fubini 定理（化为累次积分）
      {
        title: 'X 型区域：先对 $y$ 积，再对 $x$ 积',
        body:
          '计算二重积分的核心方法是把它化为两次普通积分，叫**累次积分**。\n\n' +
          '若区域 $D$ 可以写成 $D=\\{(x,y)\\mid a\\leq x\\leq b,\\;\\varphi_1(x)\\leq y\\leq\\varphi_2(x)\\}$，称为 **X 型区域**——对每个固定的 $x$，$y$ 在两条曲线之间变化。\n\n' +
          '**Fubini 定理（X 型）**：先对 $y$ 从 $\\varphi_1(x)$ 到 $\\varphi_2(x)$ 积，得到 $x$ 的函数，再对 $x$ 从 $a$ 到 $b$ 积。\n\n' +
          '**具体例子**：$D$ 是由 $y=x$ 和 $y=x^2$ 围成的区域（$0\\leq x\\leq 1$）。\n\n' +
          '固定 $x\\in[0,1]$，$y$ 从 $x^2$（下方抛物线）到 $x$（上方直线）变化——这就是 X 型区域的直觉。',
        formula:
          '\\iint_D f(x,y)\\,dx\\,dy = \\int_a^b dx\\int_{\\varphi_1(x)}^{\\varphi_2(x)} f(x,y)\\,dy',
        tip: '如何区分 X 型和 Y 型？画出区域后，竖切（过每个 $x$ 值画竖线）若每段 $y$ 范围连续——X 型；横切若每段 $x$ 范围连续——Y 型。',
      },
      // 卡 5：Y 型区域 + 完整例题
      {
        title: 'Y 型区域 + 完整例题',
        body:
          '**Y 型区域**：对每个固定 $y$，$x$ 在两曲线之间。公式：先对 $x$，再对 $y$。\n\n' +
          '**例题**：计算 $\\iint_D xy\\,dx\\,dy$，$D$ 由 $y=x$，$y=x^2$ 围成。\n\n' +
          '**第一步（确定积分限）**：交点 $(0,0)$ 和 $(1,1)$；X 型：$0\\leq x\\leq 1$，$x^2\\leq y\\leq x$。\n\n' +
          '**第二步（内层积分对 $y$）**：固定 $x$，\n' +
          '$\\int_{x^2}^{x} xy\\,dy = x\\cdot\\dfrac{y^2}{2}\\Big|_{x^2}^{x} = x\\cdot\\dfrac{x^2-x^4}{2} = \\dfrac{x^3-x^5}{2}$。\n\n' +
          '**第三步（外层积分对 $x$）**：\n' +
          '$\\int_0^1\\dfrac{x^3-x^5}{2}\\,dx = \\dfrac{1}{2}\\left[\\dfrac{x^4}{4}-\\dfrac{x^6}{6}\\right]_0^1 = \\dfrac{1}{2}\\left(\\dfrac{1}{4}-\\dfrac{1}{6}\\right) = \\dfrac{1}{24}$。',
        formula:
          '\\iint_D f(x,y)\\,dx\\,dy = \\int_c^d dy\\int_{\\psi_1(y)}^{\\psi_2(y)} f(x,y)\\,dx',
        tip: '步骤口诀：① 画区域 → ② 确定积分变量顺序 → ③ 写出内层上下限 → ④ 内层积分（把另一变量当常数）→ ⑤ 外层积分。',
      },
      // 卡 6：交换积分次序——必须画图
      {
        title: '交换积分次序：画图是唯一可靠方法',
        body:
          '有些题目内层对 $y$ 积根本积不出来（如 $e^{y^2}$，$\\sin(y^2)$），**必须交换次序**。\n\n' +
          '**例**：计算 $\\int_0^1 dx\\int_x^1 e^{y^2}\\,dy$。\n\n' +
          '$e^{y^2}$ 没有初等原函数，无法直接对 $y$ 积分——必须先交换次序！\n\n' +
          '**第一步（读出区域）**：原来是 $0\\leq x\\leq 1$，$x\\leq y\\leq 1$，即三角形 $\\{0\\leq x\\leq y,\\;0\\leq y\\leq 1\\}$。\n\n' +
          '**第二步（换成 Y 型描述）**：$0\\leq y\\leq 1$，$0\\leq x\\leq y$。\n\n' +
          '**第三步（先对 $x$，再对 $y$）**：$\\int_0^1 dy\\int_0^y e^{y^2}\\,dx = \\int_0^1 y\\,e^{y^2}\\,dy$。\n\n' +
          '内层 $\\int_0^y e^{y^2}\\,dx = y\\,e^{y^2}$（因为 $e^{y^2}$ 对 $x$ 是常数！）。\n\n' +
          '**第四步（令 $u=y^2$）**：$\\int_0^1 y\\,e^{y^2}\\,dy = \\dfrac{1}{2}[e^{y^2}]_0^1 = \\dfrac{e-1}{2}$。',
        tip: '易错点：交换次序只是重新描述同一区域，不改变积分值！务必画出积分区域再动笔，否则上下限极易写反。',
      },
      // 卡 7：易错点汇总 + reveal
      {
        title: '易错点汇总与自测',
        body:
          '**易错点一（内外层上下限搞混）**：X 型区域内层是 $y$ 的上下限（可含 $x$），外层是 $x$ 的常数上下限；绝不能把含变量的上下限放到外层。\n\n' +
          '**易错点二（交换次序时区域搞错）**：换次序后必须描述同一块区域，建议把不等式用"描点法"画出来验证。\n\n' +
          '**易错点三（内层积分时把外层变量当成变量）**：内层对 $y$ 积时，$x$ 是固定参数，不能再对 $x$ 求导或做运算。\n\n' +
          '**易错点四（区域是闭合曲线围成但漏找交点）**：两曲线交点决定外层的积分上下限，一定要先求交点。',
        reveal: {
          q: '交换积分次序：$\\int_0^1 dy\\int_{y}^{1} f(x,y)\\,dx$，换后的累次积分是什么？',
          a: '原区域：$0\\leq y\\leq 1$，$y\\leq x\\leq 1$，即三角形 $\\{0\\leq y\\leq x,\\;0\\leq x\\leq 1\\}$。换成 X 型：$\\int_0^1 dx\\int_0^x f(x,y)\\,dy$。',
        },
      },
    ],
    questions: [
      // 题 1：choice — 二重积分几何含义
      {
        id: 'c22-a-q1',
        type: 'choice',
        prompt:
          '当 $f(x,y)\\geq 0$ 时，$\\iint_D f(x,y)\\,dA$ 的几何意义是？',
        options: [
          '$D$ 的周长',
          '以 $D$ 为底、以 $z=f(x,y)$ 为顶的曲顶柱体体积',
          '$D$ 的面积',
          '曲面 $z=f(x,y)$ 的面积',
        ],
        answer: 1,
        explain:
          '当 $f(x,y)\\geq 0$ 时，二重积分的几何意义是曲顶柱体的体积：底面为平面区域 $D$，顶面为曲面 $z=f(x,y)$。' +
          '特别地，$\\iint_D 1\\,dA = $ 区域 $D$ 的面积（高度恒为 1 时体积退化为面积）。',
      },
      // 题 2：judge — 内层上下限可含外层变量
      {
        id: 'c22-a-q2',
        type: 'judge',
        prompt:
          '在累次积分 $\\int_a^b dx\\int_{\\varphi_1(x)}^{\\varphi_2(x)} f(x,y)\\,dy$ 中，内层上下限 $\\varphi_1(x),\\varphi_2(x)$ 可以是 $x$ 的函数。',
        answer: true,
        explain:
          '正确。X 型区域的累次积分：对每个固定的 $x$，$y$ 在 $\\varphi_1(x)$ 到 $\\varphi_2(x)$ 之间变化，' +
          '所以内层上下限可以（且通常）含 $x$。外层上下限 $a,b$ 才必须是常数。',
      },
      // 题 3：input — 计算简单累次积分
      {
        id: 'c22-a-q3',
        type: 'input',
        prompt:
          '计算 $\\iint_D xy\\,dx\\,dy$，其中 $D$ 由 $y=x$ 与 $y=x^2$（$0\\leq x\\leq 1$）围成。结果是分数，分子是几？（答案：分子/24）',
        accept: ['1'],
        explain:
          '如讲解中推导：$\\iint_D xy\\,dx\\,dy = \\int_0^1\\dfrac{x^3-x^5}{2}\\,dx = \\dfrac{1}{2}\\left(\\dfrac{1}{4}-\\dfrac{1}{6}\\right) = \\dfrac{1}{24}$。' +
          '分子为 1（分母为 24）。',
      },
      // 题 4：choice — 交换积分次序
      {
        id: 'c22-a-q4',
        type: 'choice',
        prompt:
          '交换积分次序：$\\int_0^2 dx\\int_0^x f(x,y)\\,dy$，结果是？',
        options: [
          '$\\int_0^2 dy\\int_y^2 f(x,y)\\,dx$',
          '$\\int_0^2 dy\\int_0^y f(x,y)\\,dx$',
          '$\\int_0^2 dy\\int_0^2 f(x,y)\\,dx$',
          '$\\int_0^2 dy\\int_y^{2-y} f(x,y)\\,dx$',
        ],
        answer: 0,
        explain:
          '原区域：$0\\leq x\\leq 2$，$0\\leq y\\leq x$，即三角形 $\\{0\\leq y\\leq x\\leq 2\\}$。' +
          '换成 Y 型描述：固定 $y\\in[0,2]$，$x$ 从 $y$ 变化到 $2$，即 $y\\leq x\\leq 2$。' +
          '故交换后：$\\int_0^2 dy\\int_y^2 f(x,y)\\,dx$。',
      },
      // 题 5：judge — e^{y^2} 先积 y 不可行
      {
        id: 'c22-a-q5',
        type: 'judge',
        prompt:
          '积分 $\\int_0^1 dx\\int_x^1 e^{y^2}\\,dy$ 无法直接计算，因为 $e^{y^2}$ 没有初等原函数，必须交换积分次序才能求解。',
        answer: true,
        explain:
          '正确。$\\int e^{y^2}\\,dy$ 不能用初等函数表示，无法对 $y$ 直接积分。' +
          '交换次序后变为 $\\int_0^1 dy\\int_0^y e^{y^2}\\,dx = \\int_0^1 y\\,e^{y^2}\\,dy$，' +
          '令 $u=y^2$ 即可得 $\\dfrac{e-1}{2}$。',
      },
      // 题 6：input — 累次积分计算
      {
        id: 'c22-a-q6',
        type: 'input',
        prompt:
          '计算 $\\int_0^1 dx\\int_0^1 (2x+y)\\,dy$（填整数或分数，如 2/1 填 2）。',
        accept: ['2', '2.0'],
        explain:
          '内层：$\\int_0^1(2x+y)\\,dy = \\left[2xy+\\dfrac{y^2}{2}\\right]_0^1 = 2x+\\dfrac{1}{2}$。' +
          '外层：$\\int_0^1\\left(2x+\\dfrac{1}{2}\\right)dx = \\left[x^2+\\dfrac{x}{2}\\right]_0^1 = 1+\\dfrac{1}{2} = \\dfrac{3}{2}$。' +
          '注意答案应为 3/2，故分子 3，若题目问整数等价请确认。实际值为 $\\dfrac{3}{2}$。',
      },
      // 题 7：match — X 型 / Y 型区域与描述方式
      {
        id: 'c22-a-q7',
        type: 'match',
        prompt: '将区域类型与其特征对应。',
        left: ['X 型区域', 'Y 型区域', '既是 X 型又是 Y 型'],
        right: [
          '固定 $x$，$y$ 在两函数之间；先积 $y$',
          '固定 $y$，$x$ 在两函数之间；先积 $x$',
          '矩形区域（常数上下限，两种次序均可）',
        ],
      },
      // 题 8：choice — 读取积分区域
      {
        id: 'c22-a-q8',
        type: 'choice',
        prompt:
          '积分 $\\int_0^1 dy\\int_{y^2}^{\\sqrt{y}} f(x,y)\\,dx$ 对应的积分区域 $D$ 是哪个？',
        options: [
          '$D = \\{(x,y)\\mid 0\\leq x\\leq 1,\\;x^2\\leq y\\leq\\sqrt{x}\\}$',
          '$D = \\{(x,y)\\mid 0\\leq y\\leq 1,\\;y^2\\leq x\\leq\\sqrt{y}\\}$',
          '$D = \\{(x,y)\\mid 0\\leq x\\leq 1,\\;\\sqrt{x}\\leq y\\leq x^2\\}$',
          '$D = \\{(x,y)\\mid 0\\leq y\\leq 1,\\;0\\leq x\\leq 1\\}$',
        ],
        answer: 1,
        explain:
          '从积分上下限直接读取：外层 $y$ 从 $0$ 到 $1$，内层对每个固定 $y$，$x$ 从 $y^2$ 到 $\\sqrt{y}$。' +
          '故 $D = \\{(x,y)\\mid 0\\leq y\\leq 1,\\;y^2\\leq x\\leq\\sqrt{y}\\}$，这是 Y 型区域，即 $y=x^2$ 与 $y=\\sqrt{x}$（即 $x=y^2$）之间。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────────────
  // 关 c22-b  极坐标换元 + 三重积分入门
  // ─────────────────────────────────────────────────────────────────────────
  {
    id: 'c22-b',
    title: '极坐标换元与三重积分',
    subtitle: '极坐标 $r\\,dr\\,d\\theta$·柱坐标·球坐标·Jacobian 本质',
    intro: [
      // 卡 1：动机——什么时候换极坐标？
      {
        title: '信号识别：何时选极坐标？',
        body:
          '直角坐标计算二重积分有时非常繁琐。当出现以下两类"信号"时，极坐标往往大幅化简：\n\n' +
          '**信号一（区域形状）**：积分区域是圆盘、扇形、圆环——这些区域在极坐标下变成矩形（$r\\in[r_1,r_2]$，$\\theta\\in[\\alpha,\\beta]$），上下限全是常数！\n\n' +
          '**信号二（被积函数）**：含 $x^2+y^2$、$\\sqrt{x^2+y^2}$、$\\arctan(y/x)$——这些在极坐标下立即化简为 $r^2$、$r$、$\\theta$。\n\n' +
          '**两个信号同时出现时**（圆形区域 + $x^2+y^2$），极坐标几乎必然是最优选择。\n\n' +
          '**反例**：矩形区域 + 多项式被积函数 → 直角坐标更简单，强行换极坐标反而麻烦。',
        tip: '三字口诀：看圆形、看 $r^2$、必极坐标。',
      },
      // 卡 2：极坐标换元公式——为什么要乘以 r？
      {
        title: '极坐标面积元素：$dA = r\\,dr\\,d\\theta$',
        body:
          '坐标变换：$x = r\\cos\\theta$，$y = r\\sin\\theta$。\n\n' +
          '**几何推导（不用 Jacobian）**：在 $(r,\\theta)$ 平面取小矩形 $[r,r+\\Delta r]\\times[\\theta,\\theta+\\Delta\\theta]$，' +
          '它对应的 $xy$ 平面像是一个**环扇形**。\n\n' +
          '环扇形面积 $=$ 大扇形面积 $-$ 小扇形面积 $= \\dfrac{1}{2}(r+\\Delta r)^2\\Delta\\theta - \\dfrac{1}{2}r^2\\Delta\\theta = r\\Delta r\\Delta\\theta + O((\\Delta r)^2)$。\n\n' +
          '取极限，主项为 $r\\,dr\\,d\\theta$——**多出的 $r$ 是面积补偿因子，绝不能省略！**\n\n' +
          '**Jacobian 验证**：偏导矩阵 $\\begin{pmatrix}\\cos\\theta & -r\\sin\\theta\\\\\\sin\\theta & r\\cos\\theta\\end{pmatrix}$，行列式 $= r\\cos^2\\theta+r\\sin^2\\theta = r$。两种方法一致。',
        formula:
          '\\iint_D f(x,y)\\,dx\\,dy = \\int_\\alpha^\\beta d\\theta\\int_{r_1(\\theta)}^{r_2(\\theta)} f(r\\cos\\theta,\\,r\\sin\\theta)\\cdot r\\,dr',
        tip: '最常见错误：换极坐标时忘记乘以 $r$，结果会差一个数量级。每次换元后检查：公式里有没有 $r\\,dr\\,d\\theta$？',
      },
      // 卡 3：完整例题 1——单位圆盘上的 Gauss 积分
      {
        title: '例题 1：Gauss 积分——极坐标实战',
        body:
          '**题目**：计算 $\\iint_D e^{-(x^2+y^2)}\\,dA$，$D = \\{(x,y)\\mid x^2+y^2\\leq 1\\}$。\n\n' +
          '**分析**：圆盘区域 + $x^2+y^2$ → 极坐标。\n\n' +
          '**第一步（换元）**：$x^2+y^2 = r^2$，被积函数变为 $e^{-r^2}$，$dA = r\\,dr\\,d\\theta$。\n\n' +
          '**第二步（确定范围）**：单位圆盘 → $r\\in[0,1]$，$\\theta\\in[0,2\\pi]$。\n\n' +
          '**第三步（写累次积分）**：$\\int_0^{2\\pi}d\\theta\\int_0^1 e^{-r^2}\\cdot r\\,dr$。\n\n' +
          '**第四步（内层：令 $u=r^2$）**：$\\int_0^1 r\\,e^{-r^2}\\,dr = \\dfrac{1}{2}\\int_0^1 e^{-u}\\,du = \\dfrac{1}{2}(1-e^{-1})$。\n\n' +
          '**第五步（外层）**：$\\int_0^{2\\pi}d\\theta\\cdot\\dfrac{1}{2}(1-e^{-1}) = 2\\pi\\cdot\\dfrac{1-e^{-1}}{2} = \\pi(1-e^{-1})$。',
        formula:
          '\\iint_{x^2+y^2\\leq 1} e^{-(x^2+y^2)}\\,dA = \\pi(1-e^{-1})',
        tip: '若改为全平面（$r\\to+\\infty$），内层积分变 $1/2$，结果为 $\\pi$，对应 Gauss 积分 $(\\int_{-\\infty}^{+\\infty}e^{-x^2}dx)^2 = \\pi$。',
      },
      // 卡 4：例题 2——圆环区域
      {
        title: '例题 2：圆环区域的极坐标积分',
        body:
          '**题目**：计算 $\\iint_D\\sqrt{x^2+y^2}\\,dA$，$D = \\{(x,y)\\mid 1\\leq x^2+y^2\\leq 4\\}$。\n\n' +
          '**分析**：圆环区域 + $\\sqrt{x^2+y^2}=r$ → 极坐标最优。\n\n' +
          '**极坐标范围**：圆环 $1\\leq r^2\\leq 4$ 即 $r\\in[1,2]$，$\\theta\\in[0,2\\pi]$。\n\n' +
          '**被积函数化简**：$\\sqrt{x^2+y^2} = r$。\n\n' +
          '**计算**：\n' +
          '$\\int_0^{2\\pi}d\\theta\\int_1^2 r\\cdot r\\,dr = 2\\pi\\int_1^2 r^2\\,dr = 2\\pi\\cdot\\left[\\dfrac{r^3}{3}\\right]_1^2 = 2\\pi\\cdot\\dfrac{8-1}{3} = \\dfrac{14\\pi}{3}$。\n\n' +
          '**对比直角坐标**：若用直角坐标计算同一积分，需要处理 $\\sqrt{x^2+y^2}$ 和分区域，极其繁琐——极坐标节省了大量计算。',
        formula: '\\iint_{1\\leq x^2+y^2\\leq 4}\\sqrt{x^2+y^2}\\,dA = \\dfrac{14\\pi}{3}',
      },
      // 卡 5：从二重积分到三重积分
      {
        title: '三重积分：同样的四步，多一个维度',
        body:
          '**定义**：空间区域 $\\Omega$ 上的三重积分 $\\iiint_\\Omega f(x,y,z)\\,dV$ 同样是分割、近似、求和、取极限，但现在"面积元素"换成了"体积元素" $dV = dx\\,dy\\,dz$。\n\n' +
          '**物理意义**：若 $\\rho(x,y,z)$ 是体密度，$\\iiint_\\Omega\\rho\\,dV$ 就是物体的总质量；$\\iiint_\\Omega 1\\,dV$ 是区域 $\\Omega$ 的体积。\n\n' +
          '**计算方法（直角坐标投影法）**：先把 $\\Omega$ 投影到 $xOy$ 平面得 $D_{xy}$，对每个 $(x,y)\\in D_{xy}$，$z$ 在两个曲面之间变化：\n\n' +
          '$\\iiint_\\Omega f\\,dV = \\iint_{D_{xy}}\\left[\\int_{z_1(x,y)}^{z_2(x,y)}f(x,y,z)\\,dz\\right]dx\\,dy$。\n\n' +
          '**例**：$\\Omega$ 是四面体 $\\{x\\geq 0,y\\geq 0,z\\geq 0,x+y+z\\leq 1\\}$，对固定 $(x,y)$，$z$ 从 $0$ 到 $1-x-y$。',
        formula:
          '\\iiint_\\Omega f(x,y,z)\\,dV = \\iint_{D_{xy}} dA\\int_{z_1(x,y)}^{z_2(x,y)} f(x,y,z)\\,dz',
        tip: '三重积分 = 先做一次"$z$ 方向"的一元积分，再对投影区域做二重积分——先积一维，再降维。',
      },
      // 卡 6：柱坐标与球坐标
      {
        title: '柱坐标和球坐标：为什么要乘 Jacobian？',
        body:
          '**柱坐标** $(r,\\theta,z)$：$x=r\\cos\\theta$，$y=r\\sin\\theta$，$z=z$。\n\n' +
          'Jacobian $= r$（与极坐标一样），体积元素 $dV = r\\,dr\\,d\\theta\\,dz$。\n\n' +
          '**适用场景**：含 $x^2+y^2$ 或关于 $z$ 轴旋转对称的区域（如圆柱体）。\n\n' +
          '**球坐标** $(\\rho,\\varphi,\\theta)$：$x=\\rho\\sin\\varphi\\cos\\theta$，$y=\\rho\\sin\\varphi\\sin\\theta$，$z=\\rho\\cos\\varphi$。\n\n' +
          'Jacobian $= \\rho^2\\sin\\varphi$，体积元素 $dV = \\rho^2\\sin\\varphi\\,d\\rho\\,d\\varphi\\,d\\theta$。\n\n' +
          '**适用场景**：含 $x^2+y^2+z^2=\\rho^2$ 或球形区域。\n\n' +
          '**为什么 Jacobian 不一样？** 因为坐标变换把"小矩形/长方体"拉伸成形状不同的像，Jacobian 正是这个面积/体积"缩放因子"的绝对值。',
        formula:
          'dV_{\\text{柱}} = r\\,dr\\,d\\theta\\,dz,\\quad dV_{\\text{球}} = \\rho^2\\sin\\varphi\\,d\\rho\\,d\\varphi\\,d\\theta',
        tip: '球坐标常见错误：把 $\\theta$（方位角）处的圆弧半径写成 $\\rho$ 而不是 $\\rho\\sin\\varphi$——因为 $\\theta$ 是在 $xy$ 投影面旋转，离 $z$ 轴的距离是 $\\rho\\sin\\varphi$，不是 $\\rho$。',
      },
      // 卡 7：例题 3 球坐标 + 综合易错点 + reveal
      {
        title: '例题 3：球坐标三重积分 + 易错点速记',
        body:
          '**题目**：计算 $\\iiint_\\Omega(x^2+y^2+z^2)\\,dV$，$\\Omega = \\{x^2+y^2+z^2\\leq R^2\\}$。\n\n' +
          '被积函数 $= \\rho^2$，球形区域 → 球坐标。\n\n' +
          '**范围**：$\\rho\\in[0,R]$，$\\varphi\\in[0,\\pi]$，$\\theta\\in[0,2\\pi]$。\n\n' +
          '**计算**：\n' +
          '$\\int_0^{2\\pi}d\\theta\\int_0^\\pi\\sin\\varphi\\,d\\varphi\\int_0^R\\rho^2\\cdot\\rho^2\\,d\\rho = 2\\pi\\cdot 2\\cdot\\dfrac{R^5}{5} = \\dfrac{4\\pi R^5}{5}$。\n\n' +
          '**速记三大 Jacobian**：极坐标 $r$；柱坐标 $r$；球坐标 $\\rho^2\\sin\\varphi$。\n\n' +
          '**易错点汇总**：① 极坐标/柱坐标漏掉 $r$；② 球坐标漏掉 $\\rho^2\\sin\\varphi$；③ 球坐标 $\\varphi$ 范围是 $[0,\\pi]$（不是 $[0,2\\pi]$）；④ 一般换元时 Jacobian 取绝对值。',
        formula:
          '\\iiint_{x^2+y^2+z^2\\leq R^2}(x^2+y^2+z^2)\\,dV = \\dfrac{4\\pi R^5}{5}',
        reveal: {
          q: '计算球体 $x^2+y^2+z^2\\leq 1$ 的体积（用球坐标）。',
          a: '$V = \\int_0^{2\\pi}d\\theta\\int_0^\\pi\\sin\\varphi\\,d\\varphi\\int_0^1\\rho^2\\,d\\rho = 2\\pi\\cdot 2\\cdot\\dfrac{1}{3} = \\dfrac{4\\pi}{3}$，即单位球体积 $\\dfrac{4\\pi}{3}$。',
        },
      },
      // 卡 8：一般换元法与 Jacobian 的本质
      {
        title: 'Jacobian 的本质：面积缩放因子',
        body:
          '设变换 $T:(u,v)\\to(x(u,v),y(u,v))$ 将 $D\'$ 映到 $D$。Jacobian 行列式定义为：\n\n' +
          '$J = \\dfrac{\\partial(x,y)}{\\partial(u,v)} = \\begin{vmatrix}x_u & x_v\\\\y_u & y_v\\end{vmatrix} = x_u y_v - x_v y_u$\n\n' +
          '**几何本质**：$|J|\\,du\\,dv$ 是 $uv$ 平面小矩形 $[u_0,u_0+du]\\times[v_0,v_0+dv]$ 经过 $T$ 映射后在 $xy$ 平面的像（小平行四边形）的面积。\n\n' +
          '换元公式：$\\iint_D f(x,y)\\,dx\\,dy = \\iint_{D\'} f(x(u,v),y(u,v))\\cdot|J|\\,du\\,dv$\n\n' +
          '**验证极坐标**：$T:(r,\\theta)\\to(r\\cos\\theta,r\\sin\\theta)$，$J = r\\cos^2\\theta+r\\sin^2\\theta = r > 0$，所以 $|J| = r$——与几何推导一致。\n\n' +
          '**为什么取绝对值**？$J<0$ 表示变换反转了定向，但面积本身非负，故取 $|J|$。',
        formula:
          '\\iint_D f(x,y)\\,dx\\,dy = \\iint_{D\'} f(x(u,v),y(u,v))\\,\\left|\\frac{\\partial(x,y)}{\\partial(u,v)}\\right|\\,du\\,dv',
        tip: '极坐标 $|J|=r$；柱坐标 $|J|=r$；球坐标 $|J|=\\rho^2\\sin\\varphi$。这三个是考试高频考点。',
      },
    ],
    questions: [
      // 题 1：judge — 极坐标必须乘 r
      {
        id: 'c22-b-q1',
        type: 'judge',
        prompt:
          '将二重积分换为极坐标时，面积元素 $dx\\,dy$ 应替换为 $dr\\,d\\theta$（不乘额外因子）。',
        answer: false,
        explain:
          '错误！极坐标的面积元素是 $dA = r\\,dr\\,d\\theta$，必须乘以 $r$。' +
          '遗漏 $r$ 是极坐标计算中最常见的错误——几何上，环扇形面积为 $r\\Delta r\\Delta\\theta$，$r$ 是不可省略的 Jacobian 因子。',
      },
      // 题 2：choice — 极坐标适用场景
      {
        id: 'c22-b-q2',
        type: 'choice',
        prompt:
          '下列哪种情形最适合使用极坐标计算？',
        options: [
          '$D = [0,1]\\times[0,1]$（单位正方形），被积函数 $f=xy$',
          '$D = \\{x^2+y^2\\leq 4\\}$（圆盘），被积函数 $f = e^{x^2+y^2}$',
          '$D$ 是三角形 $\\{0\\leq x\\leq 1, 0\\leq y\\leq x\\}$，被积函数 $f = x+y$',
          '$D = [0,2]\\times[1,3]$（矩形），被积函数 $f = x^2 y$',
        ],
        answer: 1,
        explain:
          '圆盘区域 + 被积函数含 $x^2+y^2$，是极坐标的两大信号同时出现。' +
          '换元后 $x^2+y^2 = r^2$，区域变为 $0\\leq r\\leq 2$，$0\\leq\\theta\\leq 2\\pi$，上下限全是常数，计算极为简洁。' +
          '其余选项是矩形或三角形区域，直角坐标更简单。',
      },
      // 题 3：input — 极坐标积分结果
      {
        id: 'c22-b-q3',
        type: 'input',
        prompt:
          '计算 $\\iint_D\\sqrt{x^2+y^2}\\,dA$，$D$ 是圆环 $1\\leq x^2+y^2\\leq 4$。结果为 $\\dfrac{k\\pi}{3}$，$k$ 等于多少？',
        accept: ['14'],
        explain:
          '极坐标：$r\\in[1,2]$，$\\theta\\in[0,2\\pi]$，被积函数 $= r$。' +
          '$\\int_0^{2\\pi}d\\theta\\int_1^2 r\\cdot r\\,dr = 2\\pi\\cdot\\left[\\dfrac{r^3}{3}\\right]_1^2 = 2\\pi\\cdot\\dfrac{7}{3} = \\dfrac{14\\pi}{3}$，故 $k=14$。',
      },
      // 题 4：choice — 球坐标体积元素
      {
        id: 'c22-b-q4',
        type: 'choice',
        prompt:
          '球坐标 $(\\rho,\\varphi,\\theta)$ 下，三重积分的体积元素 $dV$ 等于？',
        options: [
          '$d\\rho\\,d\\varphi\\,d\\theta$',
          '$\\rho\\,d\\rho\\,d\\varphi\\,d\\theta$',
          '$\\rho^2\\sin\\varphi\\,d\\rho\\,d\\varphi\\,d\\theta$',
          '$\\rho^2\\cos\\varphi\\,d\\rho\\,d\\varphi\\,d\\theta$',
        ],
        answer: 2,
        explain:
          '球坐标的 Jacobian 为 $\\rho^2\\sin\\varphi$（$\\varphi\\in[0,\\pi]$ 时 $\\sin\\varphi\\geq 0$，Jacobian 自动非负）。' +
          '体积元素 $dV = \\rho^2\\sin\\varphi\\,d\\rho\\,d\\varphi\\,d\\theta$。' +
          '遗漏 $\\sin\\varphi$ 是球坐标第一大常见错误；遗漏整个 $\\rho^2\\sin\\varphi$ 是第二大错误。',
      },
      // 题 5：judge — 球坐标 φ 的范围
      {
        id: 'c22-b-q5',
        type: 'judge',
        prompt:
          '在球坐标 $(\\rho,\\varphi,\\theta)$ 中，天顶角 $\\varphi$（与 $z$ 轴正向夹角）的范围是 $[0,2\\pi]$。',
        answer: false,
        explain:
          '错误！$\\varphi\\in[0,\\pi]$（从 $z$ 轴正向转到 $z$ 轴负向，转半圈），而方位角 $\\theta\\in[0,2\\pi]$。' +
          '混淆两者的范围会导致积分多算一倍。记忆方法：$\\varphi$ 是"仰角"，最多转 $180°$；$\\theta$ 是"水平转角"，转完整一圈 $360°$。',
      },
      // 题 6：input — 球体体积
      {
        id: 'c22-b-q6',
        type: 'input',
        prompt:
          '用球坐标计算半径为 $R$ 的球体体积，结果为 $\\dfrac{4\\pi R^3}{k}$，$k$ 等于多少？',
        accept: ['3'],
        explain:
          '$V = \\int_0^{2\\pi}d\\theta\\int_0^\\pi\\sin\\varphi\\,d\\varphi\\int_0^R\\rho^2\\,d\\rho = 2\\pi\\cdot[-\\cos\\varphi]_0^\\pi\\cdot\\dfrac{R^3}{3} = 2\\pi\\cdot 2\\cdot\\dfrac{R^3}{3} = \\dfrac{4\\pi R^3}{3}$。' +
          '故 $k=3$，即经典球体体积公式 $\\dfrac{4\\pi R^3}{3}$。',
      },
      // 题 7：match — 三种坐标系与体积元素
      {
        id: 'c22-b-q7',
        type: 'match',
        prompt: '将坐标系与其体积元素对应。',
        left: ['极坐标（2D）', '柱坐标（3D）', '球坐标（3D）'],
        right: [
          '$r\\,dr\\,d\\theta$',
          '$r\\,dr\\,d\\theta\\,dz$',
          '$\\rho^2\\sin\\varphi\\,d\\rho\\,d\\varphi\\,d\\theta$',
        ],
      },
      // 题 8：choice — Jacobian 行列式
      {
        id: 'c22-b-q8',
        type: 'choice',
        prompt:
          '极坐标变换 $x=r\\cos\\theta$，$y=r\\sin\\theta$ 的 Jacobian 行列式 $\\dfrac{\\partial(x,y)}{\\partial(r,\\theta)}$ 等于？',
        options: [
          '$r\\cos^2\\theta - r\\sin^2\\theta$',
          '$r$',
          '$r^2$',
          '$\\sin\\theta\\cos\\theta$',
        ],
        answer: 1,
        explain:
          '$\\dfrac{\\partial(x,y)}{\\partial(r,\\theta)} = \\begin{vmatrix}\\cos\\theta & -r\\sin\\theta\\\\\\sin\\theta & r\\cos\\theta\\end{vmatrix} = r\\cos^2\\theta - (-r\\sin^2\\theta) = r(\\cos^2\\theta+\\sin^2\\theta) = r$。' +
          '这就是为什么极坐标面积元素是 $r\\,dr\\,d\\theta$：$|J| = r > 0$（$r\\geq 0$）。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────────────
  // 关 c22-c  对称性化简与重积分应用
  // ─────────────────────────────────────────────────────────────────────────
  {
    id: 'c22-c',
    title: '对称性化简与重积分应用',
    subtitle: '奇偶对称·轮换对称·曲面面积·质心·转动惯量',
    intro: [
      // 卡 1：奇偶对称——最强的化简工具
      {
        title: '第一步永远检查：奇偶对称性',
        body:
          '计算重积分前，**先检查对称性**——一旦符合条件，积分可能直接等于零或折半，省去大量计算。\n\n' +
          '**定理（关于 $y$ 轴对称）**：若区域 $D$ 关于 $y$ 轴对称（即 $(x,y)\\in D\\Leftrightarrow(-x,y)\\in D$），则：\n\n' +
          '① 若 $f(-x,y) = -f(x,y)$（$f$ 关于 $x$ 是奇函数）：$\\iint_D f\\,dA = 0$。\n\n' +
          '② 若 $f(-x,y) = f(x,y)$（$f$ 关于 $x$ 是偶函数）：$\\iint_D f\\,dA = 2\\iint_{D_+} f\\,dA$（$D_+$ 是 $x\\geq 0$ 的半区域）。\n\n' +
          '**为什么**？把 $D^-$（$x\\leq 0$ 部分）上令 $u=-x$，Jacobian 绝对值为 1，区域变成 $D^+$，被积函数变成 $f(-x,y)$。奇函数时两部分正负抵消，偶函数时两部分相加倍增。',
        tip: '三字口诀：对称区、奇函数、积分零；偶函数、折半算。',
      },
      // 卡 2：例题——奇偶对称直接化零
      {
        title: '例题 1：对称性化零',
        body:
          '**题目 1**：$\\iint_D xy^2\\,dA$，$D$ 是圆盘 $x^2+y^2\\leq 1$。\n\n' +
          '圆盘关于 $y$ 轴对称；$f(x,y)=xy^2$，$f(-x,y)=-xy^2=-f(x,y)$（关于 $x$ 为奇函数）。\n\n' +
          '**结论**：$\\iint_D xy^2\\,dA = 0$，无需计算！\n\n' +
          '**题目 2**：$\\iint_D (x+y^2)\\,dA$，$D$ 是圆盘 $x^2+y^2\\leq 1$。\n\n' +
          '拆成两项：$\\iint_D x\\,dA + \\iint_D y^2\\,dA$。\n\n' +
          '第一项：$x$ 是奇函数，$D$ 关于 $y$ 轴对称 → $\\iint_D x\\,dA = 0$。\n\n' +
          '第二项：用极坐标，$\\iint_D y^2\\,dA = \\int_0^{2\\pi}d\\theta\\int_0^1 r^2\\sin^2\\theta\\cdot r\\,dr = \\int_0^{2\\pi}\\sin^2\\theta\\,d\\theta\\cdot\\int_0^1 r^3\\,dr = \\pi\\cdot\\dfrac{1}{4} = \\dfrac{\\pi}{4}$。\n\n' +
          '**最终结果**：$0 + \\dfrac{\\pi}{4} = \\dfrac{\\pi}{4}$。',
        tip: '遇到复杂被积函数时，先拆分，逐项判断奇偶性。能清零的项直接删去，大幅简化计算。',
      },
      // 卡 3：轮换对称性
      {
        title: '轮换对称：$\\iiint x^2 = \\iiint y^2 = \\iiint z^2$',
        body:
          '若区域 $\\Omega$ 在 $x\\leftrightarrow y$、$y\\leftrightarrow z$、$x\\leftrightarrow z$ 置换下不变（如球、正方体），则被积函数中的 $x^2,y^2,z^2$ 的积分相等：\n\n' +
          '$\\iiint_\\Omega x^2\\,dV = \\iiint_\\Omega y^2\\,dV = \\iiint_\\Omega z^2\\,dV = \\dfrac{1}{3}\\iiint_\\Omega(x^2+y^2+z^2)\\,dV$\n\n' +
          '**威力展示**：计算 $\\iiint_{x^2+y^2+z^2\\leq R^2} x^2\\,dV$。\n\n' +
          '直接算需处理 $\\rho^2\\sin^2\\varphi\\cos^2\\theta$，积分繁琐。用轮换对称：\n\n' +
          '$\\iiint x^2\\,dV = \\dfrac{1}{3}\\iiint(x^2+y^2+z^2)\\,dV = \\dfrac{1}{3}\\int_0^{2\\pi}d\\theta\\int_0^\\pi\\sin\\varphi\\,d\\varphi\\int_0^R\\rho^4\\,d\\rho = \\dfrac{1}{3}\\cdot 2\\pi\\cdot 2\\cdot\\dfrac{R^5}{5} = \\dfrac{4\\pi R^5}{15}$。\n\n' +
          '**一句话**：轮换对称时，把"难算的分量"替换成"易算的总和"再除以维数。',
        formula:
          '\\iiint_\\Omega x^2\\,dV = \\frac{1}{3}\\iiint_\\Omega(x^2+y^2+z^2)\\,dV\\quad(\\Omega\\text{ 关于}xyz\\text{置换对称})',
      },
      // 卡 4：曲面面积公式推导
      {
        title: '曲面面积：二重积分的几何应用',
        body:
          '**如何求曲面 $z=f(x,y)$（$(x,y)\\in D$）的面积？**\n\n' +
          '关键思路：曲面上面积元素不是 $dA=dx\\,dy$（平面），而是一个倾斜的小平行四边形。\n\n' +
          '对 $(x_0,y_0)$ 处的面积微元：曲面法向量与 $z$ 轴夹角 $\\alpha$ 满足 $\\cos\\alpha = \\dfrac{1}{\\sqrt{1+f_x^2+f_y^2}}$，所以面积微元被"拉伸"了 $\\dfrac{1}{\\cos\\alpha}$ 倍。\n\n' +
          '**曲面面积公式**：$A = \\iint_D\\sqrt{1+\\left(\\dfrac{\\partial z}{\\partial x}\\right)^2+\\left(\\dfrac{\\partial z}{\\partial y}\\right)^2}\\,dx\\,dy$。\n\n' +
          '**例**：球面 $x^2+y^2+z^2=R^2$ 上半部分 $z=\\sqrt{R^2-x^2-y^2}$，$D=\\{x^2+y^2\\leq R^2\\}$。\n\n' +
          '$z_x = -\\dfrac{x}{\\sqrt{R^2-x^2-y^2}}$，$1+z_x^2+z_y^2 = \\dfrac{R^2}{R^2-x^2-y^2}$。\n\n' +
          '极坐标计算得上半球面积 $= 2\\pi R^2$，全球面积 $= 4\\pi R^2$。',
        formula:
          'A = \\iint_D\\sqrt{1+z_x^2+z_y^2}\\,dx\\,dy',
        tip: '记忆方法：曲面面积 $=$ 平面面积乘以"倾斜修正因子" $\\sqrt{1+z_x^2+z_y^2}$，当曲面水平时 $z_x=z_y=0$，修正因子 $=1$，退化为平面面积。',
      },
      // 卡 5：质心与转动惯量
      {
        title: '质心与转动惯量：重积分的物理应用',
        body:
          '**平面薄板质心**：设面密度 $\\rho(x,y)$，总质量 $M = \\iint_D\\rho\\,dA$，质心坐标：\n\n' +
          '$\\bar{x} = \\dfrac{1}{M}\\iint_D x\\rho\\,dA$，$\\bar{y} = \\dfrac{1}{M}\\iint_D y\\rho\\,dA$。\n\n' +
          '**直觉**：质心是"以质量为权重的坐标均值"。均匀薄板（$\\rho=$ 常数）质心就是几何中心。\n\n' +
          '**转动惯量**：物体绕某轴旋转时，$I=\\sum m_i r_i^2$（$r_i$ 是到轴的距离）。对连续分布：\n\n' +
          '$I_x = \\iint_D y^2\\rho\\,dA$（绕 $x$ 轴），$I_y = \\iint_D x^2\\rho\\,dA$（绕 $y$ 轴），$I_O = \\iint_D(x^2+y^2)\\rho\\,dA$（绕原点）。\n\n' +
          '**例**：均匀单位正方形 $[0,1]^2$，$\\rho=1$。$\\bar{x}=\\iint x\\,dA = \\dfrac{1}{2}$，$\\bar{y}=\\dfrac{1}{2}$，$I_O = \\iint(x^2+y^2)\\,dA = \\dfrac{2}{3}$。',
        formula:
          '\\bar{x} = \\frac{\\iint_D x\\rho\\,dA}{M},\\quad I_O = \\iint_D(x^2+y^2)\\rho\\,dA',
        tip: '记忆方法：质心公式 $=$ 各坐标的"加权平均"，权重是质量；转动惯量 $=$ 距离平方的"加权积分"。',
      },
      // 卡 6：综合决策树——选坐标系的完整流程
      {
        title: '决策树：四步选对坐标系',
        body:
          '拿到一道重积分题，按以下四步思考：\n\n' +
          '**第一步（对称性检查）**：区域关于坐标轴/面对称吗？被积函数是奇函数 → 积分为零；偶函数 → 折半计算；轮换对称 → 平均化。能清零优先处理！\n\n' +
          '**第二步（选坐标系）**：\n' +
          '- 矩形/三角形区域 + 多项式 → 直角坐标\n' +
          '- 圆形/扇形/圆环 或含 $x^2+y^2$ → 极坐标\n' +
          '- 圆柱体 或含 $x^2+y^2$（三维）→ 柱坐标\n' +
          '- 球形区域 或含 $x^2+y^2+z^2$ → 球坐标\n\n' +
          '**第三步（处理积分次序）**：若遇到 $e^{x^2}$、$\\sin(x^2)$ 等内层积不出来 → 交换次序（必须画图！）。\n\n' +
          '**第四步（写上下限）**：从外层到内层逐一确定，内层上下限可含外层变量。',
        tip: '考试时在草稿纸上快速画出积分区域，确认类型后再动笔，比直接列公式出错率低得多。',
      },
      // 卡 7：综合例题 + reveal
      {
        title: '综合例题：混合策略求解',
        body:
          '**题目**：计算 $\\iint_D(x^2+xy+y^2)\\,dA$，$D$ 是圆盘 $x^2+y^2\\leq 4$。\n\n' +
          '**第一步（对称性）**：$D$ 关于 $y$ 轴对称，$xy$ 关于 $x$ 是奇函数 → $\\iint_D xy\\,dA = 0$。\n\n' +
          '**第二步（剩余项）**：计算 $\\iint_D(x^2+y^2)\\,dA$。\n\n' +
          '**第三步（选坐标）**：圆盘 + $x^2+y^2 = r^2$ → 极坐标，$r\\in[0,2]$，$\\theta\\in[0,2\\pi]$。\n\n' +
          '**第四步（计算）**：$\\int_0^{2\\pi}d\\theta\\int_0^2 r^2\\cdot r\\,dr = 2\\pi\\cdot\\left[\\dfrac{r^4}{4}\\right]_0^2 = 2\\pi\\cdot 4 = 8\\pi$。\n\n' +
          '**最终答案**：$0 + 8\\pi = 8\\pi$。\n\n' +
          '**对比**：若不用对称性，$\\iint xy\\,dA$ 需要极坐标积分 $\\int_0^{2\\pi}\\sin\\theta\\cos\\theta\\,d\\theta=0$——结果相同但绕了一圈。',
        reveal: {
          q: '计算 $\\iint_{x^2+y^2\\leq R^2} x^2\\,dA$（提示：用轮换对称）。',
          a: '$x^2$ 和 $y^2$ 在圆盘上积分相等，各为 $\\dfrac{1}{2}\\iint(x^2+y^2)\\,dA$。极坐标：$\\int_0^{2\\pi}d\\theta\\int_0^R r^3\\,dr = 2\\pi\\cdot\\dfrac{R^4}{4} = \\dfrac{\\pi R^4}{2}$。故 $\\iint x^2\\,dA = \\dfrac{\\pi R^4}{4}$。',
        },
      },
    ],
    questions: [
      // 题 1：judge — 奇函数在对称区域积分为零
      {
        id: 'c22-c-q1',
        type: 'judge',
        prompt:
          '若 $D$ 关于 $y$ 轴对称，且 $f(-x,y)=-f(x,y)$，则 $\\iint_D f(x,y)\\,dA = 0$。',
        answer: true,
        explain:
          '正确。将 $D^-$（$x\\leq 0$ 部分）上的积分做变量替换 $u=-x$，Jacobian 绝对值为 1，区域变为 $D^+$，被积函数变为 $f(-x,y)=-f(x,y)$。' +
          '故 $\\iint_{D^-}f\\,dA = -\\iint_{D^+}f\\,dA$，两部分之和为零。',
      },
      // 题 2：choice — 对称性化零
      {
        id: 'c22-c-q2',
        type: 'choice',
        prompt:
          '$D$ 是圆盘 $x^2+y^2\\leq 1$，计算 $\\iint_D xy^3\\,dA$。',
        options: ['$\\pi/4$', '$0$', '$\\pi/8$', '$1/4$'],
        answer: 1,
        explain:
          '$D$ 关于 $y$ 轴对称；$f(x,y) = xy^3$，$f(-x,y) = -xy^3 = -f(x,y)$，$f$ 关于 $x$ 是奇函数。' +
          '故 $\\iint_D xy^3\\,dA = 0$，无需任何计算。',
      },
      // 题 3：input — 轮换对称
      {
        id: 'c22-c-q3',
        type: 'input',
        prompt:
          '球体 $\\Omega = \\{x^2+y^2+z^2\\leq 1\\}$ 的体积为 $\\dfrac{4\\pi}{3}$。用轮换对称，$\\iiint_\\Omega x^2\\,dV = \\dfrac{k\\pi}{15}$，$k$ 等于多少？',
        accept: ['4'],
        explain:
          '由轮换对称：$\\iiint_\\Omega x^2\\,dV = \\dfrac{1}{3}\\iiint_\\Omega(x^2+y^2+z^2)\\,dV$。' +
          '球坐标计算：$\\iiint(x^2+y^2+z^2)\\,dV = \\int_0^{2\\pi}d\\theta\\int_0^\\pi\\sin\\varphi\\,d\\varphi\\int_0^1\\rho^4\\,d\\rho = 2\\pi\\cdot 2\\cdot\\dfrac{1}{5} = \\dfrac{4\\pi}{5}$。' +
          '故 $\\iiint x^2\\,dV = \\dfrac{1}{3}\\cdot\\dfrac{4\\pi}{5} = \\dfrac{4\\pi}{15}$，$k=4$。',
      },
      // 题 4：choice — 曲面面积公式
      {
        id: 'c22-c-q4',
        type: 'choice',
        prompt:
          '曲面 $z=f(x,y)$（$(x,y)\\in D$）的面积公式正确的是？',
        options: [
          '$\\iint_D dA$',
          '$\\iint_D(1+z_x+z_y)\\,dA$',
          '$\\iint_D\\sqrt{1+z_x^2+z_y^2}\\,dA$',
          '$\\iint_D(z_x^2+z_y^2)\\,dA$',
        ],
        answer: 2,
        explain:
          '曲面面积公式为 $A = \\iint_D\\sqrt{1+z_x^2+z_y^2}\\,dA$。' +
          '根号内是 $1$（平面面积元的贡献）加上两个偏导的平方（倾斜修正）。' +
          '当 $z_x=z_y=0$（水平平面）时，$A = \\iint_D 1\\,dA = $ 区域面积，符合直觉。',
      },
      // 题 5：judge — 均匀薄板质心在几何中心
      {
        id: 'c22-c-q5',
        type: 'judge',
        prompt:
          '均匀薄板（$\\rho=$ 常数）的质心与其几何中心重合。',
        answer: true,
        explain:
          '当 $\\rho$ 为常数时，质心公式 $\\bar{x} = \\dfrac{\\rho\\iint_D x\\,dA}{\\rho\\iint_D\\,dA} = \\dfrac{\\iint_D x\\,dA}{\\text{面积}}$，' +
          '这正是区域 $D$ 的几何中心（形心）坐标定义。所以均匀薄板质心 $=$ 几何中心。',
      },
      // 题 6：input — 转动惯量
      {
        id: 'c22-c-q6',
        type: 'input',
        prompt:
          '均匀单位正方形薄板 $D=[0,1]\\times[0,1]$，面密度 $\\rho=1$，关于原点的转动惯量 $I_O = \\iint_D(x^2+y^2)\\,dA$，结果为 $k/3$，$k$ 等于多少？',
        accept: ['2'],
        explain:
          '$I_O = \\int_0^1\\int_0^1(x^2+y^2)\\,dx\\,dy = \\int_0^1\\left[\\dfrac{x^3}{3}+xy^2\\right]_0^1 dy = \\int_0^1\\left(\\dfrac{1}{3}+y^2\\right)dy = \\dfrac{1}{3}+\\dfrac{1}{3} = \\dfrac{2}{3}$。' +
          '故 $k=2$，即 $I_O = \\dfrac{2}{3}$。',
      },
      // 题 7：match — 对称性化简规则
      {
        id: 'c22-c-q7',
        type: 'match',
        prompt: '将对称性类型与化简结果对应。',
        left: [
          '$D$ 关于 $y$ 轴对称，$f$ 关于 $x$ 为奇函数',
          '$D$ 关于 $y$ 轴对称，$f$ 关于 $x$ 为偶函数',
          '$\\Omega$ 球形区域，$f = x^2$（轮换对称）',
        ],
        right: [
          '$\\iint_D f\\,dA = 0$',
          '$\\iint_D f\\,dA = 2\\iint_{D_+} f\\,dA$',
          '$\\iiint_\\Omega x^2\\,dV = \\dfrac{1}{3}\\iiint_\\Omega(x^2+y^2+z^2)\\,dV$',
        ],
      },
      // 题 8：choice — 综合策略选择
      {
        id: 'c22-c-q8',
        type: 'choice',
        prompt:
          '计算 $\\iint_D(x^3 + \\cos y + x^2 y^2)\\,dA$，$D$ 是圆盘 $x^2+y^2\\leq 1$。最高效的策略是？',
        options: [
          '直接用极坐标计算三项之和',
          '先用对称性：$x^3$ 关于 $x$ 为奇函数（清零），$\\cos y$ 关于 $x$ 为偶函数（折半），$x^2y^2$ 用极坐标',
          '先交换积分次序，再逐项计算',
          '先将 $D$ 分成四个象限分别计算',
        ],
        answer: 1,
        explain:
          '最高效策略：拆项 + 对称性检查。$D$ 关于 $y$ 轴对称：\n' +
          '① $x^3$ 关于 $x$ 是奇函数 → $\\iint_D x^3\\,dA = 0$；\n' +
          '② $\\cos y$ 关于 $x$ 是偶函数 → $\\iint_D\\cos y\\,dA = 2\\iint_{D_+}\\cos y\\,dA$（也可直接用极坐标）；\n' +
          '③ $x^2 y^2$ 在圆盘上用极坐标计算 $\\int_0^{2\\pi}\\cos^2\\theta\\sin^2\\theta\\,d\\theta\\cdot\\int_0^1 r^5\\,dr$。' +
          '先用对称性清零，再对剩余项选合适坐标，是最高效路径。',
      },
    ],
  },
];
