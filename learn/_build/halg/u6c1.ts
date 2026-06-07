export const LESSONS = [
  {
    id: 'u6c1',
    title: '复数的概念与几何意义',
    subtitle: '$z = a + b\\mathrm{i}$，实部 $a$，虚部 $b$，模 $|z|=\\sqrt{a^2+b^2}$，共轭 $\\overline{z}=a-b\\mathrm{i}$',
    intro: [
      {
        title: '引入：为什么要扩充到复数？',
        body: '数集的每次扩张，都是为了解决"在原有数集中无法完成"的运算：\n\n- $\\mathbb{N} \\to \\mathbb{Z}$：让减法 $a - b$（$b > a$）有解\n- $\\mathbb{Z} \\to \\mathbb{Q}$：让除法 $a \\div b$（$b \\neq 0$）有意义\n- $\\mathbb{Q} \\to \\mathbb{R}$：让 $\\sqrt{2}$、$\\pi$ 有"位置"\n- $\\mathbb{R} \\to \\mathbb{C}$：让方程 $x^2 = -1$ 有解\n\n在实数范围内 $x^2 \\geq 0$ 恒成立，所以 $x^2 + 1 = 0$ 无解。于是数学家引入一个新对象 $\\mathrm{i}$，规定 $\\mathrm{i}^2 = -1$——这就是**虚数单位**的诞生。',
        tip: '扩充数集不是"凭空想象"，而是为了保证运算的封闭性。每次扩充都使数学工具更强大。',
      },
      {
        title: '虚数单位 $\\mathrm{i}$ 与高次幂规律',
        body: '**定义**：$\\mathrm{i}$ 是满足 $\\mathrm{i}^2 = -1$ 的数，称为**虚数单位**。\n\n由 $\\mathrm{i}^2 = -1$ 出发，逐步推出：\n$$\\mathrm{i}^1 = \\mathrm{i},\\quad \\mathrm{i}^2 = -1,\\quad \\mathrm{i}^3 = -\\mathrm{i},\\quad \\mathrm{i}^4 = 1$$\n\n之后 $\\mathrm{i}^5 = \\mathrm{i}$，回到起点，**以 4 为周期循环**。\n\n**速算口诀**：计算 $\\mathrm{i}^n$ 时，$n \\div 4$ 看余数：余 0 得 1，余 1 得 $\\mathrm{i}$，余 2 得 $-1$，余 3 得 $-\\mathrm{i}$。',
        formula: '\\mathrm{i}^{4k}=1,\\quad \\mathrm{i}^{4k+1}=\\mathrm{i},\\quad \\mathrm{i}^{4k+2}=-1,\\quad \\mathrm{i}^{4k+3}=-\\mathrm{i}',
        steps: [
          '将指数 $n$ 除以 4，求余数 $r = n \\bmod 4$',
          '$r=0 \\Rightarrow \\mathrm{i}^n = 1$；$r=1 \\Rightarrow \\mathrm{i}^n = \\mathrm{i}$',
          '$r=2 \\Rightarrow \\mathrm{i}^n = -1$；$r=3 \\Rightarrow \\mathrm{i}^n = -\\mathrm{i}$',
        ],
      },
      {
        title: '复数的定义与分类',
        body: '**定义**：形如 $z = a + b\\mathrm{i}$（$a, b \\in \\mathbb{R}$）的数叫做**复数**。\n- $a$ 称为 $z$ 的**实部**，记作 $\\operatorname{Re}(z) = a$\n- $b$ 称为 $z$ 的**虚部**，记作 $\\operatorname{Im}(z) = b$（虚部是实数 $b$，不是 $b\\mathrm{i}$！）\n\n**复数的分类**：\n- $b = 0$ 时，$z = a$ 是**实数**\n- $b \\neq 0$ 时，$z$ 是**虚数**\n  - $a = 0$ 且 $b \\neq 0$ 时，$z = b\\mathrm{i}$ 是**纯虚数**\n\n数集链：$\\mathbb{N} \\subset \\mathbb{Z} \\subset \\mathbb{Q} \\subset \\mathbb{R} \\subset \\mathbb{C}$',
        tip: '虚部 $b$ 是实数！$z = 3 - 2\\mathrm{i}$ 的虚部是 $-2$，不是 $-2\\mathrm{i}$。这是高考必考细节。',
      },
      {
        title: '复数相等与解方程',
        body: '**复数相等的充要条件**：\n$$a + b\\mathrm{i} = c + d\\mathrm{i} \\iff a = c \\text{ 且 } b = d$$\n\n即**实部相等且虚部相等**。特别地：$a + b\\mathrm{i} = 0 \\iff a = 0$ 且 $b = 0$。\n\n**应用**：把一个复数方程"拆解"成两个实数方程。\n\n**例**：已知 $(m-1) + (m+2)\\mathrm{i} = 0$，求 $m$。\n\n解：实部 $m - 1 = 0$，虚部 $m + 2 = 0$，解得 $m = 1$ 和 $m = -2$，两式矛盾，故无解。',
        tip: '复数没有大小关系！在 $\\mathbb{C}$ 中只能谈"相等"或"不等"，不能写 $z_1 > z_2$（除非两者都是实数）。',
      },
      {
        title: '复平面：复数的几何意义',
        body: '把复数 $z = a + b\\mathrm{i}$ 与平面点 $Z(a, b)$ 一一对应，得到**复平面**（高斯平面）：\n- $x$ 轴：**实轴**（实数都在实轴上）\n- $y$ 轴：**虚轴**（纯虚数都在虚轴上，原点例外）\n\n有两种等价的几何对应：\n1. **点的对应**：$z = a + b\\mathrm{i} \\longleftrightarrow Z(a, b)$\n2. **向量的对应**：$z = a + b\\mathrm{i} \\longleftrightarrow \\vec{OZ} = (a, b)$\n\n向量观点把"复数加减"和"向量加减"完全对应，是后续运算几何意义的基础。',
        formula: 'z = a + b\\mathrm{i} \\longleftrightarrow Z(a,\\,b) \\longleftrightarrow \\vec{OZ}=(a,b)',
        image: 'gz-p6-01-1.svg',
        imageCaption: '复平面示意：实轴（$x$ 轴）、虚轴（$y$ 轴），点 $Z(a,b)$ 对应复数 $z=a+b\\mathrm{i}$',
      },
      {
        title: '模与共轭复数',
        body: '**模**：复数 $z = a + b\\mathrm{i}$ 的模定义为 $|z| = \\sqrt{a^2 + b^2}$。\n几何意义：点 $Z(a,b)$ 到原点 $O$ 的距离。\n\n**共轭复数**：$z = a + b\\mathrm{i}$ 的共轭定义为 $\\overline{z} = a - b\\mathrm{i}$。\n几何意义：$\\overline{z}$ 对应的点是 $Z$ 关于**实轴的对称点**。\n\n**核心恒等式**：\n$$z \\cdot \\overline{z} = (a + b\\mathrm{i})(a - b\\mathrm{i}) = a^2 + b^2 = |z|^2$$\n\n这个等式是复数除法"分母实数化"的核心。',
        formula: '|z| = \\sqrt{a^2+b^2},\\quad \\overline{z} = a - b\\mathrm{i},\\quad z\\cdot\\overline{z} = |z|^2',
      },
      {
        title: '例题精讲与易错点',
        body: '**例 1（分类求参）**：$z = (m^2 - 3m - 4) + (m^2 - 5m - 6)\\mathrm{i}$，$m \\in \\mathbb{R}$。\n- $z$ 是实数 $\\Leftrightarrow$ 虚部 $= 0$：$(m-6)(m+1) = 0$，$m = 6$ 或 $m = -1$\n- $z$ 是纯虚数 $\\Leftrightarrow$ 实部 $= 0$ 且虚部 $\\neq 0$：实部 $(m-4)(m+1)=0$ 得 $m=4$ 或 $m=-1$；$m=-1$ 时虚部也为 $0$（舍），故 $m = 4$\n\n**例 2（$\\mathrm{i}$ 的幂）**：$\\mathrm{i}+\\mathrm{i}^2+\\cdots+\\mathrm{i}^{2024}$。\n连续 4 项之和 $= 0$，$2024 = 4 \\times 506$，故总和 $= 0$。\n\n**三大易错点**：① $\\mathrm{i}^2 = -1$，不是 $1$；② 虚部是实数 $b$，不是 $b\\mathrm{i}$；③ 复数没有大小关系，但**模**可以比大小。',
        reveal: {
          q: '复数 $z = (a+1) + (a^2-1)\\mathrm{i}$（$a \\in \\mathbb{R}$）对应的点在第三象限，求 $a$ 的范围。',
          a: '第三象限要求实部 $< 0$ 且虚部 $< 0$：$a + 1 < 0$ 且 $a^2 - 1 < 0$，即 $a < -1$ 且 $-1 < a < 1$，两条件矛盾，故**不存在**这样的 $a$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '下列关于复数 $z = 3 - 2\\mathrm{i}$ 的说法，正确的是（  ）。',
        options: [
          '$z$ 的虚部是 $-2\\mathrm{i}$',
          '$z$ 的虚部是 $2$',
          '$z$ 的虚部是 $-2$',
          '$z$ 是纯虚数',
        ],
        answer: 2,
        explain: '复数 $z = a + b\\mathrm{i}$ 的虚部是实数 $b$，不是 $b\\mathrm{i}$。$z = 3 - 2\\mathrm{i}$ 中 $b = -2$，故虚部为 $-2$。',
      },
      {
        type: 'judge',
        prompt: '在复数集 $\\mathbb{C}$ 中，对于任意两个复数 $z_1$、$z_2$，都可以比较大小（即 $z_1 > z_2$ 或 $z_1 < z_2$ 或 $z_1 = z_2$ 恰好有一个成立）。',
        answer: false,
        explain: '复数集中没有大小关系，只有相等与不等之分。只有当两个复数都是实数时才能比较大小。但复数的**模**是实数，可以比大小。',
      },
      {
        type: 'choice',
        prompt: '计算 $\\mathrm{i}^{2025}$ 的结果是（  ）。',
        options: [
          '$1$',
          '$\\mathrm{i}$',
          '$-1$',
          '$-\\mathrm{i}$',
        ],
        answer: 1,
        explain: '$2025 = 4 \\times 506 + 1$，余数为 $1$，故 $\\mathrm{i}^{2025} = \\mathrm{i}^1 = \\mathrm{i}$。',
      },
      {
        type: 'input',
        prompt: '设复数 $z = (m^2 - 2m - 3) + (m^2 - 4m + 3)\\mathrm{i}$（$m \\in \\mathbb{R}$）是纯虚数，则 $m$ 等于多少？',
        accept: ['1', '-1'],
        explain: '纯虚数要求实部 $= 0$ 且虚部 $\\neq 0$。实部：$m^2 - 2m - 3 = (m-3)(m+1) = 0$，得 $m = 3$ 或 $m = -1$。验证虚部：$m = 3$ 时虚部 $= 9 - 12 + 3 = 0$（不符合）；$m = -1$ 时虚部 $= 1 + 4 + 3 = 8 \\neq 0$（符合）。故 $m = -1$。',
      },
      {
        type: 'choice',
        prompt: '复数 $z = -1 + \\sqrt{3}\\mathrm{i}$ 的模 $|z|$ 等于（  ）。',
        options: [
          '$\\sqrt{2}$',
          '$2$',
          '$\\sqrt{3}$',
          '$4$',
        ],
        answer: 1,
        explain: '$|z| = \\sqrt{(-1)^2 + (\\sqrt{3})^2} = \\sqrt{1 + 3} = \\sqrt{4} = 2$。',
      },
      {
        type: 'match',
        prompt: '将下列复数与其对应在复平面上的象限或轴匹配。',
        left: [
          '$2 + 3\\mathrm{i}$',
          '$-1 - \\mathrm{i}$',
          '$3\\mathrm{i}$',
        ],
        right: [
          '虚轴正半轴',
          '第一象限',
          '第三象限',
        ],
        explain: '$2+3\\mathrm{i}$ 对应点 $(2,3)$，在第一象限；$-1-\\mathrm{i}$ 对应点 $(-1,-1)$，在第三象限；$3\\mathrm{i}$ 对应点 $(0,3)$，在虚轴正半轴。',
      },
      {
        type: 'choice',
        prompt: '复数 $z = a + b\\mathrm{i}$ 满足 $z = \\overline{z}$，则（  ）。',
        options: [
          '$z$ 必为纯虚数',
          '$z$ 必为实数',
          '$a = 0$',
          '$b = a$',
        ],
        answer: 1,
        explain: '$z = \\overline{z}$ 即 $a + b\\mathrm{i} = a - b\\mathrm{i}$，由复数相等得 $b = -b$，故 $b = 0$，$z = a$ 是实数。',
      },
      {
        type: 'input',
        prompt: '已知 $|z| = 5$，且复数 $z$ 对应点在第二象限的角平分线上（即实部与虚部互为相反数），求 $z$（填写 $a+b\\mathrm{i}$ 形式，如 $-3+3\\mathrm{i}$）。',
        accept: ['-\\frac{5\\sqrt{2}}{2}+\\frac{5\\sqrt{2}}{2}\\mathrm{i}', '-5\\sqrt{2}/2+5\\sqrt{2}/2i', '\\frac{-5\\sqrt{2}}{2}+\\frac{5\\sqrt{2}}{2}i'],
        explain: '第二象限角平分线上的点满足 $a = -b$（实部为负，虚部为正），设 $z = -t + t\\mathrm{i}$（$t > 0$）。由 $|z|^2 = t^2 + t^2 = 25$，得 $t = \\dfrac{5\\sqrt{2}}{2}$，故 $z = -\\dfrac{5\\sqrt{2}}{2} + \\dfrac{5\\sqrt{2}}{2}\\mathrm{i}$。',
      },
    ],
  },
]
