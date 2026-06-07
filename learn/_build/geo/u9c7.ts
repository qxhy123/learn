export const LESSONS = [
  {
    id: 'u9c7',
    title: '切割线模型',
    subtitle: '切线平方等于割线两段积·圆幂定理统一视角',
    intro: [
      {
        title: '引入：一道经典题',
        body:
          '圆 $O$ 外一点 $P$，$PA$ 切圆 $O$ 于点 $A$，过 $P$ 的割线交圆 $O$ 于 $B, C$ 两点（$B$ 在 $P$ 与 $C$ 之间）。已知 $PA = 6$，$PB = 4$，求 $PC$。\n\n这道题背后藏着**切割线定理**。表面上只是"切线 + 割线"的简单配置，但它给出了一个乘积关系 $PA^2 = PB \\cdot PC$，用 $36 = 4 \\cdot PC$ 解得 $PC = 9$。',
        image: 'ex-p5-07-0.svg',
        imageCaption: '切割线引入：$PA = 6$，$PB = 4$，求 $PC = 9$',
      },
      {
        title: '思维路径：为什么 $PA^2 = PB \\cdot PC$',
        body:
          '连 $AB, AC$，构造 $\\triangle PAB$ 和 $\\triangle PCA$。\n\n要证 $PA^2 = PB \\cdot PC$，等价于 $\\dfrac{PA}{PB} = \\dfrac{PC}{PA}$，即两三角形对应边成比例——需要相似。\n\n**找相似的依据**：\n1. $\\angle P$ 是公共角（$\\angle APB = \\angle CPA$）。\n2. $\\angle PAB$ 是弦切角（切线 $PA$、弦 $AB$），由**弦切角定理**，$\\angle PAB = \\angle ACB$（所夹弧 $\\overset{\\frown}{AB}$ 的圆周角）。\n\n两对角各自相等，由 **AA** 得 $\\triangle PAB \\sim \\triangle PCA$（注意顶点对应：$P \\leftrightarrow P, A \\leftrightarrow C, B \\leftrightarrow A$）。\n\n由比例关系：$\\dfrac{PA}{PC} = \\dfrac{PB}{PA}$，交叉相乘得 $PA^2 = PB \\cdot PC$。',
        formula: 'PA^2 = PB \\cdot PC',
      },
      {
        title: '切割线模型（标准版）',
        body:
          '**图形特征**：圆外一点 $P$ + 切线 $PA$（切点 $A$）+ 割线 $PBC$（$B$ 近，$C$ 远）。\n\n**核心结论**：$PA^2 = PB \\cdot PC$。\n\n**记忆口诀**：切线**平方**等于割线**两段之积**。（"切平方，割两积"）\n\n注意：$PB$ 是从 $P$ 到**近端**的距离，$PC$ 是从 $P$ 到**远端**的距离，$BC = PC - PB$ 是弦长。',
        image: 'model-tangent-secant.svg',
        imageCaption: '切割线定理：$PA^2 = PB \\cdot PC$',
      },
      {
        title: '模型变形：割线-割线版本',
        body:
          '从圆外一点 $P$ 引**两条割线** $PBC$、$PDE$（$B, D$ 近端，$C, E$ 远端），则 $PB \\cdot PC = PD \\cdot PE$。\n\n证法同样靠 AA 相似：$\\angle P$ 公共，$\\angle PBD = \\angle PEC$（圆内接四边形对角互补 $\\Rightarrow$ 外角等于内对角）。\n\n**记忆**：两割线都等于"圆幂 $PO^2 - r^2$"，所以两个乘积相等。',
        formula: 'PB \\cdot PC = PD \\cdot PE \\quad (\\text{两割线})',
        image: 'ex-p5-07-2.svg',
        imageCaption: '两割线：$PA \\cdot PB = PC \\cdot PD$',
      },
      {
        title: '模型变形：相交弦版本（点在圆内）',
        body:
          '圆**内**一点 $P$，过 $P$ 的两条弦 $AB$ 与 $CD$ 相交，则 $PA \\cdot PB = PC \\cdot PD$。\n\n证法：同弧圆周角 $\\angle A = \\angle D$（对 $\\overset{\\frown}{BC}$）+ 对顶角 $\\Rightarrow \\triangle APC \\sim \\triangle DPB \\Rightarrow$ 比例 $\\Rightarrow$ 乘积。\n\n**与切割线的统一**：当圆内点"退化"到圆外，相交弦变为割线-割线；再让一条割线旋转到与圆相切，$B, C$ 重合，乘积退化为平方——这就是切割线公式。',
        formula: 'PA \\cdot PB = PC \\cdot PD \\quad (\\text{相交弦})',
        image: 'ex-p5-07-3.svg',
        imageCaption: '圆内相交弦：$PA \\cdot PB = PC \\cdot PD$',
      },
      {
        title: '思考路标',
        body:
          '遇到圆 + 线段长度乘积问题，按顺序快速识别：\n\n- 圆外一点 + **切线 + 割线** $\\Rightarrow$ **切割线定理** $PA^2 = PB \\cdot PC$\n- 圆外一点 + **两条割线** $\\Rightarrow$ **割线-割线定理** $PB \\cdot PC = PD \\cdot PE$\n- 圆内一点 + **两弦相交** $\\Rightarrow$ **相交弦定理** $PA \\cdot PB = PC \\cdot PD$\n- 出现 $\\text{某段}^2 = \\text{某段} \\cdot \\text{某段}$ 形式 $\\Rightarrow$ 优先考虑切割线背景\n- 想证两段乘积相等却没思路 $\\Rightarrow$ 找"圆 + 公共顶点 + 相交直线"的圆幂结构',
        tip: '切线长定理（第 6 关）和切割线定理常联合出现：前者给等长，后者给乘积。两个工具配合使用。',
      },
      {
        title: '例题：过圆心的割线',
        body:
          '$PA$ 切圆 $O$ 于 $A$，过 $P$ 的割线过圆心 $O$，交圆于 $M, N$（$M$ 近 $P$），$PA = 6$，半径 $r = 4$。求圆心距 $PO$。\n\n设 $PO = d$，则 $PM = d - r = d - 4$，$PN = d + r = d + 4$。由切割线定理：$PA^2 = PM \\cdot PN$，即 $36 = (d-4)(d+4) = d^2 - 16$。$d^2 = 52$，$PO = 2\\sqrt{13}$。这也印证了"圆幂 $= PO^2 - r^2$"的结论。',
        reveal: {
          q: '圆幂是什么？',
          a: '圆外一点 $P$ 关于圆 $O$ 的"幂"等于 $PO^2 - r^2$。无论从 $P$ 引哪条切线或割线，$PA^2$（切）或 $PB \\cdot PC$（割）都等于这个固定值，这就是**圆幂定理**的统一视角。',
        },
      },
    ],
    questions: [
      {
        type: 'input',
        prompt:
          '圆 $O$ 外一点 $P$，$PT$ 切圆于 $T$，割线交圆于 $M, N$（$M$ 近 $P$），$PM = 2$，$MN = 6$。求切线长 $PT$。',
        accept: ['4'],
        explain:
          '$PN = PM + MN = 2 + 6 = 8$。由切割线定理：$PT^2 = PM \\cdot PN = 2 \\times 8 = 16$，$PT = 4$。',
      },
      {
        type: 'choice',
        prompt:
          '从圆外一点 $P$ 引两割线：一条交圆于 $A$（近）、$B$（远），$PA = 4$，$PB = 9$；另一条交圆于 $C$（近）、$D$（远），$PC = 3$。则 $PD$ 等于',
        options: ['$10$', '$11$', '$12$', '$13$'],
        answer: 2,
        explain:
          '由割线-割线定理：$PA \\cdot PB = PC \\cdot PD \\Rightarrow 4 \\times 9 = 3 \\times PD \\Rightarrow PD = 12$。',
      },
      {
        type: 'input',
        prompt:
          '圆内一点 $P$，过 $P$ 的弦 $AB$ 中 $PA = 2, PB = 6$；另一条弦 $CD$ 中 $PC = 3$。求 $PD$。',
        accept: ['4'],
        explain:
          '由相交弦定理：$PA \\cdot PB = PC \\cdot PD \\Rightarrow 2 \\times 6 = 3 \\times PD \\Rightarrow PD = 4$。弦 $CD = PC + PD = 7$。',
      },
      {
        type: 'judge',
        prompt:
          '$PA$ 切圆 $O$ 于 $A$，割线 $PBC$ 中 $PA = 6$，$PB = 4$。则 $BC = 5$。',
        answer: true,
        explain:
          '由切割线定理：$PA^2 = PB \\cdot PC \\Rightarrow 36 = 4 \\cdot PC \\Rightarrow PC = 9$。$BC = PC - PB = 9 - 4 = 5$。命题正确。',
      },
      {
        type: 'choice',
        prompt:
          '从圆外一点 $P$ 引两割线：一条交圆于 $A, B$，$PA = 3, AB = 5$；另一条交圆于 $C, D$，$PC = 4$。则 $CD$ 等于',
        options: ['$2$', '$3$', '$4$', '$6$'],
        answer: 0,
        explain:
          '$PB = PA + AB = 3 + 5 = 8$。由 $PA \\cdot PB = PC \\cdot PD$：$3 \\times 8 = 4 \\times PD$，$PD = 6$。$CD = PD - PC = 6 - 4 = 2$。',
      },
      {
        type: 'match',
        prompt: '将图形配置与适用定理配对。',
        left: ['圆外点 + 切线 + 割线', '圆外点 + 两条割线', '圆内点 + 两条弦相交'],
        right: [
          '相交弦定理：$PA \\cdot PB = PC \\cdot PD$',
          '切割线定理：$PA^2 = PB \\cdot PC$',
          '割线-割线定理：$PB \\cdot PC = PD \\cdot PE$',
        ],
        explain:
          '圆外点+切线+割线 $\\Rightarrow$ 切割线定理；圆外点+两割线 $\\Rightarrow$ 割线-割线定理；圆内点+两弦相交 $\\Rightarrow$ 相交弦定理。三者本质上都是圆幂定理在不同位置下的表达。',
      },
    ],
  },
]
