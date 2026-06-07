export const LESSONS = [
  {
    id: 'u9c6',
    title: '切线长定理与三角形的内切圆',
    subtitle: '两切线长相等·HL全等·内切圆半径公式',
    intro: [
      {
        title: '核心图形：圆外一点引两条切线',
        body:
          '圆 $\\odot O$ 外有一点 $P$，过 $P$ 引 $\\odot O$ 的两条切线 $PA, PB$（$A, B$ 是切点）。连 $OA, OB, OP$。\n\n这个图形天然**左右对称**，对称轴是 $OP$。图中同时藏着：$OA = OB = r$，$OA \\perp PA$，$OB \\perp PB$（切线性质），$OP$ 是公共斜边。这是 **HL 全等**的标准触发条件。',
        image: 'thm-p5-06-tangent-length.svg',
        imageCaption: '$PA = PB$，$OP$ 平分 $\\angle APB$ 并垂直平分弦 $AB$',
      },
      {
        title: '切线长的定义与计算',
        body:
          '**切线长**：从圆**外**一点 $P$ 到切点的**线段**长度（注意：切线是直线，切线长是线段）。\n\n在直角 $\\triangle OAP$ 中（$OA \\perp PA$，$OA = r$，$OP$ 是斜边）：$PA^2 = OP^2 - OA^2 = OP^2 - r^2$。\n\n**含义**：切线长由 $OP$ 和 $r$ 唯一决定。$OP = r$ 时切线长 $= 0$（点在圆上）；$OP < r$ 时没有实数解（点在圆内，无切线）。',
        formula: 'PA = \\sqrt{OP^2 - r^2}',
      },
      {
        title: '切线长定理',
        body:
          '**定理**：从圆外一点引圆的两条切线，两切线长**相等**，且圆心与该点的连线**平分两条切线的夹角**。\n\n符号：$PA, PB$ 切 $\\odot O$ 于 $A, B$，则 $PA = PB$，$\\angle APO = \\angle BPO$，$\\angle AOP = \\angle BOP$。\n\n**证明**：$\\triangle OAP$ 与 $\\triangle OBP$ 中，$OA = OB = r$，$\\angle OAP = \\angle OBP = 90^\\circ$，$OP$ 公共。由 **HL** 得全等，对应边 $PA = PB$，对应角 $\\angle APO = \\angle BPO$，$\\angle AOP = \\angle BOP$。',
        formula: 'PA = PB,\\quad \\angle APO = \\angle BPO,\\quad OP \\perp AB',
      },
      {
        title: '切线长定理的额外推论',
        body:
          '全等带来的四条推论（第 4 条最易被忽略）：\n1. $PA = PB$（**线段相等**）\n2. $\\angle APO = \\angle BPO$（$OP$ 平分 $\\angle APB$）\n3. $\\angle AOP = \\angle BOP$（$OP$ 平分圆心角 $\\angle AOB$）\n4. $OP \\perp AB$ 且 $OP$ **垂直平分**弦 $AB$（由对称性或 $PA = PB, OA = OB$ 直接得 $OP$ 是 $AB$ 的中垂线）\n\n图形 $OAPB$ 是"风筝形"：$OA = OB$，$PA = PB$，$OP$ 是对称轴。',
        tip: '看到"圆外一点引两条切线"，立刻画 $OP$ 并知 $OP \\perp AB$——这往往是后续解题的关键桥梁，不要忘记。',
      },
      {
        title: '三角形的内切圆',
        body:
          '**内切圆**：与三角形三条边都相切的圆；圆心叫**内心**（= 三条内角平分线交点 = 到三边等距点）。\n\n**内切圆面积公式（通用）**：设 $\\triangle ABC$ 三边 $a, b, c$，面积 $S$，半周长 $p = \\dfrac{a+b+c}{2}$，内切圆半径 $r$：$S = p \\cdot r$，即 $r = \\frac{S}{p}$。\n\n**直角三角形专用公式**（直角顶点 $C$，直角边 $a, b$，斜边 $c$）：$r = \\frac{a + b - c}{2}$。',
        formula: 'S = p\\,r;\\quad r = \\frac{a+b-c}{2}\\text{（直角三角形）}',
        image: 'thm-p5-06-incircle.svg',
        imageCaption: '内心 $I$ 是三条内角平分线交点，到三边距离均为 $r$',
      },
      {
        title: '例题：两类公式的选择',
        body:
          '**例 1（直角三角形）**：$\\triangle ABC$，$\\angle C = 90^\\circ$，$AC = 3$，$BC = 4$，$AB = 5$。$r = \\frac{3 + 4 - 5}{2} = 1$。\n验证：$S = \\frac{1}{2}\\times 3\\times 4 = 6$，$p = 6$，$r = 6/6 = 1$。两种方法一致。\n\n**例 2（等腰三角形）**：$AB = AC = 5$，$BC = 6$。求 $r$。\n先求高 $AD$：$BD = 3$，$AD = \\sqrt{25-9} = 4$，$S = \\frac{1}{2}\\times 6\\times 4 = 12$。\n半周长 $p = (5+5+6)/2 = 8$，$r = 12/8 = \\dfrac{3}{2}$。',
        image: 'ex-p5-06-2.svg',
        imageCaption: '$3$-$4$-$5$ 直角三角形内切圆半径 $r = 1$',
        reveal: {
          q: '直角三角形公式 $r = (a+b-c)/2$ 的推导思路是什么？',
          a: '设内切圆与三边切点为 $D, E, F$。直角顶点 $C$ 到两切点距离相等 $= r$（四边形 $CDIE$ 为正方形）；顶点 $B, A$ 各有切线长相等。利用 $a + b = c + 2r$ 即得 $r = (a+b-c)/2$。',
        },
      },
      {
        title: '易错点',
        body:
          '1. **"切线"与"切线长"混淆**：切线是直线，切线长是线段长度，两类问题的问法不同。\n2. **切线长定理的前提是"圆外一点"**：圆上点切线长为 $0$，圆内点无切线。\n3. **HL 全等里不能漏直角条件**：必须写 $\\angle OAP = \\angle OBP = 90^\\circ$，否则变成 SSA 无法全等。\n4. **内切圆 vs 外接圆**：内切圆在三角形内、切三边、圆心是角平分线交点；外接圆过三顶点、圆心是中垂线交点。内外互逆。\n5. **$r = (a+b-c)/2$ 仅限直角三角形**，$a, b$ 是直角边，$c$ 是斜边。\n6. **$S = pr$ 中 $p$ 是半周长**，不是周长。漏掉 $\\frac{1}{2}$ 会算大一倍。',
      },
    ],
    questions: [
      {
        type: 'input',
        prompt:
          '$\\odot O$ 半径 $r = 5$，圆外一点 $P$ 到圆心的距离 $OP = 13$。求从 $P$ 到 $\\odot O$ 的切线长 $PA$。',
        accept: ['12'],
        explain:
          '$PA = \\sqrt{OP^2 - r^2} = \\sqrt{169 - 25} = \\sqrt{144} = 12$（经典 $5$-$12$-$13$ 直角三角形）。',
      },
      {
        type: 'choice',
        prompt:
          '圆 $\\odot O$ 外一点 $P$ 引两切线 $PA, PB$（切点 $A, B$），$\\angle APB = 60^\\circ$，$PA = 4$。则 $AB$ 的长为',
        options: ['$4$', '$4\\sqrt{3}$', '$2\\sqrt{3}$', '$8$'],
        answer: 0,
        explain:
          '由切线长定理 $PA = PB = 4$，$\\triangle PAB$ 是等腰三角形，顶角 $\\angle APB = 60^\\circ$，两底角 $= (180^\\circ - 60^\\circ)/2 = 60^\\circ$，故 $\\triangle PAB$ 是等边三角形，$AB = PA = 4$。',
      },
      {
        type: 'input',
        prompt:
          '直角三角形两直角边分别为 $5$ 和 $12$，斜边为 $13$。求其内切圆半径 $r$。',
        accept: ['2'],
        explain:
          '用直角三角形公式：$r = \\dfrac{5 + 12 - 13}{2} = \\dfrac{4}{2} = 2$。验证：$S = \\frac{1}{2}\\times 5\\times 12 = 30$，$p = (5+12+13)/2 = 15$，$r = 30/15 = 2$。一致。',
      },
      {
        type: 'judge',
        prompt:
          '从圆外一点 $P$ 引两条切线 $PA, PB$（$A, B$ 为切点），则线段 $OP$ 垂直平分弦 $AB$。',
        answer: true,
        explain:
          '由切线长定理，$PA = PB$，$OA = OB$，$P$ 和 $O$ 都在 $AB$ 的中垂线上，故 $OP$ 就是 $AB$ 的中垂线，即 $OP \\perp AB$ 且 $OP$ 平分 $AB$。命题正确。',
      },
      {
        type: 'input',
        prompt:
          '三角形三边长为 $13, 14, 15$，面积为 $84$。求其内切圆半径 $r$（答案填整数）。',
        accept: ['4'],
        explain:
          '半周长 $p = (13+14+15)/2 = 21$，由 $S = pr$ 得 $r = S/p = 84/21 = 4$。',
      },
      {
        type: 'match',
        prompt: '将内切圆与外接圆的特征配对。',
        left: ['内切圆', '外接圆'],
        right: [
          '圆心是三条边的中垂线交点，圆过三顶点',
          '圆心是三条内角平分线交点，圆与三边相切',
        ],
        explain:
          '内切圆：圆在三角形内，切三边，圆心（内心）是三条**内角平分线**交点。外接圆：圆过三顶点，圆心（外心）是三条**边的中垂线**交点。一"切边"一"过顶点"，刚好相反。',
      },
      {
        type: 'choice',
        prompt:
          '从圆外一点 $P$ 引两切线 $PA, PB$（切点 $A, B$），$\\angle APB = 90^\\circ$，$PA = 6$。则圆的半径 $r$ 等于',
        options: ['$3$', '$6$', '$3\\sqrt{2}$', '$6\\sqrt{2}$'],
        answer: 1,
        explain:
          '由切线长定理 $PA = PB = 6$，$\\angle APB = 90^\\circ$，四边形 $OAPB$ 中 $\\angle OAP = \\angle OBP = 90^\\circ$，故 $\\angle AOB = 360^\\circ - 90^\\circ - 90^\\circ - 90^\\circ = 90^\\circ$。$\\triangle OAB$ 等腰直角（$OA = OB = r$），$AB = r\\sqrt{2}$。由 $\\triangle PAB$ 等腰直角（$PA = PB = 6$，$\\angle APB = 90^\\circ$），$AB = 6\\sqrt{2}$。联立：$r\\sqrt{2} = 6\\sqrt{2}$，$r = 6$。',
      },
    ],
  },
]
