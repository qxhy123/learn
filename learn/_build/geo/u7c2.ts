export const LESSONS = [
  {
    id: 'u7c2',
    title: '等边三角形',
    subtitle: '三角都 60°·高与面积公式·四线合一',
    intro: [
      {
        title: '等边三角形的基本特征',
        body:
          '**等边三角形**（正三角形）是三条边都相等的三角形。若边长为 $a$，则三个内角也全相等。由内角和 $180^\\circ$，每个内角恰好是 $60^\\circ$。\n\n等边三角形是等腰三角形的特殊情形，继承其所有性质，但更强：**每个顶点**的角都是顶角（$60^\\circ$），所以每个顶点引出的高、中线、角平分线，以及对边的中垂线，**四线合一**（而不只是三线合一）。',
        formula: 'AB = BC = CA = a \\Rightarrow \\angle A = \\angle B = \\angle C = 60^\\circ',
      },
      {
        title: '判定等边三角形的三条途径',
        body:
          '判定一个三角形是等边三角形，有以下三条路：\n\n**判定 1（定义）：** 三边相等 $\\Rightarrow$ 等边三角形。\n\n**判定 2（三角）：** 三个角都等于 $60^\\circ$ $\\Rightarrow$ 等边三角形。\n\n**判定 3（最常用）：** 等腰三角形 $+$ 一个 $60^\\circ$ 角 $\\Rightarrow$ 等边三角形。\n\n**判定 3 两种情况都成立：**\n- 若 $60^\\circ$ 是顶角：两底角 $= (180^\\circ - 60^\\circ) \\div 2 = 60^\\circ$，三角均 $60^\\circ$；\n- 若 $60^\\circ$ 是底角：另一底角也是 $60^\\circ$，顶角 $= 180^\\circ - 60^\\circ - 60^\\circ = 60^\\circ$，三角均 $60^\\circ$。',
        tip: '只要等腰三角形中有任何一个 $60^\\circ$ 角，无论是顶角还是底角，三个角就全是 $60^\\circ$，即为等边三角形。',
      },
      {
        title: '高的公式推导',
        body:
          '过顶点 $A$ 作底边 $BC$ 上的高 $AD$（同时也是中线），把等边三角形分成两个全等的 30-60-90 直角三角形。\n\n在直角 $\\triangle ABD$ 中，$AB = a$ 是斜边，$BD = a/2$ 是对 $30^\\circ$ 的短直角边，$AD = h$ 是对 $60^\\circ$ 的长直角边。\n\n由勾股定理：$h^2 = a^2 - \\left(\\dfrac{a}{2}\\right)^2 = a^2 - \\dfrac{a^2}{4} = \\dfrac{3a^2}{4}$，故 $h = \\dfrac{\\sqrt{3}}{2}a$。',
        formula: 'h = \\dfrac{\\sqrt{3}}{2}\\,a',
        image: 'thm-equilateral-height.svg',
        imageCaption: '等边三角形的高将其分为两个全等的 30-60-90 直角三角形',
      },
      {
        title: '面积公式',
        body:
          '已知高 $h = \\dfrac{\\sqrt{3}}{2}a$，底边长 $= a$，代入三角形面积公式：\n\n$S = \\dfrac{1}{2} \\times a \\times h = \\dfrac{1}{2} \\times a \\times \\dfrac{\\sqrt{3}}{2}a = \\dfrac{\\sqrt{3}}{4}a^2$\n\n**例：** 边长为 $4$ 的等边三角形：$h = \\dfrac{\\sqrt{3}}{2} \\times 4 = 2\\sqrt{3}$，$S = \\dfrac{\\sqrt{3}}{4} \\times 16 = 4\\sqrt{3}$。',
        formula: 'S = \\dfrac{\\sqrt{3}}{4}\\,a^2',
        tip: '面积公式是 $a^2$（平方），不是 $a$，心算时注意不要漏掉平方。',
      },
      {
        title: '外接圆半径与内切圆半径',
        body:
          '等边三角形的**四心（重心、内心、外心、垂心）合一**，均在中心点 $O$。\n\n重心把每条中线（即高）分成 $2:1$，靠顶点的部分是外接圆半径 $R$，靠底边的部分是内切圆半径 $r$：\n\n$R = \\dfrac{2}{3}h = \\dfrac{\\sqrt{3}}{3}a,\\qquad r = \\dfrac{1}{3}h = \\dfrac{\\sqrt{3}}{6}a$\n\n关系：$R = 2r$，$R + r = h$。\n\n**思维链：** $a \\xrightarrow{\\times \\frac{\\sqrt{3}}{2}} h \\xrightarrow{\\times \\frac{2}{3}} R = 2r$。',
        formula: 'R = \\dfrac{\\sqrt{3}}{3}\\,a,\\quad r = \\dfrac{\\sqrt{3}}{6}\\,a,\\quad R = 2r',
      },
      {
        title: '手拉手模型：两等边共顶点',
        body:
          '$\\triangle ABC$ 与 $\\triangle ADE$ 都是等边三角形且共顶点 $A$，求证 $BD = CE$。\n\n**思路：** 比较 $\\triangle ABD$ 与 $\\triangle ACE$，它们有 $AB = AC$、$AD = AE$（等边三角形腰长相等）。夹角关键：\n\n$\\angle BAD = \\angle BAC + \\angle CAD = 60^\\circ + \\angle CAD = \\angle DAE + \\angle CAD = \\angle CAE$\n\n故夹角相等，由 SAS 得 $\\triangle ABD \\cong \\triangle ACE$，对应边 $BD = CE$。',
        image: 'model-handshake-equilateral.svg',
        imageCaption: '两等边三角形共顶点 $A$（手拉手模型），$BD = CE$',
      },
      {
        title: '易错点汇总',
        body:
          '1. **判定 3 两种情形都要记**：等腰 $+$ 底角 $60^\\circ$ 同样推出等边，不要只想到顶角。\n2. **系数混淆**：$h = \\dfrac{\\sqrt{3}}{2}a$，$R = \\dfrac{\\sqrt{3}}{3}a$，$r = \\dfrac{\\sqrt{3}}{6}a$，记忆口诀"分母 2、3、6 越来越大"。\n3. **面积公式是 $a^2$**，漏写平方是常见错误。\n4. **$60^\\circ$ ≠ 等边**：必须先有"等腰"，再配一个 $60^\\circ$ 才能推等边；单独一个 $60^\\circ$ 不够。\n5. **手拉手夹角方向**：两等边同向时夹角相加，反向时相减，但 SAS 结构不变。',
        reveal: {
          q: '已知等边三角形内切圆半径 $r = 3$，求其外接圆半径 $R$、边长 $a$ 和面积 $S$。',
          a: '$R = 2r = 6$；由 $r = \\dfrac{\\sqrt{3}}{6}a$ 得 $a = \\dfrac{6r}{\\sqrt{3}} = 6\\sqrt{3}$；$S = \\dfrac{\\sqrt{3}}{4}a^2 = \\dfrac{\\sqrt{3}}{4} \\times 108 = 27\\sqrt{3}$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt:
          '边长为 $6$ 的等边三角形，其高 $h$ 为',
        options: [
          '$3\\sqrt{2}$',
          '$3\\sqrt{3}$',
          '$6\\sqrt{3}$',
          '$2\\sqrt{3}$',
        ],
        answer: 1,
        explain:
          '$h = \\dfrac{\\sqrt{3}}{2}a = \\dfrac{\\sqrt{3}}{2} \\times 6 = 3\\sqrt{3}$。注意系数是 $\\dfrac{\\sqrt{3}}{2}$，不要与 $\\dfrac{\\sqrt{2}}{2}$ 混淆。',
      },
      {
        type: 'input',
        prompt:
          '边长为 $2$ 的等边三角形面积为多少？（形如 $k\\sqrt{3}$ 填 $k$ 的值）',
        accept: ['1'],
        explain:
          '$S = \\dfrac{\\sqrt{3}}{4}a^2 = \\dfrac{\\sqrt{3}}{4} \\times 4 = \\sqrt{3}$，系数 $k = 1$，即面积为 $\\sqrt{3}$。',
      },
      {
        type: 'judge',
        prompt:
          '等腰三角形的底角为 $60^\\circ$，则该三角形一定是等边三角形。',
        answer: true,
        explain:
          '等腰三角形底角 $= 60^\\circ$，则两底角之和 $= 120^\\circ$，顶角 $= 180^\\circ - 120^\\circ = 60^\\circ$，三个角全是 $60^\\circ$，故为等边三角形。"等腰 $+$ 一个 $60^\\circ$（哪怕是底角）$\\Rightarrow$ 等边"成立。',
      },
      {
        type: 'choice',
        prompt:
          '等边三角形的外接圆半径 $R$ 与内切圆半径 $r$ 的关系是',
        options: ['$R = r$', '$R = 2r$', '$R = 3r$', '$R = \\sqrt{3}\\,r$'],
        answer: 1,
        explain:
          '由重心 $2:1$ 分高的结论：$R = \\dfrac{2}{3}h$，$r = \\dfrac{1}{3}h$，故 $R = 2r$。',
      },
      {
        type: 'input',
        prompt:
          '等边三角形的边长为 $a$，其外接圆半径 $R = \\dfrac{\\sqrt{3}}{3}a$，若 $R = 4$，则边长 $a = $___（结果含根号时保留根号形式，如 $4\\sqrt{3}$）',
        accept: ['4\\sqrt{3}', '4√3'],
        explain:
          '由 $R = \\dfrac{\\sqrt{3}}{3}a$，解出 $a = \\dfrac{3R}{\\sqrt{3}} = \\sqrt{3}\\,R = \\sqrt{3} \\times 4 = 4\\sqrt{3}$。',
      },
      {
        type: 'judge',
        prompt:
          '等边三角形的重心、内心、外心、垂心是同一个点。',
        answer: true,
        explain:
          '等边三角形具有最高的对称性（三条对称轴），其重心、内心、外心、垂心四心完全重合于同一点（中心），这是等边三角形区别于普通三角形的特殊性质。',
      },
      {
        type: 'match',
        prompt: '将等边三角形（边长 $a$）的量与其公式配对。',
        left: ['高 $h$', '面积 $S$', '外接圆半径 $R$'],
        right: [
          '$\\dfrac{\\sqrt{3}}{3}\\,a$',
          '$\\dfrac{\\sqrt{3}}{2}\\,a$',
          '$\\dfrac{\\sqrt{3}}{4}\\,a^2$',
        ],
        explain:
          '高 $h = \\dfrac{\\sqrt{3}}{2}a$（系数 $\\dfrac{1}{2}$）；面积 $S = \\dfrac{\\sqrt{3}}{4}a^2$（系数 $\\dfrac{1}{4}$，有平方）；外接圆半径 $R = \\dfrac{\\sqrt{3}}{3}a$（系数 $\\dfrac{1}{3}$）。',
      },
      {
        type: 'choice',
        prompt:
          '$\\triangle ABC$ 是等边三角形，边长为 $4$，$D$ 是 $BC$ 的中点，则 $AD = $',
        options: ['$2$', '$2\\sqrt{2}$', '$2\\sqrt{3}$', '$4\\sqrt{3}$'],
        answer: 2,
        explain:
          '$D$ 为 $BC$ 中点，$AD$ 是等边三角形的高（中线）。$AD = h = \\dfrac{\\sqrt{3}}{2} \\times 4 = 2\\sqrt{3}$。也可用勾股定理：$BD = 2$，$AB = 4$，$AD = \\sqrt{16 - 4} = \\sqrt{12} = 2\\sqrt{3}$。',
      },
    ],
  },
]
