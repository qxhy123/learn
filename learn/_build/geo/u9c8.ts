export const LESSONS = [
  {
    id: 'u9c8',
    title: '四点共圆模型',
    subtitle: '等角同弦共圆·对角互补共圆·双直角共斜边·圆幂逆用',
    intro: [
      {
        title: '引入：无圆生圆',
        body:
          '四边形 $ABCD$ 中，$\\angle BAC = \\angle BDC = 35^\\circ$，且 $A, D$ 在 $BC$ 同侧。求证 $A, B, C, D$ 四点在同一个圆上。\n\n题目里根本没有画圆，只有四个点和两个相等的角。但当我们看到"$A, D$ 分别对同一条线段 $BC$ 张出相等的角"，这正是**圆周角定理的逆命题**！一旦发现，圆就被"召唤"出来了。',
        image: 'ex-p5-08-0.svg',
        imageCaption: '同侧等角 $35^\\circ$ 召唤圆：$A, B, C, D$ 共圆',
      },
      {
        title: '思维路径：怎么发现四点共圆',
        body:
          '固定线段 $BC$，从同侧看 $BC$ 成定角的轨迹是一段**圆弧**（"定弦定角看圆弧"）。$A$ 在这段弧上，$D$ 也在 $\\Rightarrow$ 两点同弧 $\\Rightarrow$ 四点共圆。\n\n严格证明：过 $B, C, D$ 三点作圆 $\\omega$（不共线三点定一圆）。若 $A$ 不在 $\\omega$ 上，设直线 $BA$ 与 $\\omega$ 另交于 $A\'$，则圆周角 $\\angle BA\'C = \\angle BDC = \\angle BAC$，但 $\\angle BA\'C$ 与 $\\angle BAC$ 是 $\\triangle ACA\'$ 中一内角与外角，矛盾。故 $A$ 在 $\\omega$ 上，四点共圆。',
        tip: '证完"四点共圆"后，显式写出" $\\Rightarrow A, B, C, D$ 共圆"，之后所有圆性质（圆周角、对角互补、圆幂）都可直接调用，这是阅卷的关键得分点。',
      },
      {
        title: '判定一：同侧等角（圆周角逆定理）',
        body:
          '**判定 1**：$A, D$ 在直线 $BC$ 同侧，且 $\\angle BAC = \\angle BDC$ $\\Rightarrow$ $A, B, C, D$ 共圆。\n\n这是圆周角定理的逆命题："同弧所对圆周角相等"正向 $\\Rightarrow$ 逆向"同侧等角的四点共圆"。\n\n**共圆后的红利**：一旦四点共圆，$\\angle ABD$ 与 $\\angle ACD$ 同为弦 $AD$ 所对圆周角（$B, C$ 在 $AD$ 同侧）$\\Rightarrow \\angle ABD = \\angle ACD$，一口气解决所有角关系。',
        formula: '\\angle BAC = \\angle BDC,\\; A/D \\text{ 同侧} \\Rightarrow A, B, C, D \\text{ 共圆}',
      },
      {
        title: '判定二：对角互补',
        body:
          '**判定 2**：凸四边形 $ABCD$ 中，若 $\\angle A + \\angle C = 180^\\circ$（或 $\\angle B + \\angle D = 180^\\circ$），则四点共圆。\n\n这是"圆内接四边形对角互补"性质（推论 3，第 3 关）的**逆命题**。\n\n**判定 3（等价说法）**：四边形 $ABCD$ 一外角等于内对角，例如 $\\angle DCE = \\angle A$（$E$ 在 $BC$ 延长线上），则四点共圆。本质与判定 2 相同。',
        formula: '\\angle A + \\angle C = 180^\\circ \\Rightarrow A, B, C, D \\text{ 共圆}',
      },
      {
        title: '判定四：双直角共斜边（最高频特例）',
        body:
          '若 $\\angle BAC = \\angle BDC = 90^\\circ$，则 $A, D$ 都在以 $BC$ 为直径的圆上 $\\Rightarrow$ $A, B, C, D$ 共圆，且 $BC$ 是直径。\n\n这是判定 1 的特例（$90^\\circ$ 是最特殊的等角）。口诀：**"两个直角共斜边，四点共圆，斜边为直径"**。\n\n**经典场景**：$\\triangle ABC$ 中，$BD \\perp AC$ 于 $D$，$CE \\perp AB$ 于 $E$，则 $\\angle BDC = \\angle BEC = 90^\\circ$，共斜边 $BC$，$B, C, D, E$ 共圆于以 $BC$ 为直径的圆。',
        image: 'ex-p5-08-2.svg',
        imageCaption: '双直角共斜边 $BC$：$B, C, D, E$ 共圆，$BC$ 为直径',
      },
      {
        title: '判定五：圆幂逆用（乘积型）',
        body:
          '**判定 4**：两线段 $AC$ 与 $BD$ 交于 $P$，若 $PA \\cdot PC = PB \\cdot PD$，则 $A, B, C, D$ 共圆。\n\n这是相交弦定理（第 7 关）或割线定理的**逆命题**。\n\n**识别技巧**：题目给出 $PA \\cdot PC = PB \\cdot PD$ 形式的乘积等式，立刻反应判定 4，把四个端点圈成共圆。',
        formula: 'PA \\cdot PC = PB \\cdot PD \\Rightarrow A, B, C, D \\text{ 共圆}',
        image: 'ex-p5-08-3.svg',
        imageCaption: '圆幂逆用：$PA \\cdot PC = PB \\cdot PD = 12$，四点共圆',
      },
      {
        title: '思考路标与总结',
        body:
          '**快速识别四点共圆**：\n- 两角相等 + 对同一线段 + 顶点同侧 $\\Rightarrow$ 判定 1\n- 四边形对角和 $= 180^\\circ$ 或一外角等于内对角 $\\Rightarrow$ 判定 2/3\n- 两个直角共用同一斜边 $\\Rightarrow$ 判定 4（双直角特例）\n- 两线段交于一点且 $PA \\cdot PC = PB \\cdot PD$ $\\Rightarrow$ 判定 4\n\n**记忆口诀**："**等角同弦看共圆，对角互补也共圆，乘积相等还是共圆**"。',
        reveal: {
          q: '证完四点共圆后，我能用哪些工具？',
          a: '圆内接图形的全部工具都可调用：圆周角定理（同弧等角）、直径所对直角、对角互补（推论 3）、切割线/相交弦（圆幂）。这就是四点共圆的"核心收益"——把零散条件一口气串成圆上的标准结论。',
        },
      },
    ],
    questions: [
      {
        type: 'judge',
        prompt:
          '四边形 $ABCD$ 中，$\\angle DAC = \\angle DBC = 40^\\circ$，且 $A, B$ 在 $DC$ 的同侧，则 $A, B, C, D$ 四点共圆。',
        answer: true,
        explain:
          '$A, B$ 在 $DC$ 同侧，$\\angle DAC = \\angle DBC = 40^\\circ$，即 $A, B$ 对同一线段 $DC$ 张出相等的角（同侧）。由判定 1（圆周角逆定理），$A, B, C, D$ 共圆。命题正确。',
      },
      {
        type: 'choice',
        prompt:
          '凸四边形 $ABCD$ 中，$\\angle ABC = 110^\\circ$，$\\angle ADC = 70^\\circ$。则四点',
        options: ['共圆，因为对角互补', '不共圆', '共圆，因为同侧等角', '无法判断'],
        answer: 0,
        explain:
          '$\\angle ABC + \\angle ADC = 110^\\circ + 70^\\circ = 180^\\circ$，满足判定 2（对角互补），故 $A, B, C, D$ 四点共圆。',
      },
      {
        type: 'judge',
        prompt:
          '$\\triangle ABC$ 中，$BD \\perp AC$ 于 $D$，$CE \\perp AB$ 于 $E$，则 $B, C, D, E$ 四点共圆，且 $BC$ 是这个圆的直径。',
        answer: true,
        explain:
          '$\\angle BDC = 90^\\circ$（$BD \\perp AC$），$\\angle BEC = 90^\\circ$（$CE \\perp AB$）。两个直角共用斜边 $BC$，由判定 1 的直角特例（双直角共斜边），$B, C, D, E$ 共圆，且 $BC$ 是直径。命题正确。',
      },
      {
        type: 'input',
        prompt:
          '线段 $AC$ 与 $BD$ 交于 $P$，$PA = 2, PC = 6, PB = 3$。若 $A, B, C, D$ 四点共圆，求 $PD$。',
        accept: ['4'],
        explain:
          '由共圆的相交弦定理：$PA \\cdot PC = PB \\cdot PD \\Rightarrow 2 \\times 6 = 3 \\times PD \\Rightarrow PD = 4$。',
      },
      {
        type: 'choice',
        prompt:
          '$\\triangle ABC$ 中两条高 $BD, CE$ 交于垂心 $H$。则 $A, E, H, D$ 四点共圆，该圆的直径是',
        options: ['$AH$', '$BD$', '$CE$', '$EH$'],
        answer: 0,
        explain:
          '$\\angle AEH = 90^\\circ$（$CE \\perp AB$），$\\angle ADH = 90^\\circ$（$BD \\perp AC$）。两个直角共用斜边 $AH$，故 $A, E, H, D$ 共圆且 $AH$ 是直径（直角所对斜边是直径）。',
      },
      {
        type: 'match',
        prompt: '将四点共圆的判定条件与对应判定方法配对。',
        left: [
          '两角相等且对同一线段、顶点同侧',
          '四边形对角之和为 $180^\\circ$',
          '两个 $90^\\circ$ 共用同一斜边',
          '两交线段端点乘积相等',
        ],
        right: [
          '判定 4（圆幂逆命题）',
          '判定 1（圆周角逆定理）',
          '判定 2/3（对角互补逆命题）',
          '双直角共斜边（特例）',
        ],
        explain:
          '① 同侧等角 $\\Rightarrow$ 判定 1；② 对角互补 $\\Rightarrow$ 判定 2/3；③ 双直角共斜边 $\\Rightarrow$ 判定 1 的特例（直径所对直角逆命题）；④ 乘积相等 $\\Rightarrow$ 判定 4（圆幂/相交弦逆命题）。',
      },
      {
        type: 'judge',
        prompt:
          '线段 $AB, CD$ 延长线交于圆外一点 $P$，$PA = 3, PB = 8, PC = 4, PD = 6$。则 $A, B, C, D$ 四点共圆。',
        answer: true,
        explain:
          '$PA \\cdot PB = 3 \\times 8 = 24$，$PC \\cdot PD = 4 \\times 6 = 24$。两积相等，由判定 4（割线-割线逆定理），$A, B, C, D$ 四点共圆。命题正确。',
      },
    ],
  },
]
