export const LESSONS = [
  {
    id: 'u6c4',
    title: '共角共边相似模型',
    subtitle: '共顶角 + 两边成比例 → SAS 相似（直接型 vs 交叉型）',
    intro: [
      {
        title: '一图速记',
        body:
          '两个三角形**共一个顶角**，且夹这个角的两边**成比例**，就能用 SAS 判定它们相似。这是 SAS 相似的图形化版本，称为**共角共边相似模型**。关键在于：比例的写法（直接型 vs 交叉型）决定对应关系，不同的写法对应不同的相似结论。',
        image: 'model-shared-angle.svg',
        imageCaption: '共角共边相似（交叉对应）',
      },
      {
        title: '引入：直接型 vs 交叉型',
        body:
          '$\\triangle ABC$ 中，$D \\in AB$，$E \\in AC$，共顶角 $\\angle A$。\n\n- **直接型**：$\\dfrac{AD}{AB} = \\dfrac{AE}{AC}$ $\\Rightarrow$ $\\triangle ADE \\sim \\triangle ABC$（$D\\leftrightarrow B, E\\leftrightarrow C$）——这就是 A 字模型；\n- **交叉型**：$\\dfrac{AD}{AC} = \\dfrac{AE}{AB}$ $\\Rightarrow$ $\\triangle ADE \\sim \\triangle ACB$（$D\\leftrightarrow C, E\\leftrightarrow B$）——本节核心。\n\n同一张图，比例写法不同，相似结论的字母顺序就不同！',
        image: 'ex-p3s04-0.svg',
        imageCaption: '共角共边引入：交叉比例与直接比例',
      },
      {
        title: '思维路径：如何识别交叉型？',
        body:
          '看比例式 $\\dfrac{AD}{AC} = \\dfrac{AE}{AB}$——分母决定"配谁"：\n- $AD$ 的分母是 $AC$，故 $D$ 对应 $C$；\n- $AE$ 的分母是 $AB$，故 $E$ 对应 $B$。\n\n因此相似写作 $\\triangle ADE \\sim \\triangle ACB$（字母 $A$ 不动，$D$ 换到 $C$ 的位置，$E$ 换到 $B$ 的位置）。\n\n**核心心法**：分母决定对应顶点，写出对应关系后再写相似符号右侧的字母顺序。',
        tip: '交叉型等价于乘积式：$\\dfrac{AD}{AC} = \\dfrac{AE}{AB}$ $\\Leftrightarrow$ $AB \\cdot AD = AC \\cdot AE$。看到四段共顶点的乘积等式，立刻想到共角共边（交叉型）。',
      },
      {
        title: '抽象模型：直接型与交叉型对比',
        body:
          '| 比例式 | 相似结论 | 对应关系 | 模型名称 |\n|---|---|---|---|\n| $\\dfrac{AD}{AB} = \\dfrac{AE}{AC}$ | $\\triangle ADE \\sim \\triangle ABC$ | $D\\leftrightarrow B, E\\leftrightarrow C$ | A 字（直接型）|\n| $\\dfrac{AD}{AC} = \\dfrac{AE}{AB}$ | $\\triangle ADE \\sim \\triangle ACB$ | $D\\leftrightarrow C, E\\leftrightarrow B$ | 共角共边（交叉型）|',
        formula: 'AB \\cdot AD = AC \\cdot AE \\Leftrightarrow \\frac{AD}{AC} = \\frac{AE}{AB}',
      },
      {
        title: '圆中的共角共边：相交弦定理',
        body:
          '圆 $O$ 中两弦 $AB, CD$ 相交于 $P$。\n- 对顶角：$\\angle APC = \\angle DPB$；\n- 同弧 $BC$ 所对圆周角：$\\angle PAC = \\angle PDB$（圆周角定理预告）。\n\nAA 得 $\\triangle PAC \\sim \\triangle PDB$，从而对应边 $\\dfrac{PA}{PD} = \\dfrac{PC}{PB}$，交叉相乘得 $PA \\cdot PB = PC \\cdot PD$。\n\n这就是**相交弦定理**，其本质是共角共边相似（对顶角型）。',
        image: 'ex-p3s04-3.svg',
        imageCaption: '圆中相交弦定理：共角共边模型的高级应用',
        formula: 'PA \\cdot PB = PC \\cdot PD',
      },
      {
        title: '思路路标',
        body:
          '- 看到**共顶角 + 比例线段** → 共角共边相似；\n- 看到**线段乘积** $a \\cdot b = c \\cdot d$（四段共一端点）→ 化为交叉比例，共角共边（交叉型）；\n- 比例直接型 $\\Rightarrow$ A 字模型；比例交叉型 $\\Rightarrow$ 本节模型，字母顺序反转；\n- 圆中两弦相交 / 圆外割线 → 共角共边背景；\n- **写相似符号前务必先确认对应顶点**，再填字母顺序，防止字母写错导致比例式倒置。',
        reveal: {
          q: '$\\triangle ABC$ 中 $D \\in AB$，$E \\in AC$，$AD = 2, AB = 8, AE = 3, AC = 12$。$\\triangle ADE$ 与 $\\triangle ABC$ 是直接型还是交叉型相似？',
          a: '$\\dfrac{AD}{AB} = \\dfrac{2}{8} = \\dfrac{1}{4}$，$\\dfrac{AE}{AC} = \\dfrac{3}{12} = \\dfrac{1}{4}$，两者相等，是**直接型**（A 字模型），$\\triangle ADE \\sim \\triangle ABC$。',
        },
      },
      {
        title: '常见错误与易错辨析',
        body:
          '1. **字母顺序随手写成 $\\triangle ABC$**：交叉型的正确写法是 $\\triangle ADE \\sim \\triangle ACB$，右侧字母顺序不是 $ABC$ 而是 $ACB$，漏掉这一步会导致后续比例式全错；\n2. **乘积等式未转化**：看到 $AB \\cdot AD = AC \\cdot AE$，先化为 $\\dfrac{AD}{AC} = \\dfrac{AE}{AB}$ 再识别对应关系，直接从乘积式硬猜字母对应容易出错；\n3. **圆中"对顶角"混淆"公共角"**：圆内相交弦用对顶角凑 AA，圆外割线用公共角凑 AA，两种情形图形完全不同但结论形式相同（$PA \\cdot PB = PC \\cdot PD$），要分清图形来源。',
      },
    ],
    questions: [
      {
        type: 'judge',
        prompt:
          '$\\triangle ABC$ 中 $D \\in AB$，$E \\in AC$，若 $\\dfrac{AD}{AC} = \\dfrac{AE}{AB}$，则 $\\triangle ADE \\sim \\triangle ACB$。',
        answer: true,
        explain:
          '共顶角 $\\angle A$，且 $\\dfrac{AD}{AC} = \\dfrac{AE}{AB}$ 即 $\\triangle ADE$ 夹 $\\angle A$ 的两边（$AD, AE$）与 $\\triangle ACB$ 夹 $\\angle A$ 的两边（$AC, AB$）对应成比例，SAS 相似得 $\\triangle ADE \\sim \\triangle ACB$。',
      },
      {
        type: 'choice',
        prompt:
          '$\\triangle ABC$ 中，$D \\in AB$，$E \\in AC$，$AB \\cdot AD = AC \\cdot AE$，则下列相似结论正确的是',
        options: [
          '$\\triangle ADE \\sim \\triangle ABC$',
          '$\\triangle ADE \\sim \\triangle ACB$',
          '$\\triangle ADE \\sim \\triangle BAC$',
          '$\\triangle ADE \\sim \\triangle BCA$',
        ],
        answer: 1,
        explain:
          '$AB \\cdot AD = AC \\cdot AE$ 化为 $\\dfrac{AD}{AC} = \\dfrac{AE}{AB}$，是交叉型比例。共顶角 $\\angle A$，SAS 相似得 $\\triangle ADE \\sim \\triangle ACB$（$D$ 对 $C$，$E$ 对 $B$）。',
      },
      {
        type: 'input',
        prompt:
          '$\\triangle ABC$ 中，$D \\in AB$，$E \\in AC$，$\\triangle ADE \\sim \\triangle ACB$，$AD = 3$，$AC = 9$，$AB = 12$，求 $AE$。',
        accept: ['4'],
        explain:
          '$\\triangle ADE \\sim \\triangle ACB$，对应边 $AD \\leftrightarrow AC$，$AE \\leftrightarrow AB$，故 $\\dfrac{AD}{AC} = \\dfrac{AE}{AB}$，即 $\\dfrac{3}{9} = \\dfrac{AE}{12}$，解得 $AE = 4$。',
      },
      {
        type: 'match',
        prompt: '将比例式与对应的相似结论配对（$D \\in AB$，$E \\in AC$，共顶角 $\\angle A$）。',
        left: [
          '$\\dfrac{AD}{AB} = \\dfrac{AE}{AC}$（直接型）',
          '$\\dfrac{AD}{AC} = \\dfrac{AE}{AB}$（交叉型）',
          '$AB \\cdot AD = AC \\cdot AE$（乘积型）',
        ],
        right: [
          '$\\triangle ADE \\sim \\triangle ACB$',
          '$\\triangle ADE \\sim \\triangle ACB$（等价于交叉型）',
          '$\\triangle ADE \\sim \\triangle ABC$',
        ],
        explain:
          '直接型比例（分子分母同侧）对应 A 字模型 $\\triangle ADE \\sim \\triangle ABC$；交叉型比例（分子分母互换）和乘积式等价，对应 $\\triangle ADE \\sim \\triangle ACB$。',
      },
      {
        type: 'judge',
        prompt:
          '圆内两弦 $AB, CD$ 相交于 $P$，由 $\\triangle PAC \\sim \\triangle PDB$ 可得 $PA \\cdot PB = PC \\cdot PD$。',
        answer: true,
        explain:
          '$\\triangle PAC \\sim \\triangle PDB$，对应边 $\\dfrac{PA}{PD} = \\dfrac{PC}{PB}$，交叉相乘得 $PA \\cdot PB = PC \\cdot PD$，这正是相交弦定理。',
      },
      {
        type: 'choice',
        prompt:
          '已知 $\\triangle ADE \\sim \\triangle ACB$（$D \\in AB$，$E \\in AC$），下列比例式中正确的是',
        options: [
          '$\\dfrac{AD}{AB} = \\dfrac{AE}{AC}$',
          '$\\dfrac{AD}{AC} = \\dfrac{AE}{AB}$',
          '$\\dfrac{DE}{BC} = \\dfrac{AD}{AB}$',
          '$\\dfrac{AD}{AB} = \\dfrac{DE}{CB}$',
        ],
        answer: 1,
        explain:
          '$\\triangle ADE \\sim \\triangle ACB$，对应顶点 $A\\leftrightarrow A, D\\leftrightarrow C, E\\leftrightarrow B$，故对应边比为 $\\dfrac{AD}{AC} = \\dfrac{AE}{AB} = \\dfrac{DE}{CB}$。选 B。',
      },
      {
        type: 'input',
        prompt:
          '$\\triangle ABC$ 中 $D \\in AB$，$E \\in AC$，$AD = 4$，$AB = 6$，$AE = 6$，$AC = 9$。问 $\\triangle ADE$ 与 $\\triangle ABC$ 的相似比（$\\triangle ADE$ 比 $\\triangle ABC$）是多少？（写成分数形式）',
        accept: ['2/3'],
        explain:
          '$\\dfrac{AD}{AB} = \\dfrac{4}{6} = \\dfrac{2}{3}$，$\\dfrac{AE}{AC} = \\dfrac{6}{9} = \\dfrac{2}{3}$，两值相等，为直接型，$\\triangle ADE \\sim \\triangle ABC$，相似比 $= \\dfrac{2}{3}$。',
      },
    ],
  },
]
