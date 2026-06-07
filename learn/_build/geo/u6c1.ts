export const LESSONS = [
  {
    id: 'u6c1',
    title: '相似三角形判定',
    subtitle: 'AA / SAS / SSS 三大判定与性质',
    intro: [
      {
        title: '什么是相似三角形？',
        body:
          '两个三角形**形状完全相同，大小可以不同**，就称它们相似。直观上，把其中一个按某个比例放大或缩小就能得到另一个。相似比（记作 $k$）是对应边长度之比的公共值。当 $k=1$ 时，两三角形全等——全等是相似的特例。',
      },
      {
        title: '相似的两个本质特征',
        body: '两个三角形相似，等价于同时满足以下两条：\n1. **对应角相等**：三组内角分别相等；\n2. **对应边成比例**：三组对应边之比相等（公比即相似比 $k$）。\n\n记作 $\\triangle ABC \\sim \\triangle DEF$，字母顺序就是对应关系——$A\\leftrightarrow D$，$B\\leftrightarrow E$，$C\\leftrightarrow F$，不可随意调换。',
        image: 'thm-similarity-criteria.svg',
        imageCaption: 'AA、SAS、SSS 三大判定示意',
      },
      {
        title: '三大判定定理',
        body:
          '按定义验证要查"三角 + 三边"共六项，太繁琐。实际上以下任一组条件已足够：',
        formula:
          '\\text{AA：} \\angle A=\\angle D,\\ \\angle B=\\angle E \\Rightarrow \\triangle ABC \\sim \\triangle DEF',
        tip: 'SAS：两边成比例且夹角相等；SSS：三边对应成比例。三者选其一即可判定相似，无需全部验证。',
      },
      {
        title: 'AA 为何足够？',
        body:
          '三角形内角和恒为 $180°$。已知 $\\angle A=\\angle D$、$\\angle B=\\angle E$，第三角自动满足 $\\angle C = 180°-\\angle A-\\angle B = \\angle F$，无需单独验证。这与全等判定（必须含至少一组边）形成鲜明对比：相似只关心形状，不关心绝对大小，因此门槛更低。',
        formula:
          '\\angle C = 180° - \\angle A - \\angle B = 180° - \\angle D - \\angle E = \\angle F',
      },
      {
        title: '相似三角形的四条性质',
        body: '设 $\\triangle ABC \\sim \\triangle DEF$，相似比为 $k$，则：\n- 对应边之比 $= k$；\n- 对应高、对应中线、对应角平分线之比 $= k$；\n- 周长之比 $= k$（一维量线性缩放）；\n- **面积之比 $= k^2$**（二维量，两个线性维度各乘 $k$）。',
        formula: '\\frac{S_{\\triangle ABC}}{S_{\\triangle DEF}} = k^2',
        tip: '面积比是相似比的平方，开平方才能得到相似比——这是最常见的失分点。',
      },
      {
        title: '典型例题：平行线 + AA',
        body:
          '$\\triangle ABC$ 中，$DE \\parallel BC$，$D \\in AB$，$E \\in AC$，$AD=3$，$DB=2$，$BC=10$，求 $DE$。\n\n**解：** 由 $DE \\parallel BC$ 得 $\\angle ADE = \\angle ABC$，又 $\\angle A$ 公共，AA 得 $\\triangle ADE \\sim \\triangle ABC$。',
        formula:
          'k = \\frac{AD}{AB} = \\frac{3}{3+2} = \\frac{3}{5},\\quad DE = \\frac{3}{5} \\times 10 = 6',
      },
      {
        title: '典型例题：面积比反推边长',
        body:
          '$\\triangle ABC \\sim \\triangle A\'B\'C\'$，面积之比为 $9:16$，$AB=6$，求 $A\'B\'$。\n\n**解：** 面积比 $= k^2$，先开方求相似比，再反推边长。',
        formula:
          'k^2 = \\frac{9}{16} \\Rightarrow k = \\frac{3}{4},\\quad A\'B\' = \\frac{AB}{k} = \\frac{6}{3/4} = 8',
        tip: '开方时只取正值，因为相似比是长度之比，恒为正。',
      },
      {
        title: '五大易错点',
        body:
          '1. **字母顺序 = 对应关系**，不能随意调换；\n2. **SSA 不能判定相似**（歧义情形存在，直角三角形 HL 型除外）；\n3. **周长比 $= k$，面积比 $= k^2$**，切勿混淆；\n4. 相似比方向：$\\triangle ABC \\sim \\triangle DEF$ 的相似比是 $AB/DE$，前者比后者；\n5. 平行线仅提供一组角相等，必须再找第二组角才能用 AA，不可省略论证步骤。',
        reveal: {
          q: '若两三角形的周长比是 $3:5$，它们的面积比是多少？',
          a: '周长比即相似比 $k = 3/5$，面积比 $= k^2 = 9/25$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt:
          '$\\triangle ABC \\sim \\triangle DEF$，相似比 $k = \\dfrac{2}{3}$，则 $\\triangle ABC$ 与 $\\triangle DEF$ 的面积比为',
        options: ['$\\dfrac{2}{3}$', '$\\dfrac{4}{9}$', '$\\dfrac{9}{4}$', '$\\dfrac{3}{2}$'],
        answer: 1,
        explain:
          '面积比 $= k^2 = \\left(\\dfrac{2}{3}\\right)^2 = \\dfrac{4}{9}$。周长比才等于 $k$，面积比要平方。',
      },
      {
        type: 'judge',
        prompt:
          '若两个三角形有两组角分别相等，则这两个三角形一定相似。',
        answer: true,
        explain:
          '两组角相等时，由三角形内角和 $180°$，第三组角也自动相等，即 AA 条件满足，必然相似。',
      },
      {
        type: 'input',
        prompt:
          '两个相似三角形的面积之比为 $16:25$，较小三角形的一边长为 $8$，对应边长为多少？',
        accept: ['10'],
        explain:
          '面积比 $= k^2 = 16/25$，故 $k = 4/5$。较小三角形的边 $= k \\times$ 较大三角形对应边，即 $8 = \\dfrac{4}{5} \\times x$，解得 $x = 10$。',
      },
      {
        type: 'choice',
        prompt:
          '$\\triangle ABC$ 中，$DE \\parallel BC$，$AD = 4$，$DB = 6$，$DE = 6$，则 $BC = $',
        options: ['$9$', '$10$', '$15$', '$12$'],
        answer: 2,
        explain:
          '$DE \\parallel BC$，公共角 $\\angle A$，AA 得 $\\triangle ADE \\sim \\triangle ABC$。相似比 $k = \\dfrac{AD}{AB} = \\dfrac{4}{10} = \\dfrac{2}{5}$，故 $BC = \\dfrac{DE}{k} = \\dfrac{6}{2/5} = 15$。',
      },
      {
        type: 'judge',
        prompt:
          'SSA（两边成比例且其中一边所对的角相等）可以判定两三角形相似。',
        answer: false,
        explain:
          'SSA 存在"摆动歧义"，无法唯一确定三角形形状，所以 SSA 既不能判定全等，也不能判定相似。唯一例外是直角三角形的 HL 型（斜边与直角边成比例），因为直角排除了歧义。',
      },
      {
        type: 'match',
        prompt: '将判定条件与对应的定理名称配对。',
        left: [
          '两角对应相等',
          '两边成比例且夹角相等',
          '三边对应成比例',
        ],
        right: ['SSS 判定', 'AA 判定', 'SAS 判定'],
        explain:
          'AA——两角相等；SAS——两边比例且夹角相等；SSS——三边全部成比例。',
      },
      {
        type: 'input',
        prompt:
          '两个相似三角形的周长之比为 $3:4$，较大三角形面积为 $48$，较小三角形面积为多少？',
        accept: ['27'],
        explain:
          '周长比 $= k = 3/4$，面积比 $= k^2 = 9/16$。设较小面积为 $S$，则 $S/48 = 9/16$，解得 $S = 27$。',
      },
    ],
  },
]
