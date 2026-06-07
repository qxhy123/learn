export const LESSONS = [
  {
    id: 'u8c5',
    title: '正方形',
    subtitle: '第8单元第5关：矩形与菱形的"交集"',
    intro: [
      {
        title: '正方形的定义与双重身份',
        body: '四条边都相等且四个角都是直角的四边形叫做正方形。它同时是矩形（四角直角）也是菱形（四边相等），处于两者的"交集"位置。等价定义：有一组邻边相等的矩形，或有一个角是直角的菱形。',
        image: 'thm-square-properties.svg',
        imageCaption: '正方形：四边相等，对角线相等且垂直平分，对角线平分对角 $45^\\circ$',
      },
      {
        title: '正方形的完整性质',
        body: '正方形继承矩形和菱形的所有性质：四边相等、四角直角；对角线相等（来自矩形）且互相垂直平分（来自菱形）；每条对角线平分一组对角（各 $45^\\circ$）；是中心对称图形，同时有 $4$ 条对称轴——两条对角线和两条对边中点连线。',
        formula: 'AC = BD,\\quad AC \\perp BD',
        tip: '初中最常见四边形中，正方形的对称轴最多，共4条。',
      },
      {
        title: '边长与对角线的换算',
        body: '正方形对角线把它分成4个全等的 $45^\\circ$-$45^\\circ$-$90^\\circ$ 等腰直角三角形。设边长为 $a$，对角线为 $d$，由勾股定理：$d = a\\sqrt{2}$，反过来 $a = \\frac{\\sqrt{2}}{2} d$。',
        formula: 'd = a\\sqrt{2}',
        tip: '边长乘 $\\sqrt{2}$ 得对角线；对角线除以 $\\sqrt{2}$ 得边长。务必背熟！',
      },
      {
        title: '正方形的判定',
        body: '判定1：有一组邻边相等的矩形是正方形。判定2：有一个角是直角的菱形是正方形。判定3：对角线相等的菱形是正方形。判定4：对角线互相垂直的矩形是正方形。总结：先升到矩形或菱形，再补上对方独有的那条性质，就升级为正方形。',
        image: 'ex-p4-05-1.svg',
        imageCaption: '正方形 $ABCD$，$AB=6$，对角线 $AC = 6\\sqrt{2}$',
      },
      {
        title: '例题：正方形中的经典全等',
        body: '正方形 $ABCD$ 中，$E$ 在 $BC$ 上，$F$ 在 $CD$ 上，$BE = CF$。求证 $AE \\perp BF$。思路：考察 $\\triangle ABE$ 与 $\\triangle BCF$：$AB = BC$，$\\angle ABE = \\angle BCF = 90^\\circ$，$BE = CF$，由 SAS 全等，$\\angle BAE = \\angle CBF$，再由两角之和 $= 90^\\circ$ 推出垂直。',
        steps: [
          '$AB = BC$，$\\angle B = \\angle C = 90^\\circ$，$BE = CF$',
          'SAS 得 $\\triangle ABE \\cong \\triangle BCF$',
          '故 $\\angle BAE = \\angle CBF$',
          '$\\angle BAE + \\angle AEB = 90^\\circ$，代入得 $\\angle CBF + \\angle AEB = 90^\\circ$，即 $AE \\perp BF$',
        ],
      },
      {
        title: '正方形的两大经典模型',
        body: '半角模型：正方形 + 顶角的 $45^\\circ$ 半角 → 绕顶点旋转 $90^\\circ$，把两段线段"拼"到同一直线。手拉手模型：两正方形共顶点 → 找 $AB = AD$，$AE = AG$，夹角相同 → SAS 全等，连线等长且垂直。',
        image: 'ex-p4-05-3.svg',
        imageCaption: '两正方形共顶点 $A$，连接 $BE$、$DG$；由手拉手模型得 $BE = DG$ 且 $BE \\perp DG$',
        reveal: {
          q: '正方形 $ABCD$ 的对角线 $AC = 4\\sqrt{2}$，求边长和面积。',
          a: '边长 $a = \\frac{d}{\\sqrt{2}} = \\frac{4\\sqrt{2}}{\\sqrt{2}} = 4$；面积 $= a^2 = 16$。也可用 $S = \\frac{1}{2}d^2 = \\frac{1}{2}\\times 32 = 16$。',
        },
      },
      {
        title: '易错点提醒',
        body: '①正方形是矩形与菱形的交集，判定时必须两套性质都满足；②"四条边相等"只是菱形，未必是正方形；③"对角线相等且垂直"还不够，必须先确认是平行四边形或矩形/菱形；④对角线分出的4个三角形是等腰直角三角形，面积各为 $\\frac{d^2}{8}$；⑤正方形有4条对称轴（2条对角线 + 2条对边中点连线），不要只数出2条。',
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '正方形 $ABCD$ 的边长为 $5$，对角线长是多少？',
        options: ['$5$', '$5\\sqrt{2}$', '$10$', '$\\frac{5\\sqrt{2}}{2}$'],
        answer: 1,
        explain: '正方形对角线 $= $ 边长 $\\times \\sqrt{2} = 5\\sqrt{2}$。',
      },
      {
        type: 'judge',
        prompt: '对角线互相垂直且相等的四边形一定是正方形。',
        answer: false,
        explain: '对角线互相垂直且相等还不够，还需要对角线互相平分（即先确认是平行四边形）。若不互相平分，只是"垂直且相等"，可能是某些特殊四边形但不一定是正方形。',
      },
      {
        type: 'input',
        prompt: '正方形的对角线 $AC = 6\\sqrt{2}$，求边长 $AB$（填数字）。',
        accept: ['6', '6cm'],
        explain: '边长 $= \\frac{d}{\\sqrt{2}} = \\frac{6\\sqrt{2}}{\\sqrt{2}} = 6$。',
      },
      {
        type: 'choice',
        prompt: '下列说法正确的是？',
        options: [
          '菱形一定是正方形',
          '矩形一定是正方形',
          '正方形一定是矩形',
          '平行四边形一定是正方形',
        ],
        answer: 2,
        explain: '正方形 = 矩形 $\\cap$ 菱形，所以正方形一定是矩形，也一定是菱形，但反过来不成立。',
      },
      {
        type: 'judge',
        prompt: '正方形有 $4$ 条对称轴。',
        answer: true,
        explain: '正方形的4条对称轴分别是：两条对角线所在直线，以及两组对边中点连线所在直线。',
      },
      {
        type: 'match',
        prompt: '将特殊四边形与其对角线特征配对。',
        left: ['矩形', '菱形', '正方形', '一般平行四边形'],
        right: ['相等+垂直+互相平分', '相等+互相平分，不一定垂直', '仅互相平分', '垂直+互相平分，不一定相等'],
        explain: '矩形对角线相等且互相平分；菱形对角线垂直且互相平分；正方形三者兼备；一般平行四边形仅互相平分。',
      },
      {
        type: 'choice',
        prompt: '正方形 $ABCD$ 的对角线 $AC$、$BD$ 交于 $O$，$\\angle AOB$ 等于多少度？',
        options: ['$45^\\circ$', '$60^\\circ$', '$90^\\circ$', '$135^\\circ$'],
        answer: 2,
        explain: '正方形的对角线互相垂直，所以 $\\angle AOB = 90^\\circ$。',
      },
      {
        type: 'input',
        prompt: '正方形边长为 $4$，求其面积（填数字）。',
        accept: ['16', '16平方单位'],
        explain: '正方形面积 $= $ 边长 $^2 = 4^2 = 16$。也可用对角线公式：$d = 4\\sqrt{2}$，$S = \\frac{1}{2}d^2 = \\frac{1}{2}\\times 32 = 16$。',
      },
    ],
  },
]
