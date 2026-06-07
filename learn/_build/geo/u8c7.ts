export const LESSONS = [
  {
    id: 'u8c7',
    title: '中位线定理',
    subtitle: '第8单元第7关：两中点之间的神奇线段',
    intro: [
      {
        title: '中位线的定义（两种）',
        body: '三角形中位线：连接三角形两边中点的线段（两端点都是中点）。注意区别：中线是"顶点到对边中点"，中位线是"边中点到边中点"，两者完全不同。梯形中位线：连接梯形两腰中点的线段（连接的是两腰，不是两底）。',
        image: 'thm-midsegment-triangle.svg',
        imageCaption: '三角形中位线 $DE$ 与倍长中位线构造：延长 $DE$ 到 $F$ 使 $EF=DE$',
        tip: '看到"两个中点"，本能反应是"三角形中位线定理"——平行且半长两个结论打包送出。',
      },
      {
        title: '三角形中位线定理',
        body: '若 $D$、$E$ 分别是 $\\triangle ABC$ 的 $AB$、$AC$ 上的中点，则 $DE \\parallel BC$ 且 $DE = \\frac{1}{2} BC$。定理一次给出两条结论——平行和半长。解题时两条都要记得用，不要只用其中一条。',
        formula: 'DE \\parallel BC,\\quad DE = \\dfrac{1}{2}BC',
      },
      {
        title: '三角形中位线的证明思路',
        body: '倍长法：延长 $DE$ 至 $F$，使 $EF = DE$，连接 $CF$。在 $\\triangle ADE$ 与 $\\triangle CFE$ 中，$AE = CE$（$E$ 是中点），$\\angle AED = \\angle CEF$（对顶角），$DE = FE$，SAS 全等。故 $AD = CF$，$BD \\parallel CF$（内错角相等）。$BD = AD = CF$，一组对边平行且相等 → $BDFC$ 是平行四边形，$DF \\parallel BC$，$DE = \\frac{1}{2} DF = \\frac{1}{2} BC$。',
        steps: [
          '延长 $DE$ 至 $F$，使 $EF = DE$，连 $CF$',
          'SAS 全等：$\\triangle ADE \\cong \\triangle CFE$，得 $AD = CF$，$AD \\parallel CF$',
          '故 $BD = CF$，$BD \\parallel CF$ → $BDFC$ 是平行四边形',
          '$DF \\parallel BC$，$DF = BC$，$DE = \\frac{1}{2}DF = \\frac{1}{2}BC$',
        ],
      },
      {
        title: '梯形中位线定理',
        body: '梯形 $ABCD$ 中 $AD \\parallel BC$，$M$、$N$ 分别是腰 $AB$、$CD$ 的中点，则中位线 $MN \\parallel AD \\parallel BC$ 且 $MN = \\frac{a+b}{2}$（$a$、$b$ 分别是上底和下底的长度）。梯形面积可改写为 $S = MN \\cdot h$，即"中位线乘以高"。',
        image: 'thm-midsegment-trapezoid.svg',
        imageCaption: '梯形中位线 $MN$ 连两腰中点，平行于两底，长度为两底之和的一半',
        formula: 'MN = \\dfrac{a+b}{2}',
        tip: '梯形中位线公式是"两底之和的一半"，不是差——别和等腰梯形里截出的小段 $\\frac{b-a}{2}$ 搞混。',
      },
      {
        title: '几何意义',
        body: '三角形三条中位线把原三角形分成4个全等的小三角形，每个与原三角形相似，相似比 $1:2$，面积比 $1:4$。中间倒置的小三角形和三个角落的小三角形完全全等。',
      },
      {
        title: '例题：中位线嵌套',
        body: '$\\triangle ABC$ 中，$BC = 16$。$D$、$E$ 分别是 $AB$、$AC$ 的中点；$F$、$G$ 分别是 $AD$、$AE$ 的中点。第一次套用：$DE = \\frac{1}{2} BC = 8$（$\\triangle ABC$ 中位线）。第二次套用：$FG = \\frac{1}{2} DE = 4$（$\\triangle ADE$ 中位线）。每嵌套一层，长度缩短一半。',
        image: 'ex-p4-07-3.svg',
        imageCaption: '中位线嵌套：$BC=16$，$DE=8$，$FG=4$',
        reveal: {
          q: '梯形 $ABCD$ 中 $AD \\parallel BC$，中位线 $MN = 7$，$AD = 3$，求 $BC$。',
          a: '$MN = \\frac{AD + BC}{2}$，$7 = \\frac{3 + BC}{2}$，$BC = 14 - 3 = 11$。',
        },
      },
      {
        title: '易错点提醒',
        body: '①中位线连两"中点"，中线连"顶点与对边中点"，不要混淆；②只有两端点都是中点时才能套用中位线定理，一般等分点需用相似；③梯形中位线连"两腰"中点，不是"两底"中点；④中位线定理给两条结论（平行+半长），两条都要记得使用。',
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '$\\triangle ABC$ 中，$BC = 10$，$D$、$E$ 分别是 $AB$、$AC$ 的中点，则 $DE$ 等于多少？',
        options: ['$10$', '$5$', '$20$', '$\\frac{10}{3}$'],
        answer: 1,
        explain: '三角形中位线定理：$DE = \\frac{1}{2} BC = \\frac{1}{2} \\times 10 = 5$。',
      },
      {
        type: 'judge',
        prompt: '梯形的中位线连接的是两底的中点。',
        answer: false,
        explain: '梯形中位线连接的是两腰的中点，不是两底的中点。这是常见混淆点。',
      },
      {
        type: 'input',
        prompt: '梯形 $ABCD$ 中，$AD = 5$，$BC = 11$，$M$、$N$ 分别是腰 $AB$、$CD$ 的中点，求中位线 $MN$ 的长（填数字）。',
        accept: ['8', '8cm'],
        explain: '$MN = \\frac{AD + BC}{2} = \\frac{5 + 11}{2} = 8$。',
      },
      {
        type: 'choice',
        prompt: '$\\triangle ABC$ 中，三条中位线把它分成几个全等的小三角形？',
        options: ['$2$ 个', '$3$ 个', '$4$ 个', '$6$ 个'],
        answer: 2,
        explain: '三角形三条中位线把原三角形分成4个全等的小三角形，面积各为原三角形的 $\\frac{1}{4}$。',
      },
      {
        type: 'input',
        prompt: '梯形 $ABCD$ 中，中位线 $MN = 9$，上底 $AD = 6$，求下底 $BC$（填数字）。',
        accept: ['12', '12cm'],
        explain: '$MN = \\frac{AD + BC}{2}$，$9 = \\frac{6 + BC}{2}$，$BC = 18 - 6 = 12$。',
      },
      {
        type: 'judge',
        prompt: '中位线定理只给出"长度是第三边的一半"这一条结论。',
        answer: false,
        explain: '三角形中位线定理给出两条结论：平行（$DE \\parallel BC$）和半长（$DE = \\frac{1}{2}BC$）。解题时两条都可以使用。',
      },
      {
        type: 'match',
        prompt: '将线段名称与定义配对。',
        left: ['三角形中线', '三角形中位线', '梯形中位线', '梯形的高'],
        right: ['连接两腰中点的线段', '连接顶点与对边中点的线段', '上下底之间的垂直距离', '连接两边中点的线段'],
        explain: '中线→顶点到对边中点；三角形中位线→两边中点；梯形中位线→两腰中点；梯形高→垂直距离。',
      },
      {
        type: 'choice',
        prompt: '$\\triangle ABC$ 中，$BC = 16$。$D$、$E$ 是 $AB$、$AC$ 中点，$F$、$G$ 是 $AD$、$AE$ 中点，则 $FG$ 等于多少？',
        options: ['$8$', '$4$', '$2$', '$6$'],
        answer: 1,
        explain: '$DE = \\frac{1}{2}BC = 8$（$\\triangle ABC$ 中位线），$FG = \\frac{1}{2}DE = 4$（$\\triangle ADE$ 中位线）。嵌套两次，各缩半。',
      },
    ],
  },
]
