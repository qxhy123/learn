export const LESSONS = [
  {
    id: 'u8c2',
    title: '平行四边形',
    subtitle: '第8单元第2关：四边形大家族的"基础款"',
    intro: [
      {
        title: '平行四边形的定义',
        body: '两组对边分别平行的四边形叫做平行四边形，符号"▱"。▱$ABCD$ 表示 $AB \\parallel DC$、$AD \\parallel BC$。顶点字母按沿同一方向依次绕一圈的顺序排列，不能乱写。',
        image: 'thm-parallelogram-properties.svg',
        imageCaption: '▱$ABCD$：对边平行且相等，对角线互相平分于 $O$',
      },
      {
        title: '平行四边形的性质（必要条件）',
        body: '▱$ABCD$ 必满足：①两组对边分别平行且相等；②两组对角分别相等，邻角互补（$\\angle A + \\angle B = 180^\\circ$）；③对角线互相平分（$OA=OC$，$OB=OD$）；④是中心对称图形，对称中心为对角线交点 $O$。',
        formula: 'OA = OC,\\quad OB = OD',
      },
      {
        title: '平行四边形的判定（五条充分条件）',
        body: '四边形满足下列任一条件即为平行四边形：①两组对边分别平行；②两组对边分别相等；③一组对边平行且相等；④两组对角分别相等；⑤对角线互相平分。',
        image: 'thm-parallelogram-criteria.svg',
        imageCaption: '判定3：一组对边平行且相等 $\\Rightarrow$ 平行四边形',
        tip: '判定5"对角线互相平分"是最常用的"升级暗号"——一看到这个条件就立刻推出平行四边形。',
      },
      {
        title: '核心证明：对角线互相平分',
        body: '在 ▱$ABCD$ 中，对角线 $AC$、$BD$ 交于 $O$。由 $AB \\parallel CD$ 得内错角 $\\angle OAB = \\angle OCD$，$\\angle OBA = \\angle ODC$；又对边相等 $AB = CD$。由 ASA，$\\triangle AOB \\cong \\triangle COD$，故 $OA = OC$，$OB = OD$。',
        steps: [
          '$AB \\parallel CD$ 给内错角：$\\angle OAB = \\angle OCD$，$\\angle OBA = \\angle ODC$',
          '对边相等：$AB = CD$',
          'ASA 全等：$\\triangle AOB \\cong \\triangle COD$',
          '对应边相等：$OA = OC$，$OB = OD$',
        ],
      },
      {
        title: '例题1：已知边角求其余量',
        body: '▱$ABCD$ 中，$\\angle A = 70^\\circ$，$AB = 5$，$BC = 8$。由邻角互补：$\\angle B = 180^\\circ - 70^\\circ = 110^\\circ$；由对角相等：$\\angle C = 70^\\circ$，$\\angle D = 110^\\circ$；由对边相等：$CD = AB = 5$，$AD = BC = 8$。',
        image: 'ex-p4-02-1.svg',
        imageCaption: '▱$ABCD$，$\\angle A=70^\\circ$，$AB=5$，$BC=8$',
      },
      {
        title: '解题套路："条件→形态升级→性质激活"',
        body: '解平行四边形题的核心思路：看到"对角线互相平分"→ 判定5升级为 ▱ → 激活全套性质（对边平行、对边相等、对角相等）。反过来，看到"一组对边平行且相等"→ 判定3。',
        reveal: {
          q: '▱$ABCD$ 中，$\\angle A : \\angle B = 2:7$，求 $\\angle A$。',
          a: '邻角互补：$\\angle A + \\angle B = 180^\\circ$；设 $\\angle A = 2k$，$\\angle B = 7k$，则 $9k = 180^\\circ$，$k = 20^\\circ$，$\\angle A = 40^\\circ$。',
        },
      },
      {
        title: '易错点提醒',
        body: '判定3"一组对边平行且相等"中，平行与相等必须是同一组对边，两条缺一不可。若一组平行、另一组相等，可能是等腰梯形，不是平行四边形。另外，平行四边形一般不是轴对称图形，只有矩形和菱形才是。',
        tip: '"一组对边平行，另一组对边相等" 不能推平行四边形——等腰梯形是经典反例！',
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '▱$ABCD$ 中，$\\angle A = 65^\\circ$，则 $\\angle B$ 等于多少？',
        options: ['$65^\\circ$', '$115^\\circ$', '$130^\\circ$', '$90^\\circ$'],
        answer: 1,
        explain: '平行四边形邻角互补：$\\angle B = 180^\\circ - 65^\\circ = 115^\\circ$。',
      },
      {
        type: 'judge',
        prompt: '平行四边形的对角线互相垂直。',
        answer: false,
        explain: '平行四边形的对角线只保证互相平分，不保证垂直。对角线垂直是菱形的特有性质。',
      },
      {
        type: 'input',
        prompt: '▱$ABCD$ 中，对角线 $AC = 12$，$BD = 8$，交点为 $O$，求 $OA$ 的长。',
        accept: ['6', '6cm', '6单位'],
        explain: '对角线互相平分：$OA = \\frac{1}{2} AC = \\frac{1}{2} \\times 12 = 6$。',
      },
      {
        type: 'choice',
        prompt: '下列哪个条件不能直接判定四边形 $ABCD$ 是平行四边形？',
        options: [
          '对角线互相平分',
          '两组对边分别相等',
          '一组对边平行，另一组对边相等',
          '两组对角分别相等',
        ],
        answer: 2,
        explain: '"一组对边平行，另一组对边相等"不能判定平行四边形，等腰梯形满足此条件但不是平行四边形。其余三项均为有效判定条件。',
      },
      {
        type: 'match',
        prompt: '将平行四边形的性质与对应的条件配对。',
        left: ['对边关系', '对角关系', '对角线关系', '对称性'],
        right: ['互相平分', '中心对称图形', '分别相等，邻角互补', '分别平行且相等'],
        explain: '对边→平行且相等；对角→相等，邻角互补；对角线→互相平分；对称性→中心对称。',
      },
      {
        type: 'judge',
        prompt: '若四边形 $ABCD$ 中 $AB = CD$ 且 $AD = BC$，则 $ABCD$ 一定是平行四边形。',
        answer: true,
        explain: '两组对边分别相等是平行四边形的判定条件2，结论成立。',
      },
      {
        type: 'input',
        prompt: '▱$ABCD$ 中，$AB = 6$，$BC = 9$，则 $CD$ 等于多少？',
        accept: ['6', '6cm'],
        explain: '平行四边形对边相等：$CD = AB = 6$。',
      },
      {
        type: 'choice',
        prompt: '▱$ABCD$ 中，$\\angle B = 120^\\circ$，则 $\\angle D$ 等于多少？',
        options: ['$60^\\circ$', '$90^\\circ$', '$120^\\circ$', '$240^\\circ$'],
        answer: 2,
        explain: '平行四边形对角相等：$\\angle D = \\angle B = 120^\\circ$。',
      },
    ],
  },
]
