export const LESSONS = [
  {
    id: 'u8c4',
    title: '菱形',
    subtitle: '第8单元第4关：四边相等的平行四边形',
    intro: [
      {
        title: '菱形的定义',
        body: '有一组邻边相等的平行四边形叫做菱形。由于平行四边形对边相等，邻边只要相等，四条边就全部相等。因此等价定义为：四条边都相等的四边形。菱形就像一张"被推歪的正方形"，四边等长但角不一定是直角。',
        image: 'thm-rhombus-properties.svg',
        imageCaption: '菱形 $ABCD$：四边相等，对角线 $AC \\perp BD$ 于 $O$，互相平分并平分对角',
      },
      {
        title: '菱形的特有性质',
        body: '在继承平行四边形所有性质的基础上，菱形额外拥有三条特有性质：①四条边都相等；②对角线互相垂直平分，且每条对角线平分一组对角；③是轴对称图形，两条对称轴就是两条对角线所在直线。',
        formula: 'AC \\perp BD',
        tip: '矩形的对称轴是"对边中点连线"，菱形的对称轴是"对角线"——这是两者最容易混淆的地方。',
      },
      {
        title: '菱形面积公式',
        body: '菱形面积有两种算法：①底乘高（继承自平行四边形）：$S = a \\cdot h$；②对角线乘积的一半：$S = \\frac{1}{2} d_1 d_2$。公式②来源：对角线把菱形分成4个全等的直角三角形，每个面积为 $\\frac{d_1}{2} \\cdot \\frac{d_2}{2} \\cdot \\frac{1}{2}$，四个合计 $\\frac{d_1 d_2}{2}$。',
        formula: 'S = \\dfrac{1}{2} d_1 d_2',
        tip: '已知两对角线时，优先用 $\\frac{1}{2} d_1 d_2$，比"底乘高"算得更快。',
      },
      {
        title: '菱形的判定',
        body: '判定1：有一组邻边相等的平行四边形是菱形（定义）。判定2：四条边都相等的四边形是菱形。判定3：对角线互相垂直的平行四边形是菱形。注意：判定1和3都要先有平行四边形前提，判定2可以直接用。',
        image: 'ex-p4-04-1.svg',
        imageCaption: '$AC=8$、$BD=6$；对角线将菱形分成4个全等直角三角形（直角边 $4,3$）',
      },
      {
        title: '例题1：由对角线求面积与边长',
        body: '菱形 $ABCD$ 中，$AC = 8$，$BD = 6$。面积 $S = \\frac{1}{2} \\times 8 \\times 6 = 24$。对角线交于 $O$，$OA = 4$，$OB = 3$，$\\angle AOB = 90^\\circ$。由勾股定理，$AB = \\sqrt{4^2 + 3^2} = 5$。',
        steps: [
          '面积：$S = \\frac{1}{2} d_1 d_2 = \\frac{1}{2} \\times 8 \\times 6 = 24$',
          '对角线互相平分：$OA = 4$，$OB = 3$',
          '对角线垂直：$\\angle AOB = 90^\\circ$',
          '勾股定理：$AB = \\sqrt{4^2 + 3^2} = 5$',
        ],
      },
      {
        title: '例题2：菱形 + 角度 = 特殊直角三角形',
        body: '菱形 $ABCD$ 中，$\\angle BAD = 60^\\circ$，$AB = 4$。对角线 $AC$ 平分 $\\angle BAD$，故 $\\angle BAO = 30^\\circ$；又 $AC \\perp BD$，$\\triangle AOB$ 是 $30^\\circ$-$60^\\circ$-$90^\\circ$ 直角三角形。$OB = \\frac{1}{2} AB = 2$，$OA = \\sqrt{AB^2 - OB^2} = 2\\sqrt{3}$，所以 $BD = 4$，$AC = 4\\sqrt{3}$。',
        image: 'ex-p4-04-2.svg',
        imageCaption: '$\\angle BAD=60^\\circ$，$AB=4$；$\\triangle AOB$ 为 $30^\\circ$-$60^\\circ$-$90^\\circ$ 直角三角形',
      },
      {
        title: '易错点提醒',
        body: '①菱形不等于正方形，菱形只要求四边相等，角不一定是直角；②菱形对角线"垂直但不一定相等"，矩形对角线"相等但不一定垂直"——两者都同时满足才是正方形；③判定3必须先有平行四边形前提，筝形对角线也垂直，但不一定是菱形；④对角线平分对角是菱形的特有性质，矩形没有。',
        reveal: {
          q: '菱形 $ABCD$ 中，$AC = 10$，$BD = 24$，求边长和面积。',
          a: '$OA = 5$，$OB = 12$，$AB = \\sqrt{5^2 + 12^2} = 13$（勾股数5-12-13）。面积 $= \\frac{1}{2} \\times 10 \\times 24 = 120$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '菱形 $ABCD$ 中，$AC = 6$，$BD = 8$，菱形的面积是多少？',
        options: ['$24$', '$48$', '$12$', '$36$'],
        answer: 0,
        explain: '$S = \\frac{1}{2} d_1 d_2 = \\frac{1}{2} \\times 6 \\times 8 = 24$。',
      },
      {
        type: 'judge',
        prompt: '菱形的对角线相等。',
        answer: false,
        explain: '菱形的对角线互相垂直平分，但一般不相等。对角线相等是矩形的特征。只有正方形的对角线既垂直又相等。',
      },
      {
        type: 'input',
        prompt: '菱形 $ABCD$ 中，$AC = 10$，$BD = 24$，求边长 $AB$（填数字）。',
        accept: ['13', '13cm'],
        explain: '$OA = 5$，$OB = 12$，$AB = \\sqrt{5^2 + 12^2} = \\sqrt{169} = 13$（勾股数 5-12-13）。',
      },
      {
        type: 'choice',
        prompt: '下列条件中，能直接判定四边形 $ABCD$ 是菱形的是？',
        options: [
          '对角线互相垂直',
          '四条边都相等',
          '对角线互相平分',
          '两组对角分别相等',
        ],
        answer: 1,
        explain: '"四条边都相等"是菱形的判定2，不需要先证平行四边形，可以直接使用。其他选项单独成立时有反例。',
      },
      {
        type: 'judge',
        prompt: '菱形是轴对称图形，其对称轴是两条对角线所在直线。',
        answer: true,
        explain: '菱形的两条对称轴正是两条对角线所在直线，这与矩形（对称轴为对边中点连线）不同。',
      },
      {
        type: 'input',
        prompt: '菱形 $ABCD$ 中，$\\angle A = 60^\\circ$，$AB = 8$，则较短对角线 $BD$ 等于多少？',
        accept: ['8', '8cm'],
        explain: '$\\angle A = 60^\\circ$，$\\triangle ABD$ 中 $AB = AD = 8$ 且顶角 $\\angle A = 60^\\circ$，是等边三角形，故 $BD = 8$。',
      },
      {
        type: 'match',
        prompt: '将对角线特征与对应的特殊四边形配对。',
        left: ['对角线相等且互相平分', '对角线垂直且互相平分', '对角线相等、垂直且互相平分', '对角线仅互相平分'],
        right: ['菱形', '矩形', '平行四边形', '正方形'],
        explain: '平行四边形→仅互相平分；矩形→相等且互相平分；菱形→垂直且互相平分；正方形→三者兼备。',
      },
      {
        type: 'choice',
        prompt: '菱形 $ABCD$ 中，$\\angle ABC = 120^\\circ$，$BD = 6$，则边长 $AB$ 等于多少？',
        options: ['$3$', '$6$', '$3\\sqrt{3}$', '$6\\sqrt{3}$'],
        answer: 1,
        explain: '$\\angle ABO = 60^\\circ$（对角线平分对角），$OB = 3$，$\\angle AOB = 90^\\circ$，$AB = OB / \\cos 60^\\circ = 3 \\div \\frac{1}{2} = 6$。',
      },
    ],
  },
]
