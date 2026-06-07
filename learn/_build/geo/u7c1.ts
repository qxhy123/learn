export const LESSONS = [
  {
    id: 'u7c1',
    title: '等腰三角形',
    subtitle: '等边对等角·三线合一·等角对等边',
    intro: [
      {
        title: '等腰三角形的各部分名称',
        body:
          '有两条边相等的三角形叫作**等腰三角形**。统一约定 $\\triangle ABC$ 中 $AB = AC$，则：\n- **腰**：相等的两条边 $AB$、$AC$；\n- **底边**：第三条边 $BC$；\n- **顶角**：两腰所夹的角 $\\angle A$；\n- **底角**：腰与底边所成的角 $\\angle B$、$\\angle C$。\n\n等边三角形是等腰三角形的特殊情形，每个角都是 $60^\\circ$。',
      },
      {
        title: '核心性质 1：等边对等角',
        body:
          '在 $\\triangle ABC$ 中，若 $AB = AC$，则两底角相等 $\\angle B = \\angle C$。\n\n**证明思路：** 作底边 $BC$ 的中线 $AD$（$D$ 为 $BC$ 中点），在 $\\triangle ABD$ 与 $\\triangle ACD$ 中：$AB = AC$（已知），$BD = CD$（中点），$AD = AD$（公共边），由 **SSS** 得两三角形全等，从而 $\\angle B = \\angle C$。',
        formula: 'AB = AC \\Rightarrow \\angle B = \\angle C',
      },
      {
        title: '核心性质 2：三线合一',
        body:
          '等腰三角形**顶角平分线、底边上的中线、底边上的高**互相重合，统称「三线合一」。\n\n利用上面的全等 $\\triangle ABD \\cong \\triangle ACD$，可同时得到：\n- $\\angle BAD = \\angle CAD$，故 $AD$ 是顶角平分线；\n- $D$ 为 $BC$ 中点，故 $AD$ 是底边中线；\n- $\\angle ADB = \\angle ADC = 90^\\circ$，故 $AD \\perp BC$，$AD$ 是底边上的高。\n\n**前提：** 三线合一只对"底边对应的三条线"成立，腰上的高或底角平分线一般不重合，切勿乱套。',
        image: 'thm-isosceles-three-in-one.svg',
        imageCaption: '顶角平分线 $AD$ 同时是底边中线和底边上的高（三线合一）',
      },
      {
        title: '对称性',
        body:
          '等腰三角形是**轴对称图形**，其对称轴是顶角平分线所在的直线，也即底边 $BC$ 的垂直平分线。\n\n推论：若点 $E$ 在对称轴（即 $AD$）上，则 $E$ 到 $B$、$C$ 两点的距离相等，即 $EB = EC$。',
        formula: 'E \\in AD \\Rightarrow EB = EC',
      },
      {
        title: '判定：等角对等边',
        body:
          '「等边对等角」与「等角对等边」互为逆命题，两者都成立：\n\n若 $\\triangle ABC$ 中 $\\angle B = \\angle C$，则 $AB = AC$（等腰三角形）。\n\n**证明：** 作 $\\angle A$ 的平分线 $AD$，在 $\\triangle ABD$ 与 $\\triangle ACD$ 中，$\\angle BAD = \\angle CAD$（作图），$\\angle B = \\angle C$（已知），$AD = AD$（公共边），由 **AAS** 得全等，故 $AB = AC$。',
        formula: '\\angle B = \\angle C \\Rightarrow AB = AC',
      },
      {
        title: '分类讨论：给定一个角求其余角',
        body:
          '**例：** 等腰三角形的一个内角为 $50^\\circ$，求其余两个内角。\n\n题目未说是顶角还是底角，必须**分两种情形**：\n\n**情形 1（$50^\\circ$ 是顶角）：** 两底角 $= (180^\\circ - 50^\\circ) \\div 2 = 65^\\circ$，其余两角为 $65^\\circ$、$65^\\circ$。\n\n**情形 2（$50^\\circ$ 是底角）：** 另一底角也是 $50^\\circ$，顶角 $= 180^\\circ - 50^\\circ - 50^\\circ = 80^\\circ$，其余两角为 $50^\\circ$、$80^\\circ$。\n\n综上，其余两角为 $65^\\circ$、$65^\\circ$ 或 $50^\\circ$、$80^\\circ$。',
        tip: '底角必须是锐角（两个底角都 $\\ge 90^\\circ$ 则角和超过 $180^\\circ$），若题目给出的是钝角，则它只能是顶角，直接排除"它是底角"这一支。',
      },
      {
        title: '分类讨论：给定边长求周长',
        body:
          '**例：** 等腰三角形的一边长为 $4$，另一边长为 $9$，求周长。\n\n两边长 $4$ 和 $9$ 各有可能是腰或底边，分两种情形并**用三边关系验证**：\n\n**情形 1（腰 $= 4$，底 $= 9$）：** 需检验 $4 + 4 = 8 > 9$，但 $8 < 9$，**不能构成三角形，舍去。**\n\n**情形 2（腰 $= 9$，底 $= 4$）：** 检验 $9 + 9 = 18 > 4$，$9 + 4 = 13 > 9$，合法。周长 $= 9 + 9 + 4 = 22$。',
        tip: '边长分类后务必用三角不等式验证，否则可能漏掉"无解需舍去"的情形。',
        reveal: {
          q: '为什么腰 $= 4$ 时必须舍去？',
          a: '两腰之和 $4 + 4 = 8$ 小于底边 $9$，违反三角形三边关系（两边之和必须大于第三边），故不能构成三角形，该情形须舍去。',
        },
      },
      {
        title: '易错点总结',
        body:
          '1. 给定"某个角"未说明类型时，必须分**顶角**与**底角**两种情形讨论。\n2. 给定"某条边"未说明类型时，必须分**腰**与**底边**两种情形，并用三边关系验证是否合法。\n3. 「三线合一」只限底边的三条线，**腰上的高、腰上的中线、底角的平分线一般不重合**。\n4. 「等边对等角」只在**同一三角形内**有效，不跨三角形。\n5. 等边三角形是等腰三角形的特殊情形，但「等腰」推不出「等边」。',
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt:
          '等腰三角形 $\\triangle ABC$ 中 $AB = AC$，$AD$ 是底边 $BC$ 上的高，则下列结论中**错误**的是',
        options: [
          '$AD$ 是 $\\angle BAC$ 的平分线',
          '$BD = CD$',
          '$AD$ 是腰 $AB$ 上的高',
          '$\\triangle ABD \\cong \\triangle ACD$',
        ],
        answer: 2,
        explain:
          '由三线合一，$AD$（底边上的高）同时也是顶角平分线和底边中线，故选项 A、B 正确，选项 D 由全等论证也正确。选项 C 错误：$AD$ 是**底边** $BC$ 上的高，而非腰 $AB$ 上的高，两者是不同的线段。',
      },
      {
        type: 'judge',
        prompt:
          '等腰三角形的顶角平分线、底边上的中线、底边上的高三线重合，因此等腰三角形任意一条角平分线都是对边上的高。',
        answer: false,
        explain:
          '三线合一只针对「顶角的平分线」与「底边上的中线、高」，底角的平分线一般不是对边上的高，也不一定是对边上的中线，不能将三线合一随意推广。',
      },
      {
        type: 'input',
        prompt:
          '$\\triangle ABC$ 中 $AB = AC$，$AD \\perp BC$，垂足为 $D$，$BC = 10$，$AD = 12$，求腰 $AB$ 的长。（结果保留整数）',
        accept: ['13'],
        explain:
          '由三线合一，$AD$ 是底边中线，故 $BD = BC/2 = 5$。在直角 $\\triangle ABD$ 中，$AB = \\sqrt{AD^2 + BD^2} = \\sqrt{12^2 + 5^2} = \\sqrt{144 + 25} = \\sqrt{169} = 13$。',
      },
      {
        type: 'choice',
        prompt:
          '等腰三角形的一个内角为 $120^\\circ$，则其余两个底角各为',
        options: ['$60^\\circ$', '$30^\\circ$', '$40^\\circ$', '$50^\\circ$'],
        answer: 1,
        explain:
          '$120^\\circ$ 是钝角，若它是底角则两底角之和 $\\ge 240^\\circ > 180^\\circ$，不合法，故 $120^\\circ$ 只能是**顶角**。两底角 $= (180^\\circ - 120^\\circ) \\div 2 = 30^\\circ$。',
      },
      {
        type: 'judge',
        prompt:
          '等腰三角形的一边长为 $3$，另一边长为 $7$，则其周长为 $13$ 或 $17$。',
        answer: false,
        explain:
          '分两种情形：①腰 $= 3$，底 $= 7$：两腰之和 $3 + 3 = 6 < 7$，不能构成三角形，舍去；②腰 $= 7$，底 $= 3$：$7 + 7 = 14 > 3$，$7 + 3 = 10 > 7$，合法，周长 $= 7 + 7 + 3 = 17$。故只有一种情形，周长为 $17$，选项中"或 $13$"是错误的。',
      },
      {
        type: 'input',
        prompt:
          '$\\triangle ABC$ 中 $\\angle B = \\angle C = 70^\\circ$，则 $\\angle A = $___$^\\circ$，且 $\\triangle ABC$ 是等腰三角形（填顶角的度数）。',
        accept: ['40'],
        explain:
          '三角形内角和 $180^\\circ$，$\\angle A = 180^\\circ - 70^\\circ - 70^\\circ = 40^\\circ$。又 $\\angle B = \\angle C$，由等角对等边得 $AB = AC$，$\\triangle ABC$ 是以 $A$ 为顶角、$40^\\circ$ 为顶角的等腰三角形。',
      },
      {
        type: 'match',
        prompt: '将等腰三角形的线段与其对应的三线合一身份配对。',
        left: [
          '顶角 $\\angle A$ 的平分线',
          '底边 $BC$ 的中线',
          '底边 $BC$ 上的高',
        ],
        right: [
          '过 $A$ 且 $\\perp BC$ 的线段',
          '连接 $A$ 与 $BC$ 中点的线段',
          '将 $\\angle BAC$ 二等分的线段',
        ],
        explain:
          '三线合一：顶角平分线 $=$ 底边中线 $=$ 底边上的高，三者是同一条线段 $AD$（$D$ 为 $BC$ 中点）。配对：顶角平分线↔将 $\\angle BAC$ 二等分的线段；底边中线↔连接 $A$ 与 $BC$ 中点的线段；底边上的高↔过 $A$ 且 $\\perp BC$ 的线段。',
      },
      {
        type: 'choice',
        prompt:
          '如图，$\\triangle ABC$ 中 $AB = AC$，$D$ 为 $BC$ 的中点，$E$ 是线段 $AD$ 上的一点，则 $EB$ 与 $EC$ 的关系是',
        options: ['$EB > EC$', '$EB < EC$', '$EB = EC$', '无法确定'],
        answer: 2,
        explain:
          '由三线合一，$AD$ 是底边 $BC$ 的垂直平分线。$E$ 在 $AD$ 上，即 $E$ 在 $BC$ 的中垂线上，由中垂线性质，$EB = EC$。',
      },
    ],
  },
]
