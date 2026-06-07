export const LESSONS = [
  {
    id: 'u2c2',
    title: '向量在三角形中的应用',
    subtitle: '基底法、中点重心与三点共线',
    intro: [
      {
        title: '为什么用向量处理三角形？',
        body: '三角形的边、中线、高、重心……用坐标系计算很繁琐，用向量基底法却能一步到位：选好两条边作基底，其余所有向量都写成它们的线性组合，然后比较系数。这比逐点坐标计算整洁得多，也是高考向量大题的主流方法。',
      },
      {
        title: '基底法：三步框架',
        body: '向量在三角形中应用的标准流程，也叫"基底法"：',
        steps: [
          '第一步——选基底：找两条不共线的已知边，例如 $\\vec{AB}$ 和 $\\vec{AC}$，作为平面的一组基底',
          '第二步——路径分解：把目标向量写成从起点出发经若干已知点到达终点的折线路径之和，如 $\\vec{AD} = \\vec{AB} + \\vec{BD}$',
          '第三步——合并系数：将路径上每段都化为基底的倍数，合并 $\\vec{AB}$ 和 $\\vec{AC}$ 的系数，利用基底唯一性得结论',
        ],
        tip: '路径不唯一，结果唯一。遇到计算量不同的路径，选更短的那条。',
      },
      {
        title: '引入题：中线的向量表示',
        body: '在 $\\triangle ABC$ 中，$D$ 是 $BC$ 的中点，用 $\\vec{AB}$ 和 $\\vec{AC}$ 表示 $\\vec{AD}$。内心还原：',
        steps: [
          '选路径：$\\vec{AD} = \\vec{AB} + \\vec{BD}$（先走 $A\\to B$，再走 $B\\to D$）',
          '表示 $\\vec{BD}$：$D$ 是 $BC$ 中点，故 $\\vec{BD} = \\dfrac{1}{2}\\vec{BC}$',
          '转化 $\\vec{BC}$：$\\vec{BC} = \\vec{AC} - \\vec{AB}$，代入得 $\\vec{BD} = \\dfrac{1}{2}(\\vec{AC}-\\vec{AB})$',
          '合并：$\\vec{AD} = \\vec{AB} + \\dfrac{1}{2}(\\vec{AC}-\\vec{AB}) = \\dfrac{1}{2}\\vec{AB} + \\dfrac{1}{2}\\vec{AC}$',
        ],
        tip: '两系数之和 $= 1$，说明 $D$ 在 $BC$ 上——这是"点在线段上"的向量特征。',
        image: 'geo-p2-02-1.svg',
        imageCaption: '三角形中线：$\\vec{AD} = \\dfrac{1}{2}(\\vec{AB}+\\vec{AC})$',
      },
      {
        title: '中点公式与重心公式',
        body: '两个最常用的公式，必须记住：',
        formula: '\\vec{OM} = \\frac{1}{2}(\\vec{OB}+\\vec{OC}) \\quad (M \\text{ 为 } BC \\text{ 中点})',
        steps: [
          '中点：$M$ 的位置向量 $=$ 两端点位置向量的平均',
          '重心 $G$ 是三条中线的交点：$\\vec{OG} = \\dfrac{1}{3}(\\vec{OA}+\\vec{OB}+\\vec{OC})$',
          '重心恒等式：$\\vec{GA}+\\vec{GB}+\\vec{GC} = \\vec{0}$（三顶点到重心向量之和为零向量）',
          '重心性质：$G$ 将每条中线按顶点侧 $2:$ 对边中点侧 $1$ 的比例分割',
        ],
        image: 'geo-p2-02-2.svg',
        imageCaption: '重心 $G$：$\\vec{OG} = \\dfrac{1}{3}(\\vec{OA}+\\vec{OB}+\\vec{OC})$，将中线 $2:1$ 分割',
      },
      {
        title: '分点比公式',
        body: '若 $P$ 在线段 $AB$ 上且 $AP:PB = m:n$（$m,n>0$），则：',
        formula: '\\vec{OP} = \\frac{n}{m+n}\\vec{OA} + \\frac{m}{m+n}\\vec{OB}',
        steps: [
          '推导：$\\vec{OP} = \\vec{OA} + \\dfrac{m}{m+n}\\vec{AB} = \\vec{OA} + \\dfrac{m}{m+n}(\\vec{OB}-\\vec{OA})$，整理即得',
          '记忆口诀："用对方的比例"——靠近 $A$ 的系数是 $\\dfrac{n}{m+n}$，取的是 $B$ 侧的分量 $n$',
          '两系数之和恒为 $1$，这是点 $P$ 在线段 $AB$ 上的必要特征',
          '若系数出现负数，则 $P$ 在线段的延长线上（超出端点之外）',
        ],
        tip: '特例：$m=n=1$ 时退化为中点公式，验证系数 $\\dfrac{1}{2}+\\dfrac{1}{2}=1$ ✓。',
      },
      {
        title: '三点共线的向量判定',
        body: '三点 $A, B, C$ 共线有两种等价的向量表达方式，遇到题目时选用更方便的一种：',
        steps: [
          '方式一：$\\vec{AC} = t\\,\\vec{AB}$（$t \\in \\mathbb{R}$）——即 $\\vec{AC}$ 与 $\\vec{AB}$ 平行（方向相同或相反）',
          '方式二：$\\vec{OC} = \\lambda\\vec{OA}+(1-\\lambda)\\vec{OB}$——两系数之和为 $1$，$O$ 为不在该直线上的参考点',
          '判断方法：若 $\\vec{BC} = t\\,\\vec{AB}$ 中 $t$ 能唯一确定，则共线；若 $\\vec{b}$ 和 $\\vec{c}$ 系数联立矛盾，则不共线',
        ],
        tip: '陷阱：两向量共线（平行）只需满足方向相同/相反，不要求模长相等。',
      },
      {
        title: '中位线定理的向量证法',
        body: '设 $M, N$ 分别是 $\\triangle ABC$ 的边 $AB, AC$ 的中点，用向量证明 $MN \\parallel BC$ 且 $MN = \\dfrac{1}{2}BC$：',
        steps: [
          '以 $A$ 为参考点，令 $\\vec{AB} = \\vec{b}$，$\\vec{AC} = \\vec{c}$',
          '$\\vec{AM} = \\dfrac{1}{2}\\vec{b}$，$\\vec{AN} = \\dfrac{1}{2}\\vec{c}$',
          '$\\vec{MN} = \\vec{AN} - \\vec{AM} = \\dfrac{1}{2}\\vec{c} - \\dfrac{1}{2}\\vec{b} = \\dfrac{1}{2}(\\vec{c}-\\vec{b}) = \\dfrac{1}{2}\\vec{BC}$',
          '由此得 $MN \\parallel BC$ 且 $MN = \\dfrac{1}{2}BC$',
        ],
        tip: '向量法证几何定理的模板：选基底 → 路径分解 → 结果化为目标向量的数倍，读出平行与长度关系。',
      },
      {
        title: '易错辨析与条件反射',
        body: '本章高频错误：',
        steps: [
          '错误1：分点比系数记反——$AP:PB = m:n$ 中，$A$ 旁系数是 $\\dfrac{n}{m+n}$（取对方 $n$），不是 $\\dfrac{m}{m+n}$',
          '错误2：重心公式混淆——$\\vec{OG}=\\dfrac{1}{3}(\\vec{OA}+\\vec{OB}+\\vec{OC})$ 是以任意点 $O$ 为参考点；若以 $G$ 为参考点则 $\\vec{GA}+\\vec{GB}+\\vec{GC}=\\vec{0}$',
          '错误3：忽略基底不共线条件——若 $\\vec{e}_1,\\vec{e}_2$ 不共线，才能由 $p\\vec{e}_1+q\\vec{e}_2 = r\\vec{e}_1+s\\vec{e}_2$ 推出 $p=r,q=s$',
        ],
        reveal: {
          q: '已知 $\\vec{GA} = (2,1)$，$\\vec{GB} = (-1,3)$，$G$ 为 $\\triangle ABC$ 重心，求 $\\vec{GC}$。',
          a: '由 $\\vec{GA}+\\vec{GB}+\\vec{GC}=\\vec{0}$，得 $\\vec{GC} = -(2,1)-(-1,3) = (-1,-4)$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '在 $\\triangle ABC$ 中，$D$ 是 $BC$ 的中点，设 $\\vec{AB}=\\vec{b}$，$\\vec{AC}=\\vec{c}$，则 $\\vec{AD}$ 等于：',
        options: [
          '$\\dfrac{1}{2}(\\vec{b}+\\vec{c})$',
          '$\\dfrac{1}{2}(\\vec{c}-\\vec{b})$',
          '$\\vec{b}+\\vec{c}$',
          '$\\dfrac{1}{2}\\vec{b}+\\vec{c}$',
        ],
        answer: 0,
        explain: '$\\vec{AD} = \\vec{AB}+\\vec{BD} = \\vec{b}+\\dfrac{1}{2}\\vec{BC} = \\vec{b}+\\dfrac{1}{2}(\\vec{c}-\\vec{b}) = \\dfrac{1}{2}\\vec{b}+\\dfrac{1}{2}\\vec{c} = \\dfrac{1}{2}(\\vec{b}+\\vec{c})$。',
      },
      {
        type: 'choice',
        prompt: '$\\triangle ABC$ 的重心为 $G$，则 $\\vec{GA}+\\vec{GB}+\\vec{GC}$ 等于：',
        options: ['$\\vec{0}$', '$\\vec{OG}$', '$3\\vec{OG}$', '$\\dfrac{1}{3}(\\vec{OA}+\\vec{OB}+\\vec{OC})$'],
        answer: 0,
        explain: '重心恒等式：$\\vec{GA}+\\vec{GB}+\\vec{GC}=\\vec{0}$，这是重心最重要的特征性质。',
      },
      {
        type: 'input',
        prompt: '在 $\\triangle ABC$ 中，$E$ 是 $AB$ 上满足 $AE:EB = 1:2$ 的点，设 $\\vec{AB}=\\vec{b}$，$\\vec{AC}=\\vec{c}$，则 $\\vec{AE}$ 的 $\\vec{b}$ 系数为多少（填分数，如 $1/3$）？',
        accept: ['1/3'],
        explain: '$AE:EB=1:2$，故 $\\vec{AE} = \\dfrac{1}{3}\\vec{AB} = \\dfrac{1}{3}\\vec{b}$，$\\vec{b}$ 的系数为 $\\dfrac{1}{3}$。',
      },
      {
        type: 'judge',
        prompt: '若 $\\vec{OC} = \\dfrac{1}{3}\\vec{OA}+\\dfrac{2}{3}\\vec{OB}$，则 $A, B, C$ 三点共线（$O$ 不在直线 $AB$ 上）。',
        answer: true,
        explain: '两系数 $\\dfrac{1}{3}+\\dfrac{2}{3}=1$，满足三点共线的向量条件 $\\vec{OC} = \\lambda\\vec{OA}+(1-\\lambda)\\vec{OB}$（这里 $\\lambda=\\dfrac{1}{3}$），故 $A,B,C$ 共线。',
      },
      {
        type: 'choice',
        prompt: '$P$ 在线段 $AB$ 上且 $AP:PB = 2:3$，设原点为 $O$，则 $\\vec{OP}$ 等于：',
        options: [
          '$\\dfrac{3}{5}\\vec{OA}+\\dfrac{2}{5}\\vec{OB}$',
          '$\\dfrac{2}{5}\\vec{OA}+\\dfrac{3}{5}\\vec{OB}$',
          '$\\dfrac{1}{2}\\vec{OA}+\\dfrac{1}{2}\\vec{OB}$',
          '$\\dfrac{2}{3}\\vec{OA}+\\dfrac{1}{3}\\vec{OB}$',
        ],
        answer: 0,
        explain: '$AP:PB=2:3$，分点比公式：$\\vec{OP} = \\dfrac{3}{2+3}\\vec{OA}+\\dfrac{2}{2+3}\\vec{OB} = \\dfrac{3}{5}\\vec{OA}+\\dfrac{2}{5}\\vec{OB}$（靠近 $A$ 取 $B$ 侧的比例 $3$）。',
      },
      {
        type: 'judge',
        prompt: '在 $\\triangle ABC$ 中，$M,N$ 分别是 $AB,AC$ 的中点，则 $\\vec{MN} = \\dfrac{1}{2}\\vec{BC}$，故 $MN \\parallel BC$ 且 $MN = \\dfrac{1}{2}BC$。',
        answer: true,
        explain: '令 $\\vec{AB}=\\vec{b}$，$\\vec{AC}=\\vec{c}$。$\\vec{MN} = \\vec{AN}-\\vec{AM} = \\dfrac{1}{2}\\vec{c}-\\dfrac{1}{2}\\vec{b} = \\dfrac{1}{2}(\\vec{c}-\\vec{b}) = \\dfrac{1}{2}\\vec{BC}$，即 $MN \\parallel BC$ 且长为 $BC$ 的一半，正确。',
      },
      {
        type: 'match',
        prompt: '将下列特殊点与对应的向量公式配对（$O$ 为参考点）：',
        left: [
          '$BC$ 的中点 $M$',
          '$\\triangle ABC$ 的重心 $G$',
          '$AB$ 上满足 $AP:PB=1:1$ 的点 $P$',
        ],
        right: [
          '$\\vec{OM} = \\dfrac{1}{2}(\\vec{OB}+\\vec{OC})$',
          '$\\vec{OG} = \\dfrac{1}{3}(\\vec{OA}+\\vec{OB}+\\vec{OC})$',
          '$\\vec{OP} = \\dfrac{1}{2}\\vec{OA}+\\dfrac{1}{2}\\vec{OB}$',
        ],
        explain: '$BC$ 中点套中点公式；重心套重心公式；$AP:PB=1:1$ 即中点，套中点公式（与第一条形式相同）。三者一一对应。',
      },
      {
        type: 'choice',
        prompt: '在 $\\triangle ABC$ 中，$\\vec{AB}=\\vec{b}$，$\\vec{AC}=\\vec{c}$。点 $D$ 满足 $BD:DC=1:2$，则 $\\vec{AD}$ 等于：',
        options: [
          '$\\dfrac{2}{3}\\vec{b}+\\dfrac{1}{3}\\vec{c}$',
          '$\\dfrac{1}{3}\\vec{b}+\\dfrac{2}{3}\\vec{c}$',
          '$\\dfrac{1}{2}\\vec{b}+\\dfrac{1}{2}\\vec{c}$',
          '$\\vec{b}+\\dfrac{1}{3}\\vec{c}$',
        ],
        answer: 0,
        explain: '$BD:DC=1:2$，$\\vec{AD}=\\vec{AB}+\\vec{BD}=\\vec{b}+\\dfrac{1}{3}\\vec{BC}=\\vec{b}+\\dfrac{1}{3}(\\vec{c}-\\vec{b})=\\dfrac{2}{3}\\vec{b}+\\dfrac{1}{3}\\vec{c}$。系数之和 $\\dfrac{2}{3}+\\dfrac{1}{3}=1$，符合 $D$ 在 $BC$ 上。',
      },
    ],
  },
]
