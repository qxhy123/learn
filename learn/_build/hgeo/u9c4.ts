export const LESSONS = [
  {
    id: 'u9c4',
    title: '用向量证空间平行与垂直',
    subtitle: '方向向量与法向量：六大位置关系的向量判定',
    intro: [
      {
        title: '引入：向量法证线面垂直',
        body: '长方体 $ABCD$-$A_1B_1C_1D_1$，以 $A$ 为原点建系（$AB\\to x$，$AD\\to y$，$AA_1\\to z$）。传统方法证 $AA_1\\perp$ 底面 $ABCD$ 需分别证 $AA_1\\perp AB$ 和 $AA_1\\perp AD$，再用定理。\n\n向量法：$\\vec{AA_1}=(0,0,1)$，底面法向量 $\\vec{n}=(0,0,1)$（$z=0$ 平面），$\\vec{AA_1}=1\\cdot\\vec{n}$，即 $\\vec{AA_1}\\parallel\\vec{n}$，故 $AA_1\\perp$ 底面 $ABCD$。一步到位，无需辅助线。',
        image: 'geo-p9-04-1.svg',
        imageCaption: '线面垂直向量判定：方向向量 $\\vec{a}\\parallel$ 法向量 $\\vec{n}$，即 $\\vec{a}=\\lambda\\vec{n}$',
      },
      {
        title: '线线关系的向量判定',
        body: '设两直线方向向量为 $\\vec{a}$ 和 $\\vec{b}$：\n\n线线平行：$\\vec{a}\\parallel\\vec{b}$，即 $\\vec{a}=\\lambda\\vec{b}$（$\\lambda\\neq0$）。\n\n线线垂直：$\\vec{a}\\perp\\vec{b}$，即 $\\vec{a}\\cdot\\vec{b}=0$。\n\n注意：异面直线的方向向量可以既不平行也不垂直。两线"异面"不能只从方向向量判断，需结合位置信息。若 $\\vec{a}\\cdot\\vec{b}=0$，则线线垂直（包含异面垂直情形）。',
        formula: 'l_1\\parallel l_2\\Leftrightarrow\\vec{a}=\\lambda\\vec{b};\\quad l_1\\perp l_2\\Leftrightarrow\\vec{a}\\cdot\\vec{b}=0',
      },
      {
        title: '线面关系的向量判定',
        body: '设直线方向向量为 $\\vec{a}$，平面法向量为 $\\vec{n}$：\n\n线面垂直：$\\vec{a}\\parallel\\vec{n}$，即 $\\vec{a}=\\lambda\\vec{n}$（方向与平面朝向一致）。\n\n线面平行：$\\vec{a}\\perp\\vec{n}$，即 $\\vec{a}\\cdot\\vec{n}=0$，且直线上有一点不在面内。\n\n线在面内：也满足 $\\vec{a}\\cdot\\vec{n}=0$，但直线上的点在面内。所以线面平行与线在面内的向量条件相同，区别在于直线上是否有点在平面内。',
        formula: 'l\\perp\\alpha\\Leftrightarrow\\vec{a}=\\lambda\\vec{n};\\quad l\\parallel\\alpha\\Leftrightarrow\\vec{a}\\cdot\\vec{n}=0\\;(P\\notin\\alpha)',
      },
      {
        title: '面面关系的向量判定',
        body: '设两平面法向量分别为 $\\vec{n}_1$ 和 $\\vec{n}_2$：\n\n面面平行：$\\vec{n}_1\\parallel\\vec{n}_2$，即 $\\vec{n}_1=\\lambda\\vec{n}_2$（两面朝向相同）。\n\n面面垂直：$\\vec{n}_1\\perp\\vec{n}_2$，即 $\\vec{n}_1\\cdot\\vec{n}_2=0$（两面朝向互相垂直）。\n\n法向量的数乘倍数不影响结论：$\\vec{n}$ 和 $2\\vec{n}$ 描述同一平面，判定结果相同；方向相反（$\\vec{n}$ 与 $-\\vec{n}$）仍视为平行。',
        formula: '\\alpha\\parallel\\beta\\Leftrightarrow\\vec{n}_1=\\lambda\\vec{n}_2;\\quad\\alpha\\perp\\beta\\Leftrightarrow\\vec{n}_1\\cdot\\vec{n}_2=0',
        image: 'geo-p9-04-2.svg',
        imageCaption: '面面平行向量判定：两平面法向量 $\\vec{n}_1\\parallel\\vec{n}_2$，即 $\\vec{n}_1=\\lambda\\vec{n}_2$',
      },
      {
        title: '向量法证明的通用模板',
        body: '证线面垂直：建系 → 写直线方向向量 $\\vec{a}$ → 求面法向量 $\\vec{n}$ → 验证 $\\vec{a}=\\lambda\\vec{n}$ → 写结论。\n\n证线面平行：建系 → $\\vec{a}$ 和 $\\vec{n}$ → 验证 $\\vec{a}\\cdot\\vec{n}=0$ → 验证直线上一点不在面内 → 写结论。\n\n证面面平行：求 $\\vec{n}_1,\\vec{n}_2$ → 验证 $\\vec{n}_1=\\lambda\\vec{n}_2$ → 写结论。\n\n证面面垂直：求 $\\vec{n}_1,\\vec{n}_2$ → 验证 $\\vec{n}_1\\cdot\\vec{n}_2=0$ → 写结论。\n\n最后一步必须写出文字推断，不能只给出数字验算结果。',
      },
      {
        title: '正四棱锥建系例题',
        body: '正四棱锥 $P$-$ABCD$，底面边长 $2$，高 $\\sqrt{2}$，取底面中心 $O^{\\prime}$ 为原点，坐标：$A(1,1,0)$，$B(-1,1,0)$，$C(-1,-1,0)$，$D(1,-1,0)$，$P(0,0,\\sqrt{2})$。\n\n证明平面 $PAC\\perp$ 底面 $ABCD$：底面法向量 $\\vec{n}_1=(0,0,1)$；面 $PAC$ 内向量 $\\vec{AP}=(-1,-1,\\sqrt{2})$，$\\vec{AC}=(-2,-2,0)$，令 $x=1$ 解得 $\\vec{n}_2=(1,-1,0)$；$\\vec{n}_1\\cdot\\vec{n}_2=0$，故 $\\text{平面}\\ PAC\\perp\\text{底面}\\ ABCD$。',
      },
      {
        title: '易错点汇总',
        body: '易错 1：法向量方向相反仍视为平行（$\\vec{n}$ 与 $-\\vec{n}$），$\\lambda$ 可为负数。\n\n易错 2：线面平行必须额外验证"直线不在面内"，仅 $\\vec{a}\\cdot\\vec{n}=0$ 不够。\n\n易错 3：取面内两向量时选了共线的向量（分量成比例），联立方程无唯一解，需重选。\n\n易错 4：证明结尾必须写文字推断（"因为……所以……"），不能只写数字。\n\n易错 5：混淆线线垂直（$\\vec{a}\\cdot\\vec{b}=0$）与线面垂直（$\\vec{a}=\\lambda\\vec{n}$）的条件，尤其注意线面平行的条件（$\\vec{a}\\cdot\\vec{n}=0$）与线线垂直形式相同但含义不同。',
        reveal: {
          q: '正方体棱长 $1$，以 $A$ 为原点建系，侧面 $ABB_1A_1$ 与底面 $ABCD$ 的关系？',
          a: '面 $ABB_1A_1$ 在 $y=0$ 平面，法向量 $(0,1,0)$；底面 $ABCD$ 在 $z=0$，法向量 $(0,0,1)$；$(0,1,0)\\cdot(0,0,1)=0$，故两面垂直。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '直线 $l$ 的方向向量为 $\\vec{a}=(1,2,-1)$，平面 $\\alpha$ 的法向量为 $\\vec{n}=(2,4,-2)$，则 $l$ 与 $\\alpha$ 的关系为',
        options: [
          '$l\\parallel\\alpha$',
          '$l\\perp\\alpha$',
          '$l$ 在 $\\alpha$ 内',
          '$l$ 与 $\\alpha$ 斜交',
        ],
        answer: 1,
        explain: '$\\vec{n}=(2,4,-2)=2(1,2,-1)=2\\vec{a}$，故 $\\vec{a}\\parallel\\vec{n}$，即 $\\vec{a}=\\frac{1}{2}\\vec{n}$，满足线面垂直条件 $\\vec{a}=\\lambda\\vec{n}$，所以 $l\\perp\\alpha$。',
      },
      {
        type: 'choice',
        prompt: '平面 $\\alpha$ 的法向量为 $\\vec{n}_1=(1,0,2)$，平面 $\\beta$ 的法向量为 $\\vec{n}_2=(2,0,4)$，则 $\\alpha$ 与 $\\beta$ 的关系为',
        options: [
          '$\\alpha\\perp\\beta$',
          '$\\alpha\\parallel\\beta$',
          '$\\alpha$ 与 $\\beta$ 斜交',
          '无法判断',
        ],
        answer: 1,
        explain: '$\\vec{n}_2=(2,0,4)=2(1,0,2)=2\\vec{n}_1$，故 $\\vec{n}_1\\parallel\\vec{n}_2$，即两平面法向量平行，所以 $\\alpha\\parallel\\beta$。',
      },
      {
        type: 'judge',
        prompt: '若直线方向向量 $\\vec{a}$ 与平面法向量 $\\vec{n}$ 满足 $\\vec{a}\\cdot\\vec{n}=0$，则直线一定平行于该平面。',
        answer: false,
        explain: '$\\vec{a}\\cdot\\vec{n}=0$ 说明方向向量平行于平面（垂直于法向量），但直线可能在面内，也可能平行于面。需额外验证直线上一点不在平面内，才能确认是线面平行。',
      },
      {
        type: 'choice',
        prompt: '已知 $\\vec{a}=(1,-1,2)$，$\\vec{b}=(2,m,1)$，若 $\\vec{a}\\perp\\vec{b}$，则 $m=$',
        options: ['$3$', '$-3$', '$4$', '$-4$'],
        answer: 2,
        explain: '$\\vec{a}\\cdot\\vec{b}=1\\cdot2+(-1)\\cdot m+2\\cdot1=2-m+2=4-m=0$，故 $m=4$，对应选项 C（下标 $2$）。',
      },
      {
        type: 'input',
        prompt: '已知两平面法向量 $\\vec{n}_1=(1,2,0)$，$\\vec{n}_2=(0,0,3)$，计算 $\\vec{n}_1\\cdot\\vec{n}_2$ 的值',
        accept: ['0'],
        explain: '$\\vec{n}_1\\cdot\\vec{n}_2=1\\times0+2\\times0+0\\times3=0$，故两平面互相垂直（$\\vec{n}_1\\perp\\vec{n}_2$）。',
      },
      {
        type: 'match',
        prompt: '将各位置关系与对应的向量条件配对（$\\vec{a}$：方向向量，$\\vec{n}$：法向量）',
        left: [
          '线面垂直 $l\\perp\\alpha$',
          '线面平行 $l\\parallel\\alpha$（且不在面内）',
          '面面平行 $\\alpha\\parallel\\beta$',
          '面面垂直 $\\alpha\\perp\\beta$',
        ],
        right: [
          '$\\vec{n}_1\\cdot\\vec{n}_2=0$',
          '$\\vec{a}=\\lambda\\vec{n}$',
          '$\\vec{n}_1=\\lambda\\vec{n}_2$',
          '$\\vec{a}\\cdot\\vec{n}=0$',
        ],
        explain: '线面垂直→方向向量平行于法向量 $\\vec{a}=\\lambda\\vec{n}$；线面平行→方向向量垂直于法向量 $\\vec{a}\\cdot\\vec{n}=0$（且不在面内）；面面平行→两法向量平行 $\\vec{n}_1=\\lambda\\vec{n}_2$；面面垂直→两法向量垂直 $\\vec{n}_1\\cdot\\vec{n}_2=0$。',
      },
      {
        type: 'choice',
        prompt: '正方体棱长 $1$，以 $A$ 为原点建系（$AB\\to x$，$AD\\to y$，$AA_1\\to z$），体对角线 $\\vec{AC_1}=(1,1,1)$ 与侧面 $ABB_1A_1$（在 $y=0$ 平面内，法向量 $(0,1,0)$）的关系为',
        options: [
          '$AC_1\\perp$ 侧面 $ABB_1A_1$',
          '$AC_1\\parallel$ 侧面 $ABB_1A_1$',
          '$AC_1$ 在侧面 $ABB_1A_1$ 内',
          '$AC_1$ 斜交侧面 $ABB_1A_1$',
        ],
        answer: 3,
        explain: '$\\vec{AC_1}=(1,1,1)$，法向量 $\\vec{n}=(0,1,0)$。$\\vec{AC_1}\\cdot\\vec{n}=0+1+0=1\\neq0$，故不平行、不在面内；$(1,1,1)\\neq\\lambda(0,1,0)$，故不垂直。因此 $AC_1$ 斜交侧面 $ABB_1A_1$（相交但不垂直）。',
      },
    ],
  },
]
