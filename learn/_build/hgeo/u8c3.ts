export const LESSONS = [
  {
    id: 'u8c3',
    title: '直线与平面的位置关系',
    subtitle: '线面平行、线面垂直、三垂线定理——三种关系的判定与性质',
    intro: [
      {
        title: '三种线面位置关系',
        body: '直线 $l$ 与平面 $\\alpha$ 有三种位置关系：① 直线在平面内（$l \\subset \\alpha$）——直线上每点都在平面内，公共点无数个；② 直线与平面平行（$l \\parallel \\alpha$）——直线与平面无公共点；③ 直线与平面相交（$l \\cap \\alpha = P$）——恰好一个公共点 $P$（交点）。注意：直线经过平面内一点时，只能是在平面内或相交，不能平行。',
        image: 'geo-p8-03-1.svg',
        imageCaption: '直线与平面的三种位置关系',
      },
      {
        title: '线面平行判定定理',
        body: '线面平行判定定理：若直线 $l$ 在平面 $\\alpha$ 外，且 $l$ 平行于 $\\alpha$ 内的某条直线 $m$，则 $l \\parallel \\alpha$。记忆口诀："平面外 + 平面内 + 两线平行 → 线面平行"（外内平 → 线面平）。证题步骤：① 找 $\\alpha$ 内的直线 $m$，证 $l \\parallel m$；② 验证 $l \\not\\subset \\alpha$；③ 由定理得 $l \\parallel \\alpha$。',
        formula: 'l \\not\\subset \\alpha,\\quad m \\subset \\alpha,\\quad l \\parallel m \\implies l \\parallel \\alpha',
      },
      {
        title: '线面平行性质定理',
        body: '线面平行性质定理：若 $l \\parallel \\alpha$，$\\beta$ 是过 $l$ 的平面，$\\beta \\cap \\alpha = m$，则 $l \\parallel m$。用途：由线面平行推出线线平行，常用于截面性质题。理解：$l$ 平行于 $\\alpha$，故 $l$ 与 $\\alpha$ 没有公共点；$l$ 和 $m$ 同在平面 $\\beta$ 内且不相交，故平行。',
        tip: '线面平行性质把线面关系传递到线线关系，是"从面推线"的桥梁。',
      },
      {
        title: '线面垂直判定定理',
        body: '线面垂直定义：直线 $l$ 垂直于平面 $\\alpha$ 内的所有直线，记作 $l \\perp \\alpha$，交点叫垂足。判定定理：若直线 $l$ 垂直于平面 $\\alpha$ 内的两条相交直线 $m$ 和 $n$，则 $l \\perp \\alpha$。记忆口诀："两交 → 线面垂"。必须是两条相交直线——只垂直一条不够（因为有无数条平行线都与之垂直，但这些线不一定垂直整个平面）。',
        image: 'geo-p8-03-2.svg',
        imageCaption: '线面垂直判定定理示意',
        formula: 'l \\perp m,\\quad l \\perp n,\\quad m \\cap n = O,\\quad m \\subset \\alpha,\\quad n \\subset \\alpha \\implies l \\perp \\alpha',
      },
      {
        title: '线面垂直性质定理',
        body: '由线面垂直可以推出多个性质。定理 1：若 $l \\perp \\alpha$，$m \\subset \\alpha$，则 $l \\perp m$（垂直平面则垂直其内所有线）。定理 2：若 $l \\perp \\alpha$，$m \\perp \\alpha$，则 $l \\parallel m$（两线同垂一面则平行）。定理 3：若 $l \\perp \\alpha$，$m \\parallel l$，则 $m \\perp \\alpha$（平行于垂线则也垂直平面）。',
        tip: '看到"两线同垂一面"立即得出两线平行——这是证平行题的高频结论。',
      },
      {
        title: '三垂线定理',
        body: '三垂线定理：设 $PO \\perp \\alpha$（垂足为 $O$），$A$ 为 $\\alpha$ 内一点（$OA$ 为斜线 $PA$ 的射影），若 $OA \\perp l$（$l \\subset \\alpha$），则 $PA \\perp l$。逆定理同样成立：$PA \\perp l$ 且 $PO \\perp \\alpha$ 则 $OA \\perp l$。直觉：斜线与 $l$ 的关系完全由射影与 $l$ 的关系决定（因为 $PO \\perp \\alpha$ 保证了 $PO \\perp l$，从而降维到平面）。',
        tip: '三垂线定理是求斜线与平面所成角、证明空间垂直关系的利器。',
      },
      {
        title: '斜线与平面所成角',
        body: '斜线 $l$ 与平面 $\\alpha$ 所成角定义：斜线与其在 $\\alpha$ 上射影所成的锐角（或直角），范围 $(0°, 90°)$（严格不含端点，垂直时 $= 90°$ 特殊处理）。求法：① 从斜线上一点 $P$ 向 $\\alpha$ 作垂线，垂足为 $O$；② 连 $O$ 与斜线在平面内对应点 $A$，得射影 $OA$；③ $\\angle PAO$ 即为所成角（$PA$ 是斜线，$OA$ 是射影，$\\angle PAO$ 是斜线与射影夹角）。',
      },
      {
        title: '例题：正三棱锥中证线面垂直',
        body: '正三棱锥 $P$-$ABC$ 中，底面边长为 $a$，$O$ 是底面正三角形中心。证明 $PO \\perp$ 底面 $ABC$。思路：取 $M$ 为 $BC$ 中点。正三角形中 $AM \\perp BC$（中线也是高）；$PB = PC$，$M$ 为 $BC$ 中点，故 $PM \\perp BC$（等腰三角形中线也是高）。于是 $BC \\perp AM$ 且 $BC \\perp PM$，$AM \\cap PM = M$，由判定定理 $BC \\perp$ 平面 $APM$。类似地证 $AB \\perp PO$，有 $BC \\perp PO$ 且 $AB \\perp PO$（$BC \\cap AB = B$），由判定定理 $PO \\perp$ 底面 $ABC$。',
        reveal: {
          q: '线面平行判定定理为什么需要验证"直线不在平面内"？',
          a: '若直线在平面内，它与平面内另一直线平行是成立的，但此时直线属于平面（$l \\subset \\alpha$），不是线面平行。判定定理的结论是 $l \\parallel \\alpha$，前提必须排除 $l \\subset \\alpha$ 的情形。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '下列关于线面关系的说法中，正确的是？',
        options: [
          '直线 $l$ 与平面 $\\alpha$ 没有公共点，则 $l \\parallel \\alpha$',
          '直线 $l$ 不在平面 $\\alpha$ 内，则 $l \\parallel \\alpha$',
          '直线 $l$ 在平面 $\\alpha$ 外，且 $l$ 平行于 $\\alpha$ 内某直线，则 $l \\parallel \\alpha$',
          '直线 $l$ 平行于平面 $\\alpha$ 内的两条直线，则 $l \\parallel \\alpha$',
        ],
        answer: 2,
        explain: '选项 A 中，若 $l \\subset \\alpha$ 则有无数公共点，此处说的是"没有公共点"，但可能是 $l \\parallel \\alpha$ 或 $l$ 完全在 $\\alpha$ 外——实际 $l \\not\\subset \\alpha$ 且无公共点确实是平行，但表述不严谨。选项 C 是线面平行判定定理，正确。选项 D 中平面内两条直线若平行，则只约束一个方向，$l$ 可能与平面相交。',
      },
      {
        type: 'judge',
        prompt: '若直线 $l$ 垂直于平面 $\\alpha$ 内的一条直线 $m$，则 $l \\perp \\alpha$。',
        answer: false,
        explain: '线面垂直判定定理要求垂直于平面内两条相交直线。只垂直一条不够，因为有无数条与 $m$ 平行的线都与 $l$ 垂直，但 $l$ 不一定垂直整个平面。',
      },
      {
        type: 'choice',
        prompt: '已知 $l \\perp \\alpha$，$m \\perp \\alpha$，则 $l$ 与 $m$ 的关系是？',
        options: ['相交', '平行', '异面', '垂直'],
        answer: 1,
        explain: '由线面垂直性质定理：两条直线都垂直于同一平面，则这两条直线平行（$l \\parallel m$）。',
      },
      {
        type: 'judge',
        prompt: '三垂线定理的逆定理同样成立：若斜线 $PA$ 垂直于 $l$（$l \\subset \\alpha$），且 $PO \\perp \\alpha$，则射影 $OA \\perp l$。',
        answer: true,
        explain: '三垂线定理与其逆定理都成立：斜线与射影同垂某直线（只要知道垂线存在），斜线垂直于 $l$ 可以推出射影也垂直于 $l$，反之亦然。',
      },
      {
        type: 'choice',
        prompt: '正三棱柱 $ABC$-$A_1B_1C_1$ 中，下列哪条直线垂直于底面 $ABC$？',
        options: ['$AC_1$', '$AA_1$', '$AB_1$', '$BC_1$'],
        answer: 1,
        explain: '正三棱柱的侧棱（如 $AA_1$）垂直于底面，这是"直棱柱"的定义——侧面与底面垂直，侧棱即底面的法线方向。',
      },
      {
        type: 'input',
        prompt: '证明线面垂直时，需要找平面内两条____直线，分别证它们与已知直线垂直。',
        accept: ['相交', '互相相交', '交于一点'],
        explain: '线面垂直判定定理要求两条相交直线（不能平行），两条相交直线确定了平面的完整方向，从而能推出直线垂直于整个平面。',
      },
      {
        type: 'match',
        prompt: '将线面垂直的性质定理与对应结论配对。',
        left: [
          '$l \\perp \\alpha$，$m \\subset \\alpha$',
          '$l \\perp \\alpha$，$m \\perp \\alpha$',
          '$l \\perp \\alpha$，$m \\parallel l$',
        ],
        right: [
          '$l \\perp m$（垂直平面则垂直其内任意直线）',
          '$l \\parallel m$（两线同垂一面则平行）',
          '$m \\perp \\alpha$（平行于法线则也是法线）',
        ],
        explain: '三条性质定理各有用途：第一条用于从线面垂直推出线线垂直；第二条用于证平行；第三条用于传递垂直性质。',
      },
      {
        type: 'choice',
        prompt: '直线 $l$ 与平面 $\\alpha$ 所成角的范围是？',
        options: [
          '$(0°, 90°]$',
          '$[0°, 90°]$',
          '$(0°, 90°)$',
          '$[0°, 180°]$',
        ],
        answer: 0,
        explain: '斜线与平面所成角范围为 $(0°, 90°]$：下限 $0°$ 排除（$0°$ 对应平行或在平面内，不是斜线）；上限 $90°$ 包含（垂直时所成角为 $90°$）。',
      },
    ],
  },
]
