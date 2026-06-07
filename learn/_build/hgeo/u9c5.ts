export const LESSONS = [
  {
    id: 'u9c5',
    title: '用向量求空间角',
    subtitle: '线线角、线面角、二面角的向量公式与看图判正负',
    intro: [
      {
        title: '引入：正方体二面角的法向量求法',
        body: '正方体 $ABCD$-$A_1B_1C_1D_1$（棱长 $1$），求平面 $A_1BD$ 与平面 $D_1BC$ 所成二面角的余弦值。\n\n策略：以 $B$ 为原点建系 → 求两平面各自法向量 $\\vec{n}_1,\\vec{n}_2$ → 算 $\\cos\\psi$ → 看图判正负。\n\n建系后：$A(1,0,0)$，$B(0,0,0)$，$D(1,1,0)$，$A_1(1,0,1)$，$D_1(1,1,1)$，$C(0,1,0)$。',
        image: 'geo-p9-05-1.svg',
        imageCaption: '三种空间角的几何意义对比：线线角取锐角，线面角用 $\\sin$，二面角看图判正负',
      },
      {
        title: '三类空间角总览',
        body: '线线角（两直线夹角），范围 $\\theta\\in(0,\\frac{\\pi}{2}]$，公式取 $\\cos$ 加绝对值（结果永远不超过 $90°$）。\n\n线面角（直线与平面），范围 $\\varphi\\in[0,\\frac{\\pi}{2}]$，公式取 $\\sin$ 加绝对值——因为线面角是直线方向与法向量夹角的余角，$\\sin\\varphi=\\cos(\\text{余角})$，故用 $\\sin$，也永远是锐角或直角。\n\n二面角（两半平面），范围 $\\psi\\in[0,\\pi]$，公式不取绝对值——符号由法向量方向决定，必须结合图形判断正负。',
        formula: '\\cos\\theta=\\left|\\frac{\\vec{a}\\cdot\\vec{b}}{|\\vec{a}||\\vec{b}|}\\right|;\\quad\\sin\\varphi=\\left|\\frac{\\vec{l}\\cdot\\vec{n}}{|\\vec{l}||\\vec{n}|}\\right|;\\quad\\cos\\psi=\\pm\\frac{\\vec{n}_1\\cdot\\vec{n}_2}{|\\vec{n}_1||\\vec{n}_2|}',
      },
      {
        title: '线线角：取绝对值的原因',
        body: '两直线的方向向量为 $\\vec{a}$ 和 $\\vec{b}$，线线角 $\\theta\\in(0,\\frac{\\pi}{2}]$（总取锐角或直角）。若两向量夹角 $>90°$，说明方向向量指向"相反侧"，但直线本身的夹角应取其补角的补角（即取锐角）。\n\n公式：$\\cos\\theta=\\left|\\dfrac{\\vec{a}\\cdot\\vec{b}}{|\\vec{a}||\\vec{b}|}\\right|$，取绝对值后 $\\cos\\theta\\geq0$，即 $\\theta\\leq90°$。\n\n异面直线：先将方向向量平移到同一起点，计算夹角即为异面直线所成角（不需要真的找交点）。',
        formula: '\\cos\\theta=\\left|\\frac{\\vec{a}\\cdot\\vec{b}}{|\\vec{a}||\\vec{b}|}\\right|',
      },
      {
        title: '线面角：用 sin 而非 cos 的原因',
        body: '直线方向向量 $\\vec{l}$，平面法向量 $\\vec{n}$，设 $\\vec{l}$ 与 $\\vec{n}$ 夹角为 $\\alpha$。直线与平面的夹角（线面角）$\\varphi=90°-\\alpha$（线面角是直线到平面的投影角，等于与法向量夹角的余角）。\n\n所以 $\\sin\\varphi=\\cos\\alpha=\\left|\\dfrac{\\vec{l}\\cdot\\vec{n}}{|\\vec{l}||\\vec{n}|}\\right|$，用 $\\sin$ 而不用 $\\cos$，且取绝对值（线面角在 $[0°,90°]$）。\n\n特殊情形：$\\sin\\varphi=0$ 时直线平行于平面，$\\sin\\varphi=1$ 时直线垂直于平面。',
        formula: '\\sin\\varphi=\\left|\\frac{\\vec{l}\\cdot\\vec{n}}{|\\vec{l}||\\vec{n}|}\\right|',
      },
      {
        title: '二面角：看图判正负',
        body: '两平面法向量 $\\vec{n}_1,\\vec{n}_2$，二面角 $\\psi\\in[0°,180°]$：\n\n$\\cos\\psi=\\pm\\dfrac{\\vec{n}_1\\cdot\\vec{n}_2}{|\\vec{n}_1||\\vec{n}_2|}$，符号需看图：若两法向量指向棱的同侧，$\\cos\\psi>0$（锐角）；指向异侧，$\\cos\\psi<0$（钝角）；垂直时 $\\cos\\psi=0$（直角）。\n\n操作：先算出 $\\cos$ 的数值，再结合图形几何关系判断二面角是锐角还是钝角，赋予正负号。若法向量取反，$\\cos$ 变号，但角度不变，因此一定要配合图形。',
        formula: '\\cos\\psi=\\pm\\frac{\\vec{n}_1\\cdot\\vec{n}_2}{|\\vec{n}_1||\\vec{n}_2|}',
        image: 'geo-p9-05-2.svg',
        imageCaption: '二面角法向量法：两法向量同侧则 $\\cos>0$（锐角），异侧则 $\\cos<0$（钝角）',
      },
      {
        title: '引入题解答',
        body: '正方体以 $B$ 为原点建系：面 $A_1BD$ 法向量：$\\vec{BA_1}=(1,0,1)$，$\\vec{BD}=(1,1,0)$，解方程组 $x+z=0,x+y=0$，取 $x=1$ 得 $\\vec{n}_1=(1,-1,-1)$。\n\n面 $D_1BC$ 法向量：$\\vec{BD_1}=(1,1,1)$，$\\vec{BC}=(0,1,0)$，解方程组 $x+y+z=0,y=0$，取 $x=1$ 得 $\\vec{n}_2=(1,0,-1)$。\n\n$\\vec{n}_1\\cdot\\vec{n}_2=1+0+1=2$，$|\\vec{n}_1|=\\sqrt{3}$，$|\\vec{n}_2|=\\sqrt{2}$，$\\cos\\psi=\\dfrac{2}{\\sqrt{6}}=\\dfrac{\\sqrt{6}}{3}$。\n\n看图：$A_1$ 与 $D_1$ 分别在棱 $BD$ 两侧，两面成锐角，取正值，二面角余弦为 $\\dfrac{\\sqrt{6}}{3}$。',
        reveal: {
          q: '正方体棱长 $1$，求体对角线 $AC_1$ 与棱 $AB$ 所成的线线角。',
          a: '$\\vec{AC_1}=(1,1,1)$，$\\vec{AB}=(1,0,0)$，$\\cos\\theta=\\left|\\frac{1}{\\sqrt{3}\\cdot1}\\right|=\\frac{\\sqrt{3}}{3}$，$\\theta=\\arccos\\frac{\\sqrt{3}}{3}$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '长方体 $ABCD$-$A_1B_1C_1D_1$，$AB=2,AD=1,AA_1=3$，以 $A$ 为原点建系，直线 $AC_1$ 与底面 $ABCD$ 所成线面角 $\\varphi$ 满足',
        options: [
          '$\\sin\\varphi=\\dfrac{3}{\\sqrt{14}}$',
          '$\\cos\\varphi=\\dfrac{3}{\\sqrt{14}}$',
          '$\\sin\\varphi=\\dfrac{\\sqrt{5}}{\\sqrt{14}}$',
          '$\\tan\\varphi=\\dfrac{3}{\\sqrt{5}}$',
        ],
        answer: 0,
        explain: '$C_1=(2,1,3)$，$\\vec{AC_1}=(2,1,3)$，底面法向量 $\\vec{n}=(0,0,1)$；$\\sin\\varphi=\\left|\\frac{(2,1,3)\\cdot(0,0,1)}{\\sqrt{4+1+9}\\cdot1}\\right|=\\frac{3}{\\sqrt{14}}$。注意用 $\\sin$ 而非 $\\cos$，选项 A 正确。',
      },
      {
        type: 'judge',
        prompt: '求两直线所成角时，若两方向向量点积为负数，应先取绝对值再用反余弦函数求角，结果一定在 $(0°,90°]$。',
        answer: true,
        explain: '两直线所成角 $\\theta\\in(0°,90°]$，公式 $\\cos\\theta=\\left|\\frac{\\vec{a}\\cdot\\vec{b}}{|\\vec{a}||\\vec{b}|}\\right|$ 取绝对值，确保 $\\cos\\theta\\geq0$，即 $\\theta\\leq90°$。若点积为负，取绝对值后得到正的余弦值，结果确实在 $(0°,90°]$。',
      },
      {
        type: 'choice',
        prompt: '三棱锥 $P$-$ABC$ 中，$PA\\perp$ 底面，$AB\\perp BC$，$PA=AB=BC=2$。底面法向量 $\\vec{n}_1=(0,0,1)$，平面 $PBC$ 法向量 $\\vec{n}_2=(1,0,-1)$，计算 $\\vec{n}_1\\cdot\\vec{n}_2$，则二面角 $P$-$BC$-$A$ 的余弦值为',
        options: ['$\\dfrac{\\sqrt{2}}{2}$', '$-\\dfrac{\\sqrt{2}}{2}$', '$\\dfrac{1}{2}$', '$-\\dfrac{1}{2}$'],
        answer: 1,
        explain: '$\\vec{n}_1\\cdot\\vec{n}_2=0+0+(-1)=-1$，$|\\vec{n}_1|=1$，$|\\vec{n}_2|=\\sqrt{2}$，$\\cos\\psi=\\frac{-1}{\\sqrt{2}}=-\\frac{\\sqrt{2}}{2}$。因 $PA\\perp$ 底面，面 $PBC$ 与底面成钝角（$\\vec{n}_1$ 指上，$\\vec{n}_2$ 的 $z$ 分量为负，指向异侧），$\\cos\\psi<0$ 正确，二面角为 $135°$。',
      },
      {
        type: 'input',
        prompt: '正方体棱长 $1$，以 $A$ 为原点建系，面对角线 $\\vec{A_1C}=(1,1,-1)$（从 $A_1$ 到 $C$），底面法向量 $\\vec{n}=(0,0,1)$，求 $\\sin\\varphi$（线面角），填精确值（如 $\\frac{\\sqrt{3}}{3}$）',
        accept: ['√3/3', '$\\frac{\\sqrt{3}}{3}$', '1/√3'],
        explain: '$\\sin\\varphi=\\left|\\frac{(1,1,-1)\\cdot(0,0,1)}{\\sqrt{3}\\cdot1}\\right|=\\left|\\frac{-1}{\\sqrt{3}}\\right|=\\frac{1}{\\sqrt{3}}=\\frac{\\sqrt{3}}{3}$。',
      },
      {
        type: 'match',
        prompt: '将空间角类型与对应的公式及特点配对',
        left: [
          '线线角',
          '线面角',
          '二面角',
          '直线垂直平面时的线面角',
        ],
        right: [
          '$\\varphi=90°$，$\\sin\\varphi=1$',
          '$\\cos\\theta=\\left|\\frac{\\vec{a}\\cdot\\vec{b}}{|\\vec{a}||\\vec{b}|}\\right|$，取绝对值',
          '$\\sin\\varphi=\\left|\\frac{\\vec{l}\\cdot\\vec{n}}{|\\vec{l}||\\vec{n}|}\\right|$，用 $\\sin$',
          '$\\cos\\psi=\\pm\\frac{\\vec{n}_1\\cdot\\vec{n}_2}{|\\vec{n}_1||\\vec{n}_2|}$，看图判正负',
        ],
        explain: '线线角用 $\\cos$ 取绝对值（锐角）；线面角用 $\\sin$ 取绝对值（余角关系）；二面角不取绝对值，看图判正负；直线垂直平面时线面角为 $90°$，$\\sin\\varphi=1$。',
      },
      {
        type: 'choice',
        prompt: '已知两平面法向量 $\\vec{n}_1=(1,1,0)$，$\\vec{n}_2=(0,1,1)$，则两平面所成二面角的 $|\\cos\\psi|=$',
        options: ['$\\dfrac{1}{2}$', '$\\dfrac{\\sqrt{2}}{2}$', '$0$', '$1$'],
        answer: 0,
        explain: '$\\vec{n}_1\\cdot\\vec{n}_2=0+1+0=1$，$|\\vec{n}_1|=\\sqrt{2}$，$|\\vec{n}_2|=\\sqrt{2}$，$|\\cos\\psi|=\\frac{1}{2}$，即两平面所成角为 $60°$ 或 $120°$（再看图判具体是哪个）。',
      },
      {
        type: 'choice',
        prompt: '正方体棱长 $1$，以 $B$ 为原点建系（$BA\\to x$，$BC\\to y$，$BB_1\\to z$），平面 $A_1BD$ 法向量为 $\\vec{n}_1=(1,-1,-1)$，则 $|\\vec{n}_1|=$',
        options: ['$\\sqrt{2}$', '$\\sqrt{3}$', '$2$', '$\\sqrt{5}$'],
        answer: 1,
        explain: '$|\\vec{n}_1|=\\sqrt{1^2+(-1)^2+(-1)^2}=\\sqrt{1+1+1}=\\sqrt{3}$。',
      },
    ],
  },
]
