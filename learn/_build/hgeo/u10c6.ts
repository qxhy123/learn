export const LESSONS = [
  {
    id: 'u10c6',
    title: '向量综合：三角 + 圆锥曲线 + 最值',
    subtitle: '向量条件翻译为代数，结合曲线或三角求解',
    intro: [
      {
        title: '向量综合题的核心逻辑',
        body: '向量综合题本质是"翻译"：把向量条件（垂直、模长、线性组合）翻译成代数语言（点积为零、坐标方程、三角式），再结合曲线方程或三角公式化简求解。无论是向量 + 三角、向量 + 圆锥曲线，还是向量 + 最值，处理步骤都是：① 识别向量条件类型；② 翻译为坐标/三角条件；③ 结合曲线方程（韦达定理/参数化）；④ 化简求目标量。',
        tip: '核心翻译对照：$\\overrightarrow{OA}\\cdot\\overrightarrow{OB}=0 \\Leftrightarrow x_1x_2+y_1y_2=0$（垂直）；$|\\overrightarrow{PA}|=|\\overrightarrow{PB}| \\Leftrightarrow PA=PB$（等距）；$\\overrightarrow{PA}=\\lambda\\overrightarrow{PB} \\Leftrightarrow A,B,P$ 共线（线性依赖）。',
        image: 'geo-p10-06-1.svg',
        imageCaption: '向量综合交叉结构图：向量与三角、圆锥曲线、不等式的连接',
      },
      {
        title: '向量 + 三角：单位向量点积 = 夹角余弦',
        body: '若 $\\vec{a}=(\\cos\\alpha,\\sin\\alpha)$，$\\vec{b}=(\\cos\\beta,\\sin\\beta)$ 均为单位向量（模长为 $1$），则 $\\vec{a}\\cdot\\vec{b}=\\cos\\alpha\\cos\\beta+\\sin\\alpha\\sin\\beta=\\cos(\\alpha-\\beta)$。这是向量证明三角恒等式的基础。引入例题：$|\\vec{a}-\\vec{b}|^2=(\\cos\\alpha-\\cos\\beta)^2+(\\sin\\alpha-\\sin\\beta)^2=2-2\\cos(\\alpha-\\beta)=4\\sin^2\\dfrac{\\alpha-\\beta}{2}$，故 $|\\vec{a}-\\vec{b}|=2\\left|\\sin\\dfrac{\\alpha-\\beta}{2}\\right|\\leq 2$（等号在 $\\alpha-\\beta=\\pm\\pi$ 时取到）。',
        formula: 'a\\sin\\theta + b\\cos\\theta = \\sqrt{a^2+b^2}\\sin\\!\\left(\\theta+\\varphi\\right) \\leq \\sqrt{a^2+b^2}',
      },
      {
        title: '向量 + 三角最值：辅助角法的向量解释',
        body: '$f(\\theta)=a\\sin\\theta+b\\cos\\theta$，令 $\\vec{u}=(a,b)$，$\\vec{v}=(\\sin\\theta,\\cos\\theta)$（单位向量），则 $f(\\theta)=\\vec{u}\\cdot\\vec{v}\\leq|\\vec{u}||\\vec{v}|=\\sqrt{a^2+b^2}$。等号成立当且仅当 $\\vec{u}\\parallel\\vec{v}$，即 $\\tan\\theta=a/b$（方向相同）。这比辅助角法更直观，直接得出最大值 $\\sqrt{a^2+b^2}$。例：$f(\\theta)=\\cos\\theta+\\sqrt{3}\\sin\\theta=2(\\frac{1}{2}\\cos\\theta+\\frac{\\sqrt{3}}{2}\\sin\\theta)=2\\sin(\\theta+\\frac{\\pi}{6})$，最大值 $2$（取 $\\theta=\\frac{\\pi}{3}$ 时）。',
        tip: '向量 + 三角最值：直接用 $\\sqrt{a^2+b^2}$ 给出最大值，无需辅助角展开。',
      },
      {
        title: '向量 + 圆锥曲线：翻译垂直条件与韦达结合',
        body: '椭圆上 $A(x_1,y_1)$、$B(x_2,y_2)$，$\\overrightarrow{OA}\\cdot\\overrightarrow{OB}=0$ 即 $x_1x_2+y_1y_2=0$。处理步骤：设直线 $l: y=kx+m$，代入椭圆整理为二次方程，韦达定理给出 $x_1+x_2$，$x_1x_2$；计算 $y_1y_2=k^2x_1x_2+km(x_1+x_2)+m^2$（全用韦达展开）；代入垂直条件 $x_1x_2+y_1y_2=0$，解出 $m^2$ 关于 $k$ 的表达式；再代入弦长公式或其他目标量，化简求最值。常用翻译：$\\overrightarrow{OA}+\\overrightarrow{OB}=\\vec{0} \\Leftrightarrow x_1+x_2=0,y_1+y_2=0$（关于原点对称）；$k\\overrightarrow{OA}+\\overrightarrow{OB}=\\vec{0} \\Leftrightarrow x_2=-kx_1,y_2=-ky_1$。',
        formula: 'x_1x_2 + y_1y_2 = 0 \\quad \\text{（} \\overrightarrow{OA}\\perp\\overrightarrow{OB} \\text{ 的坐标条件）}',
      },
      {
        title: '参数化椭圆处理 $|\\overrightarrow{OA}|^2+|\\overrightarrow{OB}|^2$ 范围',
        body: '椭圆 $\\dfrac{x^2}{4}+y^2=1$，设 $A=(2\\cos\\alpha,\\sin\\alpha)$，$B=(2\\cos\\beta,\\sin\\beta)$。$|\\overrightarrow{OA}|^2=4\\cos^2\\alpha+\\sin^2\\alpha=3\\cos^2\\alpha+1$，$|\\overrightarrow{OB}|^2=3\\cos^2\\beta+1$，$S=3(\\cos^2\\alpha+\\cos^2\\beta)+2$。垂直条件：$4\\cos\\alpha\\cos\\beta+\\sin\\alpha\\sin\\beta=0$，即 $\\tan\\alpha\\tan\\beta=-4$（$\\sin\\alpha,\\sin\\beta\\neq 0$ 时）。当 $A=(0,\\pm1)$，$B=(\\pm2,0)$ 时，$S=3(0+1)+2=5$（最小值）。当 $\\cos^2\\alpha$、$\\cos^2\\beta$ 趋向最大时 $S\\to 8$（但端点不可达）。结论：$S\\in[5,8)$。',
        tip: '参数化椭圆的关键：$|\\overrightarrow{OP}|^2=a^2\\cos^2 t+b^2\\sin^2 t=(a^2-b^2)\\cos^2 t+b^2$，是关于 $\\cos^2 t$ 的线性函数。',
      },
      {
        title: '向量模长最值：关于参数 $t$ 的二次函数',
        body: '$|\\vec{a}+t\\vec{b}|^2=|\\vec{a}|^2+2t(\\vec{a}\\cdot\\vec{b})+t^2|\\vec{b}|^2$（关于 $t$ 的二次函数），在 $t=-\\dfrac{\\vec{a}\\cdot\\vec{b}}{|\\vec{b}|^2}$ 时取最小值 $|\\vec{a}|^2-\\dfrac{(\\vec{a}\\cdot\\vec{b})^2}{|\\vec{b}|^2}$（即 $\\vec{a}$ 到直线 $t\\vec{b}$ 的距离平方）。这本质是投影的余量，几何意义是 $\\vec{a}$ 在 $\\vec{b}$ 的垂直方向上的分量。柯西不等式的向量形式：$(\\vec{a}\\cdot\\vec{b})^2\\leq|\\vec{a}|^2|\\vec{b}|^2$，等号成立当且仅当 $\\vec{a}\\parallel\\vec{b}$。',
        reveal: {
          q: '设 $\\vec{a}=(\\cos\\theta,\\sin\\theta)$，$\\vec{b}=(1,\\sqrt{3})$，求 $f(\\theta)=\\vec{a}\\cdot\\vec{b}$ 的最大值。',
          a: '$f(\\theta)=\\cos\\theta+\\sqrt{3}\\sin\\theta$，最大值 $=|\\vec{b}|=\\sqrt{1+3}=2$（$\\vec{a}$ 为单位向量时，点积最大等于 $|\\vec{b}|$）。',
        },
      },
      {
        title: '五大易错点与思路总结',
        body: '① $\\overrightarrow{OA}\\cdot\\overrightarrow{OB}=0$ 是 $x_1x_2+y_1y_2=0$（和为零），不是两者各自为零；② 高中只有数量积（结果是数），没有向量积（叉积）；③ $|\\vec{a}+\\vec{b}|^2=|\\vec{a}|^2+|\\vec{b}|^2$ 只在 $\\vec{a}\\perp\\vec{b}$ 时成立，一般情形有 $2\\vec{a}\\cdot\\vec{b}$ 的交叉项；④ 参数化椭圆中，参数角 $t$ 不是点与 $x$ 轴的真实夹角；⑤ 柯西不等式等号成立需要 $\\vec{a}\\parallel\\vec{b}$，验证最值时必须检查。思路总结：向量条件翻译 → 代入曲线（韦达/参数化）→ 化简目标量 → 用导数/换元求最值。',
        tip: '椭圆垂直条件 $\\overrightarrow{OA}\\cdot\\overrightarrow{OB}=0$ + 韦达定理，是最高频的向量 + 圆锥曲线组合，熟练模板即可快速推进。',
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '已知 $\\vec{a}=(\\cos\\alpha,\\sin\\alpha)$，$\\vec{b}=(\\cos\\beta,\\sin\\beta)$，则 $\\vec{a}\\cdot\\vec{b}=$？',
        options: [
          '$\\cos(\\alpha-\\beta)$',
          '$\\cos(\\alpha+\\beta)$',
          '$\\sin(\\alpha-\\beta)$',
          '$\\cos\\alpha\\cos\\beta$',
        ],
        answer: 0,
        explain: '$\\vec{a}\\cdot\\vec{b}=\\cos\\alpha\\cos\\beta+\\sin\\alpha\\sin\\beta=\\cos(\\alpha-\\beta)$，这是余弦差角公式的向量形式。',
      },
      {
        type: 'choice',
        prompt: '$f(\\theta)=3\\sin\\theta+4\\cos\\theta$ 的最大值是？',
        options: [
          '$5$',
          '$7$',
          '$4$',
          '$3$',
        ],
        answer: 0,
        explain: '$f(\\theta)=\\vec{u}\\cdot\\vec{v}$，其中 $\\vec{u}=(3,4)$，$\\vec{v}=(\\sin\\theta,\\cos\\theta)$（单位向量）。最大值 $=|\\vec{u}|=\\sqrt{9+16}=5$。',
      },
      {
        type: 'judge',
        prompt: '椭圆上 $A(x_1,y_1)$、$B(x_2,y_2)$ 满足 $\\overrightarrow{OA}\\cdot\\overrightarrow{OB}=0$，等价于 $x_1x_2=0$ 且 $y_1y_2=0$。',
        answer: false,
        explain: '错误。$\\overrightarrow{OA}\\cdot\\overrightarrow{OB}=x_1x_2+y_1y_2=0$ 是两者之**和**为零，不是两者各自为零。例如 $x_1x_2=1$，$y_1y_2=-1$ 也满足条件，但 $x_1x_2\\neq 0$ 且 $y_1y_2\\neq 0$。',
      },
      {
        type: 'choice',
        prompt: '向量 $\\vec{a}=(1,t)$，$\\vec{b}=(t,4)$ 同向（$t>0$），则 $|\\vec{a}+\\vec{b}|=$？',
        options: [
          '$3\\sqrt{5}$',
          '$\\sqrt{5}$',
          '$5\\sqrt{2}$',
          '$\\sqrt{13}$',
        ],
        answer: 0,
        explain: '同向 $\\Rightarrow \\vec{b}=\\lambda\\vec{a}$（$\\lambda>0$）：$t=\\lambda\\cdot 1$，$4=\\lambda t=t^2$，$t=2$（取正）。$\\vec{a}=(1,2)$，$\\vec{b}=(2,4)$，$\\vec{a}+\\vec{b}=(3,6)$，$|\\vec{a}+\\vec{b}|=\\sqrt{9+36}=\\sqrt{45}=3\\sqrt{5}$。',
      },
      {
        type: 'input',
        prompt: '椭圆 $\\dfrac{x^2}{9}+\\dfrac{y^2}{4}=1$，焦点 $F_1(-\\sqrt{5},0)$，$F_2(\\sqrt{5},0)$，椭圆上点 $P(x_0,y_0)$（$y_0\\neq 0$）满足 $\\overrightarrow{F_1P}\\cdot\\overrightarrow{F_2P}=0$，求 $|\\overrightarrow{OP}|$。（提示：$\\overrightarrow{F_1P}\\cdot\\overrightarrow{F_2P}=x_0^2-5+y_0^2=0$。）',
        accept: ['\\sqrt{5}', 'sqrt(5)'],
        explain: '$\\overrightarrow{F_1P}=(x_0+\\sqrt{5},y_0)$，$\\overrightarrow{F_2P}=(x_0-\\sqrt{5},y_0)$，点积 $=(x_0+\\sqrt{5})(x_0-\\sqrt{5})+y_0^2=x_0^2-5+y_0^2=0$，即 $x_0^2+y_0^2=5$，$|\\overrightarrow{OP}|=\\sqrt{5}$。（验证：联立椭圆方程解出 $x_0^2=\\dfrac{9}{5}$，$y_0^2=\\dfrac{16}{5}$，$|\\overrightarrow{OP}|^2=5$，成立。）',
      },
      {
        type: 'match',
        prompt: '将向量条件与对应的几何/代数含义配对：',
        left: [
          '$\\overrightarrow{OA}\\cdot\\overrightarrow{OB}=0$',
          '$\\overrightarrow{OA}+\\overrightarrow{OB}=\\vec{0}$',
          '$\\overrightarrow{PA}=\\lambda\\overrightarrow{PB}$（$\\lambda\\neq 0,-1$）',
          '$|\\overrightarrow{OA}|=|\\overrightarrow{OB}|$',
        ],
        right: [
          '$A$、$B$ 到原点等距，$A$、$B$ 在以 $O$ 为圆心的某圆上',
          '$A$、$B$、$P$ 三点共线',
          '$A$、$B$ 关于原点 $O$ 对称（$x_1+x_2=0$，$y_1+y_2=0$）',
          '$OA\\perp OB$（$x_1x_2+y_1y_2=0$）',
        ],
        explain: '点积为零 → 垂直；和为零向量 → 关于原点对称；线性依赖 → 共线；模长相等 → 等距（在以 $O$ 为圆心的圆上）。',
      },
      {
        type: 'judge',
        prompt: '椭圆参数化 $x=a\\cos t$，$y=b\\sin t$ 中，参数 $t$ 就是点 $(x,y)$ 与原点连线和 $x$ 轴正方向的夹角。',
        answer: false,
        explain: '错误。参数角 $t$ 不是真实夹角（几何角）。真实夹角 $\\theta$ 满足 $\\tan\\theta=y/x=\\dfrac{b\\sin t}{a\\cos t}=\\dfrac{b}{a}\\tan t$，与 $t$ 的关系不是 $t=\\theta$（除非 $a=b$，即圆的情形）。这是参数化椭圆的常见误区。',
      },
      {
        type: 'choice',
        prompt: '已知 $|\\vec{a}+t\\vec{b}|^2$ 关于 $t$ 的最小值（$\\vec{a}=(1,2)$，$\\vec{b}=(2,1)$），则最小值为？',
        options: [
          '$\\dfrac{9}{5}$',
          '$5$',
          '$1$',
          '$\\dfrac{1}{5}$',
        ],
        answer: 0,
        explain: '$|\\vec{a}|^2=5$，$|\\vec{b}|^2=5$，$\\vec{a}\\cdot\\vec{b}=1\\times2+2\\times1=4$。最小值 $=|\\vec{a}|^2-\\dfrac{(\\vec{a}\\cdot\\vec{b})^2}{|\\vec{b}|^2}=5-\\dfrac{16}{5}=\\dfrac{9}{5}$。',
      },
    ],
  },
]
