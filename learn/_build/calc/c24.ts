// 第24章  曲面积分
// 关 c24-a：第一类曲面积分——对面积的积分（8 卡 7 题）
// 关 c24-b：第二类曲面积分——对坐标的积分（7 卡 7 题）
// 关 c24-c：Gauss 散度定理与 Stokes 旋度定理（8 卡 8 题）

export const LESSONS = [
  // ─────────────────────────────────────────
  // 关 c24-a  第一类曲面积分（对面积的积分）
  // ─────────────────────────────────────────
  {
    id: 'c24-a',
    title: '第一类曲面积分：对面积积分',
    subtitle: '动机·面积元·投影化为二重积分·例题·易错点',
    intro: [
      // 卡 1：动机——曲面质量问题
      {
        title: '为什么需要曲面积分？',
        body:
          '把一块薄铁板弯成曲面形状，面密度（每单位面积的质量）在各点不同，如何求总质量？\n\n' +
          '一维时，我们用"切成小段，每段质量 $\\approx$ 密度 $\\times$ 弧长，再求和"——这给出了曲线积分。\n\n' +
          '二维曲面时，思路完全一样：把曲面切成许多小片，每片面积记作 $\\Delta S_i$，该片密度记作 $\\rho(\\xi_i,\\eta_i,\\zeta_i)$，质量近似为 $\\rho_i \\Delta S_i$，加总取极限，就得到**第一类曲面积分**：\n\n' +
          '$\\iint_S f(x,y,z)\\,dS$\n\n' +
          '注意 $dS$ 是面积元，**和法向量方向无关**——这一点将来与第二类曲面积分形成关键对比。',
        tip: '记忆要点：第一类 $dS$ = 面积元，无方向；第二类 $d\\mathbf{S} = \\mathbf{n}\\,dS$，有法向。',
      },
      // 卡 2：定义
      {
        title: '定义：分割-近似-求和-极限',
        body:
          '设 $f(x,y,z)$ 在光滑曲面 $S$ 上有界。把 $S$ 任意分成 $n$ 小片 $\\Delta S_1,\\ldots,\\Delta S_n$（$\\Delta S_i$ 既表示小片也表示其面积），在每片上任取一点 $(\\xi_i,\\eta_i,\\zeta_i)$，作黎曼和：\n\n' +
          '$\\displaystyle\\sum_{i=1}^n f(\\xi_i,\\eta_i,\\zeta_i)\\,\\Delta S_i$\n\n' +
          '当最大片直径 $\\lambda \\to 0$ 时，若极限存在且与分法、取点无关，则称之为**第一类曲面积分**，记为 $\\iint_S f\\,dS$。\n\n' +
          '**物理速查**：$f=$ 面密度 $\\Rightarrow$ 结果是曲面总质量；$f=1 \\Rightarrow$ 结果是曲面面积。',
        formula: '\\iint_S f(x,y,z)\\,dS = \\lim_{\\lambda\\to 0}\\sum_{i=1}^n f(\\xi_i,\\eta_i,\\zeta_i)\\,\\Delta S_i',
      },
      // 卡 3：面积元 dS 的推导
      {
        title: '面积元 $dS$ 从哪里来？',
        body:
          '设曲面方程为 $z = z(x,y)$，$(x,y)\\in D_{xy}$，且 $z$ 有连续偏导数。\n\n' +
          '在点 $(x_0,y_0)$ 附近，曲面近似于切平面。切平面上由 $dx,dy$ 张成的小平行四边形在切平面内的面积，是平投影面积 $dx\\,dy$ 乘以斜率放大因子：\n\n' +
          '**直觉**：若曲面倾斜，同样的 $dx\\,dy$ 在曲面上对应更大的面积。具体计算可验证放大因子正好是 $\\sqrt{1+z_x^2+z_y^2}$。\n\n' +
          '因此面积元为：$dS = \\sqrt{1+z_x^2+z_y^2}\\,dx\\,dy$\n\n' +
          '**记忆口诀**：根号下是“$1$ 加 $x$ 偏导的平方 加 $y$ 偏导的平方”，即 $dS = \\sqrt{1+z_x^2+z_y^2}\\,dx\\,dy$。',
        formula: 'dS = \\sqrt{1 + z_x^2 + z_y^2}\\,dx\\,dy',
        tip: '当曲面是水平面 $z=c$ 时，$z_x=z_y=0$，$dS=dx\\,dy$，回到普通面积——符合直觉。',
      },
      // 卡 4：计算公式
      {
        title: '计算公式：投影到 $xOy$ 平面',
        body:
          '有了面积元，第一类曲面积分化为二重积分：将 $z=z(x,y)$ 代入 $f$，再乘以面积元，在投影区域 $D_{xy}$ 上积分：\n\n' +
          '$\\iint_S f(x,y,z)\\,dS = \\iint_{D_{xy}} f\\!\\left(x,y,z(x,y)\\right)\\sqrt{1+z_x^2+z_y^2}\\,dx\\,dy$\n\n' +
          '若曲面由 $x=x(y,z)$ 给出，则投影到 $yOz$ 面，面积元为 $\\sqrt{1+x_y^2+x_z^2}\\,dy\\,dz$；由 $y=y(x,z)$ 给出类似。\n\n' +
          '**步骤总结**：① 写出曲面方程并求偏导；② 算 $\\sqrt{1+z_x^2+z_y^2}$；③ 代入被积函数；④ 写出投影区域；⑤ 化为二重积分计算。',
        formula: '\\iint_S f\\,dS = \\iint_{D_{xy}} f\\bigl(x,y,z(x,y)\\bigr)\\sqrt{1+z_x^2+z_y^2}\\,dx\\,dy',
      },
      // 卡 5：例题——上半球面
      {
        title: '例题：$\\iint_S z\\,dS$，$S$ 为单位上半球面',
        body:
          '计算 $\\iint_S z\\,dS$，$S: x^2+y^2+z^2=1$，$z\\geq 0$。\n\n' +
          '上半球面方程为 $z=\\sqrt{1-x^2-y^2}$，投影 $D_{xy}: x^2+y^2\\leq 1$。',
        steps: [
          '第一步：偏导。$z_x = \\dfrac{-x}{\\sqrt{1-x^2-y^2}}$，$z_y = \\dfrac{-y}{\\sqrt{1-x^2-y^2}}$。',
          '第二步：面积元因子。$\\sqrt{1+z_x^2+z_y^2} = \\sqrt{1+\\dfrac{x^2+y^2}{1-x^2-y^2}} = \\dfrac{1}{\\sqrt{1-x^2-y^2}}$。',
          '第三步：代入。$\\iint_S z\\,dS = \\iint_{D_{xy}} \\sqrt{1-x^2-y^2}\\cdot\\dfrac{1}{\\sqrt{1-x^2-y^2}}\\,dx\\,dy = \\iint_{D_{xy}} 1\\,dx\\,dy$。',
          '第四步：$\\iint_{D_{xy}} dx\\,dy =$ 单位圆面积 $= \\pi\\cdot 1^2 = \\pi$。',
          '结果：$\\boxed{\\pi}$。',
        ],
        tip: '面积元因子与 $z$ 恰好约分，这是上半球的特殊之处。计算前先化简会省去大量运算。',
      },
      // 卡 6：例题——抛物面锥面
      {
        title: '例题：锥面 $z=\\sqrt{x^2+y^2}$（$0\\leq z\\leq 1$）的面积元',
        body:
          '求 $\\iint_S (x^2+y^2)\\,dS$，$S: z=\\sqrt{x^2+y^2}$，$0\\leq z\\leq 1$。',
        steps: [
          '第一步：偏导。$z_x=\\dfrac{x}{\\sqrt{x^2+y^2}}$，$z_y=\\dfrac{y}{\\sqrt{x^2+y^2}}$。',
          '第二步：面积元因子。$\\sqrt{1+z_x^2+z_y^2}=\\sqrt{1+\\dfrac{x^2+y^2}{x^2+y^2}}=\\sqrt{2}$（常数！）。',
          '第三步：投影区域。$z\\leq 1\\Rightarrow\\sqrt{x^2+y^2}\\leq 1$，即 $D_{xy}: x^2+y^2\\leq 1$。',
          '第四步：代入。$\\iint_S (x^2+y^2)\\,dS=\\sqrt{2}\\iint_{D_{xy}}(x^2+y^2)\\,dx\\,dy$。',
          '第五步：极坐标。$=\\sqrt{2}\\int_0^{2\\pi}d\\theta\\int_0^1 r^2\\cdot r\\,dr=\\sqrt{2}\\cdot 2\\pi\\cdot\\dfrac{1}{4}=\\dfrac{\\sqrt{2}\\pi}{2}$。',
        ],
        tip: '锥面的面积元因子是常数 $\\sqrt{2}$，这让计算大大简化——遇到锥面先算 $\\sqrt{1+z_x^2+z_y^2}$ 往往有惊喜。',
      },
      // 卡 7：性质与对称性
      {
        title: '奇偶对称性加速计算',
        body:
          '若曲面 $S$ 关于某坐标平面对称，被积函数关于对应变量是奇函数，则积分为零。\n\n' +
          '**例**：$S$ 为整个单位球面（关于三个坐标平面都对称），$f=x$（关于 $x$ 为奇函数），则 $\\iint_S x\\,dS=0$。\n\n' +
          '若 $f$ 关于 $x,y,z$ 的地位完全相同（如球面），则 $\\iint_S x^2\\,dS=\\iint_S y^2\\,dS=\\iint_S z^2\\,dS$，可将三项平均化简。\n\n' +
          '**操作**：$\\iint_S (x^2+y^2+z^2)\\,dS = 3\\iint_S z^2\\,dS$，而球面上 $x^2+y^2+z^2=R^2$ 是常数，所以 $3\\iint_S z^2\\,dS = R^2\\cdot 3 = R^2 \\cdot$ 球面面积 $/ 3 \\cdot 3$……（具体用面积 $4\\pi R^2$）。',
        tip: '考试时遇到全球面 + $x^2,y^2,z^2$ 的组合，第一反应是用对称性把三项合并为一个，再利用 $x^2+y^2+z^2=R^2$ 提取常数。',
      },
      // 卡 8：易错点 + reveal
      {
        title: '易错点与自测',
        body:
          '**易错一**：第一类曲面积分 $\\iint_S f\\,dS$ 结果**不随法向方向改变**——翻转曲面方向积分值不变（因为 $dS > 0$ 恒成立）。\n\n' +
          '**易错二**：$\\sqrt{1+z_x^2+z_y^2}$ 是面积放大因子，**必须乘上去**，不能省略——省去就变成普通二重积分，量纲都错了。\n\n' +
          '**易错三**：投影区域 $D_{xy}$ 要根据曲面约束条件来确定，不能随意写成矩形或圆。\n\n' +
          '**易错四**：对于分段定义的曲面（如上半球面 + 圆盘），要分段计算再相加。',
        reveal: {
          q: '曲面 $S: z=1$（$x^2+y^2\\leq 4$）上，$\\iint_S 3\\,dS$ 等于多少？试用面积元公式快速计算。',
          a: '$z=1$ 是水平面，$z_x=z_y=0$，$\\sqrt{1+0+0}=1$，所以 $\\iint_S 3\\,dS = 3\\iint_{D_{xy}}dx\\,dy = 3\\cdot\\pi\\cdot 2^2 = 12\\pi$。平面上第一类曲面积分就等于普通二重积分。',
        },
      },
    ],
    questions: [
      // 题 1：judge — dS 与法向方向无关
      {
        id: 'c24-a-q1',
        type: 'judge',
        prompt:
          '第一类曲面积分 $\\iint_S f\\,dS$ 的值，当曲面取上侧与取下侧时可能不同。',
        answer: false,
        explain:
          '第一类曲面积分中，面积元 $dS > 0$ 始终为正，与法向量方向无关。翻转曲面的定向（上侧变下侧）不改变积分值。这和第二类曲面积分（定向改变则变号）是根本区别。',
      },
      // 题 2：choice — 面积元
      {
        id: 'c24-a-q2',
        type: 'choice',
        prompt:
          '曲面 $z=3x+4y$（$(x,y)\\in D$）的面积元 $dS$ 等于哪项？',
        options: [
          '$dx\\,dy$',
          '$\\sqrt{1+9+16}\\,dx\\,dy = \\sqrt{26}\\,dx\\,dy$',
          '$\\sqrt{9+16}\\,dx\\,dy = 5\\,dx\\,dy$',
          '$\\sqrt{1+3+4}\\,dx\\,dy$',
        ],
        answer: 1,
        explain:
          '$z_x=3$，$z_y=4$，$\\sqrt{1+z_x^2+z_y^2}=\\sqrt{1+9+16}=\\sqrt{26}$，$dS=\\sqrt{26}\\,dx\\,dy$。注意分母里是 $z_x^2+z_y^2$，不是 $z_x+z_y$。',
      },
      // 题 3：input — 平面曲面积分
      {
        id: 'c24-a-q3',
        type: 'input',
        prompt:
          '设 $S$ 是平面 $z=0$，$0\\leq x\\leq 1$，$0\\leq y\\leq 2$。计算 $\\iint_S 5\\,dS$。',
        accept: ['10', '10.0'],
        placeholder: '输入数值',
        explain:
          '$z=0$ 是水平面，$z_x=z_y=0$，$dS=dx\\,dy$。$\\iint_S 5\\,dS = 5\\iint_D dx\\,dy = 5\\times(1\\times 2)=10$。',
      },
      // 题 4：choice — 锥面面积元
      {
        id: 'c24-a-q4',
        type: 'choice',
        prompt:
          '锥面 $z=\\sqrt{x^2+y^2}$ 的面积元因子 $\\sqrt{1+z_x^2+z_y^2}$ 等于多少？',
        options: ['$1$', '$\\sqrt{2}$', '$2$', '$\\sqrt{1+x^2+y^2}$'],
        answer: 1,
        explain:
          '$z_x=\\dfrac{x}{\\sqrt{x^2+y^2}}$，$z_y=\\dfrac{y}{\\sqrt{x^2+y^2}}$，$z_x^2+z_y^2=\\dfrac{x^2+y^2}{x^2+y^2}=1$，故 $\\sqrt{1+1}=\\sqrt{2}$。锥面的面积元因子是常数，这是常考结论。',
      },
      // 题 5：judge — f=1 时的含义
      {
        id: 'c24-a-q5',
        type: 'judge',
        prompt: '$\\iint_S 1\\,dS$ 等于曲面 $S$ 的面积。',
        answer: true,
        explain:
          '由第一类曲面积分的定义，当 $f\\equiv 1$ 时，积分等于把所有小面积元求和的极限，即曲面的总面积。这是最基本的应用，也是用积分公式计算曲面面积的理论基础。',
      },
      // 题 6：match — 面积元公式
      {
        id: 'c24-a-q6',
        type: 'match',
        prompt: '将曲面形式与对应的面积元（投影到对应坐标面）配对。',
        left: [
          '$z=z(x,y)$，投影到 $xOy$',
          '$x=x(y,z)$，投影到 $yOz$',
          '$y=y(x,z)$，投影到 $xOz$',
        ],
        right: [
          '$\\sqrt{1+z_x^2+z_y^2}\\,dx\\,dy$',
          '$\\sqrt{1+x_y^2+x_z^2}\\,dy\\,dz$',
          '$\\sqrt{1+y_x^2+y_z^2}\\,dx\\,dz$',
        ],
      },
      // 题 7：input — 上半球积分（单位球）
      {
        id: 'c24-a-q7',
        type: 'input',
        prompt:
          '计算 $\\iint_S z\\,dS$，$S$ 为单位上半球面 $x^2+y^2+z^2=1$，$z\\geq 0$。结果等于 $k\\pi$，$k$ 是多少？',
        accept: ['1', '1.0'],
        explain:
          '上半球面 $z=\\sqrt{1-x^2-y^2}$，面积元因子为 $\\dfrac{1}{\\sqrt{1-x^2-y^2}}$，代入后 $z\\cdot\\text{因子}=1$，积分化为 $\\iint_{D_{xy}}dx\\,dy=\\pi$，故 $k=1$。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c24-b  第二类曲面积分（对坐标的积分）
  // ─────────────────────────────────────────
  {
    id: 'c24-b',
    title: '第二类曲面积分：通量与定向',
    subtitle: '流量动机·曲面定向·计算公式·法向决定符号·易错',
    intro: [
      // 卡 1：动机——流体通量
      {
        title: '动机：流体穿过曲面的通量',
        body:
          '设流体速度场为 $\\mathbf{F}(x,y,z)=P\\mathbf{i}+Q\\mathbf{j}+R\\mathbf{k}$，$S$ 是流体中的一块曲面。\n\n' +
          '单位时间内穿过曲面的流量是多少？\n\n' +
          '对小面元 $\\Delta S_i$，穿过该面元的流量近似 $= \\mathbf{F}\\cdot\\mathbf{n}_i\\,\\Delta S_i$（法向分量 $\\times$ 面积）。\n\n' +
          '从法向正侧流向负侧为负，反之为正——**方向很关键**，必须确定曲面的定向。\n\n' +
          '累加取极限，就得到**第二类曲面积分**（通量积分）：$\\iint_\\Sigma \\mathbf{F}\\cdot d\\mathbf{S}$，其中 $d\\mathbf{S}=\\mathbf{n}\\,dS$ 是带法向的向量面积元。',
        tip: '与第一类对比：第一类 $\\iint_S f\\,dS$ 中 $dS>0$ 无方向；第二类 $\\iint_\\Sigma \\mathbf{F}\\cdot d\\mathbf{S}$ 中 $d\\mathbf{S}=\\mathbf{n}\\,dS$ 有方向，翻转法向则积分变号。',
      },
      // 卡 2：曲面定向
      {
        title: '曲面的定向：选择哪侧为正',
        body:
          '可定向曲面（双面曲面）上，每点都有两个单位法向量，选其一为正侧。\n\n' +
          '**封闭曲面**（如球面、正方体表面）：\n\n' +
          '- 外侧（法向指向曲面外部）——Gauss 定理标准约定\n' +
          '- 内侧（法向指向曲面内部）\n\n' +
          '**非封闭曲面** $z=z(x,y)$：\n\n' +
          '- 上侧：法向量与 $z$ 轴正方向夹角为锐角，即法向 $z$ 分量 $\\cos\\gamma>0$\n' +
          '- 下侧：$\\cos\\gamma<0$\n\n' +
          '**莫比乌斯带**是不可定向曲面的经典例子——它只有一个面，没有"两侧"之分，不能做第二类曲面积分。',
        tip: '口诀：外侧朝外，上侧朝天，下侧朝地。定向确定了，计算就固定了符号。',
      },
      // 卡 3：定义与分量写法
      {
        title: '定义与分量展开写法',
        body:
          '设 $\\Sigma$ 是有向曲面，$\\mathbf{n}=(\\cos\\alpha,\\cos\\beta,\\cos\\gamma)$ 是指定侧的单位法向量，$\\mathbf{F}=P\\mathbf{i}+Q\\mathbf{j}+R\\mathbf{k}$。\n\n' +
          '**第二类曲面积分**定义为：\n\n' +
          '$\\iint_\\Sigma \\mathbf{F}\\cdot d\\mathbf{S} = \\iint_\\Sigma (P\\cos\\alpha+Q\\cos\\beta+R\\cos\\gamma)\\,dS$\n\n' +
          '常用分量写法：令 $dy\\,dz=\\cos\\alpha\\,dS$，$dz\\,dx=\\cos\\beta\\,dS$，$dx\\,dy=\\cos\\gamma\\,dS$，则\n\n' +
          '$\\iint_\\Sigma P\\,dy\\,dz+Q\\,dz\\,dx+R\\,dx\\,dy$\n\n' +
          '**注意**：$dx\\,dy=\\cos\\gamma\\,dS$ 表示带符号的投影面积元，当取上侧时 $\\cos\\gamma>0$，$dx\\,dy>0$；取下侧时 $\\cos\\gamma<0$，$dx\\,dy<0$。',
        formula: '\\iint_\\Sigma P\\,dy\\,dz+Q\\,dz\\,dx+R\\,dx\\,dy = \\iint_\\Sigma (P\\cos\\alpha+Q\\cos\\beta+R\\cos\\gamma)\\,dS',
      },
      // 卡 4：计算公式（核心）
      {
        title: '计算公式：投影 + 符号',
        body:
          '设 $\\Sigma: z=z(x,y)$，$(x,y)\\in D_{xy}$。只讨论 $R\\,dx\\,dy$ 分量（其余类似）：\n\n' +
          '**取上侧**（$\\cos\\gamma>0$）：$\\iint_\\Sigma R(x,y,z)\\,dx\\,dy = +\\iint_{D_{xy}} R(x,y,z(x,y))\\,dx\\,dy$\n\n' +
          '**取下侧**（$\\cos\\gamma<0$）：$\\iint_\\Sigma R(x,y,z)\\,dx\\,dy = -\\iint_{D_{xy}} R(x,y,z(x,y))\\,dx\\,dy$\n\n' +
          '类似地，对 $P\\,dy\\,dz$ 投影到 $yOz$ 面，对 $Q\\,dz\\,dx$ 投影到 $xOz$ 面，上/外侧取正，下/内侧取负。\n\n' +
          '**实用四步**：① 确认曲面方程和定向；② 将 $z=z(x,y)$ 代入 $R$；③ 写出投影区域 $D_{xy}$；④ 按上侧/下侧决定 $\\pm$。',
        formula: '\\iint_\\Sigma R\\,dx\\,dy = \\pm\\iint_{D_{xy}} R(x,y,z(x,y))\\,dx\\,dy',
        tip: '符号口诀：上侧正，下侧负；外侧（对 $z$）正，内侧（对 $z$）负。遇到球面外侧时，按分量分别判断各坐标方向。',
      },
      // 卡 5：例题——抛物面通量
      {
        title: '例题：$\\iint_\\Sigma z\\,dx\\,dy$，抛物面上侧',
        body:
          '计算 $\\iint_\\Sigma z\\,dx\\,dy$，$\\Sigma: z=1-x^2-y^2$（$z\\geq 0$），取上侧。',
        steps: [
          '第一步：$z\\geq 0\\Rightarrow x^2+y^2\\leq 1$，投影区域 $D_{xy}=\\{(x,y)\\mid x^2+y^2\\leq 1\\}$。',
          '第二步：上侧，$\\cos\\gamma>0$，取正号。',
          '第三步：代入 $z=1-x^2-y^2$。$\\iint_\\Sigma z\\,dx\\,dy = \\iint_{D_{xy}}(1-x^2-y^2)\\,dx\\,dy$。',
          '第四步：极坐标。$=\\int_0^{2\\pi}d\\theta\\int_0^1(1-r^2)\\cdot r\\,dr=2\\pi\\left[\\dfrac{r^2}{2}-\\dfrac{r^4}{4}\\right]_0^1=2\\pi\\cdot\\dfrac{1}{4}=\\dfrac{\\pi}{2}$。',
        ],
        tip: '这道题只有 $R\\,dx\\,dy$ 分量，$P=Q=0$，所以只用投影到 $xOy$ 面，不需要处理 $dy\\,dz$ 和 $dz\\,dx$。',
      },
      // 卡 6：两类积分的关系
      {
        title: '两类曲面积分的关系',
        body:
          '第一类和第二类曲面积分用法向量联系起来：\n\n' +
          '$\\iint_\\Sigma P\\,dy\\,dz+Q\\,dz\\,dx+R\\,dx\\,dy = \\iint_\\Sigma (P\\cos\\alpha+Q\\cos\\beta+R\\cos\\gamma)\\,dS$\n\n' +
          '**左边**是第二类（带符号，有方向）；**右边**是用法向量展开后变成被积函数 $= \\mathbf{F}\\cdot\\mathbf{n}$ 的第一类积分（$dS>0$ 无方向，方向信息在 $\\cos\\alpha,\\cos\\beta,\\cos\\gamma$ 里）。\n\n' +
          '**实用价值**：若曲面有简洁的法向量表达式（如单位球外法向 $\\mathbf{n}=(x,y,z)$），可用右边的形式直接代入，省去分分量投影的麻烦。',
      },
      // 卡 7：易错点 + reveal
      {
        title: '易错点与自测',
        body:
          '**易错一**：定向反转，积分**变号**。$\\Sigma$ 取下侧与上侧的积分值互为相反数。\n\n' +
          '**易错二**：对 $\\iint_\\Sigma R\\,dx\\,dy$，"$dx\\,dy$"并非普通面积，而是带符号的投影面积元，下侧为负。\n\n' +
          '**易错三**：三个分量 $\\iint P\\,dy\\,dz$、$\\iint Q\\,dz\\,dx$、$\\iint R\\,dx\\,dy$ 分别投影到不同坐标面，不要都投影到 $xOy$。\n\n' +
          '**易错四**：封闭曲面外侧并非所有方向都"正"——对 $\\iint_\\Sigma R\\,dx\\,dy$，球面上方部分外法向 $z$ 分量为正，下方部分外法向 $z$ 分量为负，需分上下两部分分别处理（或用 Gauss 定理一步搞定）。',
        reveal: {
          q: '曲面 $\\Sigma: z=0$，$x^2+y^2\\leq 1$，取下侧（法向朝下，$\\mathbf{n}=(0,0,-1)$）。$\\iint_\\Sigma 3\\,dx\\,dy$ 等于多少？',
          a: '取下侧，$\\cos\\gamma=-1<0$，故 $\\iint_\\Sigma 3\\,dx\\,dy=-\\iint_{D_{xy}}3\\,dx\\,dy=-3\\pi$。若取上侧则为 $+3\\pi$，定向反转结果恰好取反。',
        },
      },
    ],
    questions: [
      // 题 1：judge — 定向改变变号
      {
        id: 'c24-b-q1',
        type: 'judge',
        prompt:
          '将曲面 $\\Sigma$ 的定向（上侧改为下侧），第二类曲面积分 $\\iint_\\Sigma P\\,dy\\,dz+Q\\,dz\\,dx+R\\,dx\\,dy$ 的值变为原来的相反数。',
        answer: true,
        explain:
          '第二类曲面积分与法向量方向有关。翻转定向等于将所有法向量取负，$\\cos\\alpha,\\cos\\beta,\\cos\\gamma$ 全变号，被积量 $P\\cos\\alpha+Q\\cos\\beta+R\\cos\\gamma$ 也变号，积分值变为相反数。',
      },
      // 题 2：choice — 上侧符号
      {
        id: 'c24-b-q2',
        type: 'choice',
        prompt:
          '曲面 $\\Sigma: z=x+y$（$0\\leq x\\leq 1$，$0\\leq y\\leq 1$）取上侧，则 $\\iint_\\Sigma R(x,y,z)\\,dx\\,dy$ 等于哪种写法？',
        options: [
          '$-\\iint_D R(x,y,x+y)\\,dx\\,dy$',
          '$+\\iint_D R(x,y,x+y)\\,dx\\,dy$',
          '$\\iint_D R(x,y,x+y)\\sqrt{1+1+1}\\,dx\\,dy$',
          '$0$',
        ],
        answer: 1,
        explain:
          '上侧意味着 $\\cos\\gamma>0$，取正号。代入 $z=x+y$ 得 $+\\iint_D R(x,y,x+y)\\,dx\\,dy$。注意这里无需乘面积元因子——$dx\\,dy$ 这个分量的公式已经是纯投影，没有根号项。',
      },
      // 题 3：input — 抛物面通量
      {
        id: 'c24-b-q3',
        type: 'input',
        prompt:
          '计算 $\\iint_\\Sigma z\\,dx\\,dy$，$\\Sigma: z=1-x^2-y^2$（$z\\geq 0$），上侧。结果写成 $\\dfrac{\\pi}{k}$ 的形式，$k$ 等于多少？',
        accept: ['2'],
        placeholder: '输入整数 k',
        explain:
          '上侧正号，代入 $z=1-x^2-y^2$：$\\iint_{D}(1-x^2-y^2)\\,dx\\,dy$，极坐标得 $2\\pi\\cdot\\frac{1}{4}=\\frac{\\pi}{2}$，故 $k=2$。',
      },
      // 题 4：judge — 第一类不变号
      {
        id: 'c24-b-q4',
        type: 'judge',
        prompt:
          '翻转曲面定向后，第一类曲面积分 $\\iint_S f\\,dS$ 的值也会变号。',
        answer: false,
        explain:
          '第一类曲面积分中，面积元 $dS$ 是标量且恒为正，与定向无关。翻转定向不影响 $dS$ 的值，因此 $\\iint_S f\\,dS$ 的值不变。这是两类积分的核心区别。',
      },
      // 题 5：choice — 两类关系
      {
        id: 'c24-b-q5',
        type: 'choice',
        prompt:
          '下列哪个等式体现了两类曲面积分的关系（$\\mathbf{n}=(\\cos\\alpha,\\cos\\beta,\\cos\\gamma)$ 为单位法向量）？',
        options: [
          '$\\iint_\\Sigma P\\,dy\\,dz = \\iint_\\Sigma P\\,dS$',
          '$\\iint_\\Sigma \\mathbf{F}\\cdot d\\mathbf{S} = \\iint_\\Sigma (P\\cos\\alpha+Q\\cos\\beta+R\\cos\\gamma)\\,dS$',
          '$\\iint_\\Sigma R\\,dx\\,dy = \\iint_\\Sigma R\\sqrt{1+z_x^2+z_y^2}\\,dS$',
          '$\\iint_\\Sigma f\\,dS = \\iint_\\Sigma f\\,dx\\,dy$',
        ],
        answer: 1,
        explain:
          '第二类积分 $\\iint_\\Sigma \\mathbf{F}\\cdot d\\mathbf{S}=\\iint_\\Sigma \\mathbf{F}\\cdot\\mathbf{n}\\,dS=\\iint_\\Sigma(P\\cos\\alpha+Q\\cos\\beta+R\\cos\\gamma)\\,dS$，将带方向的积分拆为无方向的第一类积分，方向信息藏在方向余弦里。',
      },
      // 题 6：match — 定向约定
      {
        id: 'c24-b-q6',
        type: 'match',
        prompt: '将曲面类型与常用定向约定配对。',
        left: [
          '封闭曲面（如球面）',
          '非封闭曲面 $z=z(x,y)$，法向朝上',
          '非封闭曲面 $z=z(x,y)$，法向朝下',
        ],
        right: [
          '外法向，$dx\\,dy$ 在上半部分取正、下半部分取负',
          '上侧，$\\cos\\gamma>0$，$dx\\,dy>0$',
          '下侧，$\\cos\\gamma<0$，$dx\\,dy<0$',
        ],
      },
      // 题 7：input — 平面曲面第二类
      {
        id: 'c24-b-q7',
        type: 'input',
        prompt:
          '曲面 $\\Sigma: z=0$，$0\\leq x\\leq 1$，$0\\leq y\\leq 1$，取上侧（法向朝上）。$\\iint_\\Sigma (x+y+2)\\,dx\\,dy$ 等于多少？',
        accept: ['3', '3.0'],
        placeholder: '输入数值',
        explain:
          '上侧取正号，$z=0$ 代入：$\\iint_D(x+y+2)\\,dx\\,dy=\\int_0^1\\int_0^1(x+y+2)\\,dx\\,dy=\\int_0^1[\\frac{x^2}{2}+xy+2x]_0^1\\,dy=\\int_0^1(\\frac{1}{2}+y+2)\\,dy=[\\frac{5y}{2}+\\frac{y^2}{2}]_0^1=\\frac{5}{2}+\\frac{1}{2}=3$。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c24-c  Gauss 散度定理与 Stokes 旋度定理
  // ─────────────────────────────────────────
  {
    id: 'c24-c',
    title: 'Gauss 定理与 Stokes 定理',
    subtitle: '散度·Gauss 定理·旋度·Stokes 定理·与 Green 公式的统一',
    intro: [
      // 卡 1：散度的物理直觉
      {
        title: '散度：衡量"源"的强弱',
        body:
          '考虑流体速度场 $\\mathbf{F}=(P,Q,R)$。在某点附近取一个极小的封闭区域，若单位时间流出的流体净流量大于流入，说明该点是"源"；净流量小于零则是"汇"。\n\n' +
          '**散度**正是量化这一"源强度"的指标：\n\n' +
          '$\\text{div}\\,\\mathbf{F} = \\nabla\\cdot\\mathbf{F} = \\dfrac{\\partial P}{\\partial x}+\\dfrac{\\partial Q}{\\partial y}+\\dfrac{\\partial R}{\\partial z}$\n\n' +
          '$\\nabla\\cdot\\mathbf{F}>0$：该点是源（流出多）；$\\nabla\\cdot\\mathbf{F}<0$：该点是汇（流入多）；$\\nabla\\cdot\\mathbf{F}=0$：无源（无散场，如磁场）。\n\n' +
          '**例**：$\\mathbf{F}=(x,y,z)$，$\\nabla\\cdot\\mathbf{F}=1+1+1=3$，处处是源。$\\mathbf{F}=(-x,-y,-z)$，散度为 $-3$，处处是汇。',
        formula: '\\nabla\\cdot\\mathbf{F} = \\frac{\\partial P}{\\partial x}+\\frac{\\partial Q}{\\partial y}+\\frac{\\partial R}{\\partial z}',
        tip: '速记：$\\nabla=(\\partial_x,\\partial_y,\\partial_z)$ 是"梯度算子"，散度 $\\nabla\\cdot\\mathbf{F}$ 就是形式内积——对各分量各自求偏导再加起来。',
      },
      // 卡 2：Gauss 定理陈述
      {
        title: 'Gauss 定理（散度定理）',
        body:
          '**Gauss 定理**把封闭曲面的通量与内部散度联系起来：\n\n' +
          '设空间区域 $\\Omega$ 由分片光滑的封闭曲面 $\\Sigma$（取外侧）围成，$P,Q,R$ 在 $\\Omega$ 上有连续一阶偏导数，则\n\n' +
          '$\\oiint_\\Sigma P\\,dy\\,dz+Q\\,dz\\,dx+R\\,dx\\,dy = \\iiint_\\Omega \\left(P_x+Q_y+R_z\\right)dV$\n\n' +
          '**白话**：流出封闭曲面的总通量 $=$ 曲面内部所有"源"产生的总量。\n\n' +
          '**为什么有用？** 直接计算封闭曲面通量需要分多个面逐一积分，但散度往往一句话算出来，体积分比面积分简单得多。',
        formula: '\\oiint_\\Sigma \\mathbf{F}\\cdot d\\mathbf{S} = \\iiint_\\Omega \\nabla\\cdot\\mathbf{F}\\,dV',
        tip: '使用条件：① 曲面必须是封闭的；② 取外法向；③ 被积函数在围成的区域内偏导连续。',
      },
      // 卡 3：Gauss 定理应用例题
      {
        title: '例题：Gauss 定理计算球面通量',
        body:
          '计算 $\\oiint_\\Sigma x^3\\,dy\\,dz+y^3\\,dz\\,dx+z^3\\,dx\\,dy$，$\\Sigma$ 为球面 $x^2+y^2+z^2=R^2$ 外侧。',
        steps: [
          '第一步：识别信号。封闭球面 + 通量 → 优先 Gauss 定理。',
          '第二步：计算散度。$\\nabla\\cdot\\mathbf{F}=\\partial_x(x^3)+\\partial_y(y^3)+\\partial_z(z^3)=3x^2+3y^2+3z^2=3(x^2+y^2+z^2)$。',
          '第三步：体积分。$\\oiint_\\Sigma = \\iiint_{\\Omega}3(x^2+y^2+z^2)\\,dV$，$\\Omega$ 是半径 $R$ 的球体。',
          '第四步：球坐标。$=3\\int_0^{2\\pi}d\\theta\\int_0^\\pi\\sin\\varphi\\,d\\varphi\\int_0^R \\rho^2\\cdot\\rho^2\\,d\\rho=3\\cdot 2\\pi\\cdot 2\\cdot\\dfrac{R^5}{5}=\\dfrac{12\\pi R^5}{5}$。',
        ],
        tip: '关键：散度 $3(x^2+y^2+z^2)$ 在球坐标下就是 $3\\rho^2$，与 $dV=\\rho^2\\sin\\varphi\\,d\\rho\\,d\\varphi\\,d\\theta$ 合并后恰好出现 $\\rho^4$，积分是 $\\int_0^R\\rho^4\\,d\\rho=R^5/5$。',
      },
      // 卡 4：补盖法——不封闭曲面用 Gauss
      {
        title: '补盖法：不封闭曲面借用 Gauss',
        body:
          '若曲面 $\\Sigma$ 不封闭，不能直接用 Gauss 定理。但可以补充一块"盖子" $\\Sigma_0$，使 $\\Sigma + \\Sigma_0$ 构成封闭曲面，再用 Gauss：\n\n' +
          '$\\oiint_{\\Sigma+\\Sigma_0}\\mathbf{F}\\cdot d\\mathbf{S} = \\iiint_\\Omega \\nabla\\cdot\\mathbf{F}\\,dV$\n\n' +
          '所以：$\\iint_\\Sigma = \\iiint_\\Omega\\nabla\\cdot\\mathbf{F}\\,dV - \\iint_{\\Sigma_0}$\n\n' +
          '盖子通常选平坦的坐标平面截面，计算最简单。\n\n' +
          '**步骤**：① 补盖子 $\\Sigma_0$（注意方向，封闭曲面取统一外法向）；② 用 Gauss 算体积分；③ 单独算盖子的面积分；④ 相减得 $\\Sigma$ 上的积分。',
        tip: '补盖时要注意方向一致性：整个封闭曲面取外法向，盖子的方向由此确定，不能随意取。',
      },
      // 卡 5：旋度的物理直觉
      {
        title: '旋度：衡量"旋转"的强弱',
        body:
          '想象把一个小轮子放入流场，若流场让轮子旋转，就说该点有"旋度"。旋度描述向量场绕某轴旋转的强度和方向。\n\n' +
          '**旋度**的定义（行列式形式）：\n\n' +
          '$\\nabla\\times\\mathbf{F}=\\begin{vmatrix}\\mathbf{i}&\\mathbf{j}&\\mathbf{k}\\\\\\partial_x&\\partial_y&\\partial_z\\\\P&Q&R\\end{vmatrix}=\\left(R_y-Q_z\\right)\\mathbf{i}+\\left(P_z-R_x\\right)\\mathbf{j}+\\left(Q_x-P_y\\right)\\mathbf{k}$\n\n' +
          '**关键性质**：若 $\\mathbf{F}=\\nabla\\varphi$（梯度场/保守场），则 $\\nabla\\times\\mathbf{F}=\\mathbf{0}$（无旋）。\n\n' +
          '**例**：$\\mathbf{F}=(-y,x,0)$ 描述绕 $z$ 轴旋转的流场，$\\nabla\\times\\mathbf{F}=(0,0,2)$，旋转轴为 $z$ 轴，强度为 $2$。',
        formula: '\\nabla\\times\\mathbf{F}=\\left(R_y-Q_z,\\;P_z-R_x,\\;Q_x-P_y\\right)',
        tip: '旋度的行列式写法是"神器"：一列是 $\\partial_x,\\partial_y,\\partial_z$，另一列是 $P,Q,R$，用行列式展开规则机械计算，不容易出错。',
      },
      // 卡 6：Stokes 定理陈述与右手法则
      {
        title: 'Stokes 定理：曲面旋度 = 边界环量',
        body:
          '**Stokes 定理**把曲面上旋度的积分与边界曲线上的线积分联系起来：\n\n' +
          '$\\iint_\\Sigma (\\nabla\\times\\mathbf{F})\\cdot d\\mathbf{S} = \\oint_{\\partial\\Sigma}\\mathbf{F}\\cdot d\\mathbf{r}$\n\n' +
          '其中 $\\partial\\Sigma$ 是曲面 $\\Sigma$ 的边界曲线，方向由**右手法则**确定：右手四指沿边界曲线方向弯曲，拇指指向曲面正法向。\n\n' +
          '**与 Green 公式的关系**：取 $\\Sigma$ 为 $xOy$ 面上的平面区域 $D$，法向朝上，边界逆时针，$R\\equiv 0$，Stokes 定理退化为 Green 公式 $\\iint_D(Q_x-P_y)\\,dx\\,dy=\\oint_{\\partial D}P\\,dx+Q\\,dy$。',
        formula: '\\iint_\\Sigma (\\nabla\\times\\mathbf{F})\\cdot d\\mathbf{S} = \\oint_{\\partial\\Sigma}\\mathbf{F}\\cdot d\\mathbf{r}',
        tip: '右手法则记忆：握住边界曲线，手指弯向曲线正方向，拇指所指是曲面正法向。方向弄错的话积分差一个负号。',
      },
      // 卡 7：Stokes 定理应用例题
      {
        title: '例题：Stokes 定理化线积分为面积分',
        body:
          '计算 $\\oint_C y\\,dx+z\\,dy+x\\,dz$，$C$ 是平面 $x+y+z=1$ 与三坐标平面围成三角形的边界，从 $z$ 轴正方向看逆时针。',
        steps: [
          '第一步：识别。有边界曲线 + 向量场 → Stokes 定理，把线积分化为曲面旋度积分。',
          '第二步：$P=y$，$Q=z$，$R=x$。计算旋度 $\\nabla\\times\\mathbf{F}=(R_y-Q_z,P_z-R_x,Q_x-P_y)=(0-1,0-1,0-1)=(-1,-1,-1)$。',
          '第三步：取曲面 $\\Sigma$ 为平面三角形（$x+y+z=1$，$x,y,z\\geq 0$），法向为 $\\mathbf{n}=\\dfrac{1}{\\sqrt{3}}(1,1,1)$（从 $z$ 轴正方向看逆时针决定上法向）。',
          '第四步：$(\\nabla\\times\\mathbf{F})\\cdot\\mathbf{n}=(-1,-1,-1)\\cdot\\dfrac{(1,1,1)}{\\sqrt{3}}=\\dfrac{-3}{\\sqrt{3}}=-\\sqrt{3}$。',
          '第五步：三角形面积。顶点 $(1,0,0),(0,1,0),(0,0,1)$，边长均 $\\sqrt{2}$，面积 $=\\dfrac{\\sqrt{3}}{2}$。',
          '结果：$\\oint_C = -\\sqrt{3}\\cdot\\dfrac{\\sqrt{3}}{2} = -\\dfrac{3}{2}$。',
        ],
        tip: 'Stokes 定理中，同一边界的不同曲面给出相同结果——选法：优先选平面区域（三角形、圆盘），避免参数复杂的曲面。',
      },
      // 卡 8：四大定理的统一与易错点 + reveal
      {
        title: '四大定理的统一与易错点',
        body:
          'Newton-Leibniz、Green、Gauss、Stokes 四个定理都是**"边界上低维积分 = 内部高一维微分积分"**这一思想的实例：\n\n' +
          '- Newton-Leibniz：$\\int_a^b f\'(x)\\,dx = f(b)-f(a)$（区间内导数积分 = 端点差）\n' +
          '- Green：$\\iint_D(Q_x-P_y)\\,dA = \\oint_{\\partial D}P\\,dx+Q\\,dy$（面内旋度 = 边界环量）\n' +
          '- Gauss：$\\iiint_\\Omega\\nabla\\cdot\\mathbf{F}\\,dV = \\oiint_{\\partial\\Omega}\\mathbf{F}\\cdot d\\mathbf{S}$（体内散度 = 边界通量）\n' +
          '- Stokes：$\\iint_\\Sigma(\\nabla\\times\\mathbf{F})\\cdot d\\mathbf{S} = \\oint_{\\partial\\Sigma}\\mathbf{F}\\cdot d\\mathbf{r}$（面内旋度 = 边界环量）\n\n' +
          '**常见易错**：Gauss 要求封闭曲面外法向；Stokes 边界方向由右手法则确定不能随意；旋度计算行列式展开时符号最容易出错，要慢慢按规则展开。',
        reveal: {
          q: '向量场 $\\mathbf{F}=(x,y,z)$，用 Gauss 定理计算它穿过半径为 $R$ 的球面外侧的通量。',
          a: '$\\nabla\\cdot\\mathbf{F}=1+1+1=3$，球体体积 $V=\\dfrac{4}{3}\\pi R^3$，故通量 $=\\iiint_\\Omega 3\\,dV = 3\\cdot\\dfrac{4}{3}\\pi R^3=4\\pi R^3$。也可直接验证：外法向 $\\mathbf{n}=(x,y,z)/R$，$\\mathbf{F}\\cdot\\mathbf{n}=R$，积分 $=R\\cdot 4\\pi R^2=4\\pi R^3$ 一致。',
        },
      },
    ],
    questions: [
      // 题 1：input — 散度计算
      {
        id: 'c24-c-q1',
        type: 'input',
        prompt:
          '计算向量场 $\\mathbf{F}=(x^2,y^2,z^2)$ 的散度 $\\nabla\\cdot\\mathbf{F}$ 在点 $(1,1,1)$ 处的值。',
        accept: ['6', '6.0'],
        placeholder: '输入数值',
        explain:
          '$\\nabla\\cdot\\mathbf{F}=\\partial_x(x^2)+\\partial_y(y^2)+\\partial_z(z^2)=2x+2y+2z$。在 $(1,1,1)$ 处值为 $2+2+2=6$。',
      },
      // 题 2：choice — Gauss 定理使用条件
      {
        id: 'c24-c-q2',
        type: 'choice',
        prompt: '下列哪项是使用 Gauss 定理的必要条件？',
        options: [
          '曲面是任意曲面（封闭或不封闭均可）',
          '曲面是封闭曲面，且取外法向',
          '被积函数必须是常数',
          '区域 $\\Omega$ 必须是球体',
        ],
        answer: 1,
        explain:
          'Gauss 定理要求：① 曲面是封闭的（围成一个有界区域）；② 取外法向；③ 被积函数 $P,Q,R$ 在围成区域内有连续一阶偏导数。区域形状没有限制。不封闭的曲面要先补盖子才能用 Gauss 定理。',
      },
      // 题 3：judge — 散度为零
      {
        id: 'c24-c-q3',
        type: 'judge',
        prompt:
          '若 $\\nabla\\cdot\\mathbf{F}=0$ 在区域 $\\Omega$ 内处处成立，则 $\\mathbf{F}$ 穿过围成 $\\Omega$ 的封闭曲面的净通量为零。',
        answer: true,
        explain:
          '由 Gauss 定理，$\\oiint_\\Sigma \\mathbf{F}\\cdot d\\mathbf{S}=\\iiint_\\Omega\\nabla\\cdot\\mathbf{F}\\,dV=\\iiint_\\Omega 0\\,dV=0$。这正是"无源场"的含义——流进多少，流出多少，净通量为零。',
      },
      // 题 4：input — Gauss 定理计算
      {
        id: 'c24-c-q4',
        type: 'input',
        prompt:
          '用 Gauss 定理计算 $\\oiint_\\Sigma (x^2+y)\\,dy\\,dz+(y^2+z)\\,dz\\,dx+(z^2+x)\\,dx\\,dy$，$\\Sigma$ 是单位立方体 $[0,1]^3$ 的外侧。结果等于多少？',
        accept: ['3', '3.0'],
        placeholder: '输入数值',
        explain:
          '$P=x^2+y$，$Q=y^2+z$，$R=z^2+x$。散度 $=2x+2y+2z$。$\\iiint_{[0,1]^3}2(x+y+z)\\,dV=2\\cdot 3\\cdot\\int_0^1 t\\,dt=6\\cdot\\frac{1}{2}=3$（三个方向对称）。',
      },
      // 题 5：choice — 旋度计算
      {
        id: 'c24-c-q5',
        type: 'choice',
        prompt:
          '$\\mathbf{F}=(y,-x,0)$ 的旋度 $\\nabla\\times\\mathbf{F}$ 等于哪项？',
        options: [
          '$(0,0,-2)$',
          '$(0,0,2)$',
          '$(-1,1,0)$',
          '$(0,0,0)$',
        ],
        answer: 0,
        explain:
          '$P=y$，$Q=-x$，$R=0$。$\\nabla\\times\\mathbf{F}=(R_y-Q_z,P_z-R_x,Q_x-P_y)=(0-0,0-0,-1-1)=(0,0,-2)$。该场描述顺时针旋转（从 $z$ 轴正方向看），旋度 $z$ 分量为负。',
      },
      // 题 6：judge — Stokes 与 Green 的关系
      {
        id: 'c24-c-q6',
        type: 'judge',
        prompt: 'Green 公式是 Stokes 定理在曲面退化为平面区域时的特殊情形。',
        answer: true,
        explain:
          '取 $\\Sigma$ 为 $xOy$ 面的有界区域 $D$，法向朝上（即 $(0,0,1)$），令 $R=0$，Stokes 定理中 $(\\nabla\\times\\mathbf{F})\\cdot\\mathbf{n}\\,dS=(Q_x-P_y)\\,dx\\,dy$，$\\oint_{\\partial\\Sigma}\\mathbf{F}\\cdot d\\mathbf{r}=\\oint_{\\partial D}P\\,dx+Q\\,dy$，正好就是 Green 公式。',
      },
      // 题 7：match — 定理应用场景
      {
        id: 'c24-c-q7',
        type: 'match',
        prompt: '将定理与其主要应用场景配对。',
        left: ['Gauss 定理', 'Stokes 定理', 'Green 公式'],
        right: [
          '封闭曲面通量转化为散度体积分',
          '曲面边界线积分转化为曲面旋度积分',
          '平面封闭曲线积分转化为二重积分',
        ],
      },
      // 题 8：input — Stokes 定理
      {
        id: 'c24-c-q8',
        type: 'input',
        prompt:
          '向量场 $\\mathbf{F}=(y,-x,0)$，用 Stokes 定理计算 $\\oint_C\\mathbf{F}\\cdot d\\mathbf{r}$，$C$ 是 $z=0$ 平面上单位圆逆时针方向。旋度为 $(0,0,-2)$，曲面法向朝上，面积为 $\\pi$，结果等于多少？',
        accept: ['-2pi', '-2\\pi', '-6.283', '-6.28'],
        placeholder: '如 -2pi',
        explain:
          'Stokes 定理：$\\oint_C\\mathbf{F}\\cdot d\\mathbf{r}=\\iint_\\Sigma(\\nabla\\times\\mathbf{F})\\cdot d\\mathbf{S}$。$(\\nabla\\times\\mathbf{F})\\cdot\\mathbf{n}=(0,0,-2)\\cdot(0,0,1)=-2$，积分 $=-2\\cdot\\pi\\cdot 1^2=-2\\pi$。该场是顺时针旋转，逆时针边界给出负通量，符合直觉。',
      },
    ],
  },
]
