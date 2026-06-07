export const LESSONS = [
  {
    id: 'u5c1',
    title: '弧度制与任意角',
    subtitle: '$180^\\circ = \\pi$ rad，弧长 $l = r\\theta$，扇形面积 $S = \\dfrac{1}{2}r^2\\theta$',
    intro: [
      {
        title: '引入：度量角为什么要用弧度？',
        body: '我们从小学的"度"是人为规定的——把一圈平分 360 份，每份 1°，与圆的几何结构毫无关联。\n\n**弧度制**则直接从弧长出发：当弧长恰好等于半径时，所对圆心角就是 1 rad。这样，角与弧长、半径之间自然形成简洁的比例关系 $l = r\\theta$，在微积分、物理学中极为重要。\n\n高中阶段的任务：掌握换算、记住常用弧度值、用弧度计算弧长和面积。',
        tip: '弧度制的好处：公式 $l = r\\theta$ 比度数制简洁得多，角量"纯数化"无量纲。',
      },
      {
        title: '任意角：正角、负角与零角',
        body: '初中角的范围是 $0°$ 到 $360°$。高中推广到**任意角**：\n\n- **正角**：逆时针旋转（与坐标系正方向一致）\n- **负角**：顺时针旋转\n- **零角**：旋转量为零\n\n于是 $390°$ 和 $-330°$ 都合法，且和 $30°$ 的**终边相同**。\n\n**象限角**：终边在第 $n$ 象限内称第 $n$ 象限角；终边恰在坐标轴上称**轴线角**，不属于任何象限。',
        tip: '轴线角（终边在坐标轴上的角）不属于任何象限——高考常考细节！',
      },
      {
        title: '弧度的定义与度弧换算',
        body: '**定义**：圆上弧长等于半径时，所对圆心角为 1 弧度（rad）。\n\n一整圈弧长 $= 2\\pi r$，对应圆心角 $360°$，故：\n$$360° = 2\\pi \\text{ rad} \\Rightarrow 180° = \\pi \\text{ rad}$$\n\n**实用换算**：\n- 度 → 弧度：乘以 $\\dfrac{\\pi}{180}$\n- 弧度 → 度：乘以 $\\dfrac{180}{\\pi}$\n\n**常用换算**：$30° = \\dfrac{\\pi}{6}$，$45° = \\dfrac{\\pi}{4}$，$60° = \\dfrac{\\pi}{3}$，$90° = \\dfrac{\\pi}{2}$，$180° = \\pi$，$360° = 2\\pi$。\n\n**记忆口诀**：分母对应"把 $180°$ 等分的份数"，$\\dfrac{\\pi}{6}$ 就是把 $180°$ 分成 6 份每份 $30°$。',
        formula: '180^\\circ = \\pi\\text{ rad},\\quad 1^\\circ = \\frac{\\pi}{180}\\text{ rad},\\quad 1\\text{ rad} = \\frac{180^\\circ}{\\pi} \\approx 57.3^\\circ',
        image: 'gz-p5-01-1.svg',
        imageCaption: '单位圆上弧度的几何意义：弧长等于半径时圆心角为 1 rad',
      },
      {
        title: '终边相同的角',
        body: '两个角终边相同，当且仅当它们相差整数圈（$2\\pi$ 的整数倍）。\n\n与角 $\\alpha$（弧度）终边相同的所有角构成集合：\n$$\\{\\alpha + 2k\\pi \\mid k \\in \\mathbb{Z}\\}$$\n\n**例**：与 $\\dfrac{\\pi}{3}$ 终边相同的角：$\\dfrac{\\pi}{3} + 2\\pi = \\dfrac{7\\pi}{3}$，$\\dfrac{\\pi}{3} - 2\\pi = -\\dfrac{5\\pi}{3}$，……\n\n**判断象限**：把角对 $2\\pi$ 取模，化到 $[0, 2\\pi)$ 内再判断。\n- 第一象限：$\\left(0, \\dfrac{\\pi}{2}\\right)$；第二象限：$\\left(\\dfrac{\\pi}{2}, \\pi\\right)$\n- 第三象限：$\\left(\\pi, \\dfrac{3\\pi}{2}\\right)$；第四象限：$\\left(\\dfrac{3\\pi}{2}, 2\\pi\\right)$',
        formula: '\\alpha + 2k\\pi,\\quad k \\in \\mathbb{Z}',
        tip: '不要漏写 $k \\in \\mathbb{Z}$，也不要只写正整数 $k$，负整数 $k$ 对应"回转"的情况同样合法。',
      },
      {
        title: '弧长公式与扇形面积',
        body: '由弧度定义直接得到：\n$$l = r\\theta$$\n（$r$ 为半径，$\\theta$ 为圆心角弧度数，$l$ 为弧长）\n\n**扇形面积**：扇形面积占整圆面积的比例等于圆心角占 $2\\pi$ 的比例：\n$$S_{\\text{扇}} = \\frac{\\theta}{2\\pi} \\cdot \\pi r^2 = \\frac{1}{2}r^2\\theta$$\n\n利用 $l = r\\theta$ 还可写成：\n$$S_{\\text{扇}} = \\frac{1}{2}rl$$\n\n**形象记忆**：$S = \\dfrac{1}{2}rl$ 类似三角形面积"底 $\\times$ 高 $\\div$ 2"，把弧想象成底边，半径想象成高。',
        formula: 'l = r\\theta,\\quad S_{\\text{扇}} = \\frac{1}{2}r^2\\theta = \\frac{1}{2}rl',
        steps: [
          '确认 $\\theta$ 是弧度（不是度数），若给度数先换算',
          '弧长：$l = r\\theta$',
          '面积：$S = \\dfrac{1}{2}r^2\\theta$ 或 $S = \\dfrac{1}{2}rl$（两种均可，互相验证）',
        ],
      },
      {
        title: '例题演练：度弧互换与扇形计算',
        body: '**例 1（度弧换算）**：将 $-210°$ 化为弧度；将 $\\dfrac{7\\pi}{6}$ 化为度。\n\n$$-210° = -210 \\times \\frac{\\pi}{180} = -\\frac{7\\pi}{6} \\text{ rad}$$\n$$\\frac{7\\pi}{6} \\text{ rad} = \\frac{7\\pi}{6} \\times \\frac{180°}{\\pi} = 210°$$\n\n**例 2（判断象限）**：$\\dfrac{25\\pi}{6} = 4\\pi + \\dfrac{\\pi}{6}$（整整两圈后余 $\\dfrac{\\pi}{6}$），终边与 $\\dfrac{\\pi}{6}$ 相同，在**第一象限**。\n\n**例 3（扇形）**：半径 $r = 6$ cm，圆心角 $\\theta = \\dfrac{2\\pi}{3}$ rad。\n$$l = 6 \\times \\frac{2\\pi}{3} = 4\\pi \\text{ cm},\\quad S = \\frac{1}{2} \\times 6 \\times 4\\pi = 12\\pi \\text{ cm}^2$$',
        tip: '弧长公式中 $\\theta$ **必须是弧度**，用度数代入会得到错误结果，这是高考常见失分点。',
      },
      {
        title: '易错点总结',
        body: '**易错 1**：换算方向弄反。记住：度 $\\to$ 弧度乘 $\\dfrac{\\pi}{180}$（变小）；弧度 $\\to$ 度乘 $\\dfrac{180}{\\pi}$（变大约 57.3 倍）。\n\n**易错 2**：终边相同的角漏写 $k \\in \\mathbb{Z}$，或误以为 $k$ 只能取正整数。\n\n**易错 3**：$l = r\\theta$ 中 $\\theta$ 用了度数，应先换算为弧度再代入。\n\n**易错 4**：轴线角（如 $0, \\dfrac{\\pi}{2}, \\pi, \\dfrac{3\\pi}{2}$ 等）不属于任何象限。',
        reveal: {
          q: '一个扇形弧长为 $6$ cm，面积为 $9$ cm²，求其半径和圆心角。',
          a: '由 $S = \\dfrac{1}{2}rl$：$9 = \\dfrac{1}{2} \\times r \\times 6$，解得 $r = 3$ cm。再由 $l = r\\theta$：$\\theta = \\dfrac{l}{r} = \\dfrac{6}{3} = 2$ rad。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '将 $135°$ 化为弧度，结果是（  ）。',
        options: [
          '$\\dfrac{2\\pi}{3}$',
          '$\\dfrac{3\\pi}{4}$',
          '$\\dfrac{5\\pi}{6}$',
          '$\\dfrac{4\\pi}{3}$',
        ],
        answer: 1,
        explain: '$135° = 135 \\times \\dfrac{\\pi}{180} = \\dfrac{135\\pi}{180} = \\dfrac{3\\pi}{4}$。',
      },
      {
        type: 'judge',
        prompt: '终边落在 $y$ 轴负半轴上的角属于第三象限角。',
        answer: false,
        explain: '终边在坐标轴上的角称为轴线角，不属于任何象限。$y$ 轴负半轴是坐标轴，所以该角不属于第三象限。',
      },
      {
        type: 'input',
        prompt: '与角 $\\dfrac{\\pi}{4}$ 终边相同的角的集合可表示为 $\\dfrac{\\pi}{4} + 2k\\pi$（$k \\in \\mathbb{Z}$）。角 $\\dfrac{41\\pi}{4}$ 的终边在第几象限？请填数字。',
        accept: ['1', '一'],
        explain: '$\\dfrac{41\\pi}{4} = 10\\pi + \\dfrac{\\pi}{4}$，$10\\pi = 5 \\times 2\\pi$（整整五圈），故终边与 $\\dfrac{\\pi}{4}$ 相同。$\\dfrac{\\pi}{4} \\in \\left(0, \\dfrac{\\pi}{2}\\right)$，位于第一象限。',
      },
      {
        type: 'choice',
        prompt: '一个扇形的半径为 $4$ cm，圆心角为 $\\dfrac{\\pi}{3}$ rad，则该扇形的弧长为（  ）。',
        options: [
          '$\\dfrac{4\\pi}{3}$ cm',
          '$\\dfrac{2\\pi}{3}$ cm',
          '$\\dfrac{\\pi}{3}$ cm',
          '$4\\pi$ cm',
        ],
        answer: 0,
        explain: '$l = r\\theta = 4 \\times \\dfrac{\\pi}{3} = \\dfrac{4\\pi}{3}$ cm。',
      },
      {
        type: 'choice',
        prompt: '一个扇形的弧长为 $2\\pi$ cm，半径为 $3$ cm，则扇形面积为（  ）。',
        options: [
          '$3\\pi$ cm²',
          '$\\pi$ cm²',
          '$6\\pi$ cm²',
          '$2\\pi$ cm²',
        ],
        answer: 0,
        explain: '$S = \\dfrac{1}{2}rl = \\dfrac{1}{2} \\times 3 \\times 2\\pi = 3\\pi$ cm²。',
      },
      {
        type: 'match',
        prompt: '将下列角度与其弧度值对应。',
        left: [
          '$120°$',
          '$270°$',
          '$-30°$',
        ],
        right: [
          '$\\dfrac{3\\pi}{2}$',
          '$-\\dfrac{\\pi}{6}$',
          '$\\dfrac{2\\pi}{3}$',
        ],
        explain: '$120° = \\dfrac{2\\pi}{3}$；$270° = \\dfrac{3\\pi}{2}$；$-30° = -\\dfrac{\\pi}{6}$。',
      },
      {
        type: 'judge',
        prompt: '$l = r\\theta$ 中，$\\theta$ 可以是角的度数（如 $60°$）直接代入。',
        answer: false,
        explain: '$l = r\\theta$ 中 $\\theta$ 必须是弧度数（纯数），不能直接代入度数。应先将 $60°$ 换算为 $\\dfrac{\\pi}{3}$ rad 再代入，否则结果错误。',
      },
      {
        type: 'input',
        prompt: '一个扇形的圆心角为 $2$ rad，弧长为 $6$ cm，则其面积为多少平方厘米？',
        accept: ['9'],
        explain: '由 $l = r\\theta$ 得 $r = \\dfrac{l}{\\theta} = \\dfrac{6}{2} = 3$ cm。面积 $S = \\dfrac{1}{2}r^2\\theta = \\dfrac{1}{2} \\times 9 \\times 2 = 9$ cm²。也可用 $S = \\dfrac{1}{2}rl = \\dfrac{1}{2} \\times 3 \\times 6 = 9$ cm²。',
      },
    ],
  },
]
