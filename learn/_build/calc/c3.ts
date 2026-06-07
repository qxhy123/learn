export const LESSONS = [
  // ─────────────────────────────────────────────────────────────────────────
  // 关 c3-a  弧度制与单位圆
  // ─────────────────────────────────────────────────────────────────────────
  {
    id: 'c3-a',
    title: '弧度制与单位圆',
    subtitle: '弧度换算·弧长·扇形面积·单位圆定义·参考角',
    intro: [
      // 卡 1：为什么微积分用弧度？
      {
        title: '为什么微积分只用弧度？',
        body:
          '在中学我们用"度"量角：圆是 360°，直角是 90°。但**微积分默认弧度制**，原因只有一个：弧度让极限公式变得最简洁。\n\n' +
          '最直接的体现是第一重要极限：当 $x$ 为弧度时 $\\lim_{x\\to 0}\\dfrac{\\sin x}{x}=1$，结果干净；若 $x$ 为角度，分母要乘上 $\\dfrac{\\pi}{180}$，一切导数公式都多出这个丑陋系数。\n\n' +
          '**结论**：进入微积分，角度没有写单位时，默认就是弧度，不要再换算成度。',
        tip: '弧度无量纲——它就是一个实数，这也是函数 $\\sin x$ 可以被视为"把实数映射到实数"的根本原因。',
      },
      // 卡 2：弧度的定义与换算
      {
        title: '弧度的定义：弧长除以半径',
        body:
          '在半径为 $r$ 的圆中，圆心角 $\\theta$ 对应弧长 $l$，定义弧度数为\n\n' +
          '$\\theta = \\dfrac{l}{r}$。\n\n' +
          '特别地，整圆弧长 $= 2\\pi r$，所以整圆 $= \\dfrac{2\\pi r}{r} = 2\\pi$ 弧度，即 $360° = 2\\pi$，由此推出\n\n' +
          '$180° = \\pi \\text{ rad}$，$1° = \\dfrac{\\pi}{180} \\text{ rad}$，$1 \\text{ rad} = \\dfrac{180°}{\\pi}$。\n\n' +
          '**常用角**（必须熟记）：$30° = \\dfrac{\\pi}{6}$，$45° = \\dfrac{\\pi}{4}$，$60° = \\dfrac{\\pi}{3}$，$90° = \\dfrac{\\pi}{2}$，$120° = \\dfrac{2\\pi}{3}$，$150° = \\dfrac{5\\pi}{6}$，$270° = \\dfrac{3\\pi}{2}$。',
        formula: '180^\\circ = \\pi,\\quad l = r\\theta,\\quad S = \\tfrac{1}{2}r^2\\theta',
        tip: '换算口诀：度 $\\to$ 弧度乘 $\\dfrac{\\pi}{180}$；弧度 $\\to$ 度乘 $\\dfrac{180}{\\pi}$。',
      },
      // 卡 3：弧长与扇形面积（含例题）
      {
        title: '弧长与扇形面积：两个公式一个注意点',
        body:
          '**弧长**：$l = r\\theta$，其中 $r$ 是半径，$\\theta$ 是圆心角（弧度！）。\n\n' +
          '**扇形面积**：$S = \\dfrac{1}{2}r^2\\theta = \\dfrac{1}{2}lr$。\n\n' +
          '**例**：半径 6，圆心角 $150° = \\dfrac{5\\pi}{6}$，则\n\n' +
          '弧长 $l = 6 \\times \\dfrac{5\\pi}{6} = 5\\pi$；\n\n' +
          '扇形面积 $S = \\dfrac{1}{2} \\times 36 \\times \\dfrac{5\\pi}{6} = 15\\pi$。\n\n' +
          '**易错点**：公式里的 $\\theta$ 必须是弧度数，如果题目给角度，要先换算，否则结果错误。',
        steps: [
          '角度换弧度：$150° \\times \\dfrac{\\pi}{180} = \\dfrac{5\\pi}{6}$',
          '代入弧长：$l = 6 \\times \\dfrac{5\\pi}{6} = 5\\pi$',
          '代入面积：$S = \\dfrac{1}{2} \\times 36 \\times \\dfrac{5\\pi}{6} = 15\\pi$',
        ],
      },
      // 卡 4：单位圆定义三角函数
      {
        title: '单位圆：定义比直角三角形更强',
        body:
          '以原点为圆心、半径为 1 的圆叫**单位圆**，方程为 $x^2 + y^2 = 1$。\n\n' +
          '从 $x$ 轴正方向出发，逆时针旋转角 $\\theta$，终边与单位圆交于点 $P(x, y)$，定义\n\n' +
          '$\\cos\\theta = x$（横坐标），$\\sin\\theta = y$（纵坐标），$\\tan\\theta = \\dfrac{y}{x}$（$x \\ne 0$）。\n\n' +
          '**为什么比直角三角形定义更强？** 直角三角形只能处理 $0° \\sim 90°$，单位圆让负角、钝角、超过一周的角都有定义——只要旋转一下就行。\n\n' +
          '**关键推论**：$\\sin^2\\theta + \\cos^2\\theta = x^2 + y^2 = 1$（来自单位圆方程，不需要另记）。',
        formula: '\\cos\\theta = x,\\quad \\sin\\theta = y,\\quad \\tan\\theta = \\frac{y}{x}\\;(x\\ne 0)',
        tip: '记法：$\\cos$ 对应"x 轴"，$\\sin$ 对应"y 轴"——$c$ 像"x"，$s$ 像"y"，顺序一致。',
      },
      // 卡 5：参考角与象限符号（ASTC）
      {
        title: '参考角与象限符号：ASTC 四字诀',
        body:
          '**参考角**：终边与 $x$ 轴所成的锐角（$0° \\sim 90°$），用来把任意角的三角值转化为特殊角值。\n\n' +
          '**象限符号记忆——ASTC**：第一象限 All（全正），第二象限 Sin（只有 $\\sin$ 为正），第三象限 Tan（只有 $\\tan$ 为正），第四象限 Cos（只有 $\\cos$ 为正）。\n\n' +
          '**标准四步**：① 化到 $[0, 2\\pi)$；② 找参考角；③ 查特殊角值；④ 用 ASTC 确定符号。\n\n' +
          '**例**：求 $\\sin\\dfrac{5\\pi}{4}$。$\\dfrac{5\\pi}{4} = \\pi + \\dfrac{\\pi}{4}$，第三象限，参考角 $\\dfrac{\\pi}{4}$，第三象限 $\\sin$ 为负，所以 $\\sin\\dfrac{5\\pi}{4} = -\\dfrac{\\sqrt{2}}{2}$。',
        tip: '英文助记：All Students Take Calculus（所有学生都学微积分）——这四个首字母刚好是 ASTC！',
      },
      // 卡 6：特殊角值表与余弦偶函数技巧
      {
        title: '特殊角值：三角形记忆法',
        body:
          '只需记三个特殊角的 $\\sin$ 值（$0$、$\\dfrac{\\pi}{6}$、$\\dfrac{\\pi}{4}$、$\\dfrac{\\pi}{3}$、$\\dfrac{\\pi}{2}$）：\n\n' +
          '$\\sin 0 = 0$，$\\sin\\dfrac{\\pi}{6} = \\dfrac{1}{2}$，$\\sin\\dfrac{\\pi}{4} = \\dfrac{\\sqrt{2}}{2}$，$\\sin\\dfrac{\\pi}{3} = \\dfrac{\\sqrt{3}}{2}$，$\\sin\\dfrac{\\pi}{2} = 1$。\n\n' +
          '**技巧**：把这五个值依次看作 $\\dfrac{\\sqrt{0}}{2}$、$\\dfrac{\\sqrt{1}}{2}$、$\\dfrac{\\sqrt{2}}{2}$、$\\dfrac{\\sqrt{3}}{2}$、$\\dfrac{\\sqrt{4}}{2}$，是有规律的递增序列。\n\n' +
          '$\\cos$ 值与 $\\sin$ 值"互换顺序"：$\\cos\\dfrac{\\pi}{6} = \\dfrac{\\sqrt{3}}{2}$，$\\cos\\dfrac{\\pi}{4} = \\dfrac{\\sqrt{2}}{2}$，$\\cos\\dfrac{\\pi}{3} = \\dfrac{1}{2}$。\n\n' +
          '**奇偶性实用技巧**：$\\cos(-x) = \\cos x$（偶函数），$\\sin(-x) = -\\sin x$（奇函数）——负角先去掉负号，再查表。',
        reveal: {
          q: '用四步法求 $\\cos\\left(-\\dfrac{2\\pi}{3}\\right)$。',
          a: '余弦是偶函数，$\\cos\\left(-\\dfrac{2\\pi}{3}\\right) = \\cos\\dfrac{2\\pi}{3}$。$\\dfrac{2\\pi}{3}$ 在第二象限，参考角 $= \\pi - \\dfrac{2\\pi}{3} = \\dfrac{\\pi}{3}$，第二象限 $\\cos$ 为负，所以 $\\cos\\dfrac{2\\pi}{3} = -\\cos\\dfrac{\\pi}{3} = -\\dfrac{1}{2}$。',
        },
      },
      // 卡 7：易错点汇总
      {
        title: '易错点汇总：三角求值的四大坑',
        body:
          '**坑 1——公式中 $\\theta$ 用角度**：$l = r\\theta$ 和 $S = \\dfrac{1}{2}r^2\\theta$ 里的 $\\theta$ 必须是弧度。用角度得出的答案会相差 $\\dfrac{\\pi}{180}$ 倍。\n\n' +
          '**坑 2——参考角算错**：第三象限的参考角是 $\\theta - \\pi$，而不是 $\\pi - \\theta$；第四象限是 $2\\pi - \\theta$。要先确认象限，再计算。\n\n' +
          '**坑 3——忽略奇偶性**：$\\cos(-x)$ 直接等于 $\\cos x$，不用画图，直接用偶函数性质化简。\n\n' +
          '**坑 4——混淆 $\\sin$ 和 $\\cos$ 的坐标对应**：$\\sin$ 是纵坐标，$\\cos$ 是横坐标。第二象限 $\\sin > 0$（纵坐标为正），$\\cos < 0$（横坐标为负）。',
      },
    ],
    questions: [
      // 题 1：choice — 角度换弧度
      {
        id: 'c3-a-q1',
        type: 'choice',
        prompt: '$270°$ 等于多少弧度？',
        options: [
          '$\\dfrac{\\pi}{2}$',
          '$\\pi$',
          '$\\dfrac{3\\pi}{2}$',
          '$2\\pi$',
        ],
        answer: 2,
        explain:
          '$270° \\times \\dfrac{\\pi}{180} = \\dfrac{270\\pi}{180} = \\dfrac{3\\pi}{2}$。整圆 $360° = 2\\pi$，四分之三圆就是 $\\dfrac{3\\pi}{2}$。',
      },
      // 题 2：input — 弧长计算
      {
        id: 'c3-a-q2',
        type: 'input',
        prompt: '半径为 $4$，圆心角为 $\\dfrac{\\pi}{3}$ 弧度的弧长是多少？',
        accept: ['4π/3', '4\\pi/3', '(4/3)π', '4.19', '4.188'],
        placeholder: '用分数或小数',
        explain:
          '$l = r\\theta = 4 \\times \\dfrac{\\pi}{3} = \\dfrac{4\\pi}{3}$。弧长公式 $l = r\\theta$，$\\theta$ 已是弧度，直接代入。',
      },
      // 题 3：judge — 单位圆定义
      {
        id: 'c3-a-q3',
        type: 'judge',
        prompt:
          '在单位圆中，角 $\\theta$ 对应点的横坐标等于 $\\sin\\theta$，纵坐标等于 $\\cos\\theta$。',
        answer: false,
        explain:
          '单位圆的定义恰好相反：**横坐标** $= \\cos\\theta$，**纵坐标** $= \\sin\\theta$。记法：$\\sin$ 和"纵"（y 轴）对应，$\\cos$ 和"横"（x 轴）对应。',
      },
      // 题 4：choice — ASTC 象限符号
      {
        id: 'c3-a-q4',
        type: 'choice',
        prompt: '角 $\\dfrac{7\\pi}{6}$ 在哪个象限？$\\sin\\dfrac{7\\pi}{6}$ 的符号是？',
        options: [
          '第二象限，正',
          '第三象限，负',
          '第三象限，正',
          '第四象限，负',
        ],
        answer: 1,
        explain:
          '$\\dfrac{7\\pi}{6} = \\pi + \\dfrac{\\pi}{6}$，超过 $\\pi$ 不足 $\\dfrac{3\\pi}{2}$，在第三象限。第三象限只有 $\\tan$ 为正，$\\sin$ 为负，故 $\\sin\\dfrac{7\\pi}{6} = -\\dfrac{1}{2}$。',
      },
      // 题 5：choice — 特殊角值
      {
        id: 'c3-a-q5',
        type: 'choice',
        prompt: '$\\cos\\dfrac{5\\pi}{3}$ 等于多少？',
        options: [
          '$-\\dfrac{1}{2}$',
          '$\\dfrac{\\sqrt{3}}{2}$',
          '$\\dfrac{1}{2}$',
          '$-\\dfrac{\\sqrt{3}}{2}$',
        ],
        answer: 2,
        explain:
          '$\\dfrac{5\\pi}{3} = 2\\pi - \\dfrac{\\pi}{3}$，第四象限，参考角 $\\dfrac{\\pi}{3}$，第四象限 $\\cos$ 为正，所以 $\\cos\\dfrac{5\\pi}{3} = +\\cos\\dfrac{\\pi}{3} = \\dfrac{1}{2}$。',
      },
      // 题 6：input — 扇形面积
      {
        id: 'c3-a-q6',
        type: 'input',
        prompt: '半径为 $3$，圆心角为 $\\dfrac{2\\pi}{3}$ 的扇形面积是多少（以 $\\pi$ 表示，如 $3\\pi$）？',
        accept: ['3π', '3\\pi'],
        placeholder: '形如 3π',
        explain:
          '$S = \\dfrac{1}{2}r^2\\theta = \\dfrac{1}{2} \\times 9 \\times \\dfrac{2\\pi}{3} = \\dfrac{9\\pi}{3} = 3\\pi$。',
      },
      // 题 7：match — 象限与符号
      {
        id: 'c3-a-q7',
        type: 'match',
        prompt: '将象限与其三角函数符号特征对应（ASTC 规则）。',
        left: ['第一象限', '第二象限', '第三象限', '第四象限'],
        right: [
          '$\\sin>0$，$\\cos>0$，$\\tan>0$',
          '$\\sin>0$，$\\cos<0$，$\\tan<0$',
          '$\\sin<0$，$\\cos<0$，$\\tan>0$',
          '$\\sin<0$，$\\cos>0$，$\\tan<0$',
        ],
      },
      // 题 8：judge — 余弦偶函数
      {
        id: 'c3-a-q8',
        type: 'judge',
        prompt: '$\\cos\\left(-\\dfrac{\\pi}{3}\\right) = -\\dfrac{1}{2}$。',
        answer: false,
        explain:
          '$\\cos$ 是偶函数，$\\cos(-x) = \\cos x$，所以 $\\cos\\left(-\\dfrac{\\pi}{3}\\right) = \\cos\\dfrac{\\pi}{3} = \\dfrac{1}{2}$，而非 $-\\dfrac{1}{2}$。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────────────
  // 关 c3-b  三角恒等式与倍角公式
  // ─────────────────────────────────────────────────────────────────────────
  {
    id: 'c3-b',
    title: '三角恒等式与倍角公式',
    subtitle: '平方关系·和差公式·倍角·半角·恒等式证明策略',
    intro: [
      // 卡 1：核心恒等式来自单位圆
      {
        title: '最重要的恒等式：来自单位圆方程',
        body:
          '单位圆上的点满足 $x^2 + y^2 = 1$，把 $x = \\cos\\theta$、$y = \\sin\\theta$ 代入，立刻得到\n\n' +
          '$\\sin^2\\theta + \\cos^2\\theta = 1$。\n\n' +
          '这条恒等式是**所有三角恒等式的根基**，不需要单独记忆，只要记住它来自单位圆方程就永远不会忘。\n\n' +
          '两边分别除以 $\\cos^2\\theta$ 和 $\\sin^2\\theta$，可以推出两个派生公式：\n\n' +
          '$1 + \\tan^2 x = \\sec^2 x$（分母 $\\cos^2 x$），$1 + \\cot^2 x = \\csc^2 x$（分母 $\\sin^2 x$）。\n\n' +
          '**实用技巧**：见到 $\\sin^2 x$ 或 $\\cos^2 x$，立刻想到把其中一个替换成 $1 - (\\ )^2$，这是化简和积分里最频繁的操作。',
        formula: '\\sin^2 x + \\cos^2 x = 1,\\quad 1 + \\tan^2 x = \\sec^2 x',
      },
      // 卡 2：和差角公式的直觉
      {
        title: '和差角公式：旋转矩阵的代数版本',
        body:
          '把单位向量旋转 $\\alpha$ 再旋转 $\\beta$，等价于直接旋转 $\\alpha + \\beta$。从旋转矩阵乘法推导出：\n\n' +
          '$\\sin(\\alpha \\pm \\beta) = \\sin\\alpha\\cos\\beta \\pm \\cos\\alpha\\sin\\beta$，\n\n' +
          '$\\cos(\\alpha \\pm \\beta) = \\cos\\alpha\\cos\\beta \\mp \\sin\\alpha\\sin\\beta$。\n\n' +
          '**记忆方法**：$\\sin$ 和差是"交叉相乘再加减"，$\\cos$ 和差是"相同相乘再减加"（注意符号与 $\\pm$ 相反）。\n\n' +
          '**例**：用 $75° = 45° + 30°$ 算 $\\cos 75°$：\n\n' +
          '$\\cos75° = \\cos45°\\cos30° - \\sin45°\\sin30° = \\dfrac{\\sqrt{2}}{2}\\cdot\\dfrac{\\sqrt{3}}{2} - \\dfrac{\\sqrt{2}}{2}\\cdot\\dfrac{1}{2} = \\dfrac{\\sqrt{6}-\\sqrt{2}}{4}$。',
        formula:
          '\\sin(\\alpha+\\beta)=\\sin\\alpha\\cos\\beta+\\cos\\alpha\\sin\\beta,\\quad\\cos(\\alpha+\\beta)=\\cos\\alpha\\cos\\beta-\\sin\\alpha\\sin\\beta',
        tip: '$\\tan(\\alpha+\\beta) = \\dfrac{\\tan\\alpha+\\tan\\beta}{1-\\tan\\alpha\\tan\\beta}$（分母有减号，与分子相反）。',
      },
      // 卡 3：倍角公式——来自和差公式令 β=α
      {
        title: '倍角公式：令 $\\beta = \\alpha$ 立刻推出',
        body:
          '在和差公式里令 $\\beta = \\alpha$，得到倍角公式：\n\n' +
          '$\\sin 2\\alpha = 2\\sin\\alpha\\cos\\alpha$（正弦倍角）。\n\n' +
          '$\\cos 2\\alpha = \\cos^2\\alpha - \\sin^2\\alpha$（余弦倍角——基础形式），利用 $\\sin^2\\alpha + \\cos^2\\alpha = 1$ 还可以变出另外两种形式：\n\n' +
          '$\\cos 2\\alpha = 2\\cos^2\\alpha - 1$（消去 $\\sin^2$），$\\cos 2\\alpha = 1 - 2\\sin^2\\alpha$（消去 $\\cos^2$）。\n\n' +
          '**积分中的选择**：$\\cos 2\\alpha$ 有三种写法。积分 $\\int\\cos^2 x\\,dx$ 用 $\\cos^2 x = \\dfrac{1+\\cos 2x}{2}$（由 $\\cos 2x = 2\\cos^2 x - 1$ 变形），把平方消掉才能积出来。',
        formula:
          '\\sin 2\\alpha = 2\\sin\\alpha\\cos\\alpha,\\quad \\cos 2\\alpha = 2\\cos^2\\alpha - 1 = 1 - 2\\sin^2\\alpha',
        steps: [
          '和差公式令 $\\beta = \\alpha$：$\\sin(\\alpha+\\alpha) = \\sin\\alpha\\cos\\alpha + \\cos\\alpha\\sin\\alpha = 2\\sin\\alpha\\cos\\alpha$',
          '同理 $\\cos 2\\alpha = \\cos^2\\alpha - \\sin^2\\alpha$',
          '用 $\\sin^2 = 1 - \\cos^2$ 代入：$= \\cos^2\\alpha - (1 - \\cos^2\\alpha) = 2\\cos^2\\alpha - 1$',
          '用 $\\cos^2 = 1 - \\sin^2$ 代入：$= (1-\\sin^2\\alpha) - \\sin^2\\alpha = 1 - 2\\sin^2\\alpha$',
        ],
      },
      // 卡 4：半角公式——由倍角变形
      {
        title: '半角公式：把 $\\alpha$ 换成 $\\dfrac{\\alpha}{2}$',
        body:
          '从倍角公式 $\\cos 2x = 1 - 2\\sin^2 x$，令 $x = \\dfrac{\\alpha}{2}$，得\n\n' +
          '$\\cos\\alpha = 1 - 2\\sin^2\\dfrac{\\alpha}{2}$，解出 $\\sin^2\\dfrac{\\alpha}{2} = \\dfrac{1-\\cos\\alpha}{2}$，开根号：\n\n' +
          '$\\sin\\dfrac{\\alpha}{2} = \\pm\\sqrt{\\dfrac{1-\\cos\\alpha}{2}}$。\n\n' +
          '同理从 $\\cos\\alpha = 2\\cos^2\\dfrac{\\alpha}{2} - 1$：$\\cos\\dfrac{\\alpha}{2} = \\pm\\sqrt{\\dfrac{1+\\cos\\alpha}{2}}$。\n\n' +
          '正负号由 $\\dfrac{\\alpha}{2}$ 所在象限决定——**用之前先判断象限！**\n\n' +
          '实用公式（无须判断符号）：$\\tan\\dfrac{\\alpha}{2} = \\dfrac{\\sin\\alpha}{1+\\cos\\alpha} = \\dfrac{1-\\cos\\alpha}{\\sin\\alpha}$。',
        formula:
          '\\sin\\dfrac{\\alpha}{2}=\\pm\\sqrt{\\dfrac{1-\\cos\\alpha}{2}},\\quad\\cos\\dfrac{\\alpha}{2}=\\pm\\sqrt{\\dfrac{1+\\cos\\alpha}{2}}',
        tip: '半角公式在微积分里最常见的用途：把 $\\sin^2$ 或 $\\cos^2$ 降次以便积分。',
      },
      // 卡 5：恒等式证明策略
      {
        title: '恒等式证明：5 步策略',
        body:
          '**策略 1**：从**复杂**一边出发，化向简单一边（不要两边同时动）。\n\n' +
          '**策略 2**：把所有三角函数化成 $\\sin$ 和 $\\cos$（$\\tan$、$\\sec$、$\\csc$、$\\cot$ 都是它们的商或倒数）。\n\n' +
          '**策略 3**：遇到 $\\sin^2 x$ 或 $\\cos^2 x$，想 $\\sin^2+\\cos^2=1$ 替换消去平方项。\n\n' +
          '**策略 4**：遇到 $2x$，考虑倍角分解；遇到 $\\dfrac{x}{2}$，考虑半角合并。\n\n' +
          '**策略 5**：写清楚定义域——在哪些 $x$ 处分母为零，要排除那些点。\n\n' +
          '**例**：证明 $\\dfrac{1-\\cos 2x}{\\sin 2x} = \\tan x$（$\\sin 2x \\ne 0$ 且 $\\cos x \\ne 0$ 时）：\n\n' +
          '分子 $1-\\cos 2x = 1-(1-2\\sin^2 x) = 2\\sin^2 x$；分母 $\\sin 2x = 2\\sin x\\cos x$，相除得 $\\dfrac{2\\sin^2 x}{2\\sin x\\cos x} = \\dfrac{\\sin x}{\\cos x} = \\tan x$。',
      },
      // 卡 6：诱导公式与奇变偶不变
      {
        title: '诱导公式：奇变偶不变，符号看象限',
        body:
          '把角写成 $k \\cdot \\dfrac{\\pi}{2} \\pm x$ 的形式，根据 $k$ 的奇偶性决定是否换名称，再用象限定符号。\n\n' +
          '**$k$ 为偶数（与 $\\pi$ 相关）**：函数名不变，符号看象限。\n\n' +
          '$\\sin(\\pi - x) = \\sin x$（第二象限，$\\sin$ 正），$\\cos(\\pi - x) = -\\cos x$（$\\cos$ 负）。\n\n' +
          '**$k$ 为奇数（与 $\\dfrac{\\pi}{2}$ 相关）**：$\\sin \\leftrightarrow \\cos$ 互换，再定符号。\n\n' +
          '$\\sin\\left(\\dfrac{\\pi}{2} - x\\right) = \\cos x$，$\\cos\\left(\\dfrac{\\pi}{2} - x\\right) = \\sin x$（这两个是"余角"关系）。\n\n' +
          '**实用举例**：$\\cos\\left(\\dfrac{\\pi}{2} + x\\right) = -\\sin x$（$k=1$ 奇，换名称；再看 $\\dfrac{\\pi}{2}+x$ 在第二象限，$\\cos$ 负）。',
        formula:
          '\\sin(\\pi - x) = \\sin x,\\quad \\cos(\\pi - x) = -\\cos x,\\quad \\sin\\!\\left(\\tfrac{\\pi}{2}-x\\right)=\\cos x',
        reveal: {
          q: '不查表，直接说出 $\\sin\\left(\\dfrac{\\pi}{2} + x\\right)$ 的结果。',
          a: '$k=1$（奇），函数名互换：$\\sin \\to \\cos$；$\\dfrac{\\pi}{2}+x$ 在第二象限，$\\sin$ 值为正，但我们换成了 $\\cos$，所以原式 $= \\cos x$。（另外验证：$\\sin\\left(\\dfrac{\\pi}{2}+x\\right)=\\sin\\dfrac{\\pi}{2}\\cos x+\\cos\\dfrac{\\pi}{2}\\sin x = \\cos x$ ✓。）',
        },
      },
      // 卡 7：综合易错点
      {
        title: '恒等式易错点：三个高频坑',
        body:
          '**坑 1——$\\cos 2x$ 三种形式混淆**：\n\n' +
          '$\\cos 2x = \\cos^2 x - \\sin^2 x = 2\\cos^2 x - 1 = 1 - 2\\sin^2 x$。\n\n' +
          '积分 $\\int \\sin^2 x\\,dx$ 用 $\\sin^2 x = \\dfrac{1-\\cos 2x}{2}$（降次）；积分 $\\int \\cos^2 x\\,dx$ 用 $\\cos^2 x = \\dfrac{1+\\cos 2x}{2}$。\n\n' +
          '**坑 2——$\\sin 2x$ 误写成 $2\\sin x$**：$\\sin 2x = 2\\sin x \\cos x$，漏掉 $\\cos x$ 因子。\n\n' +
          '**坑 3——证明恒等式时两边同时操作**：标准方法是变换**一边**，不能同时乘以同一个量（如果该量可能为零，会引入非法步骤）。',
      },
    ],
    questions: [
      // 题 1：input — 平方恒等式
      {
        id: 'c3-b-q1',
        type: 'input',
        prompt:
          '若 $\\sin\\theta = \\dfrac{3}{5}$，且 $\\theta$ 在第二象限，求 $\\cos\\theta$（填分数，如 -4/5）。',
        accept: ['-4/5', '-0.8'],
        placeholder: '填分数',
        explain:
          '$\\cos^2\\theta = 1 - \\sin^2\\theta = 1 - \\dfrac{9}{25} = \\dfrac{16}{25}$。第二象限 $\\cos < 0$，故 $\\cos\\theta = -\\dfrac{4}{5}$。',
      },
      // 题 2：choice — 倍角公式
      {
        id: 'c3-b-q2',
        type: 'choice',
        prompt: '$\\sin\\left(2 \\times \\dfrac{\\pi}{6}\\right)$ 等于多少？',
        options: [
          '$\\dfrac{1}{2}$',
          '$\\dfrac{\\sqrt{3}}{2}$',
          '$\\dfrac{\\sqrt{3}}{4}$',
          '$1$',
        ],
        answer: 1,
        explain:
          '$\\sin\\dfrac{\\pi}{3} = \\dfrac{\\sqrt{3}}{2}$。也可以用倍角公式验证：$\\sin 2\\cdot\\dfrac{\\pi}{6} = 2\\sin\\dfrac{\\pi}{6}\\cos\\dfrac{\\pi}{6} = 2\\cdot\\dfrac{1}{2}\\cdot\\dfrac{\\sqrt{3}}{2} = \\dfrac{\\sqrt{3}}{2}$。',
      },
      // 题 3：judge — cos 2x 与 2cos x 区别
      {
        id: 'c3-b-q3',
        type: 'judge',
        prompt: '$\\cos 2x = 2\\cos x$。',
        answer: false,
        explain:
          '$\\cos 2x = 2\\cos^2 x - 1$（或 $1 - 2\\sin^2 x$ 或 $\\cos^2 x - \\sin^2 x$），不等于 $2\\cos x$。这是倍角与倍数的混淆——自变量翻倍，函数值并不简单地翻倍。',
      },
      // 题 4：choice — 和差公式求精确值
      {
        id: 'c3-b-q4',
        type: 'choice',
        prompt: '$\\sin 75°$ 的精确值是？（提示：$75° = 45° + 30°$）',
        options: [
          '$\\dfrac{\\sqrt{6}+\\sqrt{2}}{4}$',
          '$\\dfrac{\\sqrt{6}-\\sqrt{2}}{4}$',
          '$\\dfrac{\\sqrt{3}+1}{4}$',
          '$\\dfrac{\\sqrt{3}}{2}$',
        ],
        answer: 0,
        explain:
          '$\\sin75° = \\sin(45°+30°) = \\sin45°\\cos30° + \\cos45°\\sin30°$\n' +
          '$= \\dfrac{\\sqrt{2}}{2}\\cdot\\dfrac{\\sqrt{3}}{2} + \\dfrac{\\sqrt{2}}{2}\\cdot\\dfrac{1}{2} = \\dfrac{\\sqrt{6}}{4} + \\dfrac{\\sqrt{2}}{4} = \\dfrac{\\sqrt{6}+\\sqrt{2}}{4}$。',
      },
      // 题 5：input — sin 2α 计算
      {
        id: 'c3-b-q5',
        type: 'input',
        prompt:
          '已知 $\\sin\\alpha = \\dfrac{3}{5}$，$\\alpha$ 在第二象限，$\\cos\\alpha = -\\dfrac{4}{5}$，求 $\\sin 2\\alpha$（填分数，如 -24/25）。',
        accept: ['-24/25', '-0.96'],
        placeholder: '填分数',
        explain:
          '$\\sin 2\\alpha = 2\\sin\\alpha\\cos\\alpha = 2 \\times \\dfrac{3}{5} \\times \\left(-\\dfrac{4}{5}\\right) = -\\dfrac{24}{25}$。倍角公式：$\\sin 2\\alpha = 2\\sin\\alpha\\cos\\alpha$，代入计算即可。',
      },
      // 题 6：match — 恒等式与来源
      {
        id: 'c3-b-q6',
        type: 'match',
        prompt: '将三角恒等式与其最直接的推导来源对应。',
        left: [
          '$\\sin^2 x + \\cos^2 x = 1$',
          '$\\sin 2x = 2\\sin x\\cos x$',
          '$\\cos^2 x = \\dfrac{1+\\cos 2x}{2}$',
        ],
        right: [
          '单位圆方程 $x^2+y^2=1$',
          '和差公式令 $\\beta=\\alpha$',
          '$\\cos 2x = 2\\cos^2 x-1$ 变形',
        ],
      },
      // 题 7：judge — 诱导公式
      {
        id: 'c3-b-q7',
        type: 'judge',
        prompt: '$\\sin(\\pi + x) = \\sin x$。',
        answer: false,
        explain:
          '$\\pi + x$ 在第三象限，$\\sin$ 为负；$k=2$（偶数），函数名不变，符号取负，故 $\\sin(\\pi + x) = -\\sin x$，而非 $\\sin x$。"奇变偶不变"中"偶"指函数名不变，但符号仍由象限决定。',
      },
      // 题 8：choice — 降次公式选择
      {
        id: 'c3-b-q8',
        type: 'choice',
        prompt:
          '计算积分 $\\int\\sin^2 x\\,dx$ 时，应当把 $\\sin^2 x$ 替换成哪个等价形式（降次）？',
        options: [
          '$1 - \\cos^2 x$',
          '$\\dfrac{1 - \\cos 2x}{2}$',
          '$\\dfrac{1 + \\cos 2x}{2}$',
          '$\\sin x\\cos x$',
        ],
        answer: 1,
        explain:
          '由倍角公式 $\\cos 2x = 1 - 2\\sin^2 x$，变形得 $\\sin^2 x = \\dfrac{1-\\cos 2x}{2}$。代入后积分变为 $\\int\\dfrac{1-\\cos 2x}{2}\\,dx$，可以直接积出。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────────────
  // 关 c3-c  反三角函数与主值区间
  // ─────────────────────────────────────────────────────────────────────────
  {
    id: 'c3-c',
    title: '反三角函数与主值区间',
    subtitle: '$\\arcsin$·$\\arccos$·$\\arctan$·主值折回·复合求值·微积分连接',
    intro: [
      // 卡 1：为什么需要限制主值区间？
      {
        title: '为什么三角函数没有"全局反函数"？',
        body:
          '函数有反函数的前提是：它在定义域上是**一一映射**（不同输入对应不同输出）。\n\n' +
          '$\\sin x$ 的值域只有 $[-1, 1]$，但同一个值（比如 $\\dfrac{1}{2}$）对应无穷多个 $x$（$\\dfrac{\\pi}{6}$、$\\dfrac{5\\pi}{6}$、$\\dfrac{13\\pi}{6}$……）。如果整个 $\\mathbb{R}$ 上定义 $\\arcsin$，给入 $\\dfrac{1}{2}$ 就说不清楚输出哪个。\n\n' +
          '**解决办法**：选定一段让 $\\sin x$ 单调的区间，在这段上 $\\sin x$ 就是一一映射，反函数才有意义。这段区间叫**主值区间**。\n\n' +
          '选 $\\left[-\\dfrac{\\pi}{2}, \\dfrac{\\pi}{2}\\right]$ 作为 $\\arcsin$ 的值域，因为 $\\sin x$ 在这段严格递增且包含 $0$（对称，奇函数）。',
        tip: '类比"开平方"：$x^2 = 4$ 有两个根 $\\pm 2$，但定义 $\\sqrt{4} = 2$（只取非负根）。反三角函数也是类似的"选取一支"。',
      },
      // 卡 2：三个反三角函数的主值区间
      {
        title: '三个反三角函数：值域与性质一览',
        body:
          '**反正弦** $\\arcsin x$：定义域 $[-1,1]$，值域 $\\left[-\\dfrac{\\pi}{2}, \\dfrac{\\pi}{2}\\right]$，奇函数（$\\arcsin(-x) = -\\arcsin x$），单调递增。\n\n' +
          '**反余弦** $\\arccos x$：定义域 $[-1,1]$，值域 $[0, \\pi]$，单调递减，非奇非偶。\n\n' +
          '**反正切** $\\arctan x$：定义域 $\\mathbb{R}$（所有实数），值域 $\\left(-\\dfrac{\\pi}{2}, \\dfrac{\\pi}{2}\\right)$（开区间），奇函数，单调递增，有两条水平渐近线：$\\lim_{x\\to+\\infty}\\arctan x = \\dfrac{\\pi}{2}$，$\\lim_{x\\to-\\infty}\\arctan x = -\\dfrac{\\pi}{2}$。\n\n' +
          '**重要等式**：$\\arcsin x + \\arccos x = \\dfrac{\\pi}{2}$（对 $x \\in [-1,1]$），可以互换使用。',
        formula:
          '\\arcsin x\\in\\left[-\\tfrac{\\pi}{2},\\tfrac{\\pi}{2}\\right],\\quad\\arccos x\\in[0,\\pi],\\quad\\arctan x\\in\\left(-\\tfrac{\\pi}{2},\\tfrac{\\pi}{2}\\right)',
      },
      // 卡 3：复合公式 arcsin(sin x) ——折回主值区间
      {
        title: '$\\arcsin(\\sin x)$：超出主值区间必须折回',
        body:
          '$\\arcsin(\\sin x) = x$ 只在 $x \\in \\left[-\\dfrac{\\pi}{2}, \\dfrac{\\pi}{2}\\right]$ 时直接成立。其他情况，必须先求内层 $\\sin x$ 的值，再在主值区间内找对应的角。\n\n' +
          '**例**：求 $\\arcsin\\left(\\sin\\dfrac{5\\pi}{6}\\right)$。\n\n' +
          '步骤一：$\\sin\\dfrac{5\\pi}{6} = \\sin\\left(\\pi - \\dfrac{\\pi}{6}\\right) = \\sin\\dfrac{\\pi}{6} = \\dfrac{1}{2}$。\n\n' +
          '步骤二：在 $\\left[-\\dfrac{\\pi}{2}, \\dfrac{\\pi}{2}\\right]$ 内，$\\sin$ 值等于 $\\dfrac{1}{2}$ 的角是 $\\dfrac{\\pi}{6}$。\n\n' +
          '步骤三：$\\arcsin\\left(\\sin\\dfrac{5\\pi}{6}\\right) = \\dfrac{\\pi}{6} \\ne \\dfrac{5\\pi}{6}$（原角不在主值区间内）。\n\n' +
          '**关键警示**：$\\dfrac{5\\pi}{6} > \\dfrac{\\pi}{2}$，超出主值区间，不能直接"约掉"！',
        steps: [
          '先算内层：$\\sin\\dfrac{5\\pi}{6} = \\dfrac{1}{2}$',
          '转化问题：$\\arcsin\\dfrac{1}{2} = ?$',
          '在主值区间 $[-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}]$ 内找答案：$\\dfrac{\\pi}{6}$',
          '结论：$\\arcsin\\left(\\sin\\dfrac{5\\pi}{6}\\right) = \\dfrac{\\pi}{6}$',
        ],
      },
      // 卡 4：arccos 与 arctan 的折回例题
      {
        title: '$\\arccos(\\cos x)$ 与 $\\arctan(\\tan x)$ 折回演练',
        body:
          '**$\\arccos$ 例**：求 $\\arccos\\left(\\cos\\dfrac{5\\pi}{3}\\right)$。\n\n' +
          '$\\cos\\dfrac{5\\pi}{3} = \\cos\\left(2\\pi - \\dfrac{\\pi}{3}\\right) = \\cos\\dfrac{\\pi}{3} = \\dfrac{1}{2}$。\n\n' +
          '$\\arccos$ 的值域 $[0, \\pi]$，在此区间内 $\\cos$ 值等于 $\\dfrac{1}{2}$ 的角是 $\\dfrac{\\pi}{3}$，故结果为 $\\dfrac{\\pi}{3}$。\n\n' +
          '**$\\arctan$ 例**：求 $\\arctan\\left(\\tan\\dfrac{7\\pi}{4}\\right)$。\n\n' +
          '$\\dfrac{7\\pi}{4}$ 与 $-\\dfrac{\\pi}{4}$ 终边相同（相差 $2\\pi$），$-\\dfrac{\\pi}{4} \\in \\left(-\\dfrac{\\pi}{2}, \\dfrac{\\pi}{2}\\right)$，故结果为 $-\\dfrac{\\pi}{4}$。\n\n' +
          '**统一模板**：内层 $\\to$ 求函数值 $\\to$ 找主值区间内对应角。',
        tip: '$\\arctan$ 的 $\\tan$ 周期为 $\\pi$（不是 $2\\pi$），折回时找与原角相差 $k\\pi$ 且在主值区间内的角。',
      },
      // 卡 5：反三角函数与积分的联系
      {
        title: '反三角函数出现在积分里',
        body:
          '微积分里有两个经典的含反三角函数的积分公式：\n\n' +
          '$\\displaystyle\\int\\dfrac{1}{\\sqrt{1-x^2}}\\,dx = \\arcsin x + C$，\n\n' +
          '$\\displaystyle\\int\\dfrac{1}{1+x^2}\\,dx = \\arctan x + C$。\n\n' +
          '**为什么？** 对 $\\arcsin x$ 求导，$\\left(\\arcsin x\\right)^{\\prime} = \\dfrac{1}{\\sqrt{1-x^2}}$；对 $\\arctan x$ 求导，$\\left(\\arctan x\\right)^{\\prime} = \\dfrac{1}{1+x^2}$（后面学导数时会证明）。\n\n' +
          '**$\\arctan x$ 的渐近行为**：$\\displaystyle\\lim_{x\\to+\\infty}\\arctan x = \\dfrac{\\pi}{2}$，这在计算广义积分 $\\int_0^{+\\infty}\\dfrac{1}{1+x^2}\\,dx = \\dfrac{\\pi}{2}$ 时直接用到。',
        formula:
          '\\int\\dfrac{1}{\\sqrt{1-x^2}}\\,dx=\\arcsin x+C,\\quad\\int\\dfrac{1}{1+x^2}\\,dx=\\arctan x+C',
      },
      // 卡 6：易错点——sin^{-1} 与倒数
      {
        title: '高频易错：$\\arcsin x$ 与 $\\sin^{-1} x$ 不是倒数',
        body:
          '符号 $\\sin^{-1} x$（上标 $-1$）是**反正弦函数**，与 $\\arcsin x$ 完全相同。**它不是 $\\dfrac{1}{\\sin x}$！**\n\n' +
          '$\\dfrac{1}{\\sin x}$ 是**余割函数** $\\csc x$，两者不同。同理：\n\n' +
          '$\\cos^{-1} x = \\arccos x \\ne \\dfrac{1}{\\cos x} = \\sec x$，\n\n' +
          '$\\tan^{-1} x = \\arctan x \\ne \\dfrac{1}{\\tan x} = \\cot x$。\n\n' +
          '**另一个高频坑**：$\\arcsin(\\sin x) \\ne x$（超出主值区间时需折回），而 $\\sin(\\arcsin x) = x$ **对 $x \\in [-1,1]$** 恒成立（先取反函数再取正弦等于原值）。\n\n' +
          '注意这两个复合的方向是**不对称**的！',
        reveal: {
          q: '判断：$\\arcsin\\left(\\sin 2\\right) = 2$（$2$ 为弧度）是否正确？',
          a: '**不正确。** $\\pi/2 \\approx 1.57 < 2$，所以 $2 \\notin \\left[-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right]$，不能直接"约掉"。由 $\\sin 2 = \\sin(\\pi - 2)$，且 $\\pi - 2 \\approx 1.14 \\in \\left[-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right]$，故 $\\arcsin(\\sin 2) = \\pi - 2$。',
        },
      },
      // 卡 7：综合速记与微积分铺垫
      {
        title: '本关速记：三函数三件事',
        body:
          '**记值域**（三个不同的区间）：\n\n' +
          '$\\arcsin$：$\\left[-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right]$（含两端，奇函数）；\n\n' +
          '$\\arccos$：$[0,\\pi]$（含两端，单调递减）；\n\n' +
          '$\\arctan$：$\\left(-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right)$（不含两端，$x \\to \\pm\\infty$ 时趋近边界）。\n\n' +
          '**折回口诀**：遇到 $\\arcsin(\\sin x)$、$\\arccos(\\cos x)$、$\\arctan(\\tan x)$，先算内层，再找主值区间内等值角，不能跳步直接"约掉"。\n\n' +
          '**微积分预告**：\n\n' +
          '$\\left(\\arcsin x\\right)^{\\prime} = \\dfrac{1}{\\sqrt{1-x^2}}$，$\\left(\\arctan x\\right)^{\\prime} = \\dfrac{1}{1+x^2}$；\n\n' +
          '$\\arcsin x + \\arccos x = \\dfrac{\\pi}{2}$ 在简化表达式时常用。',
      },
    ],
    questions: [
      // 题 1：choice — arcsin 的值域
      {
        id: 'c3-c-q1',
        type: 'choice',
        prompt: '$\\arcsin x$ 的值域是？',
        options: [
          '$[0, \\pi]$',
          '$\\left(-\\dfrac{\\pi}{2}, \\dfrac{\\pi}{2}\\right)$',
          '$\\left[-\\dfrac{\\pi}{2}, \\dfrac{\\pi}{2}\\right]$',
          '$[-1, 1]$',
        ],
        answer: 2,
        explain:
          '$\\arcsin x$ 的值域是闭区间 $\\left[-\\dfrac{\\pi}{2}, \\dfrac{\\pi}{2}\\right]$，包含两端点（因为端点处 $\\sin$ 有意义且取到 $\\pm 1$）。$[0,\\pi]$ 是 $\\arccos$ 的值域，$(-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2})$ 是 $\\arctan$ 的值域（开区间）。',
      },
      // 题 2：input — 折回 arcsin
      {
        id: 'c3-c-q2',
        type: 'input',
        prompt:
          '$\\arcsin\\left(\\sin\\dfrac{5\\pi}{6}\\right)$ 等于多少？（填分数形式，如 π/6）',
        accept: ['π/6', '\\pi/6', 'pi/6'],
        placeholder: '如 π/6',
        explain:
          '$\\sin\\dfrac{5\\pi}{6} = \\sin\\left(\\pi - \\dfrac{\\pi}{6}\\right) = \\sin\\dfrac{\\pi}{6} = \\dfrac{1}{2}$。在 $\\left[-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right]$ 内，$\\arcsin\\dfrac{1}{2} = \\dfrac{\\pi}{6}$。',
      },
      // 题 3：judge — arctan 定义域
      {
        id: 'c3-c-q3',
        type: 'judge',
        prompt: '$\\arctan x$ 的定义域是 $[-1, 1]$。',
        answer: false,
        explain:
          '$\\arctan x$ 对应 $\\tan$ 函数，$\\tan$ 的值域是整个 $\\mathbb{R}$，所以 $\\arctan x$ 的**定义域是 $\\mathbb{R}$**（所有实数）。$[-1,1]$ 是 $\\arcsin$ 和 $\\arccos$ 的定义域。',
      },
      // 题 4：choice — 折回 arccos
      {
        id: 'c3-c-q4',
        type: 'choice',
        prompt: '$\\arccos\\left(\\cos\\dfrac{7\\pi}{6}\\right)$ 等于？',
        options: [
          '$\\dfrac{7\\pi}{6}$',
          '$\\dfrac{5\\pi}{6}$',
          '$\\dfrac{\\pi}{6}$',
          '$-\\dfrac{\\pi}{6}$',
        ],
        answer: 1,
        explain:
          '$\\cos\\dfrac{7\\pi}{6} = \\cos(\\pi+\\dfrac{\\pi}{6}) = -\\cos\\dfrac{\\pi}{6} = -\\dfrac{\\sqrt{3}}{2}$。在 $[0,\\pi]$ 内，$\\cos$ 值等于 $-\\dfrac{\\sqrt{3}}{2}$ 的角是 $\\dfrac{5\\pi}{6}$，故 $\\arccos\\left(\\cos\\dfrac{7\\pi}{6}\\right) = \\dfrac{5\\pi}{6}$。',
      },
      // 题 5：judge — sin(arcsin) 与 arcsin(sin)
      {
        id: 'c3-c-q5',
        type: 'judge',
        prompt: '$\\sin(\\arcsin x) = x$ 对所有实数 $x$ 成立。',
        answer: false,
        explain:
          '$\\sin(\\arcsin x) = x$ 仅对 $x \\in [-1, 1]$ 成立（因为 $\\arcsin x$ 的定义域就是 $[-1,1]$，对域外的 $x$，$\\arcsin x$ 无定义）。对 $x \\in [-1,1]$，这个等式确实恒成立。',
      },
      // 题 6：input — arctan 渐近值
      {
        id: 'c3-c-q6',
        type: 'input',
        prompt:
          '$\\displaystyle\\lim_{x \\to +\\infty} \\arctan x$ 等于多少？（填 π/2 或小数保留两位）',
        accept: ['π/2', '\\pi/2', 'pi/2', '1.57'],
        placeholder: '填 π/2 或 1.57',
        explain:
          '$\\arctan x$ 的值域是 $\\left(-\\dfrac{\\pi}{2}, \\dfrac{\\pi}{2}\\right)$，$x \\to +\\infty$ 时，$\\arctan x$ 从左侧趋近于上界 $\\dfrac{\\pi}{2}$，故极限为 $\\dfrac{\\pi}{2}$。这是计算广义积分 $\\int_0^{+\\infty}\\dfrac{1}{1+x^2}\\,dx = \\dfrac{\\pi}{2}$ 的关键一步。',
      },
      // 题 7：match — 三个反三角函数的值域
      {
        id: 'c3-c-q7',
        type: 'match',
        prompt: '将反三角函数与其值域对应。',
        left: ['$\\arcsin x$', '$\\arccos x$', '$\\arctan x$'],
        right: [
          '$\\left[-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right]$（闭区间）',
          '$[0,\\pi]$（闭区间）',
          '$\\left(-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right)$（开区间）',
        ],
      },
      // 题 8：choice — arctan 折回
      {
        id: 'c3-c-q8',
        type: 'choice',
        prompt: '$\\arctan\\left(\\tan\\dfrac{7\\pi}{4}\\right)$ 等于？',
        options: [
          '$\\dfrac{7\\pi}{4}$',
          '$\\dfrac{\\pi}{4}$',
          '$-\\dfrac{\\pi}{4}$',
          '$\\dfrac{3\\pi}{4}$',
        ],
        answer: 2,
        explain:
          '$\\tan$ 的周期是 $\\pi$，$\\dfrac{7\\pi}{4} - 2\\pi = -\\dfrac{\\pi}{4}$。由于 $-\\dfrac{\\pi}{4} \\in \\left(-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right)$，是主值区间内的角，故 $\\arctan\\left(\\tan\\dfrac{7\\pi}{4}\\right) = -\\dfrac{\\pi}{4}$。',
      },
    ],
  },
];
