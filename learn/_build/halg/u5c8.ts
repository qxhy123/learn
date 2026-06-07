export const LESSONS = [
  {
    id: 'u5c8',
    title: '三角函数的应用',
    subtitle: '建模：$y = A\\sin(\\omega t + \\varphi) + b$；测高：仰角/俯角 + 联立方程；$A = \\dfrac{M-m}{2}$，$b = \\dfrac{M+m}{2}$，$T = \\dfrac{2\\pi}{\\omega}$',
    intro: [
      {
        title: '引入：周期现象的数学模型',
        body: '潮汐、气温、电压、声波……自然界中大量现象按周期变化。\n\n用函数 $y = A\\sin(\\omega t + \\varphi) + b$ 建模，关键是确定 4 个参数：\n\n| 参数 | 含义 | 由已知量推导 |\n|---|---|---|\n| $A$（振幅） | 偏离中心线的最大距离 | $A = \\dfrac{M - m}{2}$ |\n| $\\omega$（角频率） | 决定周期 $T = \\dfrac{2\\pi}{\\omega}$ | $\\omega = \\dfrac{2\\pi}{T}$ |\n| $\\varphi$（初相） | 初始状态（$t = 0$ 时的相位） | 代入初始条件解出 |\n| $b$（纵移） | 函数中心线 $y = b$ | $b = \\dfrac{M + m}{2}$ |\n\n其中 $M$ = 最大值，$m$ = 最小值。',
        formula: 'A = \\frac{M - m}{2},\\quad b = \\frac{M + m}{2},\\quad T = \\frac{2\\pi}{\\omega}',
        tip: '建模流程：先从最大最小值算 $A$ 和 $b$，再从周期算 $\\omega$，最后代初始条件（如"某时刻取最大值"）定 $\\varphi$。',
      },
      {
        title: '建模四步法：以潮汐模型为例',
        body: '**题目**：某海湾最高潮 $6.0$ m，最低潮 $0.4$ m，周期 $12$ h，$t = 2$ h 时达最高潮，建立模型。\n\n**第一步：确定 $A$ 和 $b$**：\n$$A = \\frac{6.0 - 0.4}{2} = 2.8,\\quad b = \\frac{6.0 + 0.4}{2} = 3.2$$\n\n**第二步：确定 $\\omega$**：\n$$T = 12,\\quad \\omega = \\frac{2\\pi}{12} = \\frac{\\pi}{6}$$\n\n**第三步：确定 $\\varphi$**（$t = 2$ 时取最大值，即 $\\sin(\\cdots) = 1$）：\n$$\\sin\\!\\left(\\frac{\\pi}{6} \\times 2 + \\varphi\\right) = 1 \\Rightarrow \\frac{\\pi}{3} + \\varphi = \\frac{\\pi}{2} \\Rightarrow \\varphi = \\frac{\\pi}{6}$$\n\n**模型**：$h(t) = 2.8\\sin\\!\\left(\\dfrac{\\pi}{6}t + \\dfrac{\\pi}{6}\\right) + 3.2$',
        steps: [
          '计算振幅 $A = \\dfrac{M-m}{2}$，中心线 $b = \\dfrac{M+m}{2}$',
          '由周期 $T$ 计算 $\\omega = \\dfrac{2\\pi}{T}$',
          '代入初始条件（极值点或零点）解 $\\varphi$',
          '写出最终模型，代入已知点验证',
        ],
      },
      {
        title: '从极值时刻到初相',
        body: '**确定 $\\varphi$ 的两种典型情况**：\n\n**情况 1：已知某时刻 $t_0$ 函数取最大值**\n$$\\sin(\\omega t_0 + \\varphi) = 1 \\Rightarrow \\omega t_0 + \\varphi = \\frac{\\pi}{2}$$\n解出 $\\varphi = \\dfrac{\\pi}{2} - \\omega t_0$\n\n**情况 2：已知某时刻 $t_0$ 函数取最小值**\n$$\\sin(\\omega t_0 + \\varphi) = -1 \\Rightarrow \\omega t_0 + \\varphi = -\\frac{\\pi}{2}$$\n解出 $\\varphi = -\\dfrac{\\pi}{2} - \\omega t_0$\n\n**例**：用电量 $W = 3\\sin\\!\\left(\\dfrac{\\pi}{12}t - \\dfrac{2\\pi}{3}\\right) + 5$，$t = 2$ 时最小值 $2$，$t = 14$ 时最大值 $8$。\n\n验证：$t = 2$：$W = 3\\sin(-\\dfrac{\\pi}{2}) + 5 = -3 + 5 = 2$ ✓；$t = 14$：$W = 3\\sin(\\dfrac{\\pi}{2}) + 5 = 8$ ✓。',
        tip: '确定 $\\varphi$ 时注意：若题目给了 $|\\varphi| \\leq \\dfrac{\\pi}{2}$ 的约束，要在两个候选值中选满足范围的那个。',
      },
      {
        title: '求满足条件的时间段',
        body: '建立模型后，常需要求"函数值满足某不等式的时间段"。\n\n**解不等式步骤**：\n1. 设不等式 $A\\sin(\\omega t + \\varphi) + b \\geq k$，整理为 $\\sin u \\geq c$（令 $u = \\omega t + \\varphi$）\n2. 确定 $t$ 的范围对应 $u$ 的范围\n3. 在 $u$ 的范围内解 $\\sin u \\geq c$，找区间\n4. 回代 $u = \\omega t + \\varphi$ 求 $t$\n\n**例**：模型 $W = 3\\sin\\!\\left(\\dfrac{\\pi}{12}t - \\dfrac{2\\pi}{3}\\right) + 5$，$t \\in [0, 24]$，求 $W \\geq 6.5$ 的时间段。\n\n$\\sin\\!\\left(\\dfrac{\\pi}{12}t - \\dfrac{2\\pi}{3}\\right) \\geq \\dfrac{1}{2}$，令 $u = \\dfrac{\\pi}{12}t - \\dfrac{2\\pi}{3}$，$u \\in \\left[-\\dfrac{2\\pi}{3}, \\dfrac{4\\pi}{3}\\right]$。\n\n在此范围解 $\\sin u \\geq \\dfrac{1}{2}$：$u \\in \\left[\\dfrac{\\pi}{6}, \\dfrac{5\\pi}{6}\\right]$，回代得 $t \\in [10, 18]$。',
        formula: '\\text{解} \\sin u \\geq c:\\quad u \\in \\left[\\arcsin c + 2k\\pi,\\ \\pi - \\arcsin c + 2k\\pi\\right]',
      },
      {
        title: '仰角与俯角：测高模型',
        body: '**定义**：\n- **仰角**：视线向上与水平线的夹角（$0°$ 到 $90°$）\n- **俯角**：视线向下与水平线的夹角（$0°$ 到 $90°$）\n\n**基本模型**（直角三角形）：目标高度 $h$，水平距离 $d$，$\\tan\\theta = \\dfrac{h}{d}$。\n\n**两点测高（间接测距）**：\n\n当无法直接测量时，在两个已知距离的点 $A, B$（$AB = c$ 为基线）分别观测角度，利用正弦定理或直接建立方程组求 $h$。\n\n**例**：$A, B$ 在同一直线，$AB = d$，$B$ 在 $A$ 和山脚连线上，$A$ 处仰角 $\\alpha$，$B$ 处仰角 $\\beta$（$\\beta > \\alpha$）。设山高 $h$，$B$ 到山脚距离 $x$：\n$$\\tan\\beta = \\frac{h}{x} \\Rightarrow x = \\frac{h}{\\tan\\beta}$$\n$$\\tan\\alpha = \\frac{h}{x+d} \\Rightarrow x + d = \\frac{h}{\\tan\\alpha}$$\n联立：$d = \\dfrac{h}{\\tan\\alpha} - \\dfrac{h}{\\tan\\beta}$，解得 $h = \\dfrac{d\\tan\\alpha\\tan\\beta}{\\tan\\beta - \\tan\\alpha}$。',
        steps: [
          '画出清晰示意图，标出已知角和未知量',
          '建立直角三角形中的 $\\tan$ 关系（仰角 = 对边/邻边）',
          '联立两个方程，消去一个未知量',
          '解方程求高度，代入验证',
        ],
      },
      {
        title: '典型例题：建模 + 应用',
        body: '**例 1（建模）**：某工厂用电量 $W$（万度）随时间 $t$（时，$0 \\leq t \\leq 24$），$t = 2$ 时最小 $2$ 万度，$t = 14$ 时最大 $8$ 万度。建立模型并求 $t = 8$ 时用电量。\n\n$A = 3$，$b = 5$，$T = 2(14-2) = 24$，$\\omega = \\dfrac{\\pi}{12}$，$\\varphi = -\\dfrac{2\\pi}{3}$（由最小值点 $t = 2$ 定）。\n\n$W = 3\\sin\\!\\left(\\dfrac{\\pi}{12}t - \\dfrac{2\\pi}{3}\\right) + 5$\n\n$t = 8$：$W = 3\\sin\\!\\left(\\dfrac{2\\pi}{3} - \\dfrac{2\\pi}{3}\\right) + 5 = 3 \\times 0 + 5 = 5$ 万度。\n\n**例 2（测高）**：$P$ 处仰角 $60°$，沿地面走 $50$ m 到 $R$，$R$ 处仰角 $30°$（$P$ 比 $R$ 更近山脚），设 $P$ 到山脚水平距离 $d$。\n\n$\\tan 60° = h/d$：$d = h/\\sqrt{3}$；$\\tan 30° = h/(d+50)$：$d + 50 = h\\sqrt{3}$。\n\n联立：$50 = h\\sqrt{3} - h/\\sqrt{3} = 2h/\\sqrt{3}$，$h = 25\\sqrt{3}$ m。',
      },
      {
        title: '易错点总结',
        body: '**易错 1**：确定 $\\varphi$ 时只用一个方程（如只代最大值点），忽略了可能有两个候选 $\\varphi$，需结合题意或约束范围唯一确定。\n\n**易错 2**：$\\omega$ 的单位要与 $t$ 的单位统一。若 $T$ 单位是小时，$\\omega = \\dfrac{2\\pi}{T}$ 中 $T$ 也用小时，则 $\\omega$ 的单位是 rad/h。\n\n**易错 3**：解三角不等式 $\\sin u \\geq c$ 时，先确定 $u$ 的范围，在该范围内找 $\\sin u \\geq c$ 的部分；不能直接写通解后再筛选（容易漏解）。\n\n**易错 4**：仰角/俯角问题中，建筑高度 $CD$ 垂直于地面，直角在底部 $D$，不在观测点。必须画清示意图，确认直角位置。',
        reveal: {
          q: '已知正弦模型最大值 $8$，最小值 $2$，周期 $12$，$t = 8$ 时取最大值，试建立函数模型。',
          a: '$A = 3$，$b = 5$，$\\omega = \\dfrac{\\pi}{6}$。$t = 8$ 时取最大值：$\\sin\\!\\left(\\dfrac{\\pi}{6} \\times 8 + \\varphi\\right) = 1$，$\\dfrac{4\\pi}{3} + \\varphi = \\dfrac{\\pi}{2}$，$\\varphi = \\dfrac{\\pi}{2} - \\dfrac{4\\pi}{3} = -\\dfrac{5\\pi}{6}$。模型：$y = 3\\sin\\!\\left(\\dfrac{\\pi}{6}t - \\dfrac{5\\pi}{6}\\right) + 5$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '某地月平均气温最高 $38°C$（7 月），最低 $-2°C$（1 月），周期 $12$ 月，振幅 $A = $（  ）。',
        options: [
          '$18°C$',
          '$20°C$',
          '$36°C$',
          '$40°C$',
        ],
        answer: 1,
        explain: '$A = \\dfrac{M - m}{2} = \\dfrac{38 - (-2)}{2} = \\dfrac{40}{2} = 20$ °C。',
      },
      {
        type: 'judge',
        prompt: '建立正弦模型 $y = A\\sin(\\omega t + \\varphi) + b$ 时，若 $T = 24$ 小时，则 $\\omega = \\dfrac{\\pi}{12}$。',
        answer: true,
        explain: '$\\omega = \\dfrac{2\\pi}{T} = \\dfrac{2\\pi}{24} = \\dfrac{\\pi}{12}$，正确。',
      },
      {
        type: 'choice',
        prompt: '利用模型 $W = 3\\sin\\!\\left(\\dfrac{\\pi}{12}t - \\dfrac{2\\pi}{3}\\right) + 5$，求 $t = 14$ 时的用电量（万度）。',
        options: [
          '$5$',
          '$8$',
          '$3$',
          '$2$',
        ],
        answer: 1,
        explain: '$W(14) = 3\\sin\\!\\left(\\dfrac{14\\pi}{12} - \\dfrac{2\\pi}{3}\\right) + 5 = 3\\sin\\!\\left(\\dfrac{7\\pi}{6} - \\dfrac{4\\pi}{6}\\right) + 5 = 3\\sin\\dfrac{\\pi}{2} + 5 = 3 + 5 = 8$ 万度。',
      },
      {
        type: 'input',
        prompt: '某量的最大值为 $10$，最小值为 $2$，按正弦规律变化，则中心线（竖直位移 $b$）的值为？',
        accept: ['6'],
        explain: '$b = \\dfrac{M + m}{2} = \\dfrac{10 + 2}{2} = 6$。',
      },
      {
        type: 'choice',
        prompt: '从地面 $A$ 点测山顶 $C$ 仰角 $30°$，水平走 $100$ m 到 $B$（$A, B, D$ 共线，$D$ 为山脚），从 $B$ 测仰角 $60°$，山高 $h = $（  ）。',
        options: [
          '$25\\sqrt{3}$ m',
          '$50\\sqrt{3}$ m',
          '$50$ m',
          '$100$ m',
        ],
        answer: 1,
        explain: '设 $B$ 到山脚 $D$ 水平距离为 $x$，$\\tan 60° = h/x$（$x = h/\\sqrt{3}$）；$\\tan 30° = h/(x+100)$（$x+100 = h\\sqrt{3}$）。联立：$100 = h\\sqrt{3} - h/\\sqrt{3} = 2h/\\sqrt{3}$，$h = 50\\sqrt{3}$ m。',
      },
      {
        type: 'match',
        prompt: '将各参数与其计算方式对应。',
        left: [
          '振幅 $A$',
          '角频率 $\\omega$',
          '竖直位移 $b$',
        ],
        right: [
          '$\\dfrac{M + m}{2}$',
          '$\\dfrac{M - m}{2}$',
          '$\\dfrac{2\\pi}{T}$',
        ],
        explain: '振幅 $A = \\dfrac{M-m}{2}$（最大最小之差的一半）；角频率 $\\omega = \\dfrac{2\\pi}{T}$（由周期确定）；竖直位移 $b = \\dfrac{M+m}{2}$（最大最小之和的一半，即中心线）。',
      },
      {
        type: 'judge',
        prompt: '解三角不等式 $\\sin(\\omega t + \\varphi) \\geq c$ 时，可以先写通解 $\\omega t + \\varphi \\in [\\arcsin c + 2k\\pi,\\ \\pi - \\arcsin c + 2k\\pi]$，再结合 $t$ 的范围筛选。',
        answer: true,
        explain: '这种做法是正确的。先用通解形式确定 $u = \\omega t + \\varphi$ 满足 $\\sin u \\geq c$ 的所有区间，再结合 $t$ 的实际范围筛选有效解。只要最后代回 $t$ 的范围来筛选，结果是正确的。',
      },
      {
        type: 'input',
        prompt: '某地月平均气温满足正弦模型，最高温 $32°C$（7 月），最低温 $-8°C$（1 月），周期 $12$ 月，$\\omega = $？（用分数 $\\pi$ 表示，如 pi/6）',
        accept: ['\\pi/6', 'pi/6', '\\frac{\\pi}{6}'],
        explain: '$\\omega = \\dfrac{2\\pi}{T} = \\dfrac{2\\pi}{12} = \\dfrac{\\pi}{6}$。',
      },
    ],
  },
]
