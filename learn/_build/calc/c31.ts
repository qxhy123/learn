// 第28章  随机微分方程入门
// 关 c31-a：布朗运动与 Itô 公式（8 卡 8 题）
// 关 c31-b：几何布朗运动与扩散模型（7 卡 7 题）
// 关 c31-c：反向 SDE 与扩散模型核心（6 卡 6 题）

export const LESSONS = [
  // ─────────────────────────────────────────
  // 关 c31-a  布朗运动与 Itô 公式
  // ─────────────────────────────────────────
  {
    id: 'c31-a',
    title: '布朗运动与 Itô 公式',
    subtitle: '动机·布朗运动·$(\\mathrm{d}W)^2=\\mathrm{d}t$·Itô 公式推导',
    intro: [
      // 卡 1：动机——为什么 AI 需要随机微积分？
      {
        title: '为什么 AI 要学随机微分方程？',
        body:
          '扩散模型（Stable Diffusion、DALL-E 等）能从噪声里"画出"一张图，背后靠的是一条随机微分方程（SDE）。\n\n' +
          '前向过程把真实图片**逐步加噪**变成纯高斯噪声，反向过程把噪声**逐步去噪**还原成图片。\n\n' +
          '要理解这条"去噪路径"，就必须懂 SDE：带噪声的微分方程。\n\n' +
          '而学 SDE 的第一步，是理解它的基础——**布朗运动**和 **Itô 公式**。这是本关的核心。',
        tip: '一句话：扩散模型 = 前向 SDE 加噪 + 反向 SDE 去噪，两者都由本章数学支撑。',
      },
      // 卡 2：从 ODE 到 SDE
      {
        title: 'ODE 加噪 → SDE',
        body:
          '普通常微分方程（ODE）描述**确定**轨迹：给定初值，未来唯一决定。\n\n' +
          '现实系统总含噪声，于是把 ODE 推广一步，在右边叠加一个随机项：\n\n' +
          '确定性：$\\dfrac{\\mathrm{d}X}{\\mathrm{d}t} = f(X,t)$\n\n' +
          '加噪后变成 SDE：$\\mathrm{d}X_t = f(X_t,t)\\,\\mathrm{d}t + g(X_t,t)\\,\\mathrm{d}W_t$\n\n' +
          '其中 $f$ 称为**漂移项**（平均趋势），$g$ 称为**扩散项**（噪声强度），$W_t$ 是布朗运动——一种连续随机过程。\n\n' +
          '这就像"有风的河流"：$f$ 决定水流方向，$g\\,\\mathrm{d}W_t$ 是随机的风吹浪打。',
        formula: '\\mathrm{d}X_t = f(X_t,t)\\,\\mathrm{d}t + g(X_t,t)\\,\\mathrm{d}W_t',
        tip: '关键区别：ODE 的解是一条确定曲线；SDE 的每次"运行"给出一条不同的随机轨迹。',
      },
      // 卡 3：布朗运动的四条性质
      {
        title: '布朗运动：连续但到处"抖"',
        body:
          '布朗运动 $W_t$ 是 SDE 中噪声的数学来源，满足四条核心性质：\n\n' +
          '① $W_0 = 0$（从原点出发）\n\n' +
          '② 增量服从正态分布：$W_t - W_s \\sim \\mathcal{N}(0,\\, t-s)$（增量方差 = 时间差）\n\n' +
          '③ 不相交区间上的增量**相互独立**（没有记忆）\n\n' +
          '④ 路径**连续**，但几乎处处**不可微**\n\n' +
          '性质④最奇特：轨迹你可以画出来，但在每个点都没有切线——就像无限折叠的锯齿。\n\n' +
          '**为什么不可微？** 在 $\\Delta t$ 内，$\\Delta W \\approx \\sqrt{\\Delta t}\\,\\varepsilon$（$\\varepsilon$ 是标准正态），所以"导数" $\\Delta W / \\Delta t \\approx \\varepsilon/\\sqrt{\\Delta t} \\to \\infty$。',
        formula: 'W_t - W_s \\sim \\mathcal{N}(0,\\, t-s),\\quad 0 \\le s < t',
        tip: '记口诀：增量是正态，方差是时间差。',
      },
      // 卡 4：二次变分 —— (dW)² = dt
      {
        title: 'Itô 规则：$(\\mathrm{d}W)^2 = \\mathrm{d}t$',
        body:
          '普通微积分里，高阶小量 $(\\mathrm{d}x)^2 \\to 0$ 可以直接丢弃。\n\n' +
          '但布朗运动不一样！在 $\\Delta t$ 内，$\\Delta W \\sim \\sqrt{\\Delta t}$，所以 $(\\Delta W)^2 \\sim \\Delta t$——这**不是二阶小量，而是一阶小量**，不能忽略！\n\n' +
          '随机微积分用三条代数规则浓缩这一事实：\n\n' +
          '$(\\mathrm{d}W_t)^2 = \\mathrm{d}t$，$\\quad \\mathrm{d}W_t \\cdot \\mathrm{d}t = 0$，$\\quad (\\mathrm{d}t)^2 = 0$\n\n' +
          '**物理直觉**：布朗路径抖动幅度是 $\\sqrt{\\Delta t}$，平方后变成 $\\Delta t$，和时间增量同阶，因此不能扔掉。这一条规则直接导致 Itô 公式比普通链式法则多出一项。',
        formula: '(\\mathrm{d}W_t)^2 = \\mathrm{d}t,\\quad \\mathrm{d}W_t\\cdot\\mathrm{d}t = 0,\\quad (\\mathrm{d}t)^2 = 0',
        tip: '类比：普通微积分 $(\\mathrm{d}x)^2 \\to 0$（丢），随机微积分 $(\\mathrm{d}W)^2 = \\mathrm{d}t$（保留！）。',
      },
      // 卡 5：Itô 公式
      {
        title: 'Itô 公式：随机版链式法则',
        body:
          '设随机过程满足 $\\mathrm{d}X_t = \\mu\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t$，令 $Y_t = f(X_t, t)$，则：\n\n' +
          '$\\mathrm{d}Y_t = \\left(f_t + \\mu f_x + \\dfrac{1}{2}\\sigma^2 f_{xx}\\right)\\mathrm{d}t + \\sigma f_x\\,\\mathrm{d}W_t$\n\n' +
          '与普通链式法则 $\\mathrm{d}f = f_t\\,\\mathrm{d}t + f_x\\,\\mathrm{d}X$ 对比，**多了一项**：\n\n' +
          '$\\dfrac{1}{2}\\sigma^2 f_{xx}\\,\\mathrm{d}t$\n\n' +
          '这一项来自哪里？就是展开泰勒级数时 $(\\mathrm{d}X)^2$ 项——而 $(\\mathrm{d}X)^2 \\approx \\sigma^2(\\mathrm{d}W)^2 = \\sigma^2\\,\\mathrm{d}t$ 不为零！\n\n' +
          '**5 步应用法**：①识别 $\\mu,\\sigma$；②选函数 $f$；③求 $f_t, f_x, f_{xx}$；④代入公式；⑤利用 $(\\mathrm{d}W)^2=\\mathrm{d}t$ 整理。',
        formula: '\\mathrm{d}f = \\left(f_t + \\mu f_x + \\frac{1}{2}\\sigma^2 f_{xx}\\right)\\mathrm{d}t + \\sigma f_x\\,\\mathrm{d}W_t',
        tip: '记忆口诀：Itô 公式 = 普通链式法则 + 二阶修正项 $\\frac{1}{2}\\sigma^2 f_{xx}\\,\\mathrm{d}t$。',
      },
      // 卡 6：经典例子 d(W²) 推导
      {
        title: '经典例：$\\mathrm{d}(W_t^2)$ 的推导',
        body:
          '让我们用 Itô 公式计算 $\\mathrm{d}(W_t^2)$，并和普通链式法则做对比。\n\n' +
          '取 $f(x) = x^2$，对应 $f_x = 2x$，$f_{xx} = 2$，$f_t = 0$；对 $X_t = W_t$ 有 $\\mu = 0$，$\\sigma = 1$。\n\n' +
          '**Itô 公式代入**：\n\n' +
          '$\\mathrm{d}(W_t^2) = (0 + 0 + \\frac{1}{2}\\cdot 1\\cdot 2)\\,\\mathrm{d}t + 1\\cdot 2W_t\\,\\mathrm{d}W_t = \\mathrm{d}t + 2W_t\\,\\mathrm{d}W_t$\n\n' +
          '**普通链式法则（错误答案）**：$\\mathrm{d}(W_t^2) \\stackrel{?}{=} 2W_t\\,\\mathrm{d}W_t$——漏掉了 $\\mathrm{d}t$！\n\n' +
          '**验证**：对两边取期望，Itô 积分 $\\mathbb{E}[W_t\\,\\mathrm{d}W_t] = 0$，所以 $\\mathbb{E}[\\mathrm{d}(W_t^2)] = \\mathrm{d}t$，即 $\\mathbb{E}[W_t^2] = t$，正好是布朗运动方差的定义 $\\mathrm{Var}(W_t) = t$，完全吻合。',
        formula: '\\mathrm{d}(W_t^2) = 2W_t\\,\\mathrm{d}W_t + \\mathrm{d}t',
        steps: [
          '取 $f(x)=x^2$，则 $f_x=2x$，$f_{xx}=2$，$f_t=0$',
          '对 $X_t=W_t$：$\\mu=0$，$\\sigma=1$',
          '代入 Itô 公式：$\\mathrm{d}(W_t^2)=\\bigl(0+0+\\frac{1}{2}\\cdot1\\cdot2\\bigr)\\mathrm{d}t+2W_t\\,\\mathrm{d}W_t$',
          '化简：$\\mathrm{d}(W_t^2)=2W_t\\,\\mathrm{d}W_t+\\mathrm{d}t$',
          '取期望验证：$\\mathbb{E}[W_t^2]=t=\\mathrm{Var}(W_t)$，一致！',
        ],
        tip: '普通链式法则少了 $\\mathrm{d}t$！这是 Itô 公式最经典的演示案例，必须记住。',
      },
      // 卡 7：再算一例 d(e^W) + Itô 积分期望为零
      {
        title: '再看一例：$\\mathrm{d}(e^{W_t})$ 与期望技巧',
        body:
          '取 $f(x) = e^x$，则 $f_x = e^x$，$f_{xx} = e^x$，$f_t = 0$；对 $X_t = W_t$ 有 $\\mu=0,\\sigma=1$。\n\n' +
          '**代入 Itô 公式**：\n\n' +
          '$\\mathrm{d}(e^{W_t}) = \\frac{1}{2}e^{W_t}\\,\\mathrm{d}t + e^{W_t}\\,\\mathrm{d}W_t$\n\n' +
          '**关键性质：Itô 积分期望为零**。$\\int_0^t H_s\\,\\mathrm{d}W_s$ 的期望总是 $0$（在合理条件下）。\n\n' +
          '因此对两边取期望，$\\mathrm{d}\\mathbb{E}[e^{W_t}] = \\frac{1}{2}\\mathbb{E}[e^{W_t}]\\,\\mathrm{d}t$，这是一个 ODE！\n\n' +
          '解出：$\\mathbb{E}[e^{W_t}] = e^{t/2}$（以 $\\mathbb{E}[e^{W_0}]=1$ 为初值）。\n\n' +
          '**交叉验证**：$W_t \\sim \\mathcal{N}(0,t)$，其矩母函数公式给出 $\\mathbb{E}[e^{W_t}] = e^{t/2}$。完全吻合！',
        formula: '\\mathrm{d}(e^{W_t}) = e^{W_t}\\,\\mathrm{d}W_t + \\frac{1}{2}e^{W_t}\\,\\mathrm{d}t',
        tip: 'Itô 积分的期望为零是反复用到的技巧：取期望时 $\\mathbb{E}[\\int H\\,\\mathrm{d}W]=0$，剩下 $\\mathrm{d}t$ 项变 ODE。',
      },
      // 卡 8：易错点汇总
      {
        title: '易错点：随机微积分的三大陷阱',
        body:
          '**陷阱一：用普通链式法则算布朗运动的函数。** 对布朗运动 $W_t$，$\\mathrm{d}f(W_t) \\ne f\'(W_t)\\,\\mathrm{d}W_t$。必须用 Itô 公式，额外加 $\\frac{1}{2}f\'\'(W_t)\\,\\mathrm{d}t$。\n\n' +
          '**陷阱二：把 SDE 的微分当普通微分。** $\\mathrm{d}X = \\mu\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W$ 只是积分形式的简写，$X_t$ 的路径不可微，不能直接"除以 $\\mathrm{d}t$"。\n\n' +
          '**陷阱三：遗漏 Itô 积分公式里的非对称性。** 普通 Newton-Leibniz 公式给出 $\\int_0^T x\\,\\mathrm{d}x = \\frac{1}{2}T^2$，但 Itô 积分给出 $\\int_0^T W_t\\,\\mathrm{d}W_t = \\frac{1}{2}W_T^2 - \\frac{1}{2}T$（多了修正项 $-\\frac{1}{2}T$）。\n\n' +
          '**总结**：随机微积分 = 确定性微积分 + $(\\mathrm{d}W)^2=\\mathrm{d}t$ 修正。只要记住这一点，三个陷阱都能化解。',
        reveal: {
          q: '普通链式法则计算 $\\mathrm{d}(W_t^2)$ 得到什么？正确答案是什么？',
          a: '普通链式法则给出 $2W_t\\,\\mathrm{d}W_t$（错误）。正确的 Itô 公式结果是 $2W_t\\,\\mathrm{d}W_t + \\mathrm{d}t$，多出了一项 $\\mathrm{d}t$，来自 $(\\mathrm{d}W)^2 = \\mathrm{d}t$。',
        },
        tip: '遇到"对布朗运动的函数求微分"，第一反应：用 Itô 公式，别用普通链式法则。',
      },
    ],
    questions: [
      // Q1：judge — (dW)² 是几阶小量？
      {
        type: 'judge',
        prompt: '在随机微积分中，$(\\mathrm{d}W_t)^2$ 与 $\\mathrm{d}t$ 是**同阶**的（而不是像 $(\\mathrm{d}t)^2$ 那样可以忽略）。这个说法正确吗？',
        answer: true,
        explain:
          '正确。布朗增量的量级是 $\\sqrt{\\Delta t}$，所以 $(\\Delta W)^2 \\sim \\Delta t$，是一阶小量，与 $\\mathrm{d}t$ 同阶，不可忽略。这是 Itô 规则 $(\\mathrm{d}W)^2 = \\mathrm{d}t$ 的核心依据。',
      },
      // Q2：choice — 布朗运动增量分布
      {
        type: 'choice',
        prompt: '布朗运动 $W_t$ 满足 $W_0=0$。对 $0 \\le s < t$，增量 $W_t - W_s$ 服从哪种分布？',
        options: [
          '$\\mathcal{N}(0,\\, t)$',
          '$\\mathcal{N}(0,\\, t-s)$',
          '$\\mathcal{N}(s,\\, t-s)$',
          '均匀分布 $\\mathrm{Uniform}(0, t-s)$',
        ],
        answer: 1,
        explain:
          '布朗运动增量 $W_t - W_s \\sim \\mathcal{N}(0,\\, t-s)$：均值为零，方差等于时间差 $t-s$，不是总时间 $t$。选项 A 是 $W_t$ 本身（从 0 出发）的分布，不是增量的一般形式。',
      },
      // Q3：input — d(W²) 公式
      {
        type: 'input',
        prompt: '用 Itô 公式计算 $\\mathrm{d}(W_t^2)$。答案写成 $a W_t\\,\\mathrm{d}W_t + b\\,\\mathrm{d}t$ 的形式，其中 $(a, b) = ?$（输入"2,1"）',
        accept: ['2,1', '(2,1)', 'a=2,b=1'],
        explain:
          '对 $f(x)=x^2$，$f_x=2x$，$f_{xx}=2$，$f_t=0$；$\\mu=0,\\sigma=1$。代入 Itô 公式：$\\mathrm{d}(W_t^2) = 2W_t\\,\\mathrm{d}W_t + \\frac{1}{2}\\cdot1\\cdot2\\,\\mathrm{d}t = 2W_t\\,\\mathrm{d}W_t + \\mathrm{d}t$，所以 $a=2,b=1$。',
      },
      // Q4：choice — Itô 公式多出的项
      {
        type: 'choice',
        prompt: '若 $\\mathrm{d}X_t = \\mu\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t$，$Y_t = f(X_t,t)$，Itô 公式相比普通链式法则多出哪一项？',
        options: [
          '$\\sigma f_x\\,\\mathrm{d}W_t$',
          '$\\mu f_x\\,\\mathrm{d}t$',
          '$\\dfrac{1}{2}\\sigma^2 f_{xx}\\,\\mathrm{d}t$',
          '$f_t\\,\\mathrm{d}t$',
        ],
        answer: 2,
        explain:
          '普通链式法则给出 $f_t\\,\\mathrm{d}t + f_x\\,\\mathrm{d}X$，即 $(f_t + \\mu f_x)\\,\\mathrm{d}t + \\sigma f_x\\,\\mathrm{d}W_t$。Itô 公式额外多出 $\\frac{1}{2}\\sigma^2 f_{xx}\\,\\mathrm{d}t$，来源于 $(\\mathrm{d}X)^2 = \\sigma^2(\\mathrm{d}W)^2 = \\sigma^2\\,\\mathrm{d}t$。',
      },
      // Q5：judge — Itô 积分期望为零
      {
        type: 'judge',
        prompt: '在满足适当条件下，随机积分（Itô 积分）$\\int_0^T H_t\\,\\mathrm{d}W_t$ 的期望等于零。这个说法正确吗？',
        answer: true,
        explain:
          '正确。Itô 积分是一个鞅（martingale），期望为零：$\\mathbb{E}\\!\\left[\\int_0^T H_t\\,\\mathrm{d}W_t\\right] = 0$（在 $H$ 满足平方可积等适当条件时）。这是取期望时消去随机项的关键性质。',
      },
      // Q6：match — SDE 术语配对
      {
        type: 'match',
        prompt: '将 SDE $\\mathrm{d}X_t = f(X_t,t)\\,\\mathrm{d}t + g(X_t,t)\\,\\mathrm{d}W_t$ 中的术语与含义配对：',
        left: ['$f(X_t,t)$', '$g(X_t,t)$', '$W_t$', '$\\mathrm{d}W_t$'],
        right: [
          '扩散项，描述噪声强度',
          '布朗运动，随机过程本体',
          '漂移项，描述平均趋势',
          '布朗运动的微小增量',
        ],
        answer: [2, 0, 1, 3],
        explain:
          '$f$ 是漂移项（drift），对应"平均趋势"；$g$ 是扩散项（diffusion），对应"噪声强度"；$W_t$ 是布朗运动本体；$\\mathrm{d}W_t$ 是布朗运动的微小（随机）增量。',
      },
      // Q7：choice — E[d(e^W)] 化简后得到的 ODE
      {
        type: 'choice',
        prompt: '已知 $\\mathrm{d}(e^{W_t}) = e^{W_t}\\,\\mathrm{d}W_t + \\frac{1}{2}e^{W_t}\\,\\mathrm{d}t$，两边取期望后得到哪个 ODE（令 $m(t)=\\mathbb{E}[e^{W_t}]$）？',
        options: [
          '$m\'(t) = m(t)$',
          '$m\'(t) = \\frac{1}{2}m(t)$',
          '$m\'(t) = 2m(t)$',
          '$m\'(t) = 0$',
        ],
        answer: 1,
        explain:
          '取期望后 Itô 积分项消失（期望为零），剩下 $\\mathrm{d}m = \\frac{1}{2}m\\,\\mathrm{d}t$，即 $m\'(t) = \\frac{1}{2}m(t)$。解为 $m(t) = e^{t/2}$，与正态矩母函数 $\\mathbb{E}[e^{W_t}]=e^{t/2}$ 一致。',
      },
      // Q8：input — 布朗运动在 t=4 时的标准差
      {
        type: 'input',
        prompt: '布朗运动 $W_t$ 满足 $W_0=0$。$W_4$ 的标准差是多少？（输入整数）',
        accept: ['2'],
        explain:
          '$W_4 \\sim \\mathcal{N}(0,4)$，标准差 $= \\sqrt{4} = 2$。布朗运动方差等于时间：$\\mathrm{Var}(W_t)=t$，标准差为 $\\sqrt{t}$。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c31-b  几何布朗运动与 OU 过程
  // ─────────────────────────────────────────
  {
    id: 'c31-b',
    title: '几何布朗运动与 OU 过程',
    subtitle: 'Itô 公式求解·漂移修正·OU 均值回归·Fokker-Planck 分布视角',
    intro: [
      // 卡 1：几何布朗运动的 SDE
      {
        title: '几何布朗运动：保正值的 SDE',
        body:
          '普通布朗运动会变成负数，但股票价格、浓度等物理量必须为正。**几何布朗运动**（GBM）是专门构造的保正值 SDE：\n\n' +
          '$\\mathrm{d}S_t = \\mu S_t\\,\\mathrm{d}t + \\sigma S_t\\,\\mathrm{d}W_t$\n\n' +
          '漂移项和扩散项都乘以了 $S_t$，这保证了"变化量与当前值成比例"——就像复利增长一样。\n\n' +
          '**参数含义**：$\\mu$ 是期望增长率，$\\sigma$ 是波动率。\n\n' +
          '**直觉**：如果没有噪声，$\\mathrm{d}S = \\mu S\\,\\mathrm{d}t$，解为 $S_t = S_0 e^{\\mu t}$（指数增长）。加上噪声后，路径会在指数趋势附近随机抖动，但始终保持正值。',
        formula: '\\mathrm{d}S_t = \\mu S_t\\,\\mathrm{d}t + \\sigma S_t\\,\\mathrm{d}W_t',
        tip: '关键特征：漂移和扩散系数都正比于 $S_t$，保证 $S_t > 0$（若 $S_0 > 0$）。',
      },
      // 卡 2：用 Itô 公式求解 GBM —— 对数变换
      {
        title: '求解 GBM：取对数 + Itô 公式',
        body:
          '如何求 $\\mathrm{d}S_t = \\mu S_t\\,\\mathrm{d}t + \\sigma S_t\\,\\mathrm{d}W_t$ 的显式解？\n\n' +
          '**思路**：令 $Y_t = \\ln S_t$，用 Itô 公式计算 $\\mathrm{d}Y_t$。\n\n' +
          '取 $f(x) = \\ln x$，则 $f_x = 1/x$，$f_{xx} = -1/x^2$，$f_t = 0$。对 $X_t = S_t$ 有 $\\mu_X = \\mu S_t$，$\\sigma_X = \\sigma S_t$。\n\n' +
          '代入 Itô 公式：\n\n' +
          '$\\mathrm{d}Y_t = \\frac{1}{S_t}\\cdot\\mu S_t\\,\\mathrm{d}t + \\frac{1}{S_t}\\cdot\\sigma S_t\\,\\mathrm{d}W_t + \\frac{1}{2}\\left(-\\frac{1}{S_t^2}\\right)\\sigma^2 S_t^2\\,\\mathrm{d}t$\n\n' +
          '$= \\mu\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t - \\frac{\\sigma^2}{2}\\,\\mathrm{d}t = \\left(\\mu - \\frac{\\sigma^2}{2}\\right)\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t$\n\n' +
          '这是一个**常系数 SDE**，直接积分即可。',
        formula: '\\mathrm{d}(\\ln S_t) = \\left(\\mu - \\frac{\\sigma^2}{2}\\right)\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t',
        steps: [
          '令 $Y_t = \\ln S_t$，$f(x)=\\ln x$，$f_x=1/x$，$f_{xx}=-1/x^2$',
          '代入 Itô 公式：$\\mathrm{d}Y_t = \\mu\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t + \\frac{1}{2}(-1/S_t^2)\\sigma^2 S_t^2\\,\\mathrm{d}t$',
          '化简二阶修正项：$-\\frac{\\sigma^2}{2}\\,\\mathrm{d}t$',
          '整合：$\\mathrm{d}Y_t = (\\mu - \\sigma^2/2)\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t$',
        ],
        tip: '关键：Itô 公式的二阶修正项给出了 $-\\sigma^2/2$，这就是漂移修正项的来源。',
      },
      // 卡 3：GBM 显式解与漂移修正
      {
        title: 'GBM 显式解：漂移为何有 $-\\sigma^2/2$？',
        body:
          '上一步得到 $\\mathrm{d}(\\ln S_t) = (\\mu - \\sigma^2/2)\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t$，两边从 $0$ 积到 $t$：\n\n' +
          '$\\ln S_t - \\ln S_0 = \\left(\\mu - \\frac{\\sigma^2}{2}\\right)t + \\sigma W_t$\n\n' +
          '指数化（两边取 $e$）得：\n\n' +
          '$S_t = S_0\\,\\exp\\!\\left[\\left(\\mu - \\frac{\\sigma^2}{2}\\right)t + \\sigma W_t\\right]$\n\n' +
          '**为什么指数里是 $\\mu - \\sigma^2/2$ 而不是 $\\mu$？** 这是 Itô 公式的二阶修正造成的。如果用普通微积分求解，会错误地得到指数里是 $\\mu t$，而正确结果的期望增长率恰好是 $\\mu$（可以验证 $\\mathbb{E}[S_t] = S_0 e^{\\mu t}$），修正项 $-\\sigma^2/2$ 是 Jensen 不等式的反映：$\\mathbb{E}[e^X] > e^{\\mathbb{E}[X]}$。',
        formula: 'S_t = S_0\\exp\\!\\left[\\left(\\mu - \\frac{\\sigma^2}{2}\\right)t + \\sigma W_t\\right]',
        tip: '记忆要点：GBM 的显式解，指数里漂移是 $\\mu - \\sigma^2/2$（不是 $\\mu$），$-\\sigma^2/2$ 来自 Itô 二阶修正。',
      },
      // 卡 4：OU 过程——带噪声的均值回归
      {
        title: 'Ornstein-Uhlenbeck 过程：均值回归',
        body:
          'Ornstein-Uhlenbeck（OU）过程的 SDE：\n\n' +
          '$\\mathrm{d}X_t = -\\theta X_t\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t\\quad(\\theta > 0)$\n\n' +
          '**直觉**：漂移项 $-\\theta X_t$ 像一根弹簧——当 $X_t > 0$ 时，把它拉回 $0$；当 $X_t < 0$ 时，把它推回 $0$。$\\theta$ 越大，弹力越强，回归越快。扩散项 $\\sigma\\,\\mathrm{d}W_t$ 不断注入随机噪声。\n\n' +
          '**两力平衡**：回归力最终与噪声达到平衡，分布会收敛到一个固定的（平稳）高斯分布。\n\n' +
          '**AI 应用**：OU 过程是扩散模型中前向加噪过程的常用模板——它把任意初始分布逐渐"磨平"成高斯分布，扩散模型正是利用这一性质来设计前向噪声。',
        formula: '\\mathrm{d}X_t = -\\theta X_t\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t,\\quad \\theta > 0',
        tip: '记忆口诀：OU = 弹簧回归力 + 随机噪声。弹力越强（$\\theta$ 大），收敛越快；噪声越大（$\\sigma$ 大），分布越宽。',
      },
      // 卡 5：Fokker-Planck 方程——分布视角
      {
        title: 'Fokker-Planck：从单条轨迹到分布',
        body:
          'SDE 描述的是**单条随机轨迹**，但我们更常关心"大量粒子的整体分布如何演化"。\n\n' +
          '对 SDE $\\mathrm{d}X_t = \\mu(X_t,t)\\,\\mathrm{d}t + \\sigma(X_t,t)\\,\\mathrm{d}W_t$，密度 $p(x,t)$ 满足 **Fokker-Planck 方程**：\n\n' +
          '$\\dfrac{\\partial p}{\\partial t} = -\\dfrac{\\partial}{\\partial x}(\\mu p) + \\dfrac{1}{2}\\dfrac{\\partial^2}{\\partial x^2}(\\sigma^2 p)$\n\n' +
          '两项的物理意义：\n\n' +
          '- 第一项（漂移）：像"整体搬运"，把质量沿 $\\mu$ 方向移动\n' +
          '- 第二项（扩散）：像"热扩散"，使分布向四周扩散\n\n' +
          '**重要用途**：令 $\\partial p / \\partial t = 0$ 可求**平稳分布**——长时间后分布不再变化的状态。',
        formula: '\\frac{\\partial p}{\\partial t} = -\\frac{\\partial}{\\partial x}(\\mu p) + \\frac{1}{2}\\frac{\\partial^2}{\\partial x^2}(\\sigma^2 p)',
        tip: '类比：SDE 是"追踪一个粒子"，Fokker-Planck 是"追踪所有粒子的密度云"。两者是同一物理过程的两种视角。',
      },
      // 卡 6：OU 过程的平稳分布
      {
        title: 'OU 平稳分布：高斯的来源',
        body:
          '对 OU 过程 $\\mathrm{d}X_t = -\\theta X_t\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t$，Fokker-Planck 方程是：\n\n' +
          '$\\partial_t p = \\theta \\partial_x(xp) + \\frac{\\sigma^2}{2}\\partial_{xx}p$\n\n' +
          '令 $\\partial_t p = 0$（寻找平稳分布），化简后解出：\n\n' +
          '$p^*(x) \\propto \\exp\\!\\left(-\\frac{\\theta x^2}{\\sigma^2}\\right)$\n\n' +
          '归一化后是高斯分布，均值为 $0$，方差为 $\\sigma^2/(2\\theta)$：\n\n' +
          '$p^*(x) = \\mathcal{N}\\!\\left(0,\\, \\frac{\\sigma^2}{2\\theta}\\right)$\n\n' +
          '**直觉检验**：① 噪声越大（$\\sigma$ 大），方差越大（分布越宽）——合理；② 弹力越强（$\\theta$ 大），方差越小（被"夹"得更紧）——合理。两个极限行为都符合物理直觉。',
        formula: 'p^*(x) = \\mathcal{N}\\!\\left(0,\\ \\frac{\\sigma^2}{2\\theta}\\right)',
        tip: 'OU 过程的平稳分布是高斯的。这是扩散模型设计"前向加噪最终趋于标准正态"的理论基础。',
      },
      // 卡 7：易错点 — 漂移修正 + 数值方法
      {
        title: '易错点：漂移修正与 Euler-Maruyama',
        body:
          '**易错 1：GBM 漂移忘减 $\\sigma^2/2$。** 见到 $\\mathrm{d}S=\\mu S\\,\\mathrm{d}t+\\sigma S\\,\\mathrm{d}W$，显式解指数里漂移是 $\\mu - \\sigma^2/2$，不是 $\\mu$。这一修正完全来自 Itô 公式二阶项。\n\n' +
          '**Euler-Maruyama 数值离散**：将连续 SDE 离散化：\n\n' +
          '$X_{n+1} = X_n + \\mu(X_n, t_n)\\Delta t + \\sigma(X_n, t_n)\\sqrt{\\Delta t}\\,\\varepsilon_n,\\quad \\varepsilon_n \\sim \\mathcal{N}(0,1)$\n\n' +
          '注意随机项是 $\\sigma\\sqrt{\\Delta t}$，不是 $\\sigma\\Delta t$！\n\n' +
          '**收敛阶**：强收敛阶 $1/2$（路径误差），弱收敛阶 $1$（分布误差）。步长减半时，路径误差缩小约 $1/\\sqrt{2}\\approx0.71$，而不是减半（$0.5$）。',
        reveal: {
          q: 'GBM 的显式解里，指数中漂移项为什么是 $\\mu - \\sigma^2/2$ 而不是 $\\mu$？',
          a: '因为对 $\\ln S_t$ 用 Itô 公式时，$f_{xx} = -1/x^2$ 给出了二阶修正项 $\\frac{1}{2}(-1/S_t^2)\\cdot\\sigma^2 S_t^2\\,\\mathrm{d}t = -\\sigma^2/2\\,\\mathrm{d}t$。这一修正是随机微积分特有的，普通链式法则会漏掉它。',
        },
        tip: 'Euler-Maruyama 的随机项比例系数是 $\\sqrt{\\Delta t}$（不是 $\\Delta t$），因为布朗增量的标准差是 $\\sqrt{\\Delta t}$。',
      },
    ],
    questions: [
      // Q1：choice — GBM 显式解
      {
        type: 'choice',
        prompt: '几何布朗运动 $\\mathrm{d}S_t = \\mu S_t\\,\\mathrm{d}t + \\sigma S_t\\,\\mathrm{d}W_t$（$S_0 > 0$）的显式解是？',
        options: [
          '$S_t = S_0 e^{\\mu t + \\sigma W_t}$',
          '$S_t = S_0 e^{(\\mu - \\sigma^2/2)t + \\sigma W_t}$',
          '$S_t = S_0 e^{(\\mu + \\sigma^2/2)t + \\sigma W_t}$',
          '$S_t = S_0(1 + \\mu t + \\sigma W_t)$',
        ],
        answer: 1,
        explain:
          '对 $Y_t = \\ln S_t$ 用 Itô 公式，二阶修正项给出 $-\\sigma^2/2$，因此 $\\mathrm{d}Y_t = (\\mu - \\sigma^2/2)\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t$，积分后指数化得 $S_t = S_0\\exp[(\\mu-\\sigma^2/2)t + \\sigma W_t]$。',
      },
      // Q2：input — GBM 期望
      {
        type: 'input',
        prompt: '设几何布朗运动 $S_t = S_0\\exp[(\\mu-\\sigma^2/2)t+\\sigma W_t]$，$\\mathbb{E}[S_t] = ?$（输入 "S0*e^(mu*t)" 或 "$S_0 e^{\\mu t}$"，用中文简写"S0 exp(mu t)"）',
        accept: ['S0 exp(mu t)', 'S_0 e^{mu t}', 'S0*e^(mu*t)', 'S0e^(μt)', 'S_0\\exp(\\mu t)'],
        explain:
          '利用对数正态分布矩母函数：$\\mathbb{E}[e^{\\sigma W_t}] = e^{\\sigma^2 t/2}$（$W_t\\sim\\mathcal{N}(0,t)$）。所以 $\\mathbb{E}[S_t] = S_0 e^{(\\mu-\\sigma^2/2)t}\\cdot e^{\\sigma^2 t/2} = S_0 e^{\\mu t}$。漂移修正 $-\\sigma^2/2$ 与随机项的矩恰好抵消，期望增长率就是 $\\mu$。',
      },
      // Q3：judge — OU 漂移方向
      {
        type: 'judge',
        prompt: 'OU 过程 $\\mathrm{d}X_t = -\\theta X_t\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t$（$\\theta>0$）中，当 $X_t > 0$ 时漂移项为负，会将 $X_t$ 拉向 $0$。这个说法正确吗？',
        answer: true,
        explain:
          '正确。漂移项为 $-\\theta X_t$：当 $X_t > 0$ 时为负（向左，趋向 $0$），当 $X_t < 0$ 时为正（向右，趋向 $0$）。这正是"均值回归"（mean-reverting）的机制，$\\theta$ 是回归速度。',
      },
      // Q4：choice — OU 平稳分布方差
      {
        type: 'choice',
        prompt: 'OU 过程 $\\mathrm{d}X_t = -\\theta X_t\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t$ 的平稳分布 $p^*(x)$ 的方差是？',
        options: [
          '$\\sigma^2$',
          '$\\sigma^2/\\theta$',
          '$\\sigma^2/(2\\theta)$',
          '$2\\sigma^2/\\theta$',
        ],
        answer: 2,
        explain:
          '由 Fokker-Planck 方程令 $\\partial_t p = 0$ 求解，OU 过程的平稳分布为 $\\mathcal{N}(0,\\, \\sigma^2/(2\\theta))$，方差 $= \\sigma^2/(2\\theta)$。噪声越大（$\\sigma$ 大），方差越大；弹力越强（$\\theta$ 大），方差越小。',
      },
      // Q5：match — 扩散/漂移项对应
      {
        type: 'match',
        prompt: '将以下 SDE 与其描述的模型配对：',
        left: [
          '$\\mathrm{d}S_t = \\mu S_t\\,\\mathrm{d}t + \\sigma S_t\\,\\mathrm{d}W_t$',
          '$\\mathrm{d}X_t = -\\theta X_t\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t$',
          '$\\mathrm{d}X_t = \\sigma\\,\\mathrm{d}W_t$',
          '$\\mathrm{d}X_t = 0\\cdot\\mathrm{d}t + 0\\cdot\\mathrm{d}W_t$',
        ],
        right: [
          '纯布朗运动（无漂移、恒扩散）',
          '确定性静止（退化情形）',
          '几何布朗运动（保正值）',
          'OU 过程（均值回归）',
        ],
        answer: [2, 3, 0, 1],
        explain:
          '第一行乘以 $S_t$ 保正值 → GBM；第二行有 $-\\theta X_t$ 回归力 → OU；第三行 $\\sigma\\,\\mathrm{d}W_t$ 无漂移 → 纯布朗运动；第四行所有系数为零 → 静止。',
      },
      // Q6：judge — Fokker-Planck 描述分布
      {
        type: 'judge',
        prompt: 'Fokker-Planck 方程描述的是**单条**随机轨迹的时间演化。这个说法正确吗？',
        answer: false,
        explain:
          '错误。Fokker-Planck 方程描述的是概率密度 $p(x,t)$ 的时间演化——即**所有**可能轨迹的整体分布如何随时间变化，而不是追踪某条具体的轨迹。追踪单条轨迹用的是 SDE 本身（配合数值模拟如 Euler-Maruyama）。',
      },
      // Q7：choice — Euler-Maruyama 随机项系数
      {
        type: 'choice',
        prompt: 'Euler-Maruyama 方法将 SDE $\\mathrm{d}X = \\mu\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W$ 离散化为 $X_{n+1} = X_n + \\mu\\Delta t + ?\\,\\varepsilon_n$（$\\varepsilon_n\\sim\\mathcal{N}(0,1)$），其中 $?$ 是？',
        options: [
          '$\\sigma\\Delta t$',
          '$\\sigma\\sqrt{\\Delta t}$',
          '$\\sigma(\\Delta t)^2$',
          '$\\sigma/\\sqrt{\\Delta t}$',
        ],
        answer: 1,
        explain:
          '布朗运动增量 $\\Delta W \\sim \\mathcal{N}(0,\\Delta t)$，即 $\\Delta W = \\sqrt{\\Delta t}\\,\\varepsilon$（$\\varepsilon\\sim\\mathcal{N}(0,1)$）。因此随机项系数是 $\\sigma\\sqrt{\\Delta t}$，而不是 $\\sigma\\Delta t$。这是数值 SDE 模拟最常见的写错点。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c31-c  反向 SDE 与扩散模型核心
  // ─────────────────────────────────────────
  {
    id: 'c31-c',
    title: '反向 SDE 与扩散模型',
    subtitle: '前向加噪·score function·反向 SDE·DDPM 闭式·概率流 ODE',
    intro: [
      // 卡 1：扩散模型的全局框图
      {
        title: '扩散模型的两条路',
        body:
          '扩散模型由**两个过程**组成，方向相反：\n\n' +
          '**前向（加噪）**：从真实数据 $x_0$ 出发，按 SDE 逐步加噪声，$t$ 从 $0$ 增大到 $T$，最终变成接近纯高斯噪声的 $x_T$。\n\n' +
          '**反向（去噪）**：从噪声 $x_T$ 出发，沿时间反向，逐步去掉噪声，恢复出新的数据样本 $x_0$。\n\n' +
          '训练阶段：学一个神经网络，让它知道"在当前噪声状态 $x_t$ 时，应该往哪个方向走才能回到数据"。\n\n' +
          '采样阶段：从标准高斯采一个噪声，用训练好的网络做反向 SDE 积分，生成新图片/声音/文本。\n\n' +
          '**关键数学问题**：反向 SDE 怎么写？它需要知道什么信息？',
        tip: '前向 = 加噪（确定性设计），反向 = 去噪（依赖 score function，需要学习）。',
      },
      // 卡 2：前向 SDE 与 DDPM 闭式
      {
        title: '前向 SDE 与 DDPM 单步闭式',
        body:
          '扩散模型最常用的前向 SDE（连续形式）：\n\n' +
          '$\\mathrm{d}x = -\\frac{1}{2}\\beta(t)x\\,\\mathrm{d}t + \\sqrt{\\beta(t)}\\,\\mathrm{d}W_t$\n\n' +
          '离散 DDPM 的前向更新：$q(x_t|x_{t-1}) = \\mathcal{N}(\\sqrt{\\alpha_t}\\,x_{t-1},\\,(1-\\alpha_t)I)$，其中 $\\alpha_t = 1 - \\beta_t$。\n\n' +
          '将多步高斯叠加（高斯乘高斯还是高斯），得到**单步闭式**：\n\n' +
          '$q(x_t|x_0) = \\mathcal{N}(\\sqrt{\\bar{\\alpha}_t}\\,x_0,\\,(1-\\bar{\\alpha}_t)I),\\quad \\bar{\\alpha}_t = \\prod_{s=1}^t \\alpha_s$\n\n' +
          '等价采样形式：$x_t = \\sqrt{\\bar{\\alpha}_t}\\,x_0 + \\sqrt{1-\\bar{\\alpha}_t}\\,\\varepsilon$，$\\varepsilon\\sim\\mathcal{N}(0,I)$\n\n' +
          '**这条公式非常有用**：训练时不需要逐步模拟，可以直接跳到任意时刻 $t$，随机采样对应的 $x_t$！',
        formula: 'x_t = \\sqrt{\\bar{\\alpha}_t}\\,x_0 + \\sqrt{1-\\bar{\\alpha}_t}\\,\\varepsilon,\\quad \\varepsilon\\sim\\mathcal{N}(0,I)',
        tip: '$\\bar{\\alpha}_t = \\prod_{s=1}^t \\alpha_s$ 会随 $t$ 增大而趋近 $0$，使 $x_t$ 均值趋于 $0$、方差趋于 $I$——变成标准高斯。',
      },
      // 卡 3：score function 的直觉
      {
        title: 'Score Function：概率地图的指南针',
        body:
          '**Score function** 的定义：$\\nabla_x \\log p_t(x)$\n\n' +
          '直觉上，它是对数概率密度的梯度，**指向"当前位置 $x$ 朝哪个方向，概率密度增加最快"**。\n\n' +
          '例子：如果数据分布是高斯 $\\mathcal{N}(0,1)$，则 $\\log p(x) = -x^2/2 + \\mathrm{const}$，score $= -x$——总是指向原点（概率最大处）。\n\n' +
          '**为什么重要？** 反向去噪需要知道"应该朝哪个方向走才能回到数据分布的高密度区域"，这正是 score function 提供的信息。\n\n' +
          '**学习 score 的困难**：真实的 $p_t(x)$ 不知道，所以用神经网络 $s_\\theta(x,t)$ 来拟合 $\\nabla_x \\log p_t(x)$。',
        formula: '\\nabla_x \\log p_t(x)\\;\\text{指向对数密度上升最快的方向}',
        tip: '类比：score function 是"概率地图的指南针"，告诉你从当前位置向哪走，概率更高。',
      },
      // 卡 4：反向 SDE 公式
      {
        title: '反向 SDE：用 Score 导航去噪',
        body:
          'Anderson（1982）证明：若已知前向 SDE 在每个时刻的 score function，可以写出时间反向的 SDE：\n\n' +
          '$\\mathrm{d}x = \\left[f(x,t) - g(t)^2 \\nabla_x \\log p_t(x)\\right]\\mathrm{d}t + g(t)\\,\\mathrm{d}\\bar{W}_t$\n\n' +
          '其中 $\\bar{W}_t$ 是反向时间的布朗运动，$\\mathrm{d}t < 0$（时间倒流）。\n\n' +
          '**解读漂移项**：\n\n' +
          '- $f(x,t)$：前向 SDE 的原漂移（被"取反"抵消）\n' +
          '- $-g(t)^2 \\nabla_x \\log p_t(x)$：score 项，把轨迹从噪声区域"导航"回数据高密度区域\n\n' +
          '**实践中**：真实 score 未知，用训练好的 $s_\\theta(x_t, t) \\approx \\nabla_x \\log p_t(x_t)$ 替换。',
        formula: '\\mathrm{d}x = \\bigl[f(x,t) - g(t)^2\\nabla_x\\log p_t(x)\\bigr]\\mathrm{d}t + g(t)\\,\\mathrm{d}\\bar{W}_t',
        tip: 'Score 系数 $g(t)^2$ 的物理意义：噪声越强（$g$ 大），反向时需要"更强力地"往高密度区拉，所以修正量也越大。',
      },
      // 卡 5：预测噪声等价于预测 score
      {
        title: '预测噪声 $\\varepsilon$ = 预测 Score（高斯加噪下等价）',
        body:
          '实践中，扩散模型通常不直接训练网络输出 $\\nabla_x\\log p_t(x)$，而是训练它**预测加入的噪声 $\\varepsilon$**。\n\n' +
          '两者为什么等价？在高斯加噪 $x_t = \\sqrt{\\bar{\\alpha}_t}\\,x_0 + \\sqrt{1-\\bar{\\alpha}_t}\\,\\varepsilon$ 下：\n\n' +
          '$\\nabla_x \\log p_t(x_t) \\approx -\\dfrac{\\varepsilon}{\\sqrt{1-\\bar{\\alpha}_t}}$\n\n' +
          '所以"预测噪声 $\\varepsilon$"和"预测 score"只差一个常数缩放因子，本质上是**同一件事**。\n\n' +
          '**训练目标（Denoising Score Matching）**：\n\n' +
          '$\\mathcal{L} = \\mathbb{E}_{t, x_0, \\varepsilon}\\!\\left[\\|\\varepsilon_\\theta(x_t, t) - \\varepsilon\\|^2\\right]$\n\n' +
          '这就是 DDPM 论文中的训练损失。神经网络输入噪声图像和时间步，输出预测噪声。',
        formula: '\\nabla_x \\log p_t(x_t) \\approx -\\varepsilon / \\sqrt{1-\\bar{\\alpha}_t}',
        tip: '结论：扩散模型"预测噪声"的训练目标，在数学上等价于学习 score function，只是表达更直观、训练更稳定。',
      },
      // 卡 6：概率流 ODE 与 DDIM
      {
        title: '概率流 ODE 与 DDIM：确定性去噪',
        body:
          '反向 SDE 还对应一条**确定性 ODE**（概率流 ODE）：\n\n' +
          '$\\mathrm{d}x = \\left[f(x,t) - \\dfrac{1}{2}g(t)^2 \\nabla_x \\log p_t(x)\\right]\\mathrm{d}t$\n\n' +
          '与反向 SDE 对比：随机项 $g(t)\\,\\mathrm{d}\\bar{W}_t$ 消失了，score 系数从 $g^2$ 变成 $g^2/2$。\n\n' +
          '**关键性质**：这条 ODE 与原 SDE 拥有**完全相同的边际分布** $p_t(x)$，但轨迹是确定性的（给定初始噪声，路径唯一）。\n\n' +
          '**DDIM 的本质**：DDIM（Denoising Diffusion Implicit Models）可以理解为对概率流 ODE 的数值积分。\n\n' +
          '- DDPM ≈ 随机采样（反向 SDE，含噪声）\n' +
          '- DDIM ≈ 确定性积分（概率流 ODE，无随机性）\n\n' +
          'DDIM 允许用更大的步长，20 步即可达到 DDPM 1000 步的质量，大幅加速生成。',
        formula: '\\mathrm{d}x = \\left[f(x,t) - \\frac{1}{2}g(t)^2\\nabla_x\\log p_t(x)\\right]\\mathrm{d}t',
        reveal: {
          q: '为什么 DDIM 能用比 DDPM 少得多的步数生成高质量样本？',
          a: 'DDIM 对应的是确定性概率流 ODE，无随机扰动，数值积分误差积累更慢，允许更大步长。DDPM 对应随机反向 SDE，每步都有随机扰动，步长过大会导致分布失真。ODE 求解器（如高阶 Runge-Kutta）也可直接用于 DDIM，进一步减少步数。',
        },
        tip: 'DDPM = 随机（有噪声，步数多）；DDIM = 确定（无噪声，步数少）。两者边际分布相同，轨迹统计不同。',
      },
    ],
    questions: [
      // Q1：judge — 前向过程方向
      {
        type: 'judge',
        prompt: '扩散模型的前向过程（加噪）将数据分布逐渐变成接近标准高斯噪声的分布。这个说法正确吗？',
        answer: true,
        explain:
          '正确。前向 SDE 设计为：随 $t$ 增大，数据分布被持续加噪，当 $\\bar{\\alpha}_T \\to 0$ 时，$q(x_T|x_0) \\approx \\mathcal{N}(0,I)$，即趋向标准高斯分布。这使生成模型能从简单的标准高斯出发，沿反向过程采样。',
      },
      // Q2：choice — score function 的定义
      {
        type: 'choice',
        prompt: 'Score function 的正式定义是？',
        options: [
          '$\\nabla_x p_t(x)$（概率密度的梯度）',
          '$\\nabla_x \\log p_t(x)$（对数概率密度的梯度）',
          '$\\nabla_t \\log p_t(x)$（对时间的偏导）',
          '$p_t(x) / \\nabla_x p_t(x)$',
        ],
        answer: 1,
        explain:
          'Score function 定义为 $\\nabla_x \\log p_t(x)$，是**对数概率密度**关于 $x$ 的梯度。注意不是 $p_t(x)$ 本身的梯度，取对数后更容易估计（不需要归一化常数），且指向对数密度上升方向。',
      },
      // Q3：input — DDPM 闭式均值系数
      {
        type: 'input',
        prompt: 'DDPM 前向闭式 $q(x_t|x_0) = \\mathcal{N}(\\sqrt{\\bar{\\alpha}_t}\\,x_0,\\,(1-\\bar{\\alpha}_t)I)$。当 $\\bar{\\alpha}_t = 0.25$ 时，均值系数 $\\sqrt{\\bar{\\alpha}_t}$ 等于多少？（输入小数）',
        accept: ['0.5', '0.50'],
        explain:
          '$\\sqrt{\\bar{\\alpha}_t} = \\sqrt{0.25} = 0.5$。这意味着当 $\\bar{\\alpha}_t = 0.25$ 时，$x_t$ 的均值是原始数据 $x_0$ 的 $0.5$ 倍，方差 $(1-\\bar{\\alpha}_t) = 0.75$，信噪比已经相当低了。',
      },
      // Q4：match — 前向/反向 SDE 术语配对
      {
        type: 'match',
        prompt: '将扩散模型相关概念与其含义配对：',
        left: ['前向 SDE', '反向 SDE', '概率流 ODE', 'Score function'],
        right: [
          '确定性路径，与反向 SDE 共享边际分布',
          '将数据逐步加噪变成高斯',
          '$\\nabla_x\\log p_t(x)$，指向高密度区方向',
          '从噪声去噪生成数据，含随机项',
        ],
        answer: [1, 3, 0, 2],
        explain:
          '前向 SDE 加噪；反向 SDE 含随机项 $g\\,\\mathrm{d}\\bar{W}$ 去噪；概率流 ODE 是对应的确定性版本，边际分布相同；score function 是对数密度梯度，指导反向过程方向。',
      },
      // Q5：choice — 反向 SDE 中 score 系数
      {
        type: 'choice',
        prompt: '反向 SDE $\\mathrm{d}x = [f - g^2 \\nabla_x\\log p_t]\\,\\mathrm{d}t + g\\,\\mathrm{d}\\bar{W}$ 中，score 项前面的系数 $g^2$ 说明了什么？',
        options: [
          '噪声越大，score 修正量越大，去噪"力度"越强',
          '噪声越大，score 修正量越小，去噪越保守',
          '系数 $g^2$ 与 score 方向无关，只影响步长',
          '$g^2$ 只在 $t=0$ 时才有意义',
        ],
        answer: 0,
        explain:
          '正确。$g(t)$ 是扩散项强度，噪声越大时 $g$ 越大，反向 SDE 中 score 项的系数 $g^2$ 也越大，即需要更强力地"往高密度区拉"来抵消更强的噪声。这保证了反向过程能正确逆转前向加噪。',
      },
      // Q6：judge — 预测噪声等价于预测 score
      {
        type: 'judge',
        prompt: '在高斯加噪设定下，训练神经网络预测加入的噪声 $\\varepsilon$ 与预测 score function $\\nabla_x\\log p_t(x)$ 在数学上是**等价的**（只差常数缩放）。这个说法正确吗？',
        answer: true,
        explain:
          '正确。在高斯加噪 $x_t = \\sqrt{\\bar{\\alpha}_t}x_0 + \\sqrt{1-\\bar{\\alpha}_t}\\varepsilon$ 下，$\\nabla_x\\log p_t(x_t) \\approx -\\varepsilon/\\sqrt{1-\\bar{\\alpha}_t}$，两者只差缩放因子 $-1/\\sqrt{1-\\bar{\\alpha}_t}$。因此 DDPM 的"预测噪声"训练目标在本质上就是学习 score function。',
      },
    ],
  },
]
