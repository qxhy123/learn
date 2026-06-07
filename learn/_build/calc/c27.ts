export const LESSONS = [
  // ─────────────────────────────────────────────────────────────────────────
  // 关 c27-a  二阶线性方程的结构与常系数齐次方程
  // ─────────────────────────────────────────────────────────────────────────
  {
    id: 'c27-a',
    title: '特征方程与齐次通解',
    subtitle: '方程结构·特征根三情形·通解公式',
    intro: [
      // 卡 1：为什么需要二阶方程？
      {
        title: '为什么需要二阶微分方程？',
        body:
          '弹簧振子的位移 $x(t)$，既受弹力（与位移成比例）的影响，也受阻尼力（与速度成比例）的影响，' +
          '牛顿第二定律给出 $m x^{\\prime\\prime} + c x^{\\prime} + k x = F(t)$——方程里出现了**二阶导数**。\n\n' +
          'RLC 串联电路同理：电感 $L$ 贡献二阶项 $L q^{\\prime\\prime}$，电阻贡献一阶项 $R q^{\\prime}$，电容贡献零阶项 $q/C$。\n\n' +
          '这类方程形如 $y^{\\prime\\prime} + P(x)y^{\\prime} + Q(x)y = f(x)$，称为**二阶线性微分方程**。' +
          '它的求解方法——特征方程法和待定系数法——是本章的核心。',
        tip: '遇到含加速度（$x^{\\prime\\prime}$）或二阶导数的物理/工程问题，几乎都会用到本章方法。',
      },
      // 卡 2：齐次与非齐次，通解结构
      {
        title: '齐次与非齐次：通解结构定理',
        body:
          '右端为零 $f(x)\\equiv0$，称为**齐次方程**；$f(x)\\not\\equiv0$ 称为**非齐次方程**。\n\n' +
          '**齐次通解结构**：若 $y_1,y_2$ 是齐次方程的两个**线性无关**解（即 $y_1/y_2\\neq$ 常数），则通解为\n\n' +
          '$y=C_1y_1+C_2y_2$，其中 $C_1,C_2$ 是任意常数。\n\n' +
          '**非齐次通解结构**：非齐次通解 $=$ 对应齐次方程的通解 $Y$ $+$ 非齐次方程的一个特解 $y^*$。\n\n' +
          '这个"通解 $=$ 齐次通解 $+$ 特解"的结构，是整章解题的骨架。\n\n' +
          '**易错点**：初值条件必须代入**完整通解**（含特解）后再求 $C_1,C_2$，不能只对齐次部分用初值。',
        formula: 'y = Y + y^* = C_1 y_1(x) + C_2 y_2(x) + y^*(x)',
      },
      // 卡 3：为什么设 y = e^{rx}？特征方程的由来
      {
        title: '核心思路：为什么设 $y = e^{rx}$？',
        body:
          '考虑常系数齐次方程 $y^{\\prime\\prime}+py^{\\prime}+qy=0$（$p,q$ 为常数）。\n\n' +
          '**动机**：$e^{rx}$ 求导后还是 $e^{rx}$ 的倍数，不改变"形状"——这让三项可以合并。\n\n' +
          '设 $y=e^{rx}$，则 $y^{\\prime}=re^{rx}$，$y^{\\prime\\prime}=r^2e^{rx}$，代入方程：\n\n' +
          '$r^2e^{rx}+pre^{rx}+qe^{rx}=0$\n\n' +
          '由于 $e^{rx}\\neq0$，两边除以 $e^{rx}$，得到关于 $r$ 的代数方程——这就是**特征方程**：\n\n' +
          '$r^2+pr+q=0$\n\n' +
          '解这个二次方程，得到**特征根** $r$，再由 $r$ 写出通解——求微分方程变成了解代数方程！',
        formula: 'r^2 + pr + q = 0',
        tip: '特征方程是原微分方程"去掉 $y$，把 $y^{\\prime\\prime}$ 换成 $r^2$，$y^{\\prime}$ 换成 $r$，$y$ 换成 $1$"得到的。',
      },
      // 卡 4：情形一——两个不同实根
      {
        title: '情形一：两个不同实根（$\\Delta > 0$）',
        body:
          '特征方程 $r^2+pr+q=0$ 的判别式 $\\Delta=p^2-4q>0$ 时，有两个不相等实根 $r_1\\neq r_2$。\n\n' +
          '**通解**：$y=C_1e^{r_1x}+C_2e^{r_2x}$\n\n' +
          '**完整例题**：求 $y^{\\prime\\prime}-5y^{\\prime}+6y=0$ 的通解。\n\n' +
          '写特征方程：$r^2-5r+6=0$，因式分解 $(r-2)(r-3)=0$，得 $r_1=2$，$r_2=3$。\n\n' +
          '$\\Delta=25-24=1>0$，符合情形一。\n\n' +
          '通解：$y=C_1e^{2x}+C_2e^{3x}$。\n\n' +
          '**物理含义**：若两根均为负（$r_1,r_2<0$），解随 $x\\to+\\infty$ 衰减到零；若有正根，解发散增长。',
        formula: 'y = C_1 e^{r_1 x} + C_2 e^{r_2 x}',
      },
      // 卡 5：情形二——重根
      {
        title: '情形二：重根（$\\Delta = 0$）——为什么要乘 $x$？',
        body:
          '$\\Delta=0$ 时，特征方程只有一个根 $r=r_1=r_2=-p/2$，此时 $e^{rx}$ 只能给出**一个**线性无关解，不够。\n\n' +
          '**为什么第二个解是 $xe^{rx}$？**直接代入验证：设 $y_2=xe^{rx}$，则\n\n' +
          '$y_2^{\\prime}=(1+rx)e^{rx}$，$y_2^{\\prime\\prime}=(2r+r^2x)e^{rx}$\n\n' +
          '代入 $y_2^{\\prime\\prime}+py_2^{\\prime}+qy_2$，利用 $r^2+pr+q=0$ 和 $2r+p=0$（重根条件），结果恰好等于 $0$ ✓\n\n' +
          '**通解**：$y=(C_1+C_2x)e^{rx}$\n\n' +
          '**例题**：$y^{\\prime\\prime}-4y^{\\prime}+4y=0$，特征方程 $(r-2)^2=0$，重根 $r=2$，通解 $y=(C_1+C_2x)e^{2x}$。\n\n' +
          '**易错点**：$\\Delta=0$ 时绝对不能写 $y=C_1e^{r_1x}+C_2e^{r_2x}$（因为 $r_1=r_2$，两项成比例，不是两个独立解）！',
        formula: 'y = (C_1 + C_2 x)e^{rx}',
        tip: '重根通解 $(C_1+C_2 x)e^{rx}$：多项式系数乘以指数，这是"临界阻尼"在物理中的对应。',
      },
      // 卡 6：情形三——复根
      {
        title: '情形三：复根（$\\Delta < 0$）——振荡解',
        body:
          '$\\Delta<0$ 时，特征根为共轭复数 $r_{1,2}=\\alpha\\pm\\beta i$，其中\n\n' +
          '$\\alpha=-p/2$，$\\beta=\\sqrt{4q-p^2}/2>0$\n\n' +
          '用 Euler 公式 $e^{i\\theta}=\\cos\\theta+i\\sin\\theta$ 化为实值解，两个线性无关的实值基础解为：\n\n' +
          '$y_1=e^{\\alpha x}\\cos\\beta x$，$y_2=e^{\\alpha x}\\sin\\beta x$\n\n' +
          '**通解**：$y=e^{\\alpha x}(C_1\\cos\\beta x+C_2\\sin\\beta x)$\n\n' +
          '**例题**：$y^{\\prime\\prime}+2y^{\\prime}+5y=0$，特征方程 $r^2+2r+5=0$，\n\n' +
          '$r=\\dfrac{-2\\pm\\sqrt{4-20}}{2}=-1\\pm2i$，即 $\\alpha=-1$，$\\beta=2$。\n\n' +
          '通解：$y=e^{-x}(C_1\\cos2x+C_2\\sin2x)$。\n\n' +
          '**物理含义**：$\\alpha<0$ 时解做衰减振荡（欠阻尼）；$\\alpha=0$ 时做等幅振荡（无阻尼）。\n\n' +
          '**易错点**：通解必须写成实数形式 $e^{\\alpha x}(C_1\\cos\\beta x+C_2\\sin\\beta x)$，不能写复指数——考试会判错。',
        formula: 'y = e^{\\alpha x}(C_1 \\cos \\beta x + C_2 \\sin \\beta x)',
      },
      // 卡 7：三种情形对比表 + 初值问题完整例 + reveal
      {
        title: '三情形速查与初值问题',
        body:
          '**三种情形速查表**：\n\n' +
          '- $\\Delta>0$（实异根 $r_1\\neq r_2$）：$y=C_1e^{r_1x}+C_2e^{r_2x}$\n\n' +
          '- $\\Delta=0$（重根 $r$）：$y=(C_1+C_2x)e^{rx}$\n\n' +
          '- $\\Delta<0$（复根 $\\alpha\\pm\\beta i$）：$y=e^{\\alpha x}(C_1\\cos\\beta x+C_2\\sin\\beta x)$\n\n' +
          '**初值问题例**：$y^{\\prime\\prime}+y=0$，$y(0)=1$，$y^{\\prime}(0)=0$。\n\n' +
          '特征方程 $r^2+1=0$，$r=\\pm i$（$\\alpha=0$，$\\beta=1$）。通解 $y=C_1\\cos x+C_2\\sin x$。\n\n' +
          '由 $y(0)=1$：$C_1=1$；$y^{\\prime}=-C_1\\sin x+C_2\\cos x$，$y^{\\prime}(0)=C_2=0$。\n\n' +
          '特解：$y=\\cos x$——这是最简单的简谐振动。',
        tip: '解题五步：①识别类型→②写特征方程→③解 $r$→④按三情形写通解→⑤代初值求 $C_1,C_2$。',
        reveal: {
          q: '方程 $y^{\\prime\\prime}+4y^{\\prime}+13y=0$ 的通解是什么形式？（写出 $\\alpha$ 和 $\\beta$ 的值）',
          a: '特征方程 $r^2+4r+13=0$，$r=\\dfrac{-4\\pm\\sqrt{16-52}}{2}=-2\\pm3i$，故 $\\alpha=-2$，$\\beta=3$，通解 $y=e^{-2x}(C_1\\cos3x+C_2\\sin3x)$。',
        },
      },
      // 卡 8：易错点汇总
      {
        title: '高频易错点总结',
        body:
          '**易错 1：重根忘乘 $x$。** $\\Delta=0$ 时第二个独立解是 $xe^{rx}$，不是 $e^{rx}$。漏掉 $x$ 导致"通解"只含一个任意常数，不是真正的通解。\n\n' +
          '**易错 2：复根写成复指数。** 通解必须用实值形式 $e^{\\alpha x}(C_1\\cos\\beta x+C_2\\sin\\beta x)$，不能写 $C_1e^{(\\alpha+\\beta i)x}+C_2e^{(\\alpha-\\beta i)x}$。\n\n' +
          '**易错 3：符号计算错误。** 方程 $y^{\\prime\\prime}-2y^{\\prime}+5y=0$ 的特征方程是 $r^2-2r+5=0$，$\\Delta=4-20=-16<0$，$r=1\\pm2i$——注意 $p=-2$ 代入 $-p/2=1$，$\\alpha=1>0$，解发散！\n\n' +
          '**易错 4：初值代入时机错。** 必须先写完整通解（含特解），再代初值。',
      },
    ],
    questions: [
      // 题 1：choice——识别特征方程形式
      {
        id: 'c27-a-q1',
        type: 'choice',
        prompt:
          '方程 $y^{\\prime\\prime}-3y^{\\prime}+2y=0$ 的特征方程是？',
        options: [
          '$r^2-3r+2=0$',
          '$r^2+3r+2=0$',
          '$r^2-3r-2=0$',
          '$r^2+3r-2=0$',
        ],
        answer: 0,
        explain:
          '常系数齐次方程 $y^{\\prime\\prime}+py^{\\prime}+qy=0$ 的特征方程为 $r^2+pr+q=0$。' +
          '这里 $p=-3$，$q=2$，特征方程为 $r^2-3r+2=0$。',
      },
      // 题 2：input——求特征根（整数）
      {
        id: 'c27-a-q2',
        type: 'input',
        prompt:
          '方程 $y^{\\prime\\prime}-5y^{\\prime}+6y=0$ 的特征方程为 $(r-2)(r-3)=0$，两个特征根之积等于多少？',
        accept: ['6'],
        explain:
          '特征根 $r_1=2$，$r_2=3$，乘积 $r_1\\cdot r_2=6$。这也恰好等于特征方程 $r^2-5r+6=0$ 中的常数项 $q=6$（韦达定理：根之积 $=q$）。',
      },
      // 题 3：judge——重根通解判断
      {
        id: 'c27-a-q3',
        type: 'judge',
        prompt:
          '方程 $y^{\\prime\\prime}-4y^{\\prime}+4y=0$ 的通解为 $y=C_1e^{2x}+C_2e^{2x}$（两项分别取两个任意常数）。',
        answer: false,
        explain:
          '特征方程 $(r-2)^2=0$，重根 $r=2$。两项 $C_1e^{2x}$ 与 $C_2e^{2x}$ 实质上是同一函数，不是两个线性无关解。' +
          '正确通解为 $y=(C_1+C_2x)e^{2x}$，第二个独立解必须乘以 $x$。',
      },
      // 题 4：choice——复根通解
      {
        id: 'c27-a-q4',
        type: 'choice',
        prompt:
          '方程 $y^{\\prime\\prime}+2y^{\\prime}+5y=0$ 的通解是？',
        options: [
          '$y=C_1e^{-x}\\cos2x+C_2e^{-x}\\sin2x$',
          '$y=C_1e^{x}\\cos2x+C_2e^{x}\\sin2x$',
          '$y=(C_1+C_2x)e^{-x}$',
          '$y=C_1e^{-x}+C_2e^{-5x}$',
        ],
        answer: 0,
        explain:
          '特征方程 $r^2+2r+5=0$，$\\Delta=4-20=-16<0$，$r=-1\\pm2i$（$\\alpha=-1$，$\\beta=2$）。' +
          '通解为 $y=e^{-x}(C_1\\cos2x+C_2\\sin2x)$，即选项 A。',
      },
      // 题 5：input——初值问题求系数
      {
        id: 'c27-a-q5',
        type: 'input',
        prompt:
          '初值问题 $y^{\\prime\\prime}+y=0$，$y(0)=0$，$y^{\\prime}(0)=3$。' +
          '通解为 $y=C_1\\cos x+C_2\\sin x$，代入初值后 $C_2$ 等于多少？',
        accept: ['3'],
        explain:
          '由 $y(0)=C_1=0$；$y^{\\prime}=-C_1\\sin x+C_2\\cos x$，$y^{\\prime}(0)=C_2=3$。特解为 $y=3\\sin x$。',
      },
      // 题 6：match——三情形与通解
      {
        id: 'c27-a-q6',
        type: 'match',
        prompt: '将判别式情形与对应的齐次通解形式对应。',
        left: ['$\\Delta>0$，实异根 $r_1\\neq r_2$', '$\\Delta=0$，重根 $r$', '$\\Delta<0$，复根 $\\alpha\\pm\\beta i$'],
        right: [
          '$y=C_1e^{r_1x}+C_2e^{r_2x}$',
          '$y=(C_1+C_2x)e^{rx}$',
          '$y=e^{\\alpha x}(C_1\\cos\\beta x+C_2\\sin\\beta x)$',
        ],
      },
      // 题 7：choice——判别式符号判断
      {
        id: 'c27-a-q7',
        type: 'choice',
        prompt:
          '方程 $y^{\\prime\\prime}+4y=0$ 的特征根是什么？',
        options: [
          '$r=\\pm2$（实异根）',
          '$r=2$（重根）',
          '$r=\\pm2i$（纯虚数根）',
          '$r=-2$（重根）',
        ],
        answer: 2,
        explain:
          '特征方程 $r^2+4=0$，$r^2=-4$，$r=\\pm2i$。' +
          '这是 $\\alpha=0$，$\\beta=2$ 的纯虚数根，通解为 $y=C_1\\cos2x+C_2\\sin2x$，表示等幅振荡。',
      },
      // 题 8：judge——复根通解实值性
      {
        id: 'c27-a-q8',
        type: 'judge',
        prompt:
          '复根 $r=\\alpha\\pm\\beta i$ 对应的通解 $y=e^{\\alpha x}(C_1\\cos\\beta x+C_2\\sin\\beta x)$ 是实值函数（当 $C_1,C_2$ 为实数时）。',
        answer: true,
        explain:
          '$e^{\\alpha x}$、$\\cos\\beta x$、$\\sin\\beta x$ 都是实值函数，$C_1,C_2$ 取实数时整个表达式为实值。' +
          '这正是为什么我们用 Euler 公式将复指数解化为实值形式——微分方程的实系数决定了实值通解的存在。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────────────
  // 关 c27-b  常系数非齐次方程——待定系数法
  // ─────────────────────────────────────────────────────────────────────────
  {
    id: 'c27-b',
    title: '待定系数法求特解',
    subtitle: '指数多项式型·三角函数型·共振·$k$ 值规则',
    intro: [
      // 卡 1：为什么用待定系数法？
      {
        title: '为什么可以"猜"特解？',
        body:
          '非齐次方程 $y^{\\prime\\prime}+py^{\\prime}+qy=f(x)$ 的通解 $=$ 齐次通解 $Y + $ 特解 $y^*$。' +
          '关键问题：如何求 $y^*$？\n\n' +
          '**待定系数法的思路**：当 $f(x)$ 具有"常系数线性方程能生产出来的形式"时，' +
          '猜测 $y^*$ 的函数形式（由 $f$ 决定），代入方程比较系数确定待定常数。\n\n' +
          '为什么猜得准？因为对 $e^{\\lambda x}P_m(x)$ 反复求导，形状不变——仍是 $e^{\\lambda x}$ 乘多项式。' +
          '代入方程后，左边每项都有 $e^{\\lambda x}$ 因子，约掉后变成多项式方程，可以逐项比较系数。\n\n' +
          '本节掌握两种 $f(x)$ 形式：① 指数乘多项式型；② 含三角函数型。',
        tip: '待定系数法不是"碰运气"，而是基于微分算子的严格理论——$f$ 的形式决定了 $y^*$ 必然属于同一函数族。',
      },
      // 卡 2：类型一——e^{λx}P_m(x)，k=0 的情形
      {
        title: '类型一：$f=e^{\\lambda x}P_m(x)$，$\\lambda$ 不是特征根（$k=0$）',
        body:
          '$f(x)=e^{\\lambda x}P_m(x)$ 型，其中 $P_m$ 是 $m$ 次多项式。\n\n' +
          '**规则**：若 $\\lambda$ **不是**特征根，设 $y^*=e^{\\lambda x}Q_m(x)$，其中 $Q_m$ 是 $m$ 次待定多项式。\n\n' +
          '**完整例题**：求 $y^{\\prime\\prime}-5y^{\\prime}+6y=e^{4x}$ 的特解。\n\n' +
          '特征方程 $r^2-5r+6=0$，根 $r_1=2,r_2=3$。$\\lambda=4$ 不是特征根，取 $k=0$。\n\n' +
          '设 $y^*=Ae^{4x}$（$m=0$，$Q_m=A$ 为常数）。\n\n' +
          '$(y^*)^{\\prime\\prime}=16Ae^{4x}$，$(y^*)^{\\prime}=4Ae^{4x}$，代入方程：\n\n' +
          '$16Ae^{4x}-20Ae^{4x}+6Ae^{4x}=e^{4x}$，得 $2A=1$，$A=1/2$。\n\n' +
          '特解 $y^*=\\dfrac{1}{2}e^{4x}$，通解 $y=C_1e^{2x}+C_2e^{3x}+\\dfrac{1}{2}e^{4x}$。',
        formula: 'y^* = e^{\\lambda x} Q_m(x) \\quad (\\lambda\\text{ 不是特征根})',
      },
      // 卡 3：类型一——λ 是单根（k=1）
      {
        title: '类型一：$\\lambda$ 是单特征根（$k=1$）——为什么要乘 $x$？',
        body:
          '若 $\\lambda$ 是**单特征根**（$\\ lambda=r_1\\neq r_2$），设 $y^*=xe^{\\lambda x}Q_m(x)$。\n\n' +
          '**原因**：若不乘 $x$，设 $y^*=Ae^{\\lambda x}$，代入方程后左边的 $e^{\\lambda x}$ 项被方程"吸收"（因为 $e^{\\lambda x}$ 是齐次方程的解），' +
          '导致右边无法满足——系数方程无解。乘以 $x$ 后就避免了这个冲突。\n\n' +
          '**例题**：$y^{\\prime\\prime}-3y^{\\prime}+2y=e^{2x}$。特征根 $r_1=1,r_2=2$，$\\lambda=2$ 是单根，$k=1$。\n\n' +
          '设 $y^*=Axe^{2x}$。$(Axe^{2x})^{\\prime}=A(1+2x)e^{2x}$，$(Axe^{2x})^{\\prime\\prime}=A(4+4x)e^{2x}$。\n\n' +
          '代入：$A(4+4x)e^{2x}-3A(1+2x)e^{2x}+2Axe^{2x}=e^{2x}$。\n\n' +
          '整理：$A\\cdot e^{2x}\\cdot(4+4x-3-6x+2x)=e^{2x}$，即 $Ae^{2x}=e^{2x}$，$A=1$。\n\n' +
          '特解 $y^*=xe^{2x}$。',
        formula: 'y^* = x e^{\\lambda x} Q_m(x) \\quad (\\lambda\\text{ 是单特征根})',
      },
      // 卡 4：类型一——λ 是重根（k=2）
      {
        title: '类型一：$\\lambda$ 是重特征根（$k=2$）',
        body:
          '若 $\\lambda$ 是**重特征根**（$r_1=r_2=\\lambda$），设 $y^*=x^2e^{\\lambda x}Q_m(x)$，取 $k=2$。\n\n' +
          '**$k$ 值规则总结**：\n\n' +
          '- $k=0$：$\\lambda$ 不是特征根\n\n' +
          '- $k=1$：$\\lambda$ 是单特征根\n\n' +
          '- $k=2$：$\\lambda$ 是重特征根\n\n' +
          '**例题**：$y^{\\prime\\prime}-2y^{\\prime}+y=xe^x$（第24章例题24.7）。特征方程 $(r-1)^2=0$，重根 $r=1$，$\\lambda=1$ 是重根，$k=2$，$m=1$。\n\n' +
          '设 $y^*=x^2(Ax+B)e^x=(Ax^3+Bx^2)e^x$。\n\n' +
          '逐步求导后代入，$x$ 的系数对比：$6A=1\\Rightarrow A=1/6$；常数项：$2B=0\\Rightarrow B=0$。\n\n' +
          '特解 $y^*=\\dfrac{1}{6}x^3e^x$。',
        formula: 'y^* = x^k e^{\\lambda x} Q_m(x), \\quad k=\\begin{cases}0 & \\lambda\\text{ 非特征根}\\\\1 & \\lambda\\text{ 单根}\\\\2 & \\lambda\\text{ 重根}\\end{cases}',
        tip: '$k$ 就是 $\\lambda$ 作为特征根的"重数"（不是特征根则重数为 0）。',
      },
      // 卡 5：类型二——含 cos/sin，不是特征根（k=0）
      {
        title: '类型二：含 $\\cos/\\sin$，必须同时设两项',
        body:
          '$f(x)=e^{\\alpha x}[P(x)\\cos\\beta x+Q(x)\\sin\\beta x]$ 型（$P,Q$ 为多项式，最高次 $m$）。\n\n' +
          '**规则**：若 $\\alpha+\\beta i$ 不是特征根，设\n\n' +
          '$y^*=e^{\\alpha x}[R_m(x)\\cos\\beta x+S_m(x)\\sin\\beta x]$\n\n' +
          '（即使 $f$ 只含 $\\cos$，特解也必须同时设 $\\cos$ 和 $\\sin$ 两项！）\n\n' +
          '**为什么必须同时设？** 代入方程后，$\\cos\\beta x$ 求导得 $\\sin\\beta x$，两者相互产生——只设一项方程一般无法成立。\n\n' +
          '**例题**：$y^{\\prime\\prime}+y=e^x\\cos x$，$\\alpha=1$，$\\beta=1$。特征根 $r=\\pm i$，$1+i$ 不是特征根，取 $k=0$。\n\n' +
          '设 $y^*=e^x(A\\cos x+B\\sin x)$，代入后化简：\n\n' +
          '$e^x(2B\\cos x-2A\\sin x)=e^x\\cos x$，故 $2B=1$，$-2A=0$，$A=0$，$B=1/2$。\n\n' +
          '特解 $y^*=\\dfrac{e^x\\sin x}{2}$。',
        formula: 'y^* = e^{\\alpha x}[R_m(x)\\cos\\beta x + S_m(x)\\sin\\beta x]',
      },
      // 卡 6：类型二——共振（k=1）
      {
        title: '类型二：共振！$\\alpha+\\beta i$ 是特征根（$k=1$）',
        body:
          '若 $\\alpha+\\beta i$ **是**特征根，则特解须额外乘 $x$（$k=1$）：\n\n' +
          '$y^*=xe^{\\alpha x}[R_m(x)\\cos\\beta x+S_m(x)\\sin\\beta x]$\n\n' +
          '**振幅随 $x$ 线性增长——这就是"共振"！**\n\n' +
          '**例题**：$y^{\\prime\\prime}+y=\\cos x$，$\\alpha=0$，$\\beta=1$。特征根 $r=\\pm i$，$0+i=i$ **是**特征根，$k=1$。\n\n' +
          '设 $y^*=x(A\\cos x+B\\sin x)$。\n\n' +
          '$(y^*)^{\\prime\\prime}=-2A\\sin x+2B\\cos x-x(A\\cos x+B\\sin x)$\n\n' +
          '代入 $y^{\\prime\\prime}+y$（$-x$ 项与 $+xy$ 相消）：$-2A\\sin x+2B\\cos x=\\cos x$\n\n' +
          '$-2A=0$，$2B=1$，$A=0$，$B=1/2$。特解 $y^*=\\dfrac{x}{2}\\sin x$。\n\n' +
          '**共振特征**：$f=\\cos x$ 的频率恰等于系统固有频率，特解含 $x\\sin x$，振幅 $\\propto x$，随时间线性增大。',
        formula: 'y^* = x e^{\\alpha x}[R_m(x)\\cos\\beta x + S_m(x)\\sin\\beta x] \\quad (\\alpha+\\beta i\\text{ 是特征根})',
        tip: '记忆口诀：特解 $k$ 值 $=$ "$\\lambda$（或 $\\alpha+\\beta i$）作为特征根的重数"，不是就取 0。',
      },
      // 卡 7：完整三步解题流程 + reveal
      {
        title: '完整解题流程：三步法',
        body:
          '**第一步：解齐次方程**，得到齐次通解 $Y=C_1y_1+C_2y_2$。\n\n' +
          '**第二步：用待定系数法求特解 $y^*$**\n\n' +
          '（1）识别 $f(x)$ 类型：指数多项式型还是三角函数型？\n\n' +
          '（2）确定 $\\lambda$（或 $\\alpha+\\beta i$）是否为特征根，决定 $k$ 值。\n\n' +
          '（3）写出特解模板，代入方程，比较系数解待定参数。\n\n' +
          '**第三步：写通解**：$y=Y+y^*=C_1y_1+C_2y_2+y^*$。若有初值则代入求 $C_1,C_2$。\n\n' +
          '**综合例题**（引入题）：$y^{\\prime\\prime}-5y^{\\prime}+6y=e^{4x}$\n\n' +
          '第一步：特征方程 $(r-2)(r-3)=0$，$r_1=2,r_2=3$，$Y=C_1e^{2x}+C_2e^{3x}$。\n\n' +
          '第二步：$\\lambda=4$ 不是特征根，$k=0$，设 $y^*=Ae^{4x}$，代入得 $A=1/2$。\n\n' +
          '第三步：通解 $y=C_1e^{2x}+C_2e^{3x}+\\dfrac{1}{2}e^{4x}$。',
        reveal: {
          q: '方程 $y^{\\prime\\prime}+9y=2\\cos3x$ 的特解应设为什么形式？为什么？',
          a: '特征方程 $r^2+9=0$，$r=\\pm3i$，故 $0+3i$ 是特征根，$k=1$。$f$ 含 $\\cos3x$（$\\alpha=0,\\beta=3,m=0$），设 $y^*=x(A\\cos3x+B\\sin3x)$（必须同时设两项且乘 $x$）。',
        },
      },
      // 卡 8：待定系数法易错点
      {
        title: '待定系数法——三大易错点',
        body:
          '**易错 1：只设 $\\cos$ 不设 $\\sin$（或反之）。**\n\n' +
          '即使 $f(x)=\\sin\\beta x$（没有 $\\cos$ 项），特解也必须设 $y^*=A\\cos\\beta x+B\\sin\\beta x$，缺一不可。\n\n' +
          '**易错 2：忘记判断 $\\lambda$ 是否为特征根。**\n\n' +
          '最常见的错误：看到 $f=e^{2x}$，不查特征根，直接设 $y^*=Ae^{2x}$——但若 $r=2$ 恰是特征根，代入后 $A$ 解不出来（系数方程 $0\\cdot A=1$ 无解）。必须乘 $x$ 变为 $y^*=Axe^{2x}$。\n\n' +
          '**易错 3：多项式阶次设低了。**\n\n' +
          '若 $f(x)=e^{\\lambda x}(2x^2+1)$，则 $P_m$ 是 $m=2$ 次多项式，特解 $y^*=e^{\\lambda x}(Ax^2+Bx+C)$，不能只设 $y^*=Ae^{\\lambda x}$。',
      },
    ],
    questions: [
      // 题 1：choice——确定 k 值
      {
        id: 'c27-b-q1',
        type: 'choice',
        prompt:
          '方程 $y^{\\prime\\prime}-3y^{\\prime}+2y=e^{x}$ 中，$\\lambda=1$，特征根为 $r_1=1,r_2=2$。' +
          '特解应设为哪种形式？',
        options: [
          '$y^*=Ae^{x}$',
          '$y^*=Axe^{x}$',
          '$y^*=Ax^2e^{x}$',
          '$y^*=(Ax+B)e^{x}$',
        ],
        answer: 1,
        explain:
          '$\\lambda=1$ 是特征根 $r_1=1$（单根），取 $k=1$，特解设为 $y^*=Axe^{x}$。' +
          '若取 $k=0$ 设 $Ae^{x}$，因为 $e^{x}$ 是齐次方程的解，代入方程左边变为 $0$，无法等于右边 $e^{x}$。',
      },
      // 题 2：input——求待定系数 A
      {
        id: 'c27-b-q2',
        type: 'input',
        prompt:
          '求方程 $y^{\\prime\\prime}-5y^{\\prime}+6y=e^{4x}$ 的特解中的系数 $A$（设 $y^*=Ae^{4x}$）。' +
          '代入方程化简后得 $2A=1$，$A$ 等于多少（分数请输入小数，如 $0.5$）？',
        accept: ['0.5', '1/2'],
        explain:
          '代入：$(16-20+6)Ae^{4x}=e^{4x}$，即 $2A=1$，$A=0.5$，特解 $y^*=0.5e^{4x}$。',
      },
      // 题 3：judge——cos/sin 必须同时设
      {
        id: 'c27-b-q3',
        type: 'judge',
        prompt:
          '方程 $y^{\\prime\\prime}+y=\\cos x$ 中，由于 $f(x)$ 只含 $\\cos x$，特解可以只设 $y^*=A\\cos x$，不需要 $\\sin x$ 项。',
        answer: false,
        explain:
          '$\\alpha+\\beta i=0+i=i$ 是特征根（$r=\\pm i$），且 $f$ 含 $\\cos x$，特解必须设为 $y^*=x(A\\cos x+B\\sin x)$——同时含两项且乘以 $x$。' +
          '只设 $A\\cos x$ 无法使方程成立（代入后 $\\cos$ 与 $\\sin$ 交叉项不能同时消去）。',
      },
      // 题 4：choice——选正确特解模板
      {
        id: 'c27-b-q4',
        type: 'choice',
        prompt:
          '方程 $y^{\\prime\\prime}+4y=3\\sin2x$，特征根 $r=\\pm2i$，$\\alpha+\\beta i=2i$ 是特征根。' +
          '正确的特解设法是？',
        options: [
          '$y^*=A\\sin2x$',
          '$y^*=A\\cos2x+B\\sin2x$',
          '$y^*=x(A\\cos2x+B\\sin2x)$',
          '$y^*=x^2(A\\cos2x+B\\sin2x)$',
        ],
        answer: 2,
        explain:
          '$\\alpha+\\beta i=2i$ 是单特征根，$k=1$，且 $f$ 含 $\\sin2x$，必须同时设 $\\cos$ 和 $\\sin$ 两项。' +
          '正确形式为 $y^*=x(A\\cos2x+B\\sin2x)$。',
      },
      // 题 5：input——共振特解系数
      {
        id: 'c27-b-q5',
        type: 'input',
        prompt:
          '方程 $y^{\\prime\\prime}+y=\\cos x$ 的特解为 $y^*=x(A\\cos x+B\\sin x)$。' +
          '代入化简后得 $2B\\cos x-2A\\sin x=\\cos x$，$B$ 等于多少（输入分数形式，如 $0.5$）？',
        accept: ['0.5', '1/2'],
        explain:
          '比较 $\\cos x$ 系数：$2B=1$，$B=1/2$；比较 $\\sin x$ 系数：$-2A=0$，$A=0$。' +
          '特解 $y^*=\\dfrac{x}{2}\\sin x$，这是典型的共振解——振幅随 $x$ 线性增长。',
      },
      // 题 6：match——f(x) 类型与特解模板
      {
        id: 'c27-b-q6',
        type: 'match',
        prompt:
          '将 $f(x)$ 的形式与待定系数法特解的模板（设 $\\lambda$ 不是特征根，$\\alpha+\\beta i$ 不是特征根）对应。',
        left: [
          '$f=3e^{2x}$（$\\lambda=2$，非特征根）',
          '$f=e^x(x+1)$（$\\lambda=1$，非特征根）',
          '$f=\\sin3x$（$\\alpha=0,\\beta=3$，非特征根）',
        ],
        right: [
          '$y^*=Ae^{2x}$',
          '$y^*=e^x(Ax+B)$',
          '$y^*=A\\cos3x+B\\sin3x$',
        ],
      },
      // 题 7：choice——完整通解
      {
        id: 'c27-b-q7',
        type: 'choice',
        prompt:
          '方程 $y^{\\prime\\prime}-2y^{\\prime}-3y=e^{4x}$ 的通解是？（特征根 $r_1=3,r_2=-1$，$\\lambda=4$ 不是特征根，代入得 $5A=1$）',
        options: [
          '$y=C_1e^{3x}+C_2e^{-x}+\\dfrac{1}{5}e^{4x}$',
          '$y=C_1e^{3x}+C_2e^{-x}+e^{4x}$',
          '$y=C_1e^{3x}+C_2e^{-x}+5e^{4x}$',
          '$y=C_1e^{x}+C_2e^{-3x}+\\dfrac{1}{5}e^{4x}$',
        ],
        answer: 0,
        explain:
          '特征根 $r_1=3,r_2=-1$，齐次通解 $Y=C_1e^{3x}+C_2e^{-x}$。' +
          '$\\lambda=4$ 不是特征根，设 $y^*=Ae^{4x}$，代入：$16A-8A-3A=e^{4x}$ 的系数，$5A=1$，$A=1/5$。' +
          '通解 $y=C_1e^{3x}+C_2e^{-x}+\\dfrac{1}{5}e^{4x}$。',
      },
      // 题 8：judge——k=2 的判断
      {
        id: 'c27-b-q8',
        type: 'judge',
        prompt:
          '若 $y^{\\prime\\prime}-2y^{\\prime}+y=e^x$ 的特征方程有重根 $r=1$，则 $\\lambda=1$ 是重特征根，特解应设为 $y^*=Ax^2e^x$（取 $k=2$）。',
        answer: true,
        explain:
          '特征方程 $(r-1)^2=0$，重根 $r=1$，$\\lambda=1$ 是二重特征根，重数为 2，取 $k=2$，$m=0$（$f=e^x$ 中 $P_m=1$ 是零次多项式）。' +
          '特解设为 $y^*=Ax^2e^x$，代入方程后可解出 $A=1/2$，特解 $y^*=\\dfrac{x^2}{2}e^x$。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────────────
  // 关 c27-c  初值问题综合与物理应用
  // ─────────────────────────────────────────────────────────────────────────
  {
    id: 'c27-c',
    title: '初值问题与物理应用',
    subtitle: '完整解题流程·弹簧振动·RLC电路',
    intro: [
      // 卡 1：初值问题解题流程
      {
        title: '初值问题的标准解题流程',
        body:
          '给定二阶常系数线性方程加上初值条件 $y(x_0)=y_0$，$y^{\\prime}(x_0)=v_0$，求满足这两个条件的唯一特解。\n\n' +
          '**完整五步流程**：\n\n' +
          '① 写特征方程，判断 $\\Delta$ 的正负零，求特征根。\n\n' +
          '② 按三情形写齐次通解 $Y$。\n\n' +
          '③ 若非齐次：用待定系数法求特解 $y^*$，写完整通解 $y=Y+y^*$。\n\n' +
          '④ 对完整通解求导，得 $y^{\\prime}$。\n\n' +
          '⑤ 将 $x=x_0$ 代入 $y$ 和 $y^{\\prime}$，列两个方程求 $C_1,C_2$。\n\n' +
          '**易错点**：必须等到写完 $y=Y+y^*$ 的完整通解后，再求导再代初值——不能只对 $Y$ 部分用初值！',
      },
      // 卡 2：完整初值问题例题（复根）
      {
        title: '完整例题：复根初值问题',
        body:
          '**求解**：$y^{\\prime\\prime}+4y^{\\prime}+13y=0$，$y(0)=0$，$y^{\\prime}(0)=6$。\n\n' +
          '**第一步**：特征方程 $r^2+4r+13=0$，$\\Delta=16-52=-36<0$，$r=-2\\pm3i$（$\\alpha=-2,\\beta=3$）。\n\n' +
          '**第二步**：通解 $y=e^{-2x}(C_1\\cos3x+C_2\\sin3x)$。\n\n' +
          '**第三步**：代 $y(0)=0$：$C_1=0$。化简 $y=C_2e^{-2x}\\sin3x$。\n\n' +
          '**第四步**：求导 $y^{\\prime}=C_2e^{-2x}(-2\\sin3x+3\\cos3x)$。\n\n' +
          '**第五步**：代 $y^{\\prime}(0)=6$：$3C_2=6$，$C_2=2$。\n\n' +
          '**答案**：$y=2e^{-2x}\\sin3x$。物理意义：衰减振荡，频率 $\\beta=3$，衰减率 $|\\alpha|=2$。',
      },
      // 卡 3：弹簧振动建模
      {
        title: '弹簧振动：从物理到方程',
        body:
          '设质量为 $m$ 的物体挂在弹性系数为 $k$ 的弹簧下，$x(t)$ 为相对平衡位置的位移（向下为正）。\n\n' +
          '**无阻尼**（自由振动）：弹力 $-kx$（回复力），牛顿第二定律 $mx^{\\prime\\prime}=-kx$，即\n\n' +
          '$x^{\\prime\\prime}+\\omega^2x=0$，$\\omega=\\sqrt{k/m}$（固有频率）\n\n' +
          '**有阻尼**：阻力 $-cx^{\\prime}$（与速度成比例），得\n\n' +
          '$mx^{\\prime\\prime}+cx^{\\prime}+kx=0$\n\n' +
          '**受迫振动**：外力 $F_0\\cos\\omega_0 t$，得\n\n' +
          '$mx^{\\prime\\prime}+cx^{\\prime}+kx=F_0\\cos\\omega_0 t$\n\n' +
          '当外力频率 $\\omega_0$ 等于固有频率 $\\omega$ 时，发生**共振**，特解中含 $t\\sin\\omega t$，振幅无限增长。',
        formula: 'x^{\\prime\\prime} + \\omega^2 x = 0, \\quad \\omega = \\sqrt{\\tfrac{k}{m}}',
        tip: '弹簧方程特征根 $r=\\pm\\omega i$（纯虚数），对应无阻尼等幅振荡——这正是"简谐振动"。',
      },
      // 卡 4：弹簧振动完整例题
      {
        title: '弹簧振动例题：求运动规律',
        body:
          '**题目**：弹簧振子，$m=1$ kg，$k=4$ N/m，无阻尼。初始位置在平衡位置下方 $0.1$ m 处静止释放。求 $x(t)$。\n\n' +
          '**方程**：$x^{\\prime\\prime}+4x=0$，初值 $x(0)=0.1$，$x^{\\prime}(0)=0$（静止释放）。\n\n' +
          '**特征方程**：$r^2+4=0$，$r=\\pm2i$（$\\omega=2$ rad/s）。\n\n' +
          '**通解**：$x=C_1\\cos2t+C_2\\sin2t$。\n\n' +
          '由 $x(0)=0.1$：$C_1=0.1$；$x^{\\prime}=-2C_1\\sin2t+2C_2\\cos2t$，$x^{\\prime}(0)=2C_2=0$，$C_2=0$。\n\n' +
          '**答案**：$x=0.1\\cos2t$（单位：m），周期 $T=\\pi$ s $\\approx3.14$ s。\n\n' +
          '物体以 0.1 m 为幅度做简谐振动，固有频率 $\\omega=2$ rad/s。',
      },
      // 卡 5：RLC 电路建模
      {
        title: 'RLC 串联电路：与弹簧的完美对比',
        body:
          '串联 RLC 电路中，$q(t)$ 为电容上的电荷，$i=q^{\\prime}$ 为电流。Kirchhoff 电压定律给出：\n\n' +
          '$L q^{\\prime\\prime}+Rq^{\\prime}+\\dfrac{1}{C}q=E(t)$\n\n' +
          '**对比表**：\n\n' +
          '| 弹簧 | RLC |\n' +
          '|------|-----|\n' +
          '| 质量 $m$ | 电感 $L$ |\n' +
          '| 阻尼系数 $c$ | 电阻 $R$ |\n' +
          '| 弹性系数 $k$ | 倒容 $1/C$ |\n' +
          '| 外力 $F(t)$ | 电压 $E(t)$ |\n\n' +
          '数学结构完全相同！弹簧方程的所有解法都可直接用于 RLC 电路。',
        formula: 'L q^{\\prime\\prime} + R q^{\\prime} + \\frac{1}{C}q = E(t)',
      },
      // 卡 6：RLC 完整例题
      {
        title: 'RLC 例题：复根加常数特解',
        body:
          '**题目**：$L=1$ H，$R=2$ $\\Omega$，$C=0.5$ F，$E=10$ V（直流），$q(0)=0$，$i(0)=0$。求 $q(t)$。\n\n' +
          '**方程**：$q^{\\prime\\prime}+2q^{\\prime}+2q=10$。\n\n' +
          '**齐次**：特征方程 $r^2+2r+2=0$，$r=-1\\pm i$，$Q_h=e^{-t}(C_1\\cos t+C_2\\sin t)$。\n\n' +
          '**特解**：$f(x)=10$（常数，即 $e^{0\\cdot t}\\cdot10$，$\\lambda=0$），$0$ 不是特征根，设 $q^*=A$，代入得 $2A=10$，$A=5$。\n\n' +
          '**完整通解**：$q=e^{-t}(C_1\\cos t+C_2\\sin t)+5$。\n\n' +
          '由 $q(0)=0$：$C_1+5=0$，$C_1=-5$。\n\n' +
          '$q^{\\prime}=e^{-t}[(-C_1+C_2)\\cos t+(-C_1-C_2)\\sin t]$，由 $q^{\\prime}(0)=0$：$-C_1+C_2=0$，$C_2=-5$。\n\n' +
          '**答案**：$q(t)=5[1-e^{-t}(\\cos t+\\sin t)]$ C，随 $t\\to\\infty$，$q\\to5$ C（稳态充电至 $E/(1/C)=5$ C）。',
      },
      // 卡 7：阻尼类型与物理含义 + reveal
      {
        title: '阻尼类型：特征根 $\\alpha$ 的物理意义',
        body:
          '有阻尼弹簧方程 $mx^{\\prime\\prime}+cx^{\\prime}+kx=0$，特征根实部 $\\alpha=-c/(2m)$。\n\n' +
          '- **$\\Delta<0$（欠阻尼）**：$\\alpha<0$，$r=\\alpha\\pm\\beta i$，解 $=$ 衰减振荡 $e^{\\alpha t}\\cos(\\beta t+\\phi)$。\n\n' +
          '- **$\\Delta=0$（临界阻尼）**：$\\alpha<0$，重根 $r=\\alpha$，解 $=(C_1+C_2t)e^{\\alpha t}$，最快衰减到零，无振荡。\n\n' +
          '- **$\\Delta>0$（过阻尼）**：$\\alpha<0$，两个不同负实根，解 $=$ 纯指数衰减，不振荡，慢于临界阻尼。\n\n' +
          '**共振**：$c=0$（无阻尼），受外力 $F_0\\cos\\omega t$（频率等于固有频率 $\\omega$），特解含 $t\\sin\\omega t$，振幅随时间线性增大——这是桥梁、建筑设计必须避免的场景。',
        tip: '欠阻尼：弹簧反复弹跳最终静止；临界阻尼：最快停下，不反弹——汽车减震器的设计目标。',
        reveal: {
          q: '方程 $x^{\\prime\\prime}+4x=\\cos2t$ 的物理情景是什么？特解应设为何种形式？',
          a: '无阻尼弹簧（$\\omega=2$）受频率相同的外力——发生共振！特征根 $r=\\pm2i$，$\\alpha+\\beta i=2i$ 是特征根，$k=1$。' +
            '设 $x^*=t(A\\cos2t+B\\sin2t)$，特解含 $t\\sin2t$，振幅随 $t$ 线性增长。',
        },
      },
      // 卡 8：综合易错点与方法选择
      {
        title: '综合易错点与方法选择建议',
        body:
          '**使用待定系数法的前提**：$f(x)$ 必须是 $e^{\\lambda x}P_m(x)$ 或 $e^{\\alpha x}[P\\cos\\beta x+Q\\sin\\beta x]$ 型。\n\n' +
          '若 $f(x)=\\tan x$、$\\ln x$、$1/\\cos x$ 等，待定系数法失效，需用**常数变易法**（见教材 §24.3.4）。\n\n' +
          '**初值代入四步提醒**：\n\n' +
          '① 写出含 $C_1,C_2$ 的完整通解（含 $y^*$）；\n\n' +
          '② 对通解求导得 $y^{\\prime}$；\n\n' +
          '③ 分别代 $x=x_0$，得关于 $C_1,C_2$ 的两个方程；\n\n' +
          '④ 解方程组，写出最终特解。\n\n' +
          '**验算**：将特解代回原方程和初值条件，全部满足才算正确。',
      },
    ],
    questions: [
      // 题 1：choice——弹簧方程识别
      {
        id: 'c27-c-q1',
        type: 'choice',
        prompt:
          '弹簧振子 $m=1$ kg，$k=9$ N/m，无阻尼。方程 $x^{\\prime\\prime}+9x=0$ 的固有频率 $\\omega$ 等于多少（rad/s）？',
        options: ['$\\omega=1$', '$\\omega=3$', '$\\omega=9$', '$\\omega=\\sqrt{3}$'],
        answer: 1,
        explain:
          '$\\omega=\\sqrt{k/m}=\\sqrt{9/1}=3$ rad/s。特征根 $r=\\pm3i$，通解 $x=C_1\\cos3t+C_2\\sin3t$，周期 $T=2\\pi/3$ s。',
      },
      // 题 2：input——初值问题求 C_2
      {
        id: 'c27-c-q2',
        type: 'input',
        prompt:
          '初值问题：$y^{\\prime\\prime}+4y^{\\prime}+13y=0$，$y(0)=0$，$y^{\\prime}(0)=6$。' +
          '通解 $y=e^{-2x}(C_1\\cos3x+C_2\\sin3x)$，代 $y(0)=0$ 得 $C_1=0$，' +
          '再对 $y=C_2e^{-2x}\\sin3x$ 求导代 $y^{\\prime}(0)=6$，$C_2$ 等于多少？',
        accept: ['2'],
        explain:
          '$y^{\\prime}=C_2e^{-2x}(-2\\sin3x+3\\cos3x)$，代 $x=0$：$3C_2=6$，$C_2=2$。特解 $y=2e^{-2x}\\sin3x$。',
      },
      // 题 3：judge——共振判断
      {
        id: 'c27-c-q3',
        type: 'judge',
        prompt:
          '无阻尼弹簧方程 $x^{\\prime\\prime}+4x=\\cos2t$ 中，外力频率（$\\omega_0=2$）等于固有频率（$\\omega=2$），因此会发生共振，特解含 $t\\sin2t$ 项。',
        answer: true,
        explain:
          '特征根 $r=\\pm2i$，$\\alpha+\\beta i=0+2i=2i$ 是特征根，$k=1$，特解设 $x^*=t(A\\cos2t+B\\sin2t)$。' +
          '共振时振幅随 $t$ 线性增长，工程中须严格避免。',
      },
      // 题 4：choice——RLC 方程特征根类型
      {
        id: 'c27-c-q4',
        type: 'choice',
        prompt:
          'RLC 串联电路，$L=1$ H，$R=2$ $\\Omega$，$C=0.5$ F，方程 $q^{\\prime\\prime}+2q^{\\prime}+2q=E(t)$。' +
          '齐次方程特征根 $r=-1\\pm i$ 的物理情景是？',
        options: [
          '过阻尼（无振荡，两个不同负实根）',
          '临界阻尼（重负实根）',
          '欠阻尼（衰减振荡，复根实部 $<0$）',
          '无阻尼（纯虚数根）',
        ],
        answer: 2,
        explain:
          '$r=-1\\pm i$，实部 $\\alpha=-1<0$，$\\beta=1\\neq0$，是复数根，对应欠阻尼——电路中电荷做衰减振荡趋向稳态值。',
      },
      // 题 5：input——RLC 稳态电荷
      {
        id: 'c27-c-q5',
        type: 'input',
        prompt:
          'RLC 例题中（$L=1,R=2,C=0.5,E=10$），特解（稳态项）$q^*=A$ 为常数，代入 $q^{\\prime\\prime}+2q^{\\prime}+2q=10$ 得 $A$ 等于多少？',
        accept: ['5'],
        explain:
          '设 $q^*=A$，则 $q^{*\\prime\\prime}=0$，$q^{*\\prime}=0$，代入：$0+0+2A=10$，$A=5$ C。' +
          '这是电容充满后的稳态电荷 $q_{\\infty}=CE=0.5\\times10=5$ C，与物理结果一致。',
      },
      // 题 6：match——阻尼类型与特征根形式
      {
        id: 'c27-c-q6',
        type: 'match',
        prompt: '将弹簧振子的阻尼类型与对应的特征根形式对应。',
        left: ['欠阻尼（$\\Delta<0$）', '临界阻尼（$\\Delta=0$）', '过阻尼（$\\Delta>0$）'],
        right: [
          '复根 $\\alpha\\pm\\beta i$（$\\alpha<0$），衰减振荡',
          '重负实根 $r=\\alpha$，最快衰减无振荡',
          '两个不同负实根 $r_1\\neq r_2$，纯指数衰减',
        ],
      },
      // 题 7：choice——非齐次完整通解
      {
        id: 'c27-c-q7',
        type: 'choice',
        prompt:
          '方程 $y^{\\prime\\prime}+y=e^x$（特征根 $r=\\pm i$，$\\lambda=1$ 不是特征根）的通解为？',
        options: [
          '$y=C_1\\cos x+C_2\\sin x+\\dfrac{e^x}{2}$',
          '$y=C_1\\cos x+C_2\\sin x+e^x$',
          '$y=C_1\\cos x+C_2\\sin x+\\dfrac{xe^x}{2}$',
          '$y=C_1e^x+C_2e^{-x}+\\dfrac{e^x}{2}$',
        ],
        answer: 0,
        explain:
          '$\\lambda=1$ 不是特征根（$r=\\pm i$），$k=0$，设 $y^*=Ae^x$，代入 $Ae^x+Ae^x=e^x$，$2A=1$，$A=1/2$。' +
          '通解 $y=C_1\\cos x+C_2\\sin x+\\dfrac{1}{2}e^x$。',
      },
      // 题 8：judge——完整通解后再代初值
      {
        id: 'c27-c-q8',
        type: 'judge',
        prompt:
          '求解非齐次初值问题时，可以先对齐次通解 $Y=C_1y_1+C_2y_2$ 代初值求出 $C_1,C_2$，再加上特解 $y^*$。',
        answer: false,
        explain:
          '这是常见错误。必须先写出完整通解 $y=C_1y_1+C_2y_2+y^*$，然后对完整通解求导，再代初值。' +
          '若仅对 $Y$ 代初值，所求的 $C_1,C_2$ 不满足原非齐次方程的初值条件，导致结果错误。',
      },
    ],
  },
];
