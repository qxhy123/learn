import type { Unit } from '../../src/types'

export const UNIT: Unit = {
  id: 'u8',
  title: '进阶专题',
  color: '#ffc800',
  icon: '🔬',
  blurb: '奇异值分解（SVD）把任何矩阵拆成旋转-缩放-旋转；二次型用对称矩阵刻画曲面形状与正定性；矩阵微积分建立对向量、矩阵求导的完整记法，是反向传播的数学核心。',
  lessons: [
    // ─────────────────────────────────────────────────
    // u8-l1  奇异值与几何
    // ─────────────────────────────────────────────────
    {
      id: 'u8-l1',
      title: '奇异值与几何',
      subtitle: '任何矩阵都能拆成"旋转 × 缩放 × 旋转"',
      intro: [
        {
          title: '动机：为什么需要 SVD？',
          body: '特征值分解要求矩阵是**方阵且可对角化**，现实中的矩阵（图像、评分表、权重矩阵）往往是 $m \\times n$ 的矩形阵，无法直接特征分解。\n\n**SVD** 把这个限制彻底去掉：对**任意**实矩阵 $A \\in \\mathbb{R}^{m \\times n}$，不论方不方、秩多高，都存在分解 $A = U\\Sigma V^{\\top}$。\n\n直觉上，任何线性变换都可以看作：先在输入空间做一次旋转，再沿坐标轴做拉伸/压缩，最后在输出空间再做一次旋转。这三步分别对应 $V^{\\top}$、$\\Sigma$、$U$。',
          tip: 'SVD 是谱定理（对称矩阵正交对角化）对任意矩阵的推广，是线性代数最重要的工具之一。',
        },
        {
          title: 'SVD 定理',
          body: '**SVD 定理**：任意 $A \\in \\mathbb{R}^{m \\times n}$，存在正交矩阵 $U \\in \\mathbb{R}^{m \\times m}$、$V \\in \\mathbb{R}^{n \\times n}$ 和广义对角矩阵 $\\Sigma \\in \\mathbb{R}^{m \\times n}$，使得\n\n$A = U\\Sigma V^{\\top}$\n\n- $U$ 的列 $\\mathbf{u}_1, \\ldots, \\mathbf{u}_m$：**左奇异向量**，构成 $\\mathbb{R}^m$ 的标准正交基\n- $V$ 的列 $\\mathbf{v}_1, \\ldots, \\mathbf{v}_n$：**右奇异向量**，构成 $\\mathbb{R}^n$ 的标准正交基\n- $\\Sigma$ 的对角元 $\\sigma_1 \\geq \\sigma_2 \\geq \\cdots \\geq \\sigma_r > 0$：**奇异值**（$r = \\mathrm{rank}(A)$）',
          formula: 'A = U\\Sigma V^{\\top}',
        },
        {
          title: '奇异值从何而来？三步推导',
          body: '以 $A = \\begin{pmatrix}1 & 1 \\\\ 0 & 1 \\\\ 1 & 0\\end{pmatrix} \\in \\mathbb{R}^{3 \\times 2}$ 为例：',
          steps: [
            '**第一步：计算 $A^{\\top}A$。** $A^{\\top}A = \\begin{pmatrix}2 & 1 \\\\ 1 & 2\\end{pmatrix}$（实对称半正定矩阵）。',
            '**第二步：对 $A^{\\top}A$ 特征分解。** $(2-\\lambda)^2 - 1 = 0$，解得 $\\lambda_1 = 3$，$\\lambda_2 = 1$。奇异值 $\\sigma_1 = \\sqrt{3}$，$\\sigma_2 = 1$。',
            '**第三步：求右奇异向量 $V$。** $\\lambda_1=3$ 对应 $\\mathbf{v}_1 = \\tfrac{1}{\\sqrt{2}}(1,1)^{\\top}$；$\\lambda_2=1$ 对应 $\\mathbf{v}_2 = \\tfrac{1}{\\sqrt{2}}(1,-1)^{\\top}$。',
            '**第四步：求左奇异向量 $\\mathbf{u}_i = A\\mathbf{v}_i / \\sigma_i$。** $\\mathbf{u}_1 = \\tfrac{1}{\\sqrt{6}}(2,1,1)^{\\top}$，$\\mathbf{u}_2 = \\tfrac{1}{\\sqrt{2}}(0,-1,1)^{\\top}$，再补正交的 $\\mathbf{u}_3$ 凑满基。',
          ],
          tip: '口诀：① $A^{\\top}A$ 特征值开根号 → 奇异值；② $A^{\\top}A$ 特征向量 → $V$；③ $\\mathbf{u}_i = A\\mathbf{v}_i/\\sigma_i$ → $U$。',
        },
        {
          title: '几何理解：旋转-缩放-旋转',
          body: '对任意输入向量 $\\mathbf{x} \\in \\mathbb{R}^n$，$A\\mathbf{x}$ 分三步完成：\n\n1. $V^{\\top}$：在 $\\mathbb{R}^n$ 内**旋转**，把 $\\mathbf{x}$ 投影到右奇异向量基下\n2. $\\Sigma$：**沿坐标轴拉伸**，第 $i$ 个分量乘以 $\\sigma_i$（同时完成 $\\mathbb{R}^n \\to \\mathbb{R}^m$ 的维度切换）\n3. $U$：在 $\\mathbb{R}^m$ 内再**旋转**到最终输出方向\n\n单位球 $\\|\\mathbf{x}\\| = 1$ 在 $A$ 作用下变为**椭球**，各主轴半径 $= \\sigma_i$，方向 $= \\mathbf{u}_i$。',
          formula: '\\|A\\|_2 = \\sigma_1, \\quad \\|A\\|_F = \\sqrt{\\sigma_1^2 + \\cdots + \\sigma_r^2}',
        },
        {
          title: '两个关键关系',
          body: '**$A^{\\top}A$ 与 $AA^{\\top}$ 的谱分解**：\n\n$A^{\\top}A = V\\Sigma^{\\top}\\Sigma V^{\\top}$，特征值 $\\sigma_i^2$，特征向量 $\\mathbf{v}_i$\n\n$AA^{\\top} = U\\Sigma\\Sigma^{\\top} U^{\\top}$，特征值 $\\sigma_i^2$，特征向量 $\\mathbf{u}_i$\n\n**四个基本子空间**（来自 SVD）：\n\n- 列空间 $\\mathrm{Col}(A)$：由 $\\mathbf{u}_1,\\ldots,\\mathbf{u}_r$ 张成\n- 零空间 $\\mathrm{Null}(A)$：由 $\\mathbf{v}_{r+1},\\ldots,\\mathbf{v}_n$ 张成\n- 行空间 $\\mathrm{Row}(A)$：由 $\\mathbf{v}_1,\\ldots,\\mathbf{v}_r$ 张成\n- 左零空间：由 $\\mathbf{u}_{r+1},\\ldots,\\mathbf{u}_m$ 张成\n\n**秩的 SVD 判读**：正奇异值个数 $=$ $\\mathrm{rank}(A)$。',
          tip: '$A^{\\top}A$ 和 $AA^{\\top}$ 的非零特征值相同（都等于 $\\sigma_i^2$），这不是巧合——可用迹的循环性 $\\mathrm{tr}(AB) = \\mathrm{tr}(BA)$ 理解。',
        },
        {
          title: '例题精讲',
          body: '**例 1**：设 $A = \\begin{pmatrix}3&0\\\\0&0\\end{pmatrix}$，写出其 SVD。\n\n$A^{\\top}A = \\begin{pmatrix}9&0\\\\0&0\\end{pmatrix}$，特征值 $9, 0$，奇异值 $\\sigma_1 = 3$，$\\sigma_2 = 0$（秩 $1$）。$V = I$，$U = I$，$\\Sigma = \\begin{pmatrix}3&0\\\\0&0\\end{pmatrix}$。\n\n**例 2**：$A = \\begin{pmatrix}2&1\\\\1&2\\end{pmatrix}$（对称矩阵）。特征值 $\\lambda_1=3,\\lambda_2=1$，特征向量 $\\tfrac{1}{\\sqrt{2}}(1,1)^{\\top}$ 和 $\\tfrac{1}{\\sqrt{2}}(1,-1)^{\\top}$。对称矩阵的 SVD 满足 $U = V$（当特征值均为正时），奇异值 $= $ 特征值，即 $\\sigma_1=3,\\sigma_2=1$。',
          reveal: {
            q: '对 $2 \\times 2$ 单位矩阵 $I$，其奇异值是多少？',
            a: '$A^{\\top}A = I$，特征值均为 $1$，奇异值 $\\sigma_1 = \\sigma_2 = 1$。$U = \\Sigma = V = I$——单位矩阵的 SVD 就是它自身，无旋转无缩放。',
          },
        },
        {
          title: '维度与易错点',
          body: '**易错 1**：奇异值 $\\sigma_i = \\sqrt{\\lambda_i(A^{\\top}A)}$，不是 $\\lambda_i(A)$。特征值可以为负，奇异值永远非负。\n\n**易错 2**：$U \\in \\mathbb{R}^{m \\times m}$（列数 $= m$），$V \\in \\mathbb{R}^{n \\times n}$（列数 $= n$），$\\Sigma \\in \\mathbb{R}^{m \\times n}$（矩形）。\n\n**易错 3**：$\\mathrm{rank}(A)$ 等于正奇异值个数，不等于奇异值总个数（$\\min(m,n)$）。',
          tip: '维度检查口诀：$A(m \\times n) = U(m \\times m) \\cdot \\Sigma(m \\times n) \\cdot V^{\\top}(n \\times n)$，随时用此核查计算是否出错。',
        },
      ],
      questions: [
        {
          id: 'u8-l1-q1',
          type: 'choice',
          prompt: '对矩阵 $A = \\begin{pmatrix}2&0\\\\0&3\\\\0&0\\end{pmatrix} \\in \\mathbb{R}^{3 \\times 2}$，其奇异值（降序）是',
          options: [
            '$\\sigma_1 = 3,\\ \\sigma_2 = 2$',
            '$\\sigma_1 = 2,\\ \\sigma_2 = 3$',
            '$\\sigma_1 = \\sqrt{13},\\ \\sigma_2 = 0$',
            '$\\sigma_1 = 9,\\ \\sigma_2 = 4$',
          ],
          answer: 0,
          explain: '$A^{\\top}A = \\begin{pmatrix}4&0\\\\0&9\\end{pmatrix}$，特征值 $9, 4$，降序奇异值 $\\sigma_1 = 3$，$\\sigma_2 = 2$。',
        },
        {
          id: 'u8-l1-q2',
          type: 'judge',
          prompt: '矩阵 $A$ 的秩等于其正奇异值的个数。',
          answer: true,
          explain: '$\\mathrm{rank}(A) = \\mathrm{rank}(A^{\\top}A) = $ 非零特征值个数 $=$ 正奇异值个数。',
        },
        {
          id: 'u8-l1-q3',
          type: 'choice',
          prompt: '矩阵 $A \\in \\mathbb{R}^{m \\times n}$（$m > n$）的 SVD 为 $A = U\\Sigma V^{\\top}$，则 $U$ 的大小是',
          options: [
            '$m \\times n$',
            '$n \\times n$',
            '$m \\times m$',
            '$m \\times n$ 且行正交',
          ],
          answer: 2,
          explain: 'SVD 中 $U \\in \\mathbb{R}^{m \\times m}$ 是方阵正交矩阵（列构成 $\\mathbb{R}^m$ 的完整标准正交基），$V \\in \\mathbb{R}^{n \\times n}$ 类似，$\\Sigma$ 才是 $m \\times n$ 的矩形矩阵。',
        },
        {
          id: 'u8-l1-q4',
          type: 'input',
          prompt: '矩阵 $A = \\begin{pmatrix}0&2\\\\0&0\\end{pmatrix}$ 的最大奇异值 $\\sigma_1 =$ ？（填整数）',
          accept: ['2'],
          placeholder: '输入数字',
          explain: '$A^{\\top}A = \\begin{pmatrix}0&0\\\\0&4\\end{pmatrix}$，特征值 $4, 0$，$\\sigma_1 = \\sqrt{4} = 2$。',
        },
        {
          id: 'u8-l1-q5',
          type: 'match',
          prompt: '将 SVD 的三个因子与其几何含义配对：',
          left: ['$V^{\\top}$', '$\\Sigma$', '$U$'],
          right: [
            '输入空间的旋转（或反射）',
            '沿坐标轴的拉伸缩放，并完成维度切换',
            '输出空间的旋转（或反射）',
          ],
          explain: '$V^{\\top}$ 在输入空间 $\\mathbb{R}^n$ 中旋转；$\\Sigma$ 做各轴方向的缩放同时从 $\\mathbb{R}^n$ 映射到 $\\mathbb{R}^m$；$U$ 在输出空间 $\\mathbb{R}^m$ 中旋转。',
        },
        {
          id: 'u8-l1-q6',
          type: 'judge',
          prompt: '设 $A = U\\Sigma V^{\\top}$ 是 SVD，则 $A^{\\top}A$ 的特征向量矩阵就是 $V$。',
          answer: true,
          explain: '$A^{\\top}A = V\\Sigma^{\\top}\\Sigma V^{\\top}$，这是 $A^{\\top}A$ 的谱分解，特征向量矩阵正是 $V$，特征值是 $\\sigma_i^2$。',
        },
        {
          id: 'u8-l1-q7',
          type: 'choice',
          prompt: '矩阵 $A = \\begin{pmatrix}1&0\\\\0&1\\\\0&0\\end{pmatrix}$ 的 Frobenius 范数 $\\|A\\|_F$ 是',
          options: [
            '$\\sqrt{2}$',
            '$2$',
            '$1$',
            '$\\sqrt{3}$',
          ],
          answer: 0,
          explain: '$\\|A\\|_F = \\sqrt{1^2+0+0+1^2+0+0} = \\sqrt{2}$。奇异值为 $1, 1$（$A^{\\top}A = I_2$），$\\|A\\|_F = \\sqrt{\\sigma_1^2+\\sigma_2^2} = \\sqrt{2}$，两种算法一致。',
        },
        {
          id: 'u8-l1-q8',
          type: 'choice',
          prompt: '对称矩阵 $A = \\begin{pmatrix}2&1\\\\1&2\\end{pmatrix}$ 的最大奇异值是',
          options: [
            '$3$',
            '$\\sqrt{5}$',
            '$1$',
            '$2$',
          ],
          answer: 0,
          explain: '$A$ 是对称正定矩阵，特征值 $\\lambda_1 = 3, \\lambda_2 = 1$（均为正），奇异值与特征值相等，故 $\\sigma_1 = 3$。',
        },
      ],
    },

    // ─────────────────────────────────────────────────
    // u8-l2  低秩逼近·伪逆
    // ─────────────────────────────────────────────────
    {
      id: 'u8-l2',
      title: '低秩逼近·伪逆',
      subtitle: 'Eckart-Young 定理与 Moore-Penrose 伪逆',
      intro: [
        {
          title: '截断 SVD：丢掉"小"信息',
          body: '将 $A = \\sum_{i=1}^r \\sigma_i \\mathbf{u}_i\\mathbf{v}_i^{\\top}$ 写成秩-1 矩阵的加权和，每项"权重"是奇异值 $\\sigma_i$。\n\n**截断思路**：奇异值越大，对应分量携带的"信息量"越多。丢弃前 $k$ 项之外的所有项，得到**秩-$k$ 近似**：\n\n$A_k = \\sum_{i=1}^k \\sigma_i \\mathbf{u}_i \\mathbf{v}_i^{\\top}$\n\n**例（图像压缩）**：$100 \\times 100$ 图像（10000 个数），$k=5$ 时只需存 $5 \\times (100+100+1) = 1005$ 个数，约原来的 $10\\%$。',
          formula: 'A_k = \\sum_{i=1}^{k} \\sigma_i\\, \\mathbf{u}_i \\mathbf{v}_i^{\\top}',
        },
        {
          title: 'Eckart-Young 定理：截断 SVD 是最优近似',
          body: '**定理（Eckart-Young，1936）**：在所有秩不超过 $k$ 的矩阵中，$A_k$ 是 Frobenius 范数（和谱范数）意义下最接近 $A$ 的那个：\n\n$\\|A - A_k\\|_F = \\sqrt{\\sigma_{k+1}^2 + \\cdots + \\sigma_r^2}$（最小可能误差）\n\n$\\|A - A_k\\|_2 = \\sigma_{k+1}$\n\n**直觉**：最大的 $k$ 个奇异值对应矩阵最"主要"的 $k$ 个方向，截断它们以外的部分是最优"压缩"策略。\n\n**能量保留率**：$\\dfrac{\\sigma_1^2+\\cdots+\\sigma_k^2}{\\sigma_1^2+\\cdots+\\sigma_r^2}$，选 $k$ 使该比例 $\\geq 95\\%$ 是常用实践标准。',
          formula: '\\|A - A_k\\|_F^2 = \\sigma_{k+1}^2 + \\cdots + \\sigma_r^2',
          tip: 'AI 应用：PCA 降维、推荐系统矩阵分解、LoRA 微调（$\\Delta W = BA$，$B \\in \\mathbb{R}^{d \\times r}$，$A \\in \\mathbb{R}^{r \\times d}$，$r \\ll d$）都基于低秩结构假设。',
        },
        {
          title: '存储量分析',
          body: '**原始矩阵** $A \\in \\mathbb{R}^{m \\times n}$：需要 $mn$ 个数。\n\n**秩-$k$ 近似** $A_k = U_k \\Sigma_k V_k^{\\top}$：\n- $U_k \\in \\mathbb{R}^{m \\times k}$：$mk$ 个数\n- $\\Sigma_k$（对角）：$k$ 个数\n- $V_k \\in \\mathbb{R}^{n \\times k}$：$nk$ 个数\n- 合计：$k(m+n+1)$ 个数\n\n**压缩比**：$k(m+n+1)/(mn)$。当 $k \\ll \\min(m,n)$ 时，压缩效果显著。\n\n**例**：$200 \\times 300$ 矩阵，$k=10$ 时只需 $10 \\times 501 = 5010$ 个数（vs 原来 $60000$），压缩为 $8.4\\%$。',
        },
        {
          title: '伪逆的动机',
          body: '当 $A$ 不是方阵或秩不满时，$A^{-1}$ 不存在，但我们仍希望"反解" $A\\mathbf{x} = \\mathbf{b}$：\n\n- **超定**（方程多于未知量，$m > n$）：通常无精确解，求**最小二乘解**\n- **欠定**（方程少于未知量，$m < n$）：有无穷多解，求**范数最小解**\n- **降秩**：同时做到两点：最小化残差且最小化解的范数\n\n**Moore-Penrose 伪逆** $A^+$ 在所有情形下给出唯一的"最优广义逆"。',
        },
        {
          title: '伪逆的 SVD 定义与性质',
          body: '设 $A = U\\Sigma V^{\\top}$，定义 $\\Sigma^+$ 为将 $\\Sigma$ 转置后对每个非零对角元取倒数，零元保持为零。则：\n\n$A^+ = V\\Sigma^+ U^{\\top} \\in \\mathbb{R}^{n \\times m}$\n\n**满足 Moore-Penrose 四条件**：$AA^+A=A$，$A^+AA^+=A^+$，$(AA^+)^{\\top}=AA^+$，$(A^+A)^{\\top}=A^+A$。\n\n**投影解释**：$AA^+ = U_r U_r^{\\top}$ 是向列空间的正交投影，$A^+A = V_r V_r^{\\top}$ 是向行空间的正交投影。\n\n**$A^+\\mathbf{b}$ 的含义**：先将 $\\mathbf{b}$ 投影到列空间，再通过各奇异值的逆映射回行空间——无解时求最近点，有多解时取范数最小的。',
          formula: 'A^+ = V\\Sigma^+ U^{\\top}',
        },
        {
          title: '三种情形的伪逆解',
          body: '**超定**（$m > n$，$A$ 列满秩，$r=n$）：$A^+ = (A^{\\top}A)^{-1}A^{\\top}$，$A^+\\mathbf{b}$ 是最小二乘解 $\\min\\|A\\mathbf{x}-\\mathbf{b}\\|$，满足正规方程 $A^{\\top}A\\mathbf{x}=A^{\\top}\\mathbf{b}$。\n\n**欠定**（$m < n$，$A$ 行满秩，$r=m$）：$A^+ = A^{\\top}(AA^{\\top})^{-1}$，$A^+\\mathbf{b}$ 是满足 $A\\mathbf{x}=\\mathbf{b}$ 的最小范数解。\n\n**降秩**（$r < \\min(m,n)$）：$A^+\\mathbf{b}$ 同时最小化残差 $\\|A\\mathbf{x}-\\mathbf{b}\\|$ 和解的范数 $\\|\\mathbf{x}\\|$（最小范数最小二乘解）。',
          tip: '记忆法：超定 → 左逆 $(A^{\\top}A)^{-1}A^{\\top}$；欠定 → 右逆 $A^{\\top}(AA^{\\top})^{-1}$；一般情形统一写 $V\\Sigma^+ U^{\\top}$。',
        },
        {
          title: '例题精讲',
          body: '**例 1**：$A = \\begin{pmatrix}3&0\\\\0&0\\end{pmatrix}$，求 $A^+$。\n\nSVD：$U=V=I$，$\\Sigma = \\begin{pmatrix}3&0\\\\0&0\\end{pmatrix}$，$\\Sigma^+ = \\begin{pmatrix}1/3&0\\\\0&0\\end{pmatrix}$。伪逆 $A^+ = \\begin{pmatrix}1/3&0\\\\0&0\\end{pmatrix}$。\n\n**例 2**：$100 \\times 80$ 矩阵，奇异值依次为 $10, 6, 3, 1, 0.2, \\ldots$，选 $k=3$。存储量 $3 \\times (100+80+1) = 543$（vs 原来 $8000$），压缩为 $6.8\\%$。近似误差 $\\|A-A_3\\|_F = \\sqrt{1^2+0.2^2+\\cdots}$，主要由 $\\sigma_4=1$ 贡献。',
          reveal: {
            q: '设 $A \\in \\mathbb{R}^{3 \\times 2}$，列满秩，$A^+$ 的大小是多少？',
            a: '$A^+ \\in \\mathbb{R}^{2 \\times 3}$（$n \\times m$，恰好是 $A$ 的转置同形）。超定情形下 $A^+ = (A^{\\top}A)^{-1}A^{\\top}$，$A^{\\top}A \\in \\mathbb{R}^{2 \\times 2}$ 可逆，$A^+ \\in \\mathbb{R}^{2 \\times 3}$。',
          },
        },
      ],
      questions: [
        {
          id: 'u8-l2-q1',
          type: 'choice',
          prompt: '$200 \\times 300$ 矩阵 $A$（秩 $50$）用截断 SVD $A_{10}$ 近似，存储 $A_{10}$ 需要',
          options: [
            '$10 \\times (200+300+1) = 5010$ 个数',
            '$200 \\times 300 = 60000$ 个数',
            '$50 \\times 501 = 25050$ 个数',
            '$10 \\times 10 = 100$ 个数',
          ],
          answer: 0,
          explain: '秩-$k$ 近似存储 $U_k$（$200 \\times 10$）、$\\Sigma_k$（$10$ 个对角元）、$V_k$（$300 \\times 10$），共 $k(m+n+1) = 10 \\times 501 = 5010$ 个数。',
        },
        {
          id: 'u8-l2-q2',
          type: 'judge',
          prompt: 'Eckart-Young 定理说明：在所有秩不超过 $k$ 的矩阵中，截断 SVD $A_k$ 是 Frobenius 范数意义下最接近 $A$ 的。',
          answer: true,
          explain: 'Eckart-Young 定理（1936）的精确表述：$\\min_{\\mathrm{rank}(B) \\leq k} \\|A-B\\|_F = \\|A-A_k\\|_F = \\sqrt{\\sigma_{k+1}^2+\\cdots+\\sigma_r^2}$，且在谱范数意义下也成立。',
        },
        {
          id: 'u8-l2-q3',
          type: 'input',
          prompt: '矩阵 $A$ 奇异值为 $5, 3, 1$（秩 $3$），$\\|A - A_2\\|_F =$ ？（填整数）',
          accept: ['1'],
          placeholder: '整数',
          explain: '截断 SVD 误差 $\\|A-A_2\\|_F = \\sqrt{\\sigma_3^2} = \\sqrt{1^2} = 1$。',
        },
        {
          id: 'u8-l2-q4',
          type: 'choice',
          prompt: '矩阵 $A = \\begin{pmatrix}3&0\\\\0&0\\end{pmatrix}$ 的 Moore-Penrose 伪逆 $A^+$ 是',
          options: [
            '$\\begin{pmatrix}1/3&0\\\\0&0\\end{pmatrix}$',
            '$\\begin{pmatrix}3&0\\\\0&0\\end{pmatrix}$',
            '$\\begin{pmatrix}0&0\\\\0&1/3\\end{pmatrix}$',
            '$\\begin{pmatrix}1/9&0\\\\0&0\\end{pmatrix}$',
          ],
          answer: 0,
          explain: 'SVD：$U=V=I$，$\\Sigma^+ = \\begin{pmatrix}1/3&0\\\\0&0\\end{pmatrix}^{\\top}$（形状 $2 \\times 2$，对非零元取倒数）。$A^+ = V\\Sigma^+U^{\\top} = \\begin{pmatrix}1/3&0\\\\0&0\\end{pmatrix}$。',
        },
        {
          id: 'u8-l2-q5',
          type: 'judge',
          prompt: '伪逆 $A^+$ 与 $A$ 的形状相同（均为 $m \\times n$）。',
          answer: false,
          explain: '$A \\in \\mathbb{R}^{m \\times n}$ 时，$A^+ = V\\Sigma^+ U^{\\top} \\in \\mathbb{R}^{n \\times m}$，是 $A$ 的转置同形，不是相同形状。',
        },
        {
          id: 'u8-l2-q6',
          type: 'choice',
          prompt: '超定方程组 $A\\mathbf{x} = \\mathbf{b}$（$A$ 列满秩，$m > n$）的伪逆解 $\\mathbf{x}^* = A^+\\mathbf{b}$ 等价于',
          options: [
            '最小化 $\\|A\\mathbf{x}-\\mathbf{b}\\|_2$ 的最小二乘解',
            '满足 $A\\mathbf{x}=\\mathbf{b}$ 的范数最小解',
            '$A^{-1}\\mathbf{b}$（精确解）',
            '最小化 $\\|\\mathbf{x}\\|_2$ 的解',
          ],
          answer: 0,
          explain: '列满秩超定情形下，$A^+ = (A^{\\top}A)^{-1}A^{\\top}$，$A^+\\mathbf{b} = (A^{\\top}A)^{-1}A^{\\top}\\mathbf{b}$ 是正规方程的解，即最小二乘解 $\\min\\|A\\mathbf{x}-\\mathbf{b}\\|$。',
        },
        {
          id: 'u8-l2-q7',
          type: 'match',
          prompt: '将方程组情形与伪逆解的含义配对：',
          left: ['超定（$m > n$，列满秩）', '欠定（$m < n$，行满秩）', '降秩（$r < \\min(m,n)$）'],
          right: [
            '最小二乘解（最小化残差）',
            '最小范数解（最小化 $\\|\\mathbf{x}\\|$）',
            '最小范数最小二乘解（同时最小化两者）',
          ],
          explain: '超定时方程无解，取最小二乘；欠定时有无穷解，取范数最小；降秩时兼顾两者，统一用 $A^+\\mathbf{b}$。',
        },
        {
          id: 'u8-l2-q8',
          type: 'choice',
          prompt: '能量保留率 $\\dfrac{\\sigma_1^2+\\sigma_2^2}{\\sigma_1^2+\\sigma_2^2+\\sigma_3^2}$ 对奇异值 $\\sigma_1=4, \\sigma_2=3, \\sigma_3=0$ 等于',
          options: [
            '$1$（$100\\%$）',
            '$\\dfrac{16+9}{16+9+0} = 1$',
            '$\\dfrac{25}{25} = 1$',
            '以上三项均正确',
          ],
          answer: 3,
          explain: '$\\sigma_3=0$ 说明矩阵秩为 $2$，$k=2$ 时截断 SVD 已是完整分解，能量保留 $100\\%$，误差为零。三个选项描述相同事实。',
        },
      ],
    },

    // ─────────────────────────────────────────────────
    // u8-l3  矩阵表示与配方
    // ─────────────────────────────────────────────────
    {
      id: 'u8-l3',
      title: '矩阵表示与配方',
      subtitle: '二次型的对称矩阵表示与标准化',
      intro: [
        {
          title: '什么是二次型？',
          body: '一元情形：$f(x) = ax^2$ 是最简单的二次函数，形状（碗形/倒碗）由 $a$ 的符号决定。\n\n推广到 $n$ 维：**二次型**是所有变量的二次齐次多项式，只有纯二次项 $x_i^2$ 和交叉项 $x_ix_j$，**没有**一次项或常数项。\n\n**例**：$Q(x_1, x_2) = 3x_1^2 + 4x_1x_2 - 2x_2^2$\n\n在机器学习中，损失函数在临界点附近的 Hessian 矩阵定义的曲面 $\\mathbf{x}^{\\top}H\\mathbf{x}$ 正是一个二次型——它决定损失面是碗形（极小值）还是马鞍面（鞍点）。',
          tip: '二次型 = 变量的二次齐次多项式。关键词：只有平方项和交叉项，无一次项，无常数。',
        },
        {
          title: '对称矩阵表示',
          body: '任意二次型都对应唯一一个**实对称矩阵** $A = A^{\\top}$：\n\n$Q(\\mathbf{x}) = \\mathbf{x}^{\\top}A\\mathbf{x}$\n\n**读矩阵规则**：\n- 对角元 $a_{ii}$ = $x_i^2$ 的系数\n- 非对角元 $a_{ij} = a_{ji}$ = $x_ix_j$ 系数的**一半**\n\n**例 1**：$Q = 3x_1^2 + 4x_1x_2 - 2x_2^2$\n\n$A = \\begin{pmatrix}3 & 2 \\\\ 2 & -2\\end{pmatrix}$（$4x_1x_2$ 对应 $a_{12}=a_{21}=2$）\n\n**例 2**：$Q = x_1^2 + 2x_2^2 + 3x_3^2 + 2x_1x_2 - 4x_1x_3 + 6x_2x_3$\n\n$A = \\begin{pmatrix}1&1&-2\\\\1&2&3\\\\-2&3&3\\end{pmatrix}$（交叉项 $-4x_1x_3$ 给 $a_{13}=a_{31}=-2$）',
          formula: 'Q(\\mathbf{x}) = \\mathbf{x}^{\\top}A\\mathbf{x}, \\quad a_{ij} = a_{ji} = \\frac{1}{2}\\times\\text{(}x_ix_j\\text{ 系数)}',
        },
        {
          title: '几何含义',
          body: '二次型 $Q(\\mathbf{x}) = \\mathbf{x}^{\\top}A\\mathbf{x}$ 定义了 $\\mathbb{R}^n$ 上的"高度函数"：\n\n- $n=2$ 时，等高线 $\\{Q=c\\}$ 是以原点为中心的圆锥曲线\n- $A$ 正定时，等高线为同心椭圆，原点是全局最低点（碗形）\n- $A$ 不定时，等高线为双曲线，原点是鞍点\n\n交叉项使等高线的主轴**不**与坐标轴对齐。**标准化**的目标正是旋转坐标轴，消去交叉项，使主轴与坐标轴重合。',
        },
        {
          title: '正交变换化标准形',
          body: '由谱定理，实对称 $A = Q\\Lambda Q^{\\top}$（$Q$ 为特征向量矩阵）。令 $\\mathbf{x} = Q\\mathbf{y}$（即 $\\mathbf{y} = Q^{\\top}\\mathbf{x}$），代入：\n\n$\\mathbf{x}^{\\top}A\\mathbf{x} = (Q\\mathbf{y})^{\\top}A(Q\\mathbf{y}) = \\mathbf{y}^{\\top}\\Lambda\\mathbf{y} = \\lambda_1y_1^2 + \\lambda_2y_2^2 + \\cdots + \\lambda_ny_n^2$\n\n在新坐标 $\\mathbf{y}$ 下，矩阵变为对角的，**交叉项全部消去**——称为**标准形**，系数恰好是特征值。\n\n**几何意义**：正交变换（旋转）将坐标轴旋转到与 $A$ 的主轴（特征向量方向）重合。',
          formula: '\\mathbf{x}^{\\top}A\\mathbf{x} \\xrightarrow{\\mathbf{x}=Q\\mathbf{y}} \\lambda_1 y_1^2 + \\cdots + \\lambda_n y_n^2',
        },
        {
          title: '配方法化标准形',
          body: '不需要求特征向量，**配方法**通过代数操作也能消去交叉项，得到标准形（系数不一定是特征值，但惯性指数不变）。\n\n**例**：$Q = x_1^2 + 2x_1x_2 + 3x_2^2$',
          steps: [
            '对含 $x_1$ 的项配方：$(x_1 + x_2)^2 - x_2^2 + 3x_2^2 = (x_1+x_2)^2 + 2x_2^2$。',
            '令 $y_1 = x_1+x_2$，$y_2 = x_2$，标准形为 $y_1^2 + 2y_2^2$（两个正系数）。',
            '该二次型正定：标准形系数全正，与特征值全正等价（虽然此处系数 $1, 2$ 不是特征值，但符号一致）。',
          ],
          tip: '配方法的核心策略：依次对每个变量完全配方，消去含该变量的所有交叉项，再对下一个变量重复。',
        },
        {
          title: 'Sylvester 惯性定理',
          body: '**定理（Sylvester 惯性定理，1852）**：对实二次型，无论用何种非退化线性变换化标准形，**正系数个数** $p$（正惯性指数）和**负系数个数** $q$（负惯性指数）永远不变。\n\n有序对 $(p,q)$ 称为**符号差**，完整刻画了二次型的"类型"：\n\n- $(p,q) = (n,0)$：正定\n- $(p,q) = (0,n)$：负定\n- $p > 0, q > 0$：不定\n- $q=0, p<n$：正半定\n\n**等价表述**：两个实对称矩阵合同（存在非退化 $C$ 使 $B = C^{\\top}AC$）当且仅当它们有相同的符号差 $(p,q)$。',
          tip: '惯性定理说明：无论如何换坐标，"碗形""马鞍"等形状是本质性质，不随坐标选择而改变。',
        },
        {
          title: '例题精讲',
          body: '**例（正交法）**：将 $Q = 3x_1^2 + 4x_1x_2 - x_2^2$ 化标准形。\n\n矩阵 $A = \\begin{pmatrix}3&2\\\\2&-1\\end{pmatrix}$，特征方程 $\\lambda^2 - 2\\lambda - 7 = 0$，特征值 $\\lambda = 1 \\pm 2\\sqrt{2}$。\n\n$\\lambda_1 = 1 - 2\\sqrt{2} < 0$，$\\lambda_2 = 1 + 2\\sqrt{2} > 0$，标准形 $(1-2\\sqrt{2})y_1^2 + (1+2\\sqrt{2})y_2^2$。\n\n符号差 $(1,1)$，不定，等高线为双曲线族。\n\n**例（配方法）**：$Q = 2x_1^2 - 4x_1x_2 + 3x_2^2$。\n\n$Q = 2(x_1-x_2)^2 - 2x_2^2 + 3x_2^2 = 2(x_1-x_2)^2 + x_2^2$。令 $y_1=x_1-x_2, y_2=x_2$，标准形 $2y_1^2+y_2^2$，系数全正，正定。',
          reveal: {
            q: '不定矩阵 $A = \\begin{pmatrix}1&2\\\\2&1\\end{pmatrix}$ 的符号差是什么？',
            a: '特征值 $\\lambda_1=3>0$，$\\lambda_2=-1<0$，符号差 $(1,1)$，不定。等高线为双曲线族，原点是鞍点。',
          },
        },
      ],
      questions: [
        {
          id: 'u8-l3-q1',
          type: 'choice',
          prompt: '$Q(x_1,x_2) = 5x_1^2 - 6x_1x_2 + 2x_2^2$ 对应的实对称矩阵 $A$ 是',
          options: [
            '$\\begin{pmatrix}5&-6\\\\-6&2\\end{pmatrix}$',
            '$\\begin{pmatrix}5&-3\\\\-3&2\\end{pmatrix}$',
            '$\\begin{pmatrix}5&3\\\\3&2\\end{pmatrix}$',
            '$\\begin{pmatrix}5&-6\\\\3&2\\end{pmatrix}$',
          ],
          answer: 1,
          explain: '对角元直接读取：$a_{11}=5$，$a_{22}=2$。交叉项 $-6x_1x_2$ 的系数 $-6$ 平均分配：$a_{12}=a_{21}=-3$。所以 $A=\\begin{pmatrix}5&-3\\\\-3&2\\end{pmatrix}$。',
        },
        {
          id: 'u8-l3-q2',
          type: 'judge',
          prompt: '任意二次型对应的实对称矩阵是唯一的。',
          answer: true,
          explain: '规定 $a_{ij}=a_{ji}=\\tfrac{1}{2}\\times x_ix_j$ 系数后，矩阵唯一确定。不对称表示有无数个（可将交叉项系数任意分配给 $a_{ij}$ 和 $a_{ji}$），但对称化后唯一。',
        },
        {
          id: 'u8-l3-q3',
          type: 'input',
          prompt: '二次型 $Q = x_1^2 + 4x_2^2 + 4x_1x_2$ 配方后为 $(x_1 + ax_2)^2 + bx_2^2$，则 $a + b =$ ？（填整数）',
          accept: ['2'],
          placeholder: '整数',
          explain: '$Q = (x_1+2x_2)^2 - 4x_2^2 + 4x_2^2 = (x_1+2x_2)^2$，所以 $a=2$，$b=0$，$a+b=2$。这是半正定二次型（完全平方，$\\det A = 0$）。',
        },
        {
          id: 'u8-l3-q4',
          type: 'choice',
          prompt: '二次型 $Q = \\mathbf{x}^{\\top}A\\mathbf{x}$ 经正交变换 $\\mathbf{x}=Q\\mathbf{y}$（$Q$ 为特征向量矩阵）化为标准形，标准形系数是',
          options: [
            '$A$ 的奇异值',
            '$A$ 的特征值',
            '$A$ 的对角元',
            '$A$ 的主子式',
          ],
          answer: 1,
          explain: '$\\mathbf{x}^{\\top}A\\mathbf{x} = \\mathbf{y}^{\\top}(Q^{\\top}AQ)\\mathbf{y} = \\mathbf{y}^{\\top}\\Lambda\\mathbf{y} = \\sum \\lambda_i y_i^2$，系数恰好是 $A$ 的特征值。',
        },
        {
          id: 'u8-l3-q5',
          type: 'judge',
          prompt: 'Sylvester 惯性定理保证：无论用哪种非退化线性变换化标准形，正系数个数（正惯性指数）不变。',
          answer: true,
          explain: '惯性定理的核心：非退化线性变换（合同变换）保持正惯性指数 $p$ 和负惯性指数 $q$ 不变。标准形系数的具体值可以不同，但正负个数是不变量。',
        },
        {
          id: 'u8-l3-q6',
          type: 'choice',
          prompt: '二次型 $Q = x_1^2 + 2x_1x_2 + 3x_2^2$ 经配方后为 $(x_1+x_2)^2 + 2x_2^2$。其符号差是',
          options: [
            '$(2,0)$',
            '$(1,1)$',
            '$(0,2)$',
            '$(2,1)$',
          ],
          answer: 0,
          explain: '标准形 $(x_1+x_2)^2 + 2x_2^2$ 的系数均为正（$1$ 和 $2$），正系数个数 $p=2$，负系数个数 $q=0$，符号差 $(2,0)$，正定。',
        },
        {
          id: 'u8-l3-q7',
          type: 'match',
          prompt: '将符号差与二次型类型配对：',
          left: ['$(n,0)$', '$(0,n)$', '$(p,q)$，$p,q>0$', '$(p,0)$，$p<n$'],
          right: ['正定', '负定', '不定', '正半定'],
          explain: '正定要求所有方向均为正（$q=0$，$p=n$）；负定全负；不定有正有负；正半定无负方向但有零方向。',
        },
        {
          id: 'u8-l3-q8',
          type: 'choice',
          prompt: '两个实对称矩阵合同（存在非退化 $C$ 使 $B=C^{\\top}AC$），则它们',
          options: [
            '特征值完全相同',
            '符号差 $(p,q)$ 相同',
            '行列式相等',
            '迹相等',
          ],
          answer: 1,
          explain: '合同变换保持符号差（惯性定理等价表述）。特征值一般不同（$C$ 不一定是正交矩阵），行列式和迹也可能改变。',
        },
      ],
    },

    // ─────────────────────────────────────────────────
    // u8-l4  正定判定
    // ─────────────────────────────────────────────────
    {
      id: 'u8-l4',
      title: '正定判定',
      subtitle: 'Sylvester 准则、Rayleigh 商与 Hessian 分析',
      intro: [
        {
          title: '正定的几何直觉',
          body: '**正定（positive definite，$A \\succ 0$）**是最重要的类型：$\\mathbf{x}^{\\top}A\\mathbf{x} > 0$ 对所有 $\\mathbf{x} \\neq \\mathbf{0}$。\n\n几何上，正定二次型是"多维碗形曲面"：\n- 原点是唯一全局最低点（函数值为 $0$）\n- 向任何方向移动，函数值都严格增加\n- 等高面 $\\{Q=c\\}$（$c>0$）是以原点为中心的**椭球面**\n\n**椭球主轴**：方向 $=$ 特征向量，半径 $= 1/\\sqrt{\\lambda_i}$（特征值越大，对应轴越短）。',
          tip: '正定矩阵类比多维"碗"：任意方向均上弯。这正是损失函数在极小值处的形状。',
        },
        {
          title: '五个等价判别条件',
          body: '以下五个条件对实对称矩阵 $A$ **等价**：\n\n1. 二次型 $\\mathbf{x}^{\\top}A\\mathbf{x} > 0$，对所有 $\\mathbf{x} \\neq \\mathbf{0}$\n2. 所有特征值 $\\lambda_i > 0$\n3. **顺序主子式全正**（Sylvester 准则）：$\\Delta_1 > 0$，$\\Delta_2 > 0$，……，$\\Delta_n > 0$\n4. Cholesky 分解存在：$A = LL^{\\top}$，$L$ 下三角且对角元为正\n5. 存在列满秩矩阵 $B$ 使 $A = B^{\\top}B$\n\n**Sylvester 准则最实用**：$\\Delta_k$ 是 $A$ 的左上角 $k \\times k$ 子矩阵的行列式（顺序主子式）。',
          formula: 'A \\succ 0 \\iff \\text{所有顺序主子式} > 0 \\iff \\text{所有特征值} > 0',
        },
        {
          title: '各类型的判别对比',
          body: '| 类型 | 特征值条件 | 顺序主子式 |\n|:---|:---|:---|\n| 正定 | 全 $> 0$ | 全 $> 0$ |\n| 负定 | 全 $< 0$ | 交错变号（$\\Delta_1 < 0$，$\\Delta_2 > 0$，……）|\n| 正半定 | 全 $\\geq 0$，有零 | 全 $\\geq 0$，$\\det(A)=0$ |\n| 不定 | 有正有负 | 不满足上述模式 |\n\n**正定矩阵的运算性质**：\n- 若 $A \\succ 0$，则 $A^{-1} \\succ 0$（逆正定）\n- 若 $A \\succ 0$ 且 $B \\succ 0$，则 $A+B \\succ 0$\n- 若 $A \\succ 0$ 且 $C$ 列满秩，则 $C^{\\top}AC \\succ 0$（合同变换保持正定）',
        },
        {
          title: 'Rayleigh 商极值定理',
          body: '**问题**：在单位球面 $\\|\\mathbf{x}\\|=1$ 上，$Q(\\mathbf{x}) = \\mathbf{x}^{\\top}A\\mathbf{x}$ 的最大值和最小值是多少？\n\n**答（Rayleigh 商定理）**：令 $\\mathbf{y} = Q^{\\top}\\mathbf{x}$（$Q$ 为特征向量矩阵），则 $\\|\\mathbf{y}\\|=\\|\\mathbf{x}\\|=1$，\n\n$\\mathbf{x}^{\\top}A\\mathbf{x} = \\sum_i \\lambda_i y_i^2 \\in [\\lambda_{\\min}, \\lambda_{\\max}]$\n\n$\\min_{\\|\\mathbf{x}\\|=1} \\mathbf{x}^{\\top}A\\mathbf{x} = \\lambda_{\\min}(A)$，最优点 $= \\mathbf{q}_{\\min}$\n\n$\\max_{\\|\\mathbf{x}\\|=1} \\mathbf{x}^{\\top}A\\mathbf{x} = \\lambda_{\\max}(A)$，最优点 $= \\mathbf{q}_{\\max}$\n\n$\\mathbf{q}_{\\min}$、$\\mathbf{q}_{\\max}$ 是 $A$ 的最小和最大特征值对应的单位特征向量。',
          formula: '\\lambda_{\\min} \\leq \\frac{\\mathbf{x}^{\\top}A\\mathbf{x}}{\\mathbf{x}^{\\top}\\mathbf{x}} \\leq \\lambda_{\\max}',
        },
        {
          title: 'Hessian 矩阵与临界点类型',
          body: '损失函数 $\\mathcal{L}(\\boldsymbol{\\theta})$ 在临界点 $\\boldsymbol{\\theta}^*$ 附近：\n\n$\\mathcal{L}(\\boldsymbol{\\theta}^* + \\Delta\\boldsymbol{\\theta}) \\approx \\mathcal{L}(\\boldsymbol{\\theta}^*) + \\tfrac{1}{2}\\Delta\\boldsymbol{\\theta}^{\\top}H\\Delta\\boldsymbol{\\theta}$\n\nHessian 矩阵 $H$ 的符号差决定临界点类型：\n\n| Hessian 符号差 | 临界点类型 |\n|:---|:---|\n| $(n,0)$，$H \\succ 0$ | 局部极小值 |\n| $(0,n)$，$H \\prec 0$ | 局部极大值 |\n| $(p,q)$，$p,q>0$ | 鞍点 |\n\n**条件数与学习率**：$\\kappa = \\lambda_{\\max}/\\lambda_{\\min}$ 决定梯度下降收敛速度。梯度下降稳定要求 $\\eta < 2/\\lambda_{\\max}$。',
          tip: '大型神经网络中，高维参数空间的临界点几乎都是鞍点而非局部极大值——所有特征值同号（均负）的概率随维度指数下降。',
        },
        {
          title: '例题精讲',
          body: '**例 1（Sylvester 准则）**：判断 $Q = 2x_1^2 + x_2^2 + 3x_3^2 - 2x_1x_2$ 的正定性。\n\n$A = \\begin{pmatrix}2&-1&0\\\\-1&1&0\\\\0&0&3\\end{pmatrix}$。\n\n$\\Delta_1 = 2 > 0$，$\\Delta_2 = 2\\times1-1 = 1 > 0$，$\\Delta_3 = 3\\times\\Delta_2 = 3 > 0$。三个全正，$Q$ **正定**。\n\n**例 2（Rayleigh 商）**：$A = \\begin{pmatrix}5&2\\\\2&2\\end{pmatrix}$，Hessian。$\\Delta_1=5>0$，$\\Delta_2=10-4=6>0$，正定，临界点是极小值。特征多项式 $\\lambda^2-7\\lambda+6=0$，$\\lambda_1=1, \\lambda_2=6$。最优学习率 $\\eta^* = 2/(\\lambda_1+\\lambda_2) = 2/7$，稳定要求 $\\eta < 2/6 = 1/3$。',
          reveal: {
            q: '矩阵 $A = \\begin{pmatrix}5&4\\\\4&2\\end{pmatrix}$，判断其正定性。',
            a: '$\\Delta_1 = 5 > 0$，$\\Delta_2 = 10-16 = -6 < 0$，不满足全正条件，$A$ **不定**，对应临界点是鞍点。',
          },
        },
      ],
      questions: [
        {
          id: 'u8-l4-q1',
          type: 'judge',
          prompt: '若实对称矩阵 $A$ 的顺序主子式全正，则 $A$ 正定。',
          answer: true,
          explain: '这就是 Sylvester 准则：所有 $k$ 阶顺序主子式 $\\Delta_k > 0$（$k=1,\\ldots,n$）$\\Leftrightarrow$ $A \\succ 0$。',
        },
        {
          id: 'u8-l4-q2',
          type: 'choice',
          prompt: '判断 $A = \\begin{pmatrix}1&2\\\\2&1\\end{pmatrix}$ 对应的二次型类型：',
          options: [
            '正定',
            '负定',
            '不定',
            '正半定',
          ],
          answer: 2,
          explain: '$\\Delta_1=1>0$，$\\Delta_2=1-4=-3<0$。行列式为负说明特征值一正一负，符号差 $(1,1)$，二次型**不定**。',
        },
        {
          id: 'u8-l4-q3',
          type: 'choice',
          prompt: '矩阵 $A = \\begin{pmatrix}2&1\\\\1&1\\end{pmatrix}$，在约束 $\\|\\mathbf{x}\\|=1$ 下，$\\mathbf{x}^{\\top}A\\mathbf{x}$ 的最小值等于',
          options: [
            '$\\lambda_{\\min}(A)$',
            '$\\lambda_{\\max}(A)$',
            '$\\det(A)$',
            '$\\mathrm{tr}(A)/2$',
          ],
          answer: 0,
          explain: 'Rayleigh 商极值定理：$\\min_{\\|\\mathbf{x}\\|=1} \\mathbf{x}^{\\top}A\\mathbf{x} = \\lambda_{\\min}(A)$，最优点是最小特征值对应的单位特征向量。',
        },
        {
          id: 'u8-l4-q4',
          type: 'judge',
          prompt: '正定矩阵 $A \\succ 0$ 的逆矩阵 $A^{-1}$ 也正定。',
          answer: true,
          explain: '若 $A \\succ 0$，则所有特征值 $\\lambda_i > 0$，$A^{-1}$ 的特征值为 $1/\\lambda_i > 0$，故 $A^{-1} \\succ 0$。',
        },
        {
          id: 'u8-l4-q5',
          type: 'choice',
          prompt: 'Hessian 矩阵 $H = \\begin{pmatrix}5&2\\\\2&2\\end{pmatrix}$ 描述的临界点类型是',
          options: [
            '局部极小值',
            '局部极大值',
            '鞍点',
            '无法判断',
          ],
          answer: 0,
          explain: '$\\Delta_1 = 5 > 0$，$\\Delta_2 = 10-4 = 6 > 0$，顺序主子式全正，$H \\succ 0$，临界点是**局部极小值**。',
        },
        {
          id: 'u8-l4-q6',
          type: 'match',
          prompt: '将二次型对应矩阵的特征值情况与其类型配对：',
          left: ['所有特征值 $> 0$', '所有特征值 $< 0$', '特征值有正有负', '特征值全 $\\geq 0$，有零特征值'],
          right: ['正定', '负定', '不定', '正半定'],
          explain: '正定: 所有 $\\lambda>0$；负定: 所有 $\\lambda<0$；不定: 有正有负；正半定: 全非负但有零。',
        },
        {
          id: 'u8-l4-q7',
          type: 'input',
          prompt: '二次型 $Q = x_1^2 + 4x_2^2 - 2x_1x_2$，顺序主子式 $\\Delta_2 =$ ？（填整数）',
          accept: ['3'],
          placeholder: '整数',
          explain: '$A = \\begin{pmatrix}1&-1\\\\-1&4\\end{pmatrix}$，$\\Delta_2 = \\det(A) = 1 \\times 4 - (-1)^2 = 4-1 = 3$。$\\Delta_1=1>0$，$\\Delta_2=3>0$，正定。',
        },
        {
          id: 'u8-l4-q8',
          type: 'choice',
          prompt: '矩阵 $A = \\begin{pmatrix}2&0\\\\0&-1\\end{pmatrix}$，对应 Hessian 的临界点是',
          options: [
            '局部极小值',
            '局部极大值',
            '鞍点',
            '全局极大值',
          ],
          answer: 2,
          explain: '特征值 $2 > 0$ 和 $-1 < 0$，符号差 $(1,1)$，不定，临界点是**鞍点**：沿 $x_1$ 方向上弯，沿 $x_2$ 方向下弯。',
        },
      ],
    },

    // ─────────────────────────────────────────────────
    // u8-l5  梯度与 Jacobian
    // ─────────────────────────────────────────────────
    {
      id: 'u8-l5',
      title: '梯度与 Jacobian',
      subtitle: '对向量和矩阵求导的三种情形',
      intro: [
        {
          title: '动机：为什么要对矩阵求导？',
          body: '梯度下降法需要计算损失函数 $\\mathcal{L}$ 对每个参数的导数。当参数是矩阵（如神经网络权重 $W \\in \\mathbb{R}^{m \\times n}$）时，我们需要"对矩阵求导"，结果还是一个矩阵。\n\n**矩阵微积分**专门处理"向量/矩阵作为自变量或因变量"的导数记法与规则。\n\n三种核心情形：\n- 向量函数对**向量**求导 → **Jacobian 矩阵**\n- 标量函数对**向量**求导 → **梯度向量**\n- 标量函数对**矩阵**求导 → **梯度矩阵**',
          tip: '黄金法则：导数的**形状**必须与被求导变量的形状一致。$f \\in \\mathbb{R}$，$\\mathbf{x} \\in \\mathbb{R}^n$ 则 $\\nabla_{\\mathbf{x}}f \\in \\mathbb{R}^n$。',
        },
        {
          title: 'Jacobian 矩阵：向量对向量的导数',
          body: '设 $\\mathbf{f}: \\mathbb{R}^n \\to \\mathbb{R}^m$，Jacobian 矩阵定义为：\n\n$J_{ij} = \\dfrac{\\partial f_i}{\\partial x_j}$，$J \\in \\mathbb{R}^{m \\times n}$\n\n**分子布局约定**（本课程统一使用）：行由**输出** $\\mathbf{f}$ 的维度决定，列由**输入** $\\mathbf{x}$ 的维度决定。\n\n**常用结果**：\n- $\\mathbf{f} = A\\mathbf{x}$（$A$ 常数）：$\\partial(A\\mathbf{x})/\\partial\\mathbf{x} = A$\n- $\\mathbf{f} = \\mathbf{x}$（恒等映射）：$\\partial\\mathbf{x}/\\partial\\mathbf{x} = I_n$\n- $\\mathbf{f} = \\mathbf{x} \\odot \\mathbf{x}$（逐元素平方）：$J = \\mathrm{diag}(2\\mathbf{x})$\n\n**几何意义**：Jacobian 是 $\\mathbf{f}$ 在该点的**最佳线性近似**；$|\\det J|$ 是局部体积缩放比。',
          formula: 'J = \\frac{\\partial \\mathbf{f}}{\\partial \\mathbf{x}} \\in \\mathbb{R}^{m \\times n}',
        },
        {
          title: '梯度向量：标量对向量的导数',
          body: '设 $f: \\mathbb{R}^n \\to \\mathbb{R}$，梯度是列向量 $\\nabla_{\\mathbf{x}} f \\in \\mathbb{R}^n$。\n\n**梯度指向增长最快的方向**；负梯度指向下降最陡方向，是梯度下降的几何基础。\n\n**核心公式**：\n\n$\\partial(\\mathbf{a}^{\\top}\\mathbf{x})/\\partial\\mathbf{x} = \\mathbf{a}$（线性函数）\n\n$\\partial(\\|\\mathbf{x}\\|^2)/\\partial\\mathbf{x} = 2\\mathbf{x}$（取 $A=I$）\n\n$\\partial(\\mathbf{x}^{\\top}A\\mathbf{x})/\\partial\\mathbf{x} = 2A\\mathbf{x}$（$A$ 对称，由 $\\partial f/\\partial x_k = (A\\mathbf{x})_k + (A^{\\top}\\mathbf{x})_k = 2(A\\mathbf{x})_k$ 推出）\n\n$\\partial(\\|A\\mathbf{x}-\\mathbf{b}\\|^2)/\\partial\\mathbf{x} = 2A^{\\top}(A\\mathbf{x}-\\mathbf{b})$（令为零得正规方程）',
          formula: '\\frac{\\partial(\\mathbf{x}^{\\top}A\\mathbf{x})}{\\partial \\mathbf{x}} = 2A\\mathbf{x} \\quad (A \\text{ 对称})',
        },
        {
          title: '梯度矩阵：标量对矩阵的导数',
          body: '设 $f: \\mathbb{R}^{m \\times n} \\to \\mathbb{R}$，则 $\\partial f/\\partial A \\in \\mathbb{R}^{m \\times n}$，$(i,j)$ 元素是 $\\partial f/\\partial a_{ij}$——与 $A$ 同形。\n\n**迹技巧**：利用 $\\mathrm{tr}(A^{\\top}B) = \\sum_{i,j}a_{ij}b_{ij}$（Frobenius 内积），将微分写成\n\n$df = \\mathrm{tr}(G^{\\top}\\,dA)$\n\n则 $G$ 就是梯度矩阵 $\\partial f/\\partial A$。\n\n**关键公式**：\n- $\\partial\\,\\mathrm{tr}(AB)/\\partial A = B^{\\top}$\n- $\\partial\\|A\\|_F^2/\\partial A = 2A$\n- $\\partial\\log\\det(A)/\\partial A = A^{-\\top}$（$A$ 可逆）\n- $\\partial(\\mathbf{a}^{\\top}A\\mathbf{b})/\\partial A = \\mathbf{a}\\mathbf{b}^{\\top}$',
          formula: '\\frac{\\partial\\,\\mathrm{tr}(AB)}{\\partial A} = B^{\\top}, \\quad \\frac{\\partial\\log\\det(A)}{\\partial A} = A^{-\\top}',
          tip: '迹技巧的核心：凑出 $\\mathrm{tr}(G^{\\top}dA)$ 的形式，括号里的 $G$ 就是梯度矩阵。',
        },
        {
          title: '常用公式汇总',
          body: '**标量对向量**（$f: \\mathbb{R}^n \\to \\mathbb{R}$）：\n\n$\\partial(\\mathbf{a}^{\\top}\\mathbf{x})/\\partial\\mathbf{x} = \\mathbf{a}$\n\n$\\partial(\\mathbf{x}^{\\top}A\\mathbf{x})/\\partial\\mathbf{x} = (A+A^{\\top})\\mathbf{x}$（$A$ 不对称时）\n\n**标量对矩阵**（$f: \\mathbb{R}^{m\\times n} \\to \\mathbb{R}$）：\n\n$\\partial\\,\\mathrm{tr}(A)/\\partial A = I$（方阵 $A$）\n\n$\\partial\\,\\mathrm{tr}(A^{\\top}B)/\\partial A = B$\n\n$\\partial(\\|A\\mathbf{x}-\\mathbf{b}\\|^2)/\\partial A = 2(A\\mathbf{x}-\\mathbf{b})\\mathbf{x}^{\\top}$（对 $A$ 求导）\n\n**维度一致性**：每次推导完毕，检查梯度形状是否与被求导变量完全一致。',
        },
        {
          title: '例题精讲',
          body: '**例 1**：$f(\\mathbf{x}) = \\mathbf{c}^{\\top}\\mathbf{x} + \\mathbf{x}^{\\top}\\mathbf{x}$，求 $\\nabla_{\\mathbf{x}} f$。\n\n$\\nabla(\\mathbf{c}^{\\top}\\mathbf{x}) = \\mathbf{c}$，$\\nabla(\\mathbf{x}^{\\top}\\mathbf{x}) = 2\\mathbf{x}$，故 $\\nabla_{\\mathbf{x}} f = \\mathbf{c} + 2\\mathbf{x}$。令梯度为零：$\\mathbf{x}^* = -\\mathbf{c}/2$（全局极小值）。\n\n**例 2**：$f(A) = \\mathrm{tr}(A^{\\top}WA)$（$W$ 对称，$A \\in \\mathbb{R}^{n \\times k}$），求 $\\partial f/\\partial A$。\n\n$df = \\mathrm{tr}(dA^{\\top}\\cdot WA) + \\mathrm{tr}(A^{\\top}W\\cdot dA) = 2\\mathrm{tr}((WA)^{\\top}dA)$（$W$ 对称故 $W^{\\top}=W$）。\n\n对比识别形式 $df = \\mathrm{tr}(G^{\\top}dA)$，得 $\\partial f/\\partial A = 2WA$。',
          reveal: {
            q: '若 $f = \\mathbf{x}^{\\top}A\\mathbf{x}$ 而 $A$ 不对称，梯度公式是什么？',
            a: '$\\nabla_{\\mathbf{x}} f = (A + A^{\\top})\\mathbf{x}$。当 $A = A^{\\top}$ 时退化为 $2A\\mathbf{x}$，与对称情形一致。',
          },
        },
      ],
      questions: [
        {
          id: 'u8-l5-q1',
          type: 'choice',
          prompt: '设 $f(\\mathbf{x}) = \\mathbf{a}^{\\top}\\mathbf{x}$，$\\mathbf{a} \\in \\mathbb{R}^n$ 为常数向量，则 $\\nabla_{\\mathbf{x}} f$ 是',
          options: [
            '$\\mathbf{a}^{\\top}$（行向量）',
            '$\\mathbf{a}$（列向量）',
            '$\\mathbf{0}$',
            '$\\|\\mathbf{a}\\|$（标量）',
          ],
          answer: 1,
          explain: '$f = \\sum_i a_i x_i$，$\\partial f/\\partial x_j = a_j$，梯度列向量 $\\nabla f = \\mathbf{a} \\in \\mathbb{R}^n$（与输入同形）。',
        },
        {
          id: 'u8-l5-q2',
          type: 'judge',
          prompt: '若 $A$ 是对称矩阵，则 $\\nabla_{\\mathbf{x}}(\\mathbf{x}^{\\top}A\\mathbf{x}) = 2A\\mathbf{x}$。',
          answer: true,
          explain: '展开 $f=\\sum_{i,j}a_{ij}x_ix_j$，$\\partial f/\\partial x_k = (A\\mathbf{x})_k + (A^{\\top}\\mathbf{x})_k = 2(A\\mathbf{x})_k$（利用 $A=A^{\\top}$），汇总得 $\\nabla f = 2A\\mathbf{x}$。',
        },
        {
          id: 'u8-l5-q3',
          type: 'choice',
          prompt: '函数 $\\mathbf{f}(\\mathbf{x}) = A\\mathbf{x}$（$A \\in \\mathbb{R}^{m \\times n}$ 为常数）的 Jacobian 矩阵是',
          options: [
            '$A^{\\top}$（$n \\times m$）',
            '$A$（$m \\times n$）',
            '$I_m$',
            '$I_n$',
          ],
          answer: 1,
          explain: '$(A\\mathbf{x})_i = \\sum_k a_{ik}x_k$，$\\partial(A\\mathbf{x})_i/\\partial x_j = a_{ij}$，Jacobian 的 $(i,j)$ 元素就是 $a_{ij}$，故 Jacobian $= A \\in \\mathbb{R}^{m \\times n}$。',
        },
        {
          id: 'u8-l5-q4',
          type: 'choice',
          prompt: '$\\partial\\,\\mathrm{tr}(AB)/\\partial A$（$B$ 为常数方阵）等于',
          options: [
            '$B$',
            '$B^{\\top}$',
            '$A^{\\top}$',
            '$\\mathrm{tr}(B)\\cdot I$',
          ],
          answer: 1,
          explain: '由迹技巧：$d\\,\\mathrm{tr}(AB) = \\mathrm{tr}(dA\\cdot B) = \\mathrm{tr}(B\\,dA)$。对比识别形式 $df = \\mathrm{tr}(G^{\\top}dA)$，得 $G^{\\top} = B$，即梯度 $G = B^{\\top}$。',
        },
        {
          id: 'u8-l5-q5',
          type: 'input',
          prompt: '设 $f(\\mathbf{x}) = \\|\\mathbf{x}\\|^2$，在 $\\mathbf{x}=(1,2,3)^{\\top}$ 处，$\\|\\nabla_{\\mathbf{x}} f\\|^2 =$ ？（填整数）',
          accept: ['56'],
          placeholder: '整数',
          explain: '$\\nabla_{\\mathbf{x}} f = 2\\mathbf{x} = (2,4,6)^{\\top}$，$\\|\\nabla f\\|^2 = 4+16+36 = 56$。',
        },
        {
          id: 'u8-l5-q6',
          type: 'match',
          prompt: '将求导场景与结果形状配对（$f$ 为标量，$\\mathbf{f}$ 为 $m$ 维向量，$\\mathbf{x} \\in \\mathbb{R}^n$，$A \\in \\mathbb{R}^{m \\times n}$）：',
          left: [
            '$\\partial f/\\partial \\mathbf{x}$',
            '$\\partial \\mathbf{f}/\\partial \\mathbf{x}$（Jacobian）',
            '$\\partial f/\\partial A$',
          ],
          right: [
            '$n \\times 1$（与 $\\mathbf{x}$ 同形）',
            '$m \\times n$（Jacobian 矩阵）',
            '$m \\times n$（与 $A$ 同形）',
          ],
          explain: '黄金法则：导数形状与被求导变量形状一致。$\\partial f/\\partial \\mathbf{x}$ 与 $\\mathbf{x}$ 同形；Jacobian $\\partial\\mathbf{f}/\\partial\\mathbf{x}$ 形状为输出 $\\times$ 输入；$\\partial f/\\partial A$ 与 $A$ 同形。',
        },
        {
          id: 'u8-l5-q7',
          type: 'judge',
          prompt: '神经网络权重矩阵 $W$ 的梯度 $\\partial\\mathcal{L}/\\partial W$ 与 $W$ 的形状相同。',
          answer: true,
          explain: '矩阵梯度的基本规则：$\\mathcal{L}$ 对矩阵 $W \\in \\mathbb{R}^{m \\times n}$ 的梯度形状为 $m \\times n$，这样才能做更新 $W \\leftarrow W - \\eta\\,\\partial\\mathcal{L}/\\partial W$。',
        },
        {
          id: 'u8-l5-q8',
          type: 'choice',
          prompt: '$\\partial\\log\\det(A)/\\partial A$（$A$ 可逆）等于',
          options: [
            '$A^{-1}$',
            '$A^{-\\top}$',
            '$\\mathrm{tr}(A^{-1})\\cdot I$',
            '$\\log(A^{-1})$',
          ],
          answer: 1,
          explain: '由迹技巧：$d\\log\\det(A) = \\mathrm{tr}(A^{-1}dA)$，对比 $df = \\mathrm{tr}(G^{\\top}dA)$ 得 $G^{\\top} = A^{-1}$，故梯度 $= A^{-\\top}$。当 $A$ 对称时，$A^{-\\top} = A^{-1}$。',
        },
      ],
    },

    // ─────────────────────────────────────────────────
    // u8-l6  链式法则与应用
    // ─────────────────────────────────────────────────
    {
      id: 'u8-l6',
      title: '链式法则与应用',
      subtitle: '矩阵链式法则与反向传播推导',
      intro: [
        {
          title: '矩阵链式法则',
          body: '单变量链式法则 $\\dfrac{dz}{dx} = \\dfrac{dz}{dy} \\cdot \\dfrac{dy}{dx}$ 推广到向量/矩阵形式：\n\n设 $\\mathbf{x} \\in \\mathbb{R}^n \\xrightarrow{\\mathbf{g}} \\mathbf{y} \\in \\mathbb{R}^m \\xrightarrow{\\mathbf{f}} \\mathbf{z} \\in \\mathbb{R}^k$，则：\n\n$\\dfrac{\\partial \\mathbf{z}}{\\partial \\mathbf{x}} = \\dfrac{\\partial \\mathbf{z}}{\\partial \\mathbf{y}} \\cdot \\dfrac{\\partial \\mathbf{y}}{\\partial \\mathbf{x}} \\in \\mathbb{R}^{k \\times n}$\n\n（$\\mathbb{R}^{k \\times m} \\cdot \\mathbb{R}^{m \\times n} = \\mathbb{R}^{k \\times n}$，维度自动匹配）\n\n**当 $\\mathbf{z}$ 为标量**（$k=1$）时，梯度为：\n\n$\\nabla_{\\mathbf{x}} z = J_{\\mathbf{g}}^{\\top} \\nabla_{\\mathbf{y}} z$\n\n其中 $J_{\\mathbf{g}} = \\partial\\mathbf{y}/\\partial\\mathbf{x} \\in \\mathbb{R}^{m \\times n}$ 是前段 Jacobian，转置后与 $\\nabla_{\\mathbf{y}}z \\in \\mathbb{R}^m$ 相乘得 $\\mathbb{R}^n$ 向量。',
          formula: '\\nabla_{\\mathbf{x}} z = J_{\\mathbf{g}}^{\\top} \\nabla_{\\mathbf{y}} z',
          tip: '矩阵链式法则中，Jacobian 要转置（从分子布局的行向量形式换回列梯度格式）——这是与标量链式法则最容易搞错的地方。',
        },
        {
          title: '计算图与前向/反向传播',
          body: '**计算图（computational graph）**是描述计算的有向无环图，每个节点代表中间变量，每条边代表函数关系。\n\n以 $z = f(g(\\mathbf{x}))$ 为例：\n\n$\\mathbf{x} \\to [g] \\to \\mathbf{y} \\to [f] \\to z$\n\n**前向传播**：从左至右计算各节点的值。\n\n**反向传播**：从右至左，用链式法则累积梯度：\n\n$\\partial z/\\partial x_i = \\sum_j (\\partial z/\\partial y_j)(\\partial y_j/\\partial x_i)$\n\n对整个网络，梯度通过 Jacobian 矩阵的乘积从输出层反向流向输入层。\n\n**自动微分（autograd）**正是系统化地实现这一过程，对每个节点存储前向值、注册反向函数。',
        },
        {
          title: '线性层的梯度推导',
          body: '设 $\\mathbf{y} = W\\mathbf{x} + \\mathbf{b}$，损失 $\\mathcal{L} = \\tfrac{1}{2}\\|\\mathbf{y}-\\mathbf{t}\\|^2$，设误差 $\\boldsymbol{\\delta} = \\mathbf{y} - \\mathbf{t}$。\n\n**对 $W$ 的梯度**：$[\\partial\\mathcal{L}/\\partial W]_{ij} = (\\partial\\mathcal{L}/\\partial y_i)(\\partial y_i/\\partial w_{ij}) = \\delta_i \\cdot x_j$，合并为外积：\n\n$\\dfrac{\\partial\\mathcal{L}}{\\partial W} = \\boldsymbol{\\delta}\\mathbf{x}^{\\top} \\in \\mathbb{R}^{m \\times n}$\n\n**对 $\\mathbf{x}$ 的梯度**：$\\partial\\mathcal{L}/\\partial\\mathbf{x} = W^{\\top}\\boldsymbol{\\delta} \\in \\mathbb{R}^n$（Jacobian 转置 $W^{\\top}$ 乘以后向误差 $\\boldsymbol{\\delta}$）\n\n**对 $\\mathbf{b}$ 的梯度**：$\\partial\\mathcal{L}/\\partial\\mathbf{b} = \\boldsymbol{\\delta} \\in \\mathbb{R}^m$',
          formula: '\\frac{\\partial\\mathcal{L}}{\\partial W} = \\boldsymbol{\\delta}\\mathbf{x}^{\\top}, \\quad \\boldsymbol{\\delta} = \\mathbf{y} - \\mathbf{t}',
        },
        {
          title: '两层网络的完整反向传播',
          body: '设两层全连接网络：$\\mathbf{z}^{(1)} = W^{(1)}\\mathbf{x}$，$\\mathbf{a}^{(1)} = \\sigma(\\mathbf{z}^{(1)})$，$\\mathbf{z}^{(2)} = W^{(2)}\\mathbf{a}^{(1)}$，$\\mathcal{L} = \\tfrac{1}{2}\\|\\mathbf{z}^{(2)}-\\mathbf{t}\\|^2$。',
          steps: [
            '**第一步**：输出层误差 $\\boldsymbol{\\delta}^{(2)} = \\partial\\mathcal{L}/\\partial\\mathbf{z}^{(2)} = \\mathbf{z}^{(2)} - \\mathbf{t}$（激活为恒等时）。',
            '**第二步**：第二层权重梯度 $\\partial\\mathcal{L}/\\partial W^{(2)} = \\boldsymbol{\\delta}^{(2)}(\\mathbf{a}^{(1)})^{\\top}$（外积）。',
            '**第三步**：误差传回隐藏层 $\\partial\\mathcal{L}/\\partial\\mathbf{a}^{(1)} = (W^{(2)})^{\\top}\\boldsymbol{\\delta}^{(2)}$（注意 $W$ 转置！）。',
            '**第四步**：通过激活函数 $\\boldsymbol{\\delta}^{(1)} = [(W^{(2)})^{\\top}\\boldsymbol{\\delta}^{(2)}] \\odot \\sigma\'(\\mathbf{z}^{(1)})$（逐元素乘激活导数）。',
            '**第五步**：第一层权重梯度 $\\partial\\mathcal{L}/\\partial W^{(1)} = \\boldsymbol{\\delta}^{(1)}\\mathbf{x}^{\\top}$。',
          ],
          tip: '反向传播的模式：每层的权重梯度 $=$ 该层误差向量 $\\otimes$ 该层输入向量（外积）。',
        },
        {
          title: '链式法则的分步例子',
          body: '**例**：$z = \\|W\\mathbf{x} - \\mathbf{b}\\|^2$，分解计算图：\n\n$\\mathbf{u} = W\\mathbf{x}$，$\\mathbf{v} = \\mathbf{u} - \\mathbf{b}$，$z = \\|\\mathbf{v}\\|^2$\n\n各段梯度：$\\partial z/\\partial \\mathbf{v} = 2\\mathbf{v}$，$\\partial\\mathbf{v}/\\partial\\mathbf{u} = I$，$\\partial\\mathbf{u}/\\partial\\mathbf{x} = W$\n\n链式法则（标量对向量）：\n\n$\\nabla_{\\mathbf{x}} z = W^{\\top} \\cdot I^{\\top} \\cdot 2\\mathbf{v} = 2W^{\\top}(W\\mathbf{x} - \\mathbf{b})$\n\n与直接展开 $z = \\mathbf{x}^{\\top}W^{\\top}W\\mathbf{x} - 2\\mathbf{b}^{\\top}W\\mathbf{x} + \\|\\mathbf{b}\\|^2$，对 $\\mathbf{x}$ 求梯度得 $2W^{\\top}W\\mathbf{x} - 2W^{\\top}\\mathbf{b} = 2W^{\\top}(W\\mathbf{x}-\\mathbf{b})$，一致。',
        },
        {
          title: '易错点总结',
          body: '**易错 1**：分子布局下，标量对列向量 $\\mathbf{x}$ 的梯度是**列向量**；而 Jacobian 的每行对应一个输出分量。不同教材布局不同，务必先确认约定。\n\n**易错 2**：矩阵链式法则中，标量对 $\\mathbf{x}$ 的梯度 $= J_{\\mathbf{g}}^{\\top}\\nabla_{\\mathbf{y}}z$，Jacobian 要**转置**。\n\n**易错 3**：$\\partial(\\mathbf{a}^{\\top}A\\mathbf{b})/\\partial A = \\mathbf{a}\\mathbf{b}^{\\top}$（外积），不是 $\\mathbf{b}\\mathbf{a}^{\\top}$。\n\n**易错 4**：反向传播中，误差从后层传到前层要乘 $W^{\\top}$（不是 $W$），这来自 Jacobian 的转置。',
          tip: '维度检验是最可靠的自查手段：每次推导完毕，检查梯度/Jacobian 的形状是否与被求导变量完全一致。',
        },
      ],
      questions: [
        {
          id: 'u8-l6-q1',
          type: 'choice',
          prompt: '设线性层 $\\mathbf{y} = W\\mathbf{x} + \\mathbf{b}$，$W \\in \\mathbb{R}^{m \\times n}$，损失 $\\mathcal{L} = \\tfrac{1}{2}\\|\\mathbf{y} - \\mathbf{t}\\|^2$。设 $\\boldsymbol{\\delta} = \\mathbf{y} - \\mathbf{t}$，则 $\\partial\\mathcal{L}/\\partial W$ 是',
          options: [
            '$\\boldsymbol{\\delta}^{\\top}\\mathbf{x}$（$1 \\times n$）',
            '$\\boldsymbol{\\delta}\\mathbf{x}^{\\top}$（$m \\times n$ 矩阵）',
            '$\\mathbf{x}\\boldsymbol{\\delta}^{\\top}$（$n \\times m$ 矩阵）',
            '$W^{\\top}\\boldsymbol{\\delta}$（$n \\times 1$ 向量）',
          ],
          answer: 1,
          explain: '$[\\partial\\mathcal{L}/\\partial W]_{ij} = \\delta_i \\cdot x_j$，合并得外积 $\\boldsymbol{\\delta}\\mathbf{x}^{\\top} \\in \\mathbb{R}^{m \\times n}$（与 $W$ 同形）。',
        },
        {
          id: 'u8-l6-q2',
          type: 'judge',
          prompt: '在多层网络的反向传播中，误差从后层 $\\boldsymbol{\\delta}^{(2)}$ 传到前层时要乘 $(W^{(2)})^{\\top}$，不是 $W^{(2)}$。',
          answer: true,
          explain: '由链式法则，$\\partial\\mathcal{L}/\\partial\\mathbf{a}^{(1)} = (W^{(2)})^{\\top}\\boldsymbol{\\delta}^{(2)}$。Jacobian $\\partial(W\\mathbf{a})/\\partial\\mathbf{a} = W$，在链式法则中需转置，故乘 $W^{\\top}$。',
        },
        {
          id: 'u8-l6-q3',
          type: 'choice',
          prompt: '矩阵链式法则：$\\mathbf{x} \\in \\mathbb{R}^n \\to \\mathbf{y} \\in \\mathbb{R}^m \\to z \\in \\mathbb{R}$，则 $\\nabla_{\\mathbf{x}} z$ 等于',
          options: [
            '$J_{\\mathbf{g}} \\nabla_{\\mathbf{y}} z$（$m \\times n$ 乘 $m \\times 1$，维度不匹配）',
            '$J_{\\mathbf{g}}^{\\top} \\nabla_{\\mathbf{y}} z$（$n \\times m$ 乘 $m \\times 1 = n \\times 1$）',
            '$\\nabla_{\\mathbf{y}} z \\cdot J_{\\mathbf{g}}$',
            '$J_{\\mathbf{g}} / \\nabla_{\\mathbf{y}} z$',
          ],
          answer: 1,
          explain: '$J_{\\mathbf{g}} = \\partial\\mathbf{y}/\\partial\\mathbf{x} \\in \\mathbb{R}^{m \\times n}$，$\\nabla_{\\mathbf{y}} z \\in \\mathbb{R}^m$。$\\nabla_{\\mathbf{x}} z = J_{\\mathbf{g}}^{\\top} \\nabla_{\\mathbf{y}} z \\in \\mathbb{R}^n$，维度正确。',
        },
        {
          id: 'u8-l6-q4',
          type: 'choice',
          prompt: '同题：$\\partial\\mathcal{L}/\\partial\\mathbf{x}$（对输入 $\\mathbf{x}$ 求梯度，$\\mathbf{y}=W\\mathbf{x}$，$\\boldsymbol{\\delta} = \\partial\\mathcal{L}/\\partial\\mathbf{y}$）是',
          options: [
            '$W\\boldsymbol{\\delta}$',
            '$W^{\\top}\\boldsymbol{\\delta}$',
            '$\\boldsymbol{\\delta}^{\\top}W$',
            '$W^{-1}\\boldsymbol{\\delta}$',
          ],
          answer: 1,
          explain: 'Jacobian $\\partial(W\\mathbf{x})/\\partial\\mathbf{x} = W \\in \\mathbb{R}^{m \\times n}$，由链式法则 $\\nabla_{\\mathbf{x}}\\mathcal{L} = W^{\\top}\\boldsymbol{\\delta} \\in \\mathbb{R}^n$（$W^{\\top}$ 将 $m$ 维误差映回 $n$ 维输入空间）。',
        },
        {
          id: 'u8-l6-q5',
          type: 'judge',
          prompt: '自动微分（autograd）与矩阵链式法则的手工推导在数学上等价。',
          answer: true,
          explain: '自动微分正是系统化地执行矩阵链式法则：前向传播存中间值，反向传播通过每层注册的 Jacobian 函数累积梯度，与手工推导完全等价。',
        },
        {
          id: 'u8-l6-q6',
          type: 'input',
          prompt: '设 $z = \\|W\\mathbf{x}\\|^2$，$W \\in \\mathbb{R}^{2 \\times 2}$，$\\mathbf{x} = (1,0)^{\\top}$，$W = \\begin{pmatrix}1&0\\\\0&2\\end{pmatrix}$，则 $z =$ ？（填整数）',
          accept: ['1'],
          placeholder: '整数',
          explain: '$W\\mathbf{x} = (1,0)^{\\top}$，$z = \\|(1,0)^{\\top}\\|^2 = 1$。验证：$z = \\mathbf{x}^{\\top}W^{\\top}W\\mathbf{x} = (1,0)\\begin{pmatrix}1&0\\\\0&4\\end{pmatrix}(1,0)^{\\top} = 1$。',
        },
        {
          id: 'u8-l6-q7',
          type: 'match',
          prompt: '将反向传播步骤与其数学表达式配对（单层 $\\mathbf{y}=W\\mathbf{x}$，$\\mathcal{L}=\\tfrac{1}{2}\\|\\mathbf{y}-\\mathbf{t}\\|^2$）：',
          left: [
            '输出层误差',
            '权重梯度',
            '输入梯度（传到前层）',
          ],
          right: [
            '$\\boldsymbol{\\delta} = \\mathbf{y} - \\mathbf{t}$',
            '$\\boldsymbol{\\delta}\\mathbf{x}^{\\top}$（外积）',
            '$W^{\\top}\\boldsymbol{\\delta}$',
          ],
          explain: '三个核心公式：误差向量 $\\boldsymbol{\\delta}$；权重梯度 $= $ 误差 $\\otimes$ 输入（外积）；输入梯度 $= W^{\\top}\\boldsymbol{\\delta}$（Jacobian 转置作用于误差）。',
        },
        {
          id: 'u8-l6-q8',
          type: 'choice',
          prompt: '两层网络 $\\mathbf{a}^{(1)} = \\sigma(W^{(1)}\\mathbf{x})$，$\\mathbf{z}^{(2)} = W^{(2)}\\mathbf{a}^{(1)}$，第一层权重梯度 $\\partial\\mathcal{L}/\\partial W^{(1)}$ 等于',
          options: [
            '$\\boldsymbol{\\delta}^{(1)}\\mathbf{x}^{\\top}$',
            '$\\mathbf{x}(\\boldsymbol{\\delta}^{(1)})^{\\top}$',
            '$(W^{(2)})^{\\top}\\boldsymbol{\\delta}^{(2)}\\mathbf{x}^{\\top}$',
            '$\\boldsymbol{\\delta}^{(2)}(\\mathbf{a}^{(1)})^{\\top}$',
          ],
          answer: 0,
          explain: '$\\boldsymbol{\\delta}^{(1)} = [(W^{(2)})^{\\top}\\boldsymbol{\\delta}^{(2)}] \\odot \\sigma\'(W^{(1)}\\mathbf{x})$ 是第一层的有效误差，权重梯度 $= \\boldsymbol{\\delta}^{(1)}\\mathbf{x}^{\\top}$（与任一层权重梯度的通用公式一致：误差 $\\otimes$ 该层输入）。',
        },
      ],
    },
  ],
}
