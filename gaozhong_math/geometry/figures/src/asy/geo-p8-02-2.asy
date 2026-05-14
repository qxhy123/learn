import three;
size(9cm);
currentprojection = orthographic(4, 3, 2);

// 正方体 ABCD-A1B1C1D1，棱长 = 1
triple A  = (0,0,0);
triple B  = (1,0,0);
triple C  = (1,1,0);
triple D  = (0,1,0);
triple A1 = (0,0,1);
triple B1 = (1,0,1);
triple C1 = (1,1,1);
triple D1 = (0,1,1);

// 底面
draw(A--B, black+linewidth(1));
draw(B--C, black+linewidth(1));
draw(D--A, black+linewidth(1));
draw(C--D, gray+linewidth(0.8)+dashed);

// 顶面
draw(A1--B1--C1--D1--cycle, black+linewidth(1));

// 竖棱
draw(A--A1, black+linewidth(1));
draw(B--B1, black+linewidth(1));
draw(C--C1, black+linewidth(1));
draw(D--D1, gray+linewidth(0.8)+dashed);

// ===== AB1：原始异面直线之一（红色） =====
draw(A--B1, red+linewidth(2.5));

// ===== BC1（蓝色虚线，原始位置） =====
draw(B--C1, blue+linewidth(1.5)+dashed);

// ===== 平移 BC1 → AD1（蓝色粗线，平行替代） =====
// BC1: B(1,0,0) → C1(1,1,1)，方向向量 (0,1,1)
// 平移到 A(0,0,0) → D1(0,1,1)
draw(A--D1, blue+linewidth(2.5));

// ===== 三角形 AB1D1（绿色，表示所成角） =====
draw(A--B1, red+linewidth(2.5));       // 已画
draw(B1--D1, green+linewidth(2));
draw(D1--A, blue+linewidth(2.5));      // 已画

// 在 A 处标注角度
// A--B1 方向 (1,0,1)，A--D1 方向 (0,1,1)，夹角 π/3
// 用小弧表示
path3 arc1 = arc(A, 0.25*unit(B1-A), 0.25*unit(D1-A));
draw(arc1, black+linewidth(1));
label("$\theta$", A + 0.3*(unit(B1-A)+unit(D1-A)), fontsize(9));

// 端点加粗
dot(A,  black+linewidth(4));
dot(B1, red+linewidth(4));
dot(D1, blue+linewidth(4));

// 顶点标签
label("$A$",  A,  SW);
label("$B$",  B,  S);
label("$C$",  C,  SE);
label("$D$",  D,  W);
label("$A_1$", A1, NW);
label("$B_1$", B1, N);
label("$C_1$", C1, NE);
label("$D_1$", D1, W);

// 注释
label("$BC_1 \parallel AD_1$", (0.6, 1.32, 0.5), fontsize(9));
label("$\triangle AB_1D_1$ 为等边三角形", (0.6, 1.32, 0.35), fontsize(9));
