import three;
size(8cm);
currentprojection = orthographic(4, 3, 2);

// 正方体 ABCD-A1B1C1D1，棱长 = 1
// 底面：A(0,0,0), B(1,0,0), C(1,1,0), D(0,1,0)
// 顶面：A1(0,0,1), B1(1,0,1), C1(1,1,1), D1(0,1,1)
triple A  = (0,0,0);
triple B  = (1,0,0);
triple C  = (1,1,0);
triple D  = (0,1,0);
triple A1 = (0,0,1);
triple B1 = (1,0,1);
triple C1 = (1,1,1);
triple D1 = (0,1,1);

// 底面（实线）
draw(A--B, black+linewidth(1));
draw(B--C, black+linewidth(1));
draw(D--A, black+linewidth(1));
// 底面隐藏棱
draw(C--D, gray+linewidth(0.8)+dashed);

// 顶面
draw(A1--B1--C1--D1--cycle, black+linewidth(1));

// 竖棱（可见）
draw(A--A1, black+linewidth(1));
draw(B--B1, black+linewidth(1));
draw(C--C1, black+linewidth(1));
// 竖棱（隐藏）
draw(D--D1, gray+linewidth(0.8)+dashed);

// ===== 异面直线 AB1（红色粗线） =====
draw(A--B1, red+linewidth(2.5));

// ===== 异面直线 BC1（蓝色粗线） =====
draw(B--C1, blue+linewidth(2.5));

// 端点
dot(A,  red+linewidth(4));
dot(B1, red+linewidth(4));
dot(B,  blue+linewidth(4));
dot(C1, blue+linewidth(4));

// 顶点标签
label("$A$",  A,  SW);
label("$B$",  B,  S);
label("$C$",  C,  SE);
label("$D$",  D,  W);
label("$A_1$", A1, NW);
label("$B_1$", B1, N);
label("$C_1$", C1, NE);
label("$D_1$", D1, W);

// 图例
label("{\footnotesize 红：$AB_1$}", (0.05, 1.25, 0.5), red+fontsize(9));
label("{\footnotesize 蓝：$BC_1$}", (0.05, 1.25, 0.35), blue+fontsize(9));
