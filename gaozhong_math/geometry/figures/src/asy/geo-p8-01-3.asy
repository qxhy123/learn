import three;
size(8cm);
currentprojection = orthographic(3, 2, 1.5);

// 正四面体：边长 = 1
// 顶点坐标（标准正四面体，底面在 z=0）
triple A = (0, 0, 0);
triple B = (1, 0, 0);
triple C = (0.5, sqrt(3)/2, 0);
triple D = (0.5, sqrt(3)/6, sqrt(6)/3);

// 颜色：底面黑色实线，侧棱黑色实线，隐藏棱灰色虚线
// 底面 ABC
draw(A--B, black+linewidth(1.2));
draw(B--C, black+linewidth(1.2));
draw(C--A, gray+linewidth(0.8)+dashed);  // 隐藏棱

// 侧棱 DA, DB, DC
draw(D--A, black+linewidth(1.2));
draw(D--B, black+linewidth(1.2));
draw(D--C, black+linewidth(1.2));

// 顶点
dot(A, black+linewidth(3));
dot(B, black+linewidth(3));
dot(C, black+linewidth(3));
dot(D, black+linewidth(3));

// 标签
label("$A$", A, SW);
label("$B$", B, SE);
label("$C$", C, E);
label("$D$", D, N);

// 底部标注
label("正四面体（四个等边三角形面）", (0.5, sqrt(3)/2*0.15, -0.18), fontsize(9));
