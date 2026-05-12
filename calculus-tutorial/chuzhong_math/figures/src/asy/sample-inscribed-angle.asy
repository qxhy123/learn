// 样例 3：圆周角定理（part5/03 一图速记）
// Asymptote 画圆 + 圆心角 2θ + 圆周角 θ + 自动角弧标注
size(8cm);
import geometry;

pair O = (0, 0);
real r = 2;
draw(circle(O, r));

pair A = r * dir(210);
pair B = r * dir(330);
pair C = r * dir(90);

// 弦
draw(A -- B);
draw(A -- C);
draw(B -- C);

// 半径（虚线）
draw(O -- A, dashed);
draw(O -- B, dashed);

// 角弧 + 标签
markangle(Label("$2\theta$", Relative(0.5)), A, O, B, radius=0.6cm);
markangle(Label("$\theta$",  Relative(0.5)), A, C, B, radius=0.5cm);

// 顶点标签
dot(O); label("$O$", O, NE);
dot(A); label("$A$", A, SW);
dot(B); label("$B$", B, SE);
dot(C); label("$C$", C, N);
