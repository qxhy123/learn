// p10-04 例 2：y = -1/2 x^2 + 3/2 x + 2, A(-1,0) B(4,0) C(0,2)
// 抛物线上 P 使 △PBC 直角：P1=(-5,-18) 太远；P3=(1,3)。这里只画 P3，并示意 P1 在远处
size(10cm);
import graph;

real f(real x) { return -0.5*x*x + 1.5*x + 2; }

draw((-2.5,0)--(5,0), arrow=Arrow(TeXHead));
draw((0,-1.5)--(0,3.5), arrow=Arrow(TeXHead));
label("$x$", (5,0), E);
label("$y$", (0,3.5), N);
label("$O$", (0,0), SW);

draw(graph(f, -1.8, 4.5), blue);

pair A = (-1,0);
pair B = (4,0);
pair C = (0,2);
pair P3 = (1, 3);

dot(A); label("$A$", A, SW);
dot(B); label("$B$", B, SE);
dot(C); label("$C$", C, W);
dot(P3); label("$P_3$", P3, N);

draw(B--C, black+linewidth(1pt));
draw(P3--B, red);
draw(P3--C, red);

// 在 P3 处标直角
// 单位向量 P3→C = (-1,-1)/√2 ; P3→B = (3,-3)/√18 = (1,-1)/√2
// 直角符号
pair u1 = unit(C - P3) * 0.22;
pair u2 = unit(B - P3) * 0.22;
draw(P3 + u1 -- P3 + u1 + u2 -- P3 + u2, red);

label("$P_1(-5,-18)$ off-figure", (2.5, -1.2), gray+fontsize(8pt));
