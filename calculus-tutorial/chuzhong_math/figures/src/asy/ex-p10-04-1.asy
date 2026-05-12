// p10-04 例 1：y=x^2-4x+3, A(1,0) B(3,0) C(0,3)；y 轴上 P 使 △PAB 等腰
// 2 个解：P(0, ±√3)
size(9cm);
import graph;

real f(real x) { return x*x - 4*x + 3; }

draw((-1,0)--(4.5,0), arrow=Arrow(TeXHead));
draw((0,-2)--(0,3.8), arrow=Arrow(TeXHead));
label("$x$", (4.5,0), E);
label("$y$", (0,3.8), N);
label("$O$", (0,0), SW);

draw(graph(f, -0.3, 4.2), blue);

pair A = (1,0);
pair B = (3,0);
pair C = (0,3);
pair P1 = (0, sqrt(3));
pair P2 = (0, -sqrt(3));

dot(A); label("$A$", A, N);
dot(B); label("$B$", B, N);
dot(C); label("$C$", C, W);

draw(A--B, black+linewidth(1pt));
draw(P1--A--B--cycle, red+dashed);
draw(P2--A--B--cycle, red+dashed);

dot(P1); label("$P_1(0,\sqrt 3)$", P1, W);
dot(P2); label("$P_2(0,-\sqrt 3)$", P2, W);
