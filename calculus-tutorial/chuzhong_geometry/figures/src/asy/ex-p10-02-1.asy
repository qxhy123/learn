// p10-02 例 1：y = -x^2 + 2x + 3，A(-1,0) B(3,0) C(0,3)，P 在第一象限抛物线上
// 最优 P=(3/2, 15/4)
size(9cm);
import graph;

real f(real x) { return -x*x + 2*x + 3; }

draw((-2.2,0)--(4.2,0), arrow=Arrow(TeXHead));
draw((0,-0.5)--(0,4.7), arrow=Arrow(TeXHead));
label("$x$", (4.2,0), E);
label("$y$", (0,4.7), N);
label("$O$", (0,0), SW);

draw(graph(f, -1.4, 3.4), blue);

pair A = (-1,0);
pair B = (3,0);
pair C = (0,3);
pair P = (1.5, f(1.5));     // (1.5, 3.75)
pair H = (1.5, -1.5 + 3);   // 直线 BC: y=-x+3，H=(1.5,1.5)

draw((-0.3,3.3)--(3.5,-0.5), gray+dashed);

draw(B--C, black);
draw(P--B, red);
draw(P--C, red);
draw(P--H, red+linewidth(1pt));

dot(A); label("$A$", A, SW);
dot(B); label("$B$", B, SE);
dot(C); label("$C$", C, W);
dot(P); label("$P$", P, N);
dot(H); label("$H$", H, SE);

label("$BC$", (-0.3,3.3), NW, gray);
