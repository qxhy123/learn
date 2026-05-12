// p10-02 引入题：y = x^2 - 2x - 3，A(-1,0) B(3,0) C(0,-3)，P 在 BC 下方
// 求 △PBC 面积最大，P=(3/2, -15/4)
size(9cm);
import graph;

real f(real x) { return x*x - 2*x - 3; }

draw((-2.2,0)--(4.2,0), arrow=Arrow(TeXHead));
draw((0,-4.5)--(0,2), arrow=Arrow(TeXHead));
label("$x$", (4.2,0), E);
label("$y$", (0,2), N);
label("$O$", (0,0), NE);

draw(graph(f, -1.8, 3.8), blue);

pair A = (-1,0);
pair B = (3,0);
pair C = (0,-3);
pair P = (1.5, f(1.5));     // (1.5, -3.75)
pair H = (1.5, 1.5 - 3);    // 直线 BC: y=x-3，H=(1.5,-1.5)

// 直线 BC
draw((-0.3,-3.3)--(3.5,0.5), gray+dashed);

// △PBC
draw(B--C, black);
draw(P--B, red);
draw(P--C, red);

// 竖直 PH
draw(P--H, red+linewidth(1pt));

dot(A); label("$A$", A, NW);
dot(B); label("$B$", B, NE);
dot(C); label("$C$", C, W);
dot(P); label("$P$", P, S);
dot(H); label("$H$", H, NE);

label("$BC$", (3,0.5), gray);
