// E.8 y = x²/2 - 3x/2 - 2. 因式: (x-4)(x+1)/2 → A(-1,0), B(4,0), C(0,-2)
// 对称轴 x=3/2. 顶点 (1.5, -25/8) = (1.5, -3.125)
size(10cm);
import graph;

real f(real x) { return 0.5*x*x - 1.5*x - 2; }

draw((-2.5,0)--(5,0), arrow=Arrow(TeXHead));
draw((0,-4)--(0,2), arrow=Arrow(TeXHead));
label("$x$", (5,0), E);
label("$y$", (0,2), N);
label("$O$", (0,0), NE);

draw(graph(f, -1.7, 4.7), blue);

pair A = (-1,0);
pair B = (4,0);
pair C = (0,-2);

draw(B--C, black);
draw((1.5,-3.5)--(1.5,1.0), gray+dashed);

dot(A); label("$A$", A, NW);
dot(B); label("$B$", B, NE);
dot(C); label("$C$", C, SW);
label("$x=\frac{3}{2}$", (1.5,1.0), NE, fontsize(8));
