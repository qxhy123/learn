// E.9 y=x²+bx+c, A(-1,0), B(3,0). → (x+1)(x-3)=x²-2x-3, so b=-2, c=-3. C(0,-3).
// 对称轴 x=1. 顶点 (1, -4).
size(10cm);
import graph;

real f(real x) { return x*x - 2*x - 3; }

draw((-2.5,0)--(4.5,0), arrow=Arrow(TeXHead));
draw((0,-5)--(0,2), arrow=Arrow(TeXHead));
label("$x$", (4.5,0), E);
label("$y$", (0,2), N);
label("$O$", (0,0), NE);

draw(graph(f, -1.8, 3.8), blue);

pair A = (-1,0);
pair B = (3,0);
pair C = (0,-3);

draw(A--C, black);
draw((1,-4.5)--(1,1.5), gray+dashed);

dot(A); label("$A$", A, NW);
dot(B); label("$B$", B, NE);
dot(C); label("$C$", C, SW);
label("$x=1$", (1,1.5), NE, fontsize(8));
