// E.20 y=x²/4 - 1. A(-2,0), B(2,0), C(0,-1). M=(0,0)=O, r=2.
// 以 AB 直径作 ⊙M. C 在 ⊙M 内 (OC=1 < 2).
// P 在抛物线, PT 切 ⊙M, T 切点. PT=√(PM²-4). 最小当 PM 最小.
// 示例 P=(2, 0). 取 P=(3, 5/4) on parabola? 9/4-1=5/4 ✓. PM=√(9+25/16)=√(169/16)=13/4. PT=√(169/16-4)=√(105/16)
import graph;
size(11cm);

real f(real x) { return x*x/4 - 1; }

draw((-4,0)--(4,0), arrow=Arrow(TeXHead));
draw((0,-2)--(0,3), arrow=Arrow(TeXHead));
label("$x$", (4,0), E);
label("$y$", (0,3), N);

draw(graph(f, -3.2, 3.2), blue);
draw(circle((0,0), 2), red);

pair A = (-2,0);
pair B = (2,0);
pair C = (0,-1);
pair M = (0,0);
pair P = (3, 5/4);

// 切线: T 在圆上, MT⊥PT, |MT|=2, |MP|=13/4
// T: 从 M 出发, MT 垂直 PT, |MT|=2
// 在直角△MTP: cos(∠TMP)=2/(13/4)=8/13. ∠TMP=arccos(8/13)
real ang = degrees(atan2(P.y, P.x));
real dang = degrees(acos(8/13));
pair T = M + 2*dir(ang - dang);

draw(P--T, black);
draw(M--T, gray+dashed);

dot(A); label("$A$", A, SW);
dot(B); label("$B$", B, SE);
dot(C); label("$C$", C, SW);
dot(M); label("$M$", M, NW);
dot(P); label("$P$", P, NE);
dot(T); label("$T$", T, NW);
