// 相交弦 例3：圆内 P, PA=2, PB=6, PC=3 求 PD=4
size(8cm);
import geometry;

pair O = (0, 0);
real r = 3;
draw(circle(O, r));

// 选两条弦使其相交于 P. 让 P=(0.5, 0). 弦 AB 水平: A=(-x,0) 不行, 让 P 不在圆心
pair P = (0.5, 0);
// 弦 AB: 通过 P, 方向 (1, 0.3)
pair dir1 = unit((1, 0.3));
pair[] s1 = intersectionpoints(line(P, P + dir1, extendA=true, extendB=true), circle(O, r));
pair A = s1[0]; pair B = s1[1];
// 弦 CD: 通过 P, 方向 (1, -1.5)
pair dir2 = unit((1, -1.5));
pair[] s2 = intersectionpoints(line(P, P + dir2, extendA=true, extendB=true), circle(O, r));
pair C = s2[0]; pair D = s2[1];

draw(A -- B);
draw(C -- D);

dot(O); label("$O$", O, NW);
dot(P); label("$P$", P, NE);
dot(A); label("$A$", A, W);
dot(B); label("$B$", B, E);
dot(C); label("$C$", C, NE);
dot(D); label("$D$", D, SW);

label("$PA \cdot PB = PC \cdot PD$", (0, -3.6));
