// 切割线 例2：两割线 PA=4, PB=9, PC=3 求 PD=12
size(9cm);
import geometry;

pair O = (0, 0);
real r = 2.5;
draw(circle(O, r));

pair P = (6.5, 0);

pair dir1 = unit((-1, 0.3));
pair[] s1 = intersectionpoints(line(P, P + dir1, extendA=true, extendB=true), circle(O, r));
pair A = (length(s1[0] - P) < length(s1[1] - P)) ? s1[0] : s1[1];
pair B = (length(s1[0] - P) < length(s1[1] - P)) ? s1[1] : s1[0];

pair dir2 = unit((-1, -0.4));
pair[] s2 = intersectionpoints(line(P, P + dir2, extendA=true, extendB=true), circle(O, r));
pair C = (length(s2[0] - P) < length(s2[1] - P)) ? s2[0] : s2[1];
pair D = (length(s2[0] - P) < length(s2[1] - P)) ? s2[1] : s2[0];

draw(P -- B);
draw(P -- D);

dot(O); label("$O$", O, S);
dot(P); label("$P$", P, E);
dot(A); label("$A$", A, NE);
dot(B); label("$B$", B, NW);
dot(C); label("$C$", C, SE);
dot(D); label("$D$", D, SW);

label("$PA \cdot PB = PC \cdot PD$", (0, -3.5));
