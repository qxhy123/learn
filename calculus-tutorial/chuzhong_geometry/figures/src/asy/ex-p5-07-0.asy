// 切割线 引入：PA=6, PB=4, 求 PC=9
size(9cm);
import geometry;

pair O = (0, 0);
real r = 2.5;
draw(circle(O, r));

pair P = (5.5, -1.5);
pair M = (O + P)/2;
real rM = length(O - P)/2;
pair[] inters = intersectionpoints(circle(O, r), circle(M, rM));
pair A = (inters[0].y > inters[1].y) ? inters[0] : inters[1];

// 割线 PBC: choose direction to pass through circle
pair dir1 = unit((-3.5, 1.2));
pair[] sec = intersectionpoints(line(P, P + dir1, extendA=true, extendB=true), circle(O, r));
pair B0, C0;
if (length(sec[0] - P) < length(sec[1] - P)) { B0 = sec[0]; C0 = sec[1]; }
else { B0 = sec[1]; C0 = sec[0]; }

draw(P -- A);
draw(O -- A, dashed+gray);
draw(P -- C0);

markrightangle(O, A, P, size=0.18cm);

dot(O); label("$O$", O, SW);
dot(P); label("$P$", P, E);
dot(A); label("$A$", A, NW);
dot(B0); label("$B$", B0, SE);
dot(C0); label("$C$", C0, NW);

label("$PA=6$", (P+A)/2, NE);
label("$PB=4$", (P+B0)/2 + (0.2, -0.3), S);
label("$PC=?$", (P+C0)/2 + (-0.5, 0.4), N);
