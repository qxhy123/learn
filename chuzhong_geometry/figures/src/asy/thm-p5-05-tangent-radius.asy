// 切线 ⊥ 过切点的半径
size(8cm);
import geometry;

pair O = (0, 0);
real r = 2;
draw(circle(O, r));

pair A = r*dir(60);

// tangent line at A: perpendicular to OA
pair t = unit(rotate(90)*A);
pair P1 = A - 2.2*t;
pair P2 = A + 2.2*t;
draw(P1 -- P2);

draw(O -- A);
markrightangle(P2, A, O, size=0.18cm);

dot(O); label("$O$", O, SW);
dot(A); label("$A$", A, dir(60));
label("$r$", (O+A)/2, NW);
label("tangent $\ell$", P2, E);
