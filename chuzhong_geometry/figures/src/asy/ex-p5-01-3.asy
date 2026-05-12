// 例3: 半径 r=6, 圆心角 60°，求弧长与扇形面积
size(7cm);
import geometry;

pair O = (0, 0);
real r = 2.5;
draw(circle(O, r), gray+dashed);

pair A = r * dir(60);
pair B = r * dir(0);

// sector
path sector = O -- A{dir(150)} .. {dir(-30)}arc(O, r, 60, 0) -- cycle;
filldraw(sector, lightgray, black);

// redraw radii on top
draw(O -- A);
draw(O -- B);

markangle(Label("$60^\circ$", Relative(0.5)), B, O, A, radius=0.5cm);

dot(O); label("$O$", O, SW);
dot(A); label("$A$", A, NE);
dot(B); label("$B$", B, E);

label("$r=6$", (O+A)/2 + (-0.1, 0.05), NW);
