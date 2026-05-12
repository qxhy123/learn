// 例2: 3-4-5 直角三角形内切圆 r=1
size(8cm);
import geometry;

real s = 0.7;
pair Cpt = (0, 0);
pair B = (4*s, 0);
pair A = (0, 3*s);

// incircle radius (scaled): r = 1*s; incenter at (r, r) for right triangle with legs on axes
real r = 1*s;
pair I = (r, r);

draw(circle(I, r));
draw(A--B--Cpt--cycle);

markrightangle(B, Cpt, A, size=0.15cm);

dot(A); label("$A$", A, NW);
dot(B); label("$B$", B, SE);
dot(Cpt); label("$C$", Cpt, SW);
dot(I); label("$I$", I, NE);

label("$3$", (A+Cpt)/2, W);
label("$4$", (B+Cpt)/2, S);
label("$5$", (A+B)/2, NE);
label("$r=1$", I + (0.3, 0.05), E);
