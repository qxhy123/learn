// 例3: 直角△ABC，∠C=90°, AC=3, BC=4, AB=5。以 AB 为直径作圆 ⊙O (O=AB中点, r=2.5)
// 过 O 作 OD ⊥ BC 于 D, OD=1.5 < 2.5 → 相交
size(9cm);
import geometry;

pair Cpt = (0, 0);
pair B = (4*0.6, 0);     // visual scale 0.6
pair A = (0, 3*0.6);
pair O = (A + B)/2;
real r = length(A - B)/2;

draw(circle(O, r));
draw(A--B); draw(B--Cpt); draw(Cpt--A);

// foot of perp from O to BC (x-axis): D = (O.x, 0)
pair D = (O.x, 0);
draw(O -- D, red);
markrightangle(B, D, O, size=0.15cm);
markrightangle(B, Cpt, A, size=0.15cm);

dot(A); label("$A$", A, NW);
dot(B); label("$B$", B, SE);
dot(Cpt); label("$C$", Cpt, SW);
dot(O); label("$O$", O, NE);
dot(D); label("$D$", D, S);

label("$3$", (A+Cpt)/2, W);
label("$4$", (B+Cpt)/2, S);
label("$5$", (A+B)/2 + (0.15, 0.15), NE);
label("$OD=\frac{3}{2}$", D + (0.1, 0.4), E, red);
