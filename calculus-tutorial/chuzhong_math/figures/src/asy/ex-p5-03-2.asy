// 例2: AB 直径, AC=6, BC=8, ∠ACB=90°, r=5
size(8cm);
import geometry;

pair O = (0, 0);
real r = 2.5;
draw(circle(O, r));

// AB horizontal diameter; place C so that AC=6, BC=8, AB=10 → C such that AC/AB=3/5
// In coords: A=(-r,0), B=(r,0), C on circle with foot of altitude from C to AB at x where AD = AC^2/AB = 36/10 = 3.6 → from A. So Cx = -r + 3.6*(2r/10) = -r + 0.72r? Use scaled.
// Let AB length = 2r = 5 in visual? Actually use real 6-8-10 mapped. Set r=2.5 visual, full AB=5 visual.
// In coords: A=(-2.5, 0), B=(2.5, 0). For 6-8-10 triangle: foot of altitude from C divides AB s.t. AD=AC^2/AB=3.6 (real). Scaled: 3.6/10 * 5 = 1.8 from A. So D at A.x+1.8 = -0.7.
// Height CD = AC*BC/AB = 48/10 = 4.8. Scaled = 4.8/10*5 = 2.4
pair A = (-r, 0);
pair B = ( r, 0);
pair Cpt = (-0.7, 2.4);

draw(A--B);
draw(A--Cpt);
draw(B--Cpt);

markrightangle(B, Cpt, A, size=0.18cm);

dot(O); label("$O$", O, S);
dot(A); label("$A$", A, SW);
dot(B); label("$B$", B, SE);
dot(Cpt); label("$C$", Cpt, N);

label("$6$", (A+Cpt)/2, NW);
label("$8$", (B+Cpt)/2, NE);
label("$10$", (A+B)/2, S);
