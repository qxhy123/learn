// 例1: PA=PB=4, ∠APB=60° → AB=4 等边
size(8cm);
import geometry;

// place so that ∠APB = 60° and PA=PB=4 (visual scale 0.5)
real s = 0.6;
pair P = (0, 0);
pair A = P + s*4 * dir(150);   // 60° between
pair B = P + s*4 * dir(210);

// determine center O on bisector of ∠APB (along negative x), so that OA ⊥ PA
// At A, tangent direction is along PA direction (P→A); radius OA ⊥ PA.
// O lies on perpendicular at A to PA. By symmetry O is on x-axis with x<0.
pair dirPA = unit(A - P);
pair perpA = rotate(90)*dirPA;
// Solve A + t*perpA = (x, 0). y-coord: A.y + t*perpA.y = 0 → t = -A.y/perpA.y
real t = -A.y/perpA.y;
pair O = A + t*perpA;
real r = length(O - A);

draw(circle(O, r));

draw(P--A); draw(P--B); draw(A--B, red);
draw(O--A, gray); draw(O--B, gray);

markrightangle(P, A, O, size=0.12cm);
markrightangle(O, B, P, size=0.12cm);
markangle(Label("$60^\circ$", Relative(0.5)), A, P, B, radius=0.5cm);

dot(P); label("$P$", P, E);
dot(A); label("$A$", A, NW);
dot(B); label("$B$", B, SW);
dot(O); label("$O$", O, W);

label("$4$", (P+A)/2, N);
label("$4$", (P+B)/2, S);
label("$AB=4$", (A+B)/2, W, red);
