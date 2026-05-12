// 圆基本概念：圆 O、半径、弦 AB、圆心角 ∠AOB、优/劣弧
size(8cm);
import geometry;

pair O = (0, 0);
real r = 2;
draw(circle(O, r));

pair A = r * dir(200);
pair B = r * dir(340);

// chord
draw(A -- B);
// radii
draw(O -- A, gray);
draw(O -- B, gray);

// central angle arc
markangle(Label("$\angle AOB$", Relative(0.5)), B, O, A, radius=0.5cm);

// minor arc highlight (below chord)
path minor = arc(O, r, 200, 340, CW);
draw(minor, red+linewidth(1.2));
// major arc -> rest (the default circle already drawn). Add a small label on major
label("minor arc", r*dir(270) + (0, -0.35), S, red);
label("major arc", r*dir(90) + (0, 0.25), N);

dot(O); label("$O$", O, NW);
dot(A); label("$A$", A, W);
dot(B); label("$B$", B, E);

// midpoint of chord to indicate "chord-center distance" optional
pair M = (A + B)/2;
draw(O -- M, dashed+gray);
label("$d$", (O+M)/2, E, gray);
