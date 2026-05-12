// 例2: 赵州桥 — 拱跨 36, 拱高 7.2, 求半径 r=26.1
size(11cm);
import geometry;

// scale: 1 unit = 2 meters; arc width 18, height 3.6 visual
real scale = 0.5;
real halfL = 18 * scale;   // 9
real h = 7.2 * scale;       // 3.6
real R = 26.1 * scale;      // 13.05

// center O is below the chord by R - h
pair M = (0, 0);          // chord midpoint at origin
pair O = (0, -(R - h));   // below
pair A = (-halfL, 0);
pair B = ( halfL, 0);
pair C = (0, h);          // arch top

// draw arc (upper portion only) from A to B
real angA = degrees(atan2(A.y - O.y, A.x - O.x));
real angB = degrees(atan2(B.y - O.y, B.x - O.x));
draw(arc(O, R, angA, angB));

// chord
draw(A -- B);
// arch height
draw(M -- C, red);
// radii (dashed)
draw(O -- A, gray+dashed);
draw(O -- C, gray+dashed);

markrightangle(O, M, A, size=0.18cm);

dot(A); label("$A$", A, SW);
dot(B); label("$B$", B, SE);
dot(M); label("$M$", M, SE);
dot(C); label("$C$", C, N);
dot(O); label("$O$", O, S);

label("$36$ m", (A+B)/2, S);
label("$7.2$ m", (M+C)/2, E, red);
label("$r$", (O+A)/2 + (-0.15, 0), W, gray);
