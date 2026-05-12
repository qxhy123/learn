// E.18 ⊙O r=5, PA=2 (P 在 BA 延长线). PO=7. 切线 PC: PC²=PA·PB=2·12=24 → PC=2√6
// O(0,0), A(-5,0), B(5,0), P(-7,0). C 是切点: OC⊥PC. OC=5, OP=7 → PC=√(49-25)=√24=2√6
// C: 在圆上, OC⊥PC. C 坐标: 圆 x²+y²=25 与 (x+7)·x+y²=0 → x²+7x+y²=0, 即 25+7x=0 → x=-25/7, y²=25-625/49=600/49, y=10√6/7≈3.499
// CD⊥AB 于 D → D=(-25/7, 0)
import graph;
size(11cm);

pair O = (0,0);
pair A = (-5,0);
pair B = (5,0);
pair P = (-7,0);
pair C = (-25/7, 10*sqrt(6)/7);
pair D = (-25/7, 0);

draw(circle(O, 5), blue);
draw(P--B, black);
draw(P--C, red);
draw(C--D, red+dashed);
draw(O--C, gray);

dot(O); label("$O$", O, S);
dot(A); label("$A$", A, S);
dot(B); label("$B$", B, SE);
dot(P); label("$P$", P, SW);
dot(C); label("$C$", C, NW);
dot(D); label("$D$", D, S);

// 直角 OC⊥PC
real cAngle = degrees(atan2(C.y-O.y, C.x-O.x));
// 直角 CD⊥AB
draw((-25/7+0.25, 0)--(-25/7+0.25, 0.25)--(-25/7, 0.25));
