// 四点共圆：圆周角定理推论 —— 同弧 AD 对的圆周角 ∠ABD = ∠ACD
size(9cm);
import geometry;

pair O = (0, 0);
real r = 2;
draw(circle(O, r));

// 四点按顺时针顺序，避免均匀分布
pair A = r * dir(155);
pair B = r * dir(70);
pair C = r * dir(-10);
pair D = r * dir(-110);

// 四边
draw(A -- B);
draw(B -- C);
draw(C -- D);
draw(D -- A);

// 两条对角线
draw(A -- C, gray);
draw(B -- D, gray);

// 同弧 AD 对的两个圆周角：∠ABD 和 ∠ACD
markangle(Label("$\alpha$", Relative(0.5)), A, B, D, radius=0.55cm);
markangle(Label("$\alpha$", Relative(0.5)), A, C, D, radius=0.55cm);

dot(O); label("$O$", O, SE);
dot(A); label("$A$", A, NW);
dot(B); label("$B$", B, NE);
dot(C); label("$C$", C, E);
dot(D); label("$D$", D, SW);
