// 切割线定理：圆 O 外一点 P，切线 PA、割线 PBC → PA^2 = PB·PC
size(9cm);
import geometry;

pair O = (0, 0);
real r = 2;
draw(circle(O, r));

// 外点 P：放在 (4, -1.5)，距 O 约 4.27
pair P = (4, -1.5);

// 切点 A：满足 PA ⊥ OA。|OP|^2 = |OA|^2 + |PA|^2 → |PA| = sqrt(|OP|^2 - r^2)
// 切点位于以 OP 中点为圆心、半径=|OP|/2 的圆与原圆的交点（上方那个）
pair M = (O + P)/2;
real rM = length(O - P)/2;
pair[] inters = intersectionpoints(circle(O, r), circle(M, rM));
// 取 y 较大的点作为 A
pair A = (inters[0].y > inters[1].y) ? inters[0] : inters[1];

// 割线 PBC：取一条过 P 且穿过圆的直线；方向向左偏上
pair dir1 = unit((-3.5, 1.0));
// 求该直线与圆交点
pair[] sec = intersectionpoints(line(P, P + dir1, extendA=true, extendB=true), circle(O, r));
// B 距 P 近，C 远
pair B0, C0;
if (length(sec[0] - P) < length(sec[1] - P)) { B0 = sec[0]; C0 = sec[1]; }
else { B0 = sec[1]; C0 = sec[0]; }

// 画切线 PA
draw(P -- A);
// 半径 OA 虚线
draw(O -- A, dashed);
// 割线（从 P 穿过 B 到 C，稍微延长一点）
draw(P -- C0);

// 直角符号 ∠OAP
markrightangle(O, A, P, size=0.18cm);

// 点与标签
dot(O); label("$O$", O, SW);
dot(P); label("$P$", P, E);
dot(A); label("$A$", A, NW);
dot(B0); label("$B$", B0, SE);
dot(C0); label("$C$", C0, NW);

// 公式标签放在图下方
label("$PA^2 = PB \cdot PC$", (1, -3.0));
