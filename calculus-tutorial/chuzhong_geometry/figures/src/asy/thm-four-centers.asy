// 四心：重心 G / 内心 I / 外心 O / 垂心 H 各一张图
settings.tex = "xelatex";
texpreamble("\usepackage{ctex}\usepackage{amsmath}");
size(16cm);
import geometry;

// 公共三角形顶点
pair A = (0, 3);
pair B = (-2.4, 0);
pair C = (2.6, 0);

// 偏移函数：把一个三角形 + 其特殊点画在指定偏移
void drawTri(pair off, string title) {
  draw(shift(off)*(A--B--C--cycle), black+linewidth(1pt));
  label(title, off + (0.1, -0.6), S, fontsize(10pt));
}

// 重心 G = (A+B+C)/3
pair G = (A + B + C) / 3;
// 中点
pair Ma = (B + C)/2;
pair Mb = (A + C)/2;
pair Mc = (A + B)/2;

// ---- 重心 ----
pair off1 = (0, 0);
drawTri(off1, "重心 $G$（三中线交点）");
draw(shift(off1)*(A--Ma), red+dashed);
draw(shift(off1)*(B--Mb), red+dashed);
draw(shift(off1)*(C--Mc), red+dashed);
dot(off1 + G); label("$G$", off1 + G, NE);
dot(off1 + A); label("$A$", off1 + A, N);
dot(off1 + B); label("$B$", off1 + B, SW);
dot(off1 + C); label("$C$", off1 + C, SE);

// ---- 内心 I (角平分线交点) ----
pair off2 = (8, 0);
// 内心：用边长加权
real a = length(B - C), b = length(A - C), c = length(A - B);
pair I = (a*A + b*B + c*C) / (a + b + c);
drawTri(off2, "内心 $I$（三角平分线交点）");
draw(shift(off2)*(A--I), red+dashed);
draw(shift(off2)*(B--I), red+dashed);
draw(shift(off2)*(C--I), red+dashed);
// 内切圆
real rin = abs((cross(B - A, C - A))) / (a + b + c);
draw(shift(off2)*circle(I, rin), gray+dashed);
dot(off2 + I); label("$I$", off2 + I, NE);
dot(off2 + A); label("$A$", off2 + A, N);
dot(off2 + B); label("$B$", off2 + B, SW);
dot(off2 + C); label("$C$", off2 + C, SE);

// ---- 外心 O (中垂线交点) ----
pair off3 = (0, -5);
// 外心：解中垂线交点。用公式：圆心 = ?
// 对 △ABC 顶点的外接圆心
pair ext_center(pair P1, pair P2, pair P3) {
  real ax = P1.x, ay = P1.y, bx = P2.x, by = P2.y, cx = P3.x, cy = P3.y;
  real d = 2*(ax*(by - cy) + bx*(cy - ay) + cx*(ay - by));
  real ux = ((ax^2 + ay^2)*(by - cy) + (bx^2 + by^2)*(cy - ay) + (cx^2 + cy^2)*(ay - by)) / d;
  real uy = ((ax^2 + ay^2)*(cx - bx) + (bx^2 + by^2)*(ax - cx) + (cx^2 + cy^2)*(bx - ax)) / d;
  return (ux, uy);
}
pair Oc = ext_center(A, B, C);
real R = length(A - Oc);
drawTri(off3, "外心 $O$（三中垂线交点）");
draw(shift(off3)*circle(Oc, R), gray+dashed);
// 中垂线 (从中点垂直边)
draw(shift(off3)*(Ma--Oc), red+dashed);
draw(shift(off3)*(Mb--Oc), red+dashed);
draw(shift(off3)*(Mc--Oc), red+dashed);
dot(off3 + Oc); label("$O$", off3 + Oc, NE);
dot(off3 + A); label("$A$", off3 + A, N);
dot(off3 + B); label("$B$", off3 + B, SW);
dot(off3 + C); label("$C$", off3 + C, SE);

// ---- 垂心 H ----
pair off4 = (8, -5);
// 垂心：三高交点。
// 从 A 作 BC 的垂足
pair foot(pair P, pair L1, pair L2) {
  pair d = L2 - L1;
  real t = dot(P - L1, d) / dot(d, d);
  return L1 + t * d;
}
pair Ha = foot(A, B, C);
pair Hb = foot(B, A, C);
pair Hc = foot(C, A, B);
// 垂心 = 两高线交点
pair H;
{
  // 解 A + s(Ha - A) = B + t(Hb - B)
  pair d1 = Ha - A;
  pair d2 = Hb - B;
  // s*d1 - t*d2 = B - A
  real det = d1.x * (-d2.y) - d1.y * (-d2.x);
  real s = ((B.x - A.x) * (-d2.y) - (B.y - A.y) * (-d2.x)) / det;
  H = A + s * d1;
}
drawTri(off4, "垂心 $H$（三高交点）");
draw(shift(off4)*(A--Ha), red+dashed);
draw(shift(off4)*(B--Hb), red+dashed);
draw(shift(off4)*(C--Hc), red+dashed);
dot(off4 + H); label("$H$", off4 + H, NE);
dot(off4 + A); label("$A$", off4 + A, N);
dot(off4 + B); label("$B$", off4 + B, SW);
dot(off4 + C); label("$C$", off4 + C, SE);
