// p10-03 例 3 阿氏圆：r=2, OB=4 (外), OB'=1 (内), OA=5
// 关键相似：OB·OB'=r²，△OB'P~△OPB，比 1/2，故 PB'=½PB
settings.tex = "xelatex";
texpreamble("\usepackage{ctex}\usepackage{amsmath}");
size(11cm);
import graph;

real r = 2;
pair O = (0,0);
pair Bp = (1, 0);      // B' 反演点（圆内）
pair B = (4, 0);
pair A = (5, 0);

// 选一个示意 P（用于画相似三角形 / 折线）
pair P = (r * Cos(60), r * Sin(60));  // ≈ (1, 1.732), Cos/Sin 用度

// 圆
draw(circle(O, r), black);
dot(O); label("$O$", O, SW);
// x 轴示意
draw((-2.5, 0) -- (5.7, 0), gray+linewidth(0.4));

// 点
dot(A); label("$A$", A, S);
dot(B); label("$B$", B, S);
dot(Bp); label("$B'$", Bp, S);
dot(P); label("$P$", P, NW);

// 半径
draw(O--P, gray+dashed);
label("$r=2$", (O+P)/2, NW, fontsize(8pt));

// PA, PB, PB'
draw(P--A, red+linewidth(1pt));
draw(P--B, blue+linewidth(1pt));
draw(P--Bp, deepgreen+linewidth(1pt));

label("$PA$", (P+A)/2, NE, fontsize(9pt));
label("$PB$", (P+B)/2, N, fontsize(9pt));
label("$PB'$", (P+Bp)/2, W, fontsize(9pt));

// 标距离
label("$OB'=1$", (Bp.x/2, -0.4), fontsize(8pt));
label("$OB=4$", (2, -0.7), fontsize(8pt));
label("$OA=5$", (2.5, -1.1), fontsize(8pt));

// 相似三角形提示
label("$\triangle OB'P \sim \triangle OPB$，比 $\dfrac{1}{2}$", (1.5, 2.4), fontsize(9pt));
label("$\Rightarrow PB' = \dfrac{1}{2} PB$", (1.5, 1.9), fontsize(9pt));
