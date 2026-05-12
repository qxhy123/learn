// 四点共圆 例3：圆幂逆用 PA=2 PC=6 PB=3 PD=4 -> 12=12 共圆
size(8cm);
import geometry;

pair P = (0, 0);
pair A = 2 * dir(150);
pair C = 6 * dir(-30);   // 与 A 对顶, AC 过 P
pair B = 3 * dir(60);
pair D = 4 * dir(-120);

draw(A -- C);
draw(B -- D);

// 共圆：四点共圆，画出来
path circ = circle(A, B, C);
draw(circ, gray+dashed);

dot(P); label("$P$", P, NE);
dot(A); label("$A$", A, NW);
dot(B); label("$B$", B, NE);
dot(C); label("$C$", C, SE);
dot(D); label("$D$", D, SW);

label("$PA=2$", (P+A)/2, NE);
label("$PC=6$", (P+C)/2, SW);
label("$PB=3$", (P+B)/2, NW);
label("$PD=4$", (P+D)/2, NE);
