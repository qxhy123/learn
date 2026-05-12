// 四点共圆 引入：∠BAC=∠BDC=35°, A,D 在 BC 同侧 -> 共圆
size(9cm);
import geometry;

pair B = (-2.5, 0);
pair C = (2.5, 0);
// A, D 在 BC 同侧, 对 BC 张角都是 35° -> 都在以 BC 为弦张角 35° 的弧上
// 该弧的圆: 半径 R = BC / (2 sin 35°) = 5/(2*0.5736) ≈ 4.358
// 圆心在 BC 中垂线上, 与 BC 距离 = BC/2 * cot(35°) = 2.5 * 1.428 ≈ 3.57
real R = 2.5 / sin(35 * pi/180);
pair O = (0, 2.5/tan(35*pi/180));  // 圆心在上方
draw(circle(O, R), gray+dashed);

// A 和 D 在圆上
pair A = O + R * dir(180 + 50);  // 左下
pair D = O + R * dir(-50);        // 右下
// 确保 A, D 在 BC 同侧 (在 BC 上方)

draw(A -- B);
draw(A -- C);
draw(D -- B);
draw(D -- C);
draw(B -- C);

// 角弧
draw(arc(A, 0.4, degrees(B - A), degrees(C - A)), red);
draw(arc(D, 0.4, degrees(B - D), degrees(C - D)), red);

dot(A); label("$A$", A, W);
dot(B); label("$B$", B, SW);
dot(C); label("$C$", C, SE);
dot(D); label("$D$", D, E);

label("$35^\circ$", A + 0.5*dir(degrees((B-A+C-A)/2)));
label("$35^\circ$", D + 0.5*dir(degrees((B-D+C-D)/2)));
