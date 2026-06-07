# 重组 english 课 u1：4 关 → 9 关（词汇构建扩成 总览/词根/前缀/后缀/语义场/同义词辨析），分配 id、校验。
import json, pathlib

LC = "/Users/yangyang/ai_projs/math/learn"
old = json.loads(pathlib.Path(f"{LC}/_build/eng/u1.json.bak").read_text(encoding="utf-8"))["lessons"]
v1 = json.loads(pathlib.Path(f"{LC}/_build/eng/u1_v1.json").read_text(encoding="utf-8"))  # 总览, 词根
v2 = json.loads(pathlib.Path(f"{LC}/_build/eng/u1_v2.json").read_text(encoding="utf-8"))  # 前缀, 后缀
v3 = json.loads(pathlib.Path(f"{LC}/_build/eng/u1_v3.json").read_text(encoding="utf-8"))  # 语义场, 同义词辨析

lessons = [
    old[0],   # 英语概览与字母系统
    old[1],   # 音标与发音规则
    v1[0],    # 词汇构建策略（总览）
    v1[1],    # 词根
    v2[0],    # 前缀
    v2[1],    # 后缀
    v3[0],    # 语义场
    v3[1],    # 同义词辨析
    old[3],   # 句子基础：SVO结构（原 l4）
]
assert len(lessons) == 9

problems = []
for i, l in enumerate(lessons, 1):
    lid = f"u1-l{i}"
    l["id"] = lid
    if not (isinstance(l.get("title"), str) and l["title"].strip()): problems.append(f"{lid} 缺 title")
    if not l.get("intro"): problems.append(f"{lid} 缺 intro")
    for j, q in enumerate(l.get("questions", []), 1):
        q["id"] = f"{lid}-q{j}"
        t = q.get("type")
        if t not in ("choice", "input", "judge", "match"): problems.append(f"{q['id']} type={t}")
        if not (isinstance(q.get("prompt"), str) and q["prompt"].strip()): problems.append(f"{q['id']} prompt 空")
        if t == "choice":
            if not (isinstance(q.get("options"), list) and len(q["options"]) >= 2): problems.append(f"{q['id']} options 缺")
            elif not (isinstance(q.get("answer"), int) and 0 <= q["answer"] < len(q["options"])): problems.append(f"{q['id']} answer 越界")
        elif t == "input":
            if not (isinstance(q.get("accept"), list) and q["accept"]): problems.append(f"{q['id']} accept 缺")
        elif t == "judge":
            if not isinstance(q.get("answer"), bool): problems.append(f"{q['id']} answer 非 bool")
        elif t == "match":
            L, R = q.get("left"), q.get("right")
            if not (isinstance(L, list) and isinstance(R, list) and len(L) == len(R) and len(L) >= 2): problems.append(f"{q['id']} match 不等长")

out = {"id": "u1", "lessons": lessons}
pathlib.Path(f"{LC}/public/courses/english/units/u1.json").write_text(
    json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

tot = sum(len(l["questions"]) for l in lessons)
print(f"写入 u1.json：9 关，{tot} 题")
for i, l in enumerate(lessons, 1):
    print(f"  u1-l{i} {l['title']:<16} intro={len(l['intro'])} q={len(l['questions'])}")
print("结构问题：", problems if problems else "无")
print("散落 $：", "有" if "$" in json.dumps(lessons, ensure_ascii=False) else "无")
