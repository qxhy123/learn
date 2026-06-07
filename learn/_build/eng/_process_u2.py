# 重组 english 课 u2：4 关 → 9 关（动词单拆 + 非谓语 4 关），分配 id、校验、清散落 $。
import json, re, pathlib

LC = "/Users/yangyang/ai_projs/math/learn"
old = json.loads(pathlib.Path(f"{LC}/_build/eng/u2.json.bak").read_text(encoding="utf-8"))["lessons"]
va = json.loads(pathlib.Path(f"{LC}/_build/eng/u2_va.json").read_text(encoding="utf-8"))    # 动词, 副词
nf1 = json.loads(pathlib.Path(f"{LC}/_build/eng/u2_nf1.json").read_text(encoding="utf-8"))  # 非谓语总览, 不定式
nf2 = json.loads(pathlib.Path(f"{LC}/_build/eng/u2_nf2.json").read_text(encoding="utf-8"))  # 动名词, 分词

# 期望顺序
lessons = [
    old[0],   # 名词与冠词
    old[1],   # 代词与形容词
    va[0],    # 动词
    va[1],    # 副词
    old[3],   # 介词、连词与感叹词（原 l4）
    nf1[0],   # 非谓语动词总览
    nf1[1],   # 不定式
    nf2[0],   # 动名词
    nf2[1],   # 分词
]
assert len(lessons) == 9

def fix_dollar(s):
    return s.replace("$", "") if isinstance(s, str) and "$" in s else s
def walk(o):
    if isinstance(o, str): return fix_dollar(o)
    if isinstance(o, list): return [walk(x) for x in o]
    if isinstance(o, dict): return {k: walk(v) for k, v in o.items()}
    return o
lessons = walk(lessons)

problems = []
for i, l in enumerate(lessons, 1):
    lid = f"u2-l{i}"
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

out = {"id": "u2", "lessons": lessons}
pathlib.Path(f"{LC}/public/courses/english/units/u2.json").write_text(
    json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

tot = sum(len(l["questions"]) for l in lessons)
print(f"写入 u2.json：9 关，{tot} 题")
for i, l in enumerate(lessons, 1):
    print(f"  u2-l{i} {l['title']:<14} intro={len(l['intro'])} q={len(l['questions'])}")
print("结构问题：", problems if problems else "无")
print("散落 $：", "无" if "$" not in json.dumps(lessons, ensure_ascii=False) else "有")
