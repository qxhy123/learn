# 把拆好的 4 个分片拼成 english 课 u3.json（13 关），分配 id、校验、清掉散落 $。
import json, re, pathlib

LC = "/Users/yangyang/ai_projs/math/learn"
parts = ["a", "b", "c", "d"]
lessons = []
for p in parts:
    arr = json.loads(pathlib.Path(f"{LC}/_build/eng/u3_{p}.json").read_text(encoding="utf-8"))
    lessons.extend(arr)

assert len(lessons) == 13, f"应 13 关，实为 {len(lessons)}"

def fix_dollar(s):
    if not isinstance(s, str) or "$" not in s:
        return s
    s = s.replace("$1 million", "one million dollars")
    # 兜底：剩余的孤立 $ 直接去掉前导 $（英语课不该有 $）
    return s

def walk(obj):
    if isinstance(obj, str):
        return fix_dollar(obj)
    if isinstance(obj, list):
        return [walk(x) for x in obj]
    if isinstance(obj, dict):
        return {k: walk(v) for k, v in obj.items()}
    return obj

lessons = walk(lessons)

problems = []
for i, l in enumerate(lessons, 1):
    lid = f"u3-l{i}"
    l["id"] = lid
    if not (isinstance(l.get("title"), str) and l["title"].strip()):
        problems.append(f"{lid} 缺 title")
    if not l.get("intro"):
        problems.append(f"{lid} 缺 intro")
    qs = l.get("questions", [])
    if len(qs) < 1:
        problems.append(f"{lid} 无题")
    for j, q in enumerate(qs, 1):
        q["id"] = f"{lid}-q{j}"
        t = q.get("type")
        if t not in ("choice", "input", "judge", "match"):
            problems.append(f"{q['id']} 非法 type={t}")
        if not (isinstance(q.get("prompt"), str) and q["prompt"].strip()):
            problems.append(f"{q['id']} prompt 空")
        if t == "choice":
            if not (isinstance(q.get("options"), list) and len(q["options"]) >= 2):
                problems.append(f"{q['id']} options 缺")
            elif not (isinstance(q.get("answer"), int) and 0 <= q["answer"] < len(q["options"])):
                problems.append(f"{q['id']} answer 越界 {q.get('answer')}")
        elif t == "input":
            if not (isinstance(q.get("accept"), list) and q["accept"]):
                problems.append(f"{q['id']} accept 缺")
        elif t == "judge":
            if not isinstance(q.get("answer"), bool):
                problems.append(f"{q['id']} answer 非 bool")
        elif t == "match":
            L, R = q.get("left"), q.get("right")
            if not (isinstance(L, list) and isinstance(R, list) and len(L) == len(R) and len(L) >= 2):
                problems.append(f"{q['id']} match 左右不等长/过短")

# 散落 $ 复查
stray = []
s_all = json.dumps(lessons, ensure_ascii=False)
if s_all.count("$"):
    for m in re.finditer(r".{0,15}\$.{0,15}", s_all):
        stray.append(m.group(0))

out = {"id": "u3", "lessons": lessons}
pathlib.Path(f"{LC}/public/courses/english/units/u3.json").write_text(
    json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

tot_q = sum(len(l["questions"]) for l in lessons)
print(f"写入 u3.json：13 关，{tot_q} 题")
for i, l in enumerate(lessons, 1):
    print(f"  u3-l{i:<2} {l['title']:<16} intro={len(l['intro'])} q={len(l['questions'])}")
print("结构问题：", problems if problems else "无")
print("散落 $：", stray if stray else "无")
