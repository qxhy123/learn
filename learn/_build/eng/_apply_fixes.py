# 把 _FIX_1..4 的修正应用到 english 各 units 文件，permutation 安全网：只接受真排列且与原不同的。
import json, glob, pathlib

LC = "/Users/yangyang/ai_projs/math/learn"
docs, byid, fileof = {}, {}, {}
for f in glob.glob(f"{LC}/public/courses/english/units/u*.json"):
    d = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
    docs[f] = d
    for l in d["lessons"]:
        for q in l["questions"]:
            if q.get("type") == "match":
                byid[q["id"]] = q
                fileof[q["id"]] = f

fixes = {}
for i in range(1, 5):
    for it in json.loads(pathlib.Path(f"{LC}/_build/eng/_FIX_{i}.json").read_text(encoding="utf-8"))["items"]:
        fixes[it["id"]] = it

applied, skip_same, badperm, missing = [], 0, [], []
changed = set()
for qid, it in fixes.items():
    if qid not in byid:
        missing.append(qid); continue
    cur = byid[qid]["right"]
    new = it.get("right")
    if not isinstance(new, list) or sorted(map(str, new)) != sorted(map(str, cur)):
        badperm.append(qid); continue
    if new == cur:
        skip_same += 1; continue
    byid[qid]["right"] = new
    applied.append(qid); changed.add(fileof[qid])

for f in changed:
    pathlib.Path(f).write_text(json.dumps(docs[f], ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

print("应用修正(真错位):", len(applied))
print("  ", sorted(applied))
print("跳过(已对齐):", skip_same, " 拒绝(非排列):", badperm, " 缺失id:", missing)
print("改动文件:", sorted(pathlib.Path(f).name for f in changed))
