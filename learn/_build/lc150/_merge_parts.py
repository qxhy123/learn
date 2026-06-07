# 把每单元的分片 uN_1.json, uN_2.json ... 按 part 顺序拼成 uN.json（供 assemble-lc150.mjs 读取）。
import json, glob, re, os, pathlib

LC = "/Users/yangyang/ai_projs/math/learn/_build/lc150"
plan = json.load(open(f"{LC}/_chunkplan.json"))
units = {}
for c in plan:
    units.setdefault(c["uid"], []).append(c)

done, missing = [], []
for uid, chunks in units.items():
    chunks.sort(key=lambda c: c["part"])
    lessons = []
    ok = True
    for c in chunks:
        f = pathlib.Path(LC) / f"{c['cid']}.json"
        if not f.exists():
            missing.append(c["cid"]); ok = False; continue
        arr = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(arr, list):
            missing.append(c["cid"] + "(not-array)"); ok = False; continue
        lessons.extend(arr)
    if ok and lessons:
        json.dump(lessons, open(f"{LC}/{uid}.json", "w"), ensure_ascii=False)
        done.append(f"{uid}:{len(lessons)}")
print("merged units:", " ".join(sorted(done, key=lambda s: int(s.split(':')[0][1:]))))
if missing:
    print("MISSING/!array parts:", missing)
