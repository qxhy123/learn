import pathlib
# 把 6 个深度导入 assembler 的 cleanRich 占位符从 ` N `（会误吞散文数字）改成私用区字符。幂等。
ROOT = "/Users/yangyang/ai_projs/math/learn/_build"
PH_OLD = "return ` ${fences.length - 1} `"
PH_NEW = "return `\\uE000${fences.length - 1}\\uE001`"
RS_OLD = "s = s.replace(/ (\\d+) /g, (_, i) => fences[+i])"
RS_NEW = "s = s.replace(/\\uE000(\\d+)\\uE001/g, (_, i) => fences[+i])"
for name in ["lc150", "tdd", "ts", "git", "ddd", "harness"]:
    p = pathlib.Path(f"{ROOT}/assemble-{name}.mjs")
    s = p.read_text(encoding="utf-8")
    before = s
    s = s.replace(PH_OLD, PH_NEW).replace(RS_OLD, RS_NEW)
    p.write_text(s, encoding="utf-8")
    ok = ("\\uE000" in s) and (RS_OLD not in s)
    print(f"assemble-{name}.mjs: changed={before != s}  has_uE000={'\\uE000' in s}  buggy_restore_gone={RS_OLD not in s}  -> {'OK' if ok else 'CHECK'}")
