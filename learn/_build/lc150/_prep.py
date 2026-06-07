# 准备 LeetCode150 批量生产：生成 23 个 _meta_uN.json、抽取金标准例子、锁定 u2 第1关、产出 chunk 计划。
import yaml, json, pathlib, glob, os

REPO = "/Users/yangyang/ai_projs/math/leetcode150"
LC = "/Users/yangyang/ai_projs/math/learn/_build/lc150"
FIGDIR = "/Users/yangyang/ai_projs/math/learn/public/courses/leetcode150/figures"

d = yaml.safe_load(open(f"{REPO}/data/top_interview_150.yaml"))
ps = d["problems"]
order = []
for p in ps:
    if p["official_group"] not in order:
        order.append(p["official_group"])
uid_of = {g: f"u{i+1}" for i, g in enumerate(order)}

SLUG = {
    "Array / String": "array-string", "Two Pointers": "two-pointers", "Sliding Window": "sliding-window",
    "Matrix": "matrix", "Hashmap": "hash", "Intervals": "intervals", "Stack": "stack",
    "Linked List": "linked-list", "Binary Tree General": "bt-dfs", "Binary Tree BFS": "bt-bfs",
    "Binary Search Tree": "bst", "Graph General": "graph-general", "Graph BFS": "graph-bfs", "Trie": "trie",
    "Backtracking": "backtracking", "Divide & Conquer": "divide-conquer", "Kadane's Algorithm": "kadane",
    "Binary Search": "binary-search", "Heap": "heap", "Bit Manipulation": "bit", "Math": "math",
    "1D DP": "dp-1d", "Multidimensional DP": "dp-multidim",
}
CN = {
    "Array / String": "数组与字符串", "Two Pointers": "双指针", "Sliding Window": "滑动窗口", "Matrix": "矩阵",
    "Hashmap": "哈希表", "Intervals": "区间", "Stack": "栈", "Linked List": "链表",
    "Binary Tree General": "二叉树（DFS）", "Binary Tree BFS": "二叉树（BFS）", "Binary Search Tree": "二叉搜索树",
    "Graph General": "图论", "Graph BFS": "图（BFS）", "Trie": "字典树 Trie", "Backtracking": "回溯",
    "Divide & Conquer": "分治", "Kadane's Algorithm": "Kadane 算法", "Binary Search": "二分查找", "Heap": "堆",
    "Bit Manipulation": "位运算", "Math": "数学", "1D DP": "一维动态规划", "Multidimensional DP": "多维动态规划",
}
SIZES = {
    "Array / String": [6, 6, 6, 6], "Two Pointers": [1, 4], "Sliding Window": [4], "Matrix": [5],
    "Hashmap": [5, 4], "Intervals": [4], "Stack": [5], "Linked List": [6, 5], "Binary Tree General": [7, 7],
    "Binary Tree BFS": [4], "Binary Search Tree": [3], "Graph General": [6], "Graph BFS": [3], "Trie": [3],
    "Backtracking": [7], "Divide & Conquer": [4], "Kadane's Algorithm": [2], "Binary Search": [7], "Heap": [4],
    "Bit Manipulation": [6], "Math": [6], "1D DP": [5], "Multidimensional DP": [5, 4],
}

# 1) 每单元 meta（含 doc_abs）
for g in order:
    uid = uid_of[g]
    items = []
    for p in ps:
        if p["official_group"] != g:
            continue
        it = {k: p.get(k) for k in ["number", "title", "slug", "difficulty", "signature", "leetcode_url", "patterns", "examples", "constraints_summary"]}
        it["doc_abs"] = f"{REPO}/{p['doc_path']}"
        items.append(it)
    json.dump(items, open(f"{LC}/_meta_{uid}.json", "w"), ensure_ascii=False, indent=1)

# 2) 抽取金标准例子 + 锁定 u2 第1关
u2 = json.loads(pathlib.Path(f"{LC}/u2.json").read_text(encoding="utf-8"))
gold = u2[0]
json.dump(gold, open(f"{LC}/_EXAMPLE_LESSON.json", "w"), ensure_ascii=False, indent=1)
json.dump([gold], open(f"{LC}/u2_1.json", "w"), ensure_ascii=False)  # locked part

# 3) chunk 计划
def figs(g):
    s = SLUG[g]
    return sorted(os.path.basename(f) for f in glob.glob(f"{FIGDIR}/lc-{s}-*.svg"))

plan = []
for g in order:
    uid = uid_of[g]
    gps = [p for p in ps if p["official_group"] == g]
    sizes = SIZES[g]
    assert sum(sizes) == len(gps), f"{g}: sizes {sizes} != {len(gps)}"
    idx = 0
    for part, sz in enumerate(sizes, 1):
        sub = gps[idx: idx + sz]
        idx += sz
        locked = (g == "Two Pointers" and part == 1)
        plan.append({
            "cid": f"{uid}_{part}", "uid": uid, "part": part, "locked": locked,
            "unit_cn": CN[g], "slug": SLUG[g], "figures": figs(g),
            "out": f"_build/lc150/{uid}_{part}.json",
            "problems": [{"number": p["number"], "title": p["title"], "difficulty": p["difficulty"], "doc_abs": f"{REPO}/{p['doc_path']}"} for p in sub],
        })
json.dump(plan, open(f"{LC}/_chunkplan.json", "w"), ensure_ascii=False, indent=1)

todo = [c for c in plan if not c["locked"]]
print(f"units={len(order)} problems={len(ps)} total_chunks={len(plan)} agent_chunks={len(todo)}")
print("=== agent chunks (cid | unit | #q | nums | figures) ===")
for c in todo:
    nums = ",".join(str(p["number"]) for p in c["problems"])
    print(f"{c['cid']:7} | {c['unit_cn']:10} | {len(c['problems'])} | {nums:28} | {','.join(c['figures'])}")
