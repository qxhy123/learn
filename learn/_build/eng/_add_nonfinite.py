# 给 不定式/动名词/分词 三关增加更多"单词级"练习（动词搭配、固定结构、-ing/-ed 形容词）。
# 追加到 u2-l7/l8/l9，重排 q id。所有搭配/形式人工核对正确。
import json, pathlib

F = "/Users/yangyang/ai_projs/math/learn/public/courses/english/units/u2.json"
d = json.loads(pathlib.Path(F).read_text(encoding="utf-8"))
by = {l["id"]: l for l in d["lessons"]}

def I(p, a, e): return {"type": "input", "prompt": p, "accept": a, "explain": e}
def C(p, o, ans, e): return {"type": "choice", "prompt": p, "options": o, "answer": ans, "explain": e}
def J(p, a, e): return {"type": "judge", "prompt": p, "answer": a, "explain": e}
def M(p, l, r, e): return {"type": "match", "prompt": p, "left": l, "right": r, "explain": e}

# ===== u2-l7 不定式：更多接 to do 的动词 + 结构 =====
l7 = [
    I("【不定式专项】decide 后接不定式：*She decided ___ (leave) early.*", ["to leave"], "decide 只接不定式：decide *to leave*（决定离开）。"),
    I("【不定式专项】refuse 后接不定式：*He refused ___ (help) us.*", ["to help"], "refuse *to help*（拒绝帮忙）。refuse / agree / promise / offer 都接 to do。"),
    I("【不定式专项】manage 后接不定式：*They managed ___ (finish) on time.*", ["to finish"], "manage *to finish*（设法完成）。manage to do 强调“努力做成了”。"),
    I("【不定式专项】too...to 结构：*The box is too heavy ___ (carry).*", ["to carry"], "too + 形容词 + to do = 太…以致不能…：too heavy *to carry*（太重搬不动）。"),
    I("【不定式专项】enough to 结构：*She is old enough ___ (drive).*", ["to drive"], "形容词 + enough + to do：old enough *to drive*（够大可以开车）。"),
    I("【不定式专项】疑问词 + 不定式：*I don't know what ___ (do) next.*", ["to do"], "疑问词 + to do 作宾语：what *to do*（做什么）、how to、where to。"),
    I("【不定式专项】It is + 形容词 + to do：*It is important ___ (be) honest.*", ["to be"], "It 作形式主语，真正的主语是不定式：It is important *to be* honest。"),
    C("【不定式专项】下列哪个动词后面接不定式（to do），不接动名词？", ["enjoy", "avoid", "promise", "finish"], 2,
      "promise *to do*（答应做）。enjoy / avoid / finish 都只接动名词 doing。"),
    J("【不定式专项】句子 *I hope **to seeing** you soon.* 是正确的。", False,
      "hope 接不定式：hope *to see*。to 后面要用动词原形，不能用 -ing：I hope **to see** you。"),
    M("【不定式专项】把动词与它接不定式的正确例句配对：", ["want", "agree", "offer", "pretend"],
      ["*I **want to go** home.*", "*They **agreed to help**.*", "*He **offered to pay**.*", "*She **pretended to sleep**.*"],
      "want / agree / offer / pretend 都接 to do。配对的是各动词最自然的不定式例句。"),
]

# ===== u2-l8 动名词：更多接 doing 的动词 / 介词 / 固定结构 + to do vs doing 对比 =====
l8 = [
    I("【动名词专项】avoid 后接动名词：*He avoided ___ (make) the same mistake.*", ["making"], "avoid 只接动名词：avoid *making*（避免犯）。"),
    I("【动名词专项】suggest 后接动名词：*She suggested ___ (go) by train.*", ["going"], "suggest *doing*（建议做）。suggest 不接 to do。"),
    I("【动名词专项】介词 in 后用动名词：*I'm interested in ___ (learn) Spanish.*", ["learning"], "介词后必用动名词：interested in *learning*。介词 + doing 是铁律。"),
    I("【动名词专项】look forward to 的 to 是介词：*We look forward to ___ (hear) from you.*", ["hearing"], "这里 to 是介词，后接动名词：look forward to *hearing*（期待收到）。"),
    I("【动名词专项】can't help 后接动名词：*She couldn't help ___ (laugh).*", ["laughing"], "can't help *doing* = 忍不住做：couldn't help *laughing*（忍不住笑）。"),
    I("【动名词专项】be worth 后接动名词：*This book is worth ___ (read).*", ["reading"], "be worth *doing* = 值得做：worth *reading*（值得一读），主动形式表被动义。"),
    I("【动名词专项】spend time doing：*I spent an hour ___ (clean) the room.*", ["cleaning"], "spend 时间/钱 (in) *doing*：spent an hour *cleaning*（花一小时打扫）。"),
    I("【动名词专项】need + doing（= need to be done）：*The car needs ___ (wash).*", ["washing"], "need *doing* 表被动：“这车需要洗” = The car needs *washing*（= needs to be washed）。"),
    C("【动名词专项】下列哪个动词后面只接动名词（doing）？", ["decide", "admit", "hope", "agree"], 1,
      "admit *doing*（承认做过）。decide / hope / agree 都接 to do。"),
    J("【动名词专项】句子 *I am looking forward to **see** you.* 是正确的。", False,
      "look forward to 的 to 是介词，后面要用动名词：looking forward to *seeing* you。"),
    M("【to do vs doing】把句子和它的正确含义配对（注意意义差别）：",
      ["*Remember **to lock** the door.*", "*I remember **locking** the door.*", "*He **stopped to smoke**.*", "*He **stopped smoking**.*"],
      ["提醒：记得去锁门（事还没做）", "回忆：记得锁过门了（事已做过）", "停下手里别的事，去抽根烟", "把吸烟这个习惯戒了"],
      "remember / stop + to do 指“去做（尚未做）”，+ doing 指“做过 / 正在做的事”。意义完全不同。"),
]

# ===== u2-l9 分词：更多 -ing/-ed 形容词 + 分词作定语/状语 =====
l9 = [
    C("【-ing/-ed 形容词】*The movie was so ___.*（这电影令人兴奋）", ["exciting", "excited"], 0,
      "形容事物“令人…的”用 -ing：The movie was *exciting*。"),
    C("【-ing/-ed 形容词】*The children were ___ about the trip.*（孩子们感到兴奋）", ["exciting", "excited"], 1,
      "形容人“感到…的”用 -ed：The children were *excited*。"),
    I("【-ing/-ed 形容词】用 bore 的正确形式：*I was ___ during the long speech.*（我感到无聊）", ["bored"], "人感到无聊用 -ed：I was *bored*。物“令人无聊”才用 boring。"),
    I("【-ing/-ed 形容词】用 disappoint 的正确形式：*The result was ___.*（结果令人失望）", ["disappointing"], "事物“令人失望”用 -ing：The result was *disappointing*。"),
    I("【-ing/-ed 形容词】用 embarrass 的正确形式：*She felt ___ when she forgot his name.*（她感到尴尬）", ["embarrassed"], "人“感到尴尬”用 -ed：felt *embarrassed*。"),
    I("【-ing/-ed 形容词】用 surprise 的正确形式：*It was a ___ result.*（一个令人惊讶的结果）", ["surprising"], "修饰物、表“令人惊讶的”用 -ing：a *surprising* result。"),
    J("【-ing/-ed 形容词】用 *I am very **boring** in this lesson.* 来表达“我觉得这节课很无聊”是正确的。", False,
      "boring 是“令人无聊的”，I am boring = 我这个人很无趣。要表达“我感到无聊”应用 *bored*：I am **bored**。"),
    M("【-ing/-ed 形容词】把形容词与它的含义配对：", ["exciting", "excited", "confusing", "confused"],
      ["令人兴奋的（修饰事物）", "感到兴奋的（修饰人）", "令人困惑的（修饰事物）", "感到困惑的（修饰人）"],
      "-ing 表“令人…的”（主动，修饰引起感受的事物）；-ed 表“感到…的”（被动，修饰感受者）。"),
    I("【分词作定语】现在分词表主动 / 进行：*the man ___ (stand) at the door*（站在门口的人）", ["standing"], "现在分词作定语表主动：the man *standing*（= who is standing）。"),
    I("【分词作定语】过去分词表被动 / 完成：*a window ___ (break) by the storm*（被暴风雨打破的窗）", ["broken"], "过去分词作定语表被动：a window *broken*（= which was broken）。"),
    C("【分词作状语】*___ from the plane, the fields look like a chessboard.*", ["Seeing", "Seen", "To see", "See"], 1,
      "田野是“被看”，与逻辑主语 the fields 是被动关系，用过去分词：*Seen* from the plane（从飞机上看）。"),
]

adds = {"u2-l7": l7, "u2-l8": l8, "u2-l9": l9}
for lid, newqs in adds.items():
    les = by[lid]
    les["questions"].extend(newqs)
    for j, q in enumerate(les["questions"], 1):
        q["id"] = f"{lid}-q{j}"
    print(f"{lid} {les['title']}: +{len(newqs)} 题 → 共 {len(les['questions'])} 题")

s = json.dumps(d, ensure_ascii=False)
assert "$" not in s, "出现散落 $"
pathlib.Path(F).write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("u2 现在共题数：", sum(len(l["questions"]) for l in d["lessons"]), "| 散落$：无")
