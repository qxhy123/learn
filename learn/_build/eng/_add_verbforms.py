# 给时态练习增加动词变形专项题：单三、不规则过去式、不规则过去分词、现在分词(-ing)。
# 追加到对应"本家"时态关，重排该关 q id。所有形式人工核对正确。
import json, pathlib

F = "/Users/yangyang/ai_projs/math/learn/public/courses/english/units/u3.json"
d = json.loads(pathlib.Path(F).read_text(encoding="utf-8"))
by = {l["id"]: l for l in d["lessons"]}

def I(prompt, accept, explain):
    return {"type": "input", "prompt": prompt, "accept": accept, "explain": explain}
def C(prompt, options, ans, explain):
    return {"type": "choice", "prompt": prompt, "options": options, "answer": ans, "explain": explain}
def J(prompt, ans, explain):
    return {"type": "judge", "prompt": prompt, "answer": ans, "explain": explain}
def M(prompt, left, right, explain):
    return {"type": "match", "prompt": prompt, "left": left, "right": right, "explain": explain}

# ===== u3-l1 一般现在时：第三人称单数形式 =====
l1 = [
    I("【单三专项】写出 go 的第三人称单数形式：", ["goes"], "以 o 结尾的动词，单三加 -es：go → *goes*。同类还有 do → does。"),
    I("【单三专项】写出 study 的第三人称单数形式：", ["studies"], "辅音字母 + y 结尾：变 y 为 i 再加 -es：study → *studies*。"),
    I("【单三专项】写出 watch 的第三人称单数形式：", ["watches"], "以 ch 结尾，加 -es：watch → *watches*。以 s/x/ch/sh 结尾都加 -es。"),
    I("【单三专项】写出 have 的第三人称单数形式：", ["has"], "have 是不规则的：he/she/it 用 *has*，不是 haves。"),
    I("【单三专项】写出 fix 的第三人称单数形式：", ["fixes"], "以 x 结尾，加 -es：fix → *fixes*。"),
    C("【单三专项】fly 的第三人称单数是哪个？", ["flys", "flies", "flyes", "flying"], 1,
      "辅音字母 + y（fly）要变 y 为 i 再加 -es：*flies*。flying 是现在分词，不是单三。"),
    M("【单三专项】把动词与它的第三人称单数形式配对：", ["do", "miss", "carry", "go"], ["does", "misses", "carries", "goes"],
      "do→does、go→goes（特殊）；miss 以 s 结尾→misses；carry 辅音+y→carries。"),
    J("【单三专项】句子 *He **do** his homework every evening.* 是正确的。", False,
      "主语 He 是第三人称单数，do 要用单三 *does*：He **does** his homework。"),
    C("【单三专项】下列哪一类动词，第三人称单数要加 -es（而非只加 -s）？",
      ["以不发音 e 结尾", "以 s / x / ch / sh / o 结尾", "以元音字母 + y 结尾", "以辅音字母 + 重读元音结尾"], 1,
      "以 s、x、ch、sh、o 结尾的动词单三加 -es（misses, fixes, watches, washes, goes）。元音+y 直接加 -s（plays）。"),
]

# ===== u3-l2 一般过去时：不规则动词过去式（V2）=====
l2 = [
    I("【不规则过去式】写出 go 的过去式：", ["went"], "go 的过去式是 *went*（完全不规则）。"),
    I("【不规则过去式】写出 eat 的过去式：", ["ate"], "eat → *ate*（过去式），过去分词是 eaten。"),
    I("【不规则过去式】写出 buy 的过去式：", ["bought"], "buy → *bought*。过去式和过去分词同形（ABB 型）。"),
    I("【不规则过去式】写出 take 的过去式：", ["took"], "take → *took*（过去式），过去分词是 taken（ABC 型）。"),
    I("【不规则过去式】写出 think 的过去式：", ["thought"], "think → *thought*。过去式与过去分词同形。"),
    I("【不规则过去式】写出 begin 的过去式：", ["began"], "begin → *began*（过去式）→ begun（过去分词），三者各不相同（ABC 型）。"),
    M("【不规则过去式】把动词与它的过去式配对：", ["see", "come", "write", "bring"], ["saw", "came", "wrote", "brought"],
      "see→saw、come→came、write→wrote、bring→brought，都是不规则变化。"),
    C("【不规则过去式】catch 的过去式是哪个？", ["catched", "caught", "cotched", "catches"], 1,
      "catch 是不规则动词，过去式是 *caught*，不能加 -ed 写成 catched。"),
    J("【不规则过去式】句子 *She **gived** me a gift yesterday.* 是正确的。", False,
      "give 是不规则动词，过去式是 *gave*：She **gave** me a gift。没有 gived 这种形式。"),
]

# ===== u3-l4 现在进行时：现在分词(-ing) 的拼写变化 =====
l4 = [
    I("【-ing 专项】写出 run 的现在分词：", ["running"], "重读闭音节、末尾单辅音：双写辅音再加 -ing：run → *running*。"),
    I("【-ing 专项】写出 make 的现在分词：", ["making"], "以不发音 e 结尾：去 e 加 -ing：make → *making*。"),
    I("【-ing 专项】写出 lie 的现在分词（躺 / 说谎）：", ["lying"], "以 ie 结尾：变 ie 为 y 再加 -ing：lie → *lying*。同类 die → dying。"),
    I("【-ing 专项】写出 sit 的现在分词：", ["sitting"], "重读闭音节末尾单辅音，双写：sit → *sitting*。"),
    I("【-ing 专项】写出 begin 的现在分词：", ["beginning"], "重读在最后一个音节、末尾单辅音 n，双写：begin → *beginning*。"),
    C("【-ing 专项】die 的现在分词是哪个？", ["dieing", "dying", "diing", "dieding"], 1,
      "以 ie 结尾：变 ie 为 y 加 -ing：die → *dying*。"),
    M("【-ing 专项】把动词与它的现在分词配对：", ["swim", "write", "plan", "study"], ["swimming", "writing", "planning", "studying"],
      "swim/plan 双写辅音→swimming/planning；write 去 e→writing；study 直接加→studying（y 保留）。"),
    J("【-ing 专项】句子 *I am **writeing** a letter now.* 是正确的。", False,
      "write 以不发音 e 结尾，要去 e 加 -ing：*writing*，不是 writeing。"),
    I("【-ing 专项】写出 travel 的现在分词（英式拼写，双写 l）：", ["travelling", "traveling"], "英式双写 l：travel → *travelling*；美式不双写 traveling，两种都接受。"),
]

# ===== u3-l7 现在完成时：不规则动词过去分词（V3）=====
l7 = [
    I("【不规则过去分词】写出 eat 的过去分词：", ["eaten"], "eat → ate（过去式）→ *eaten*（过去分词）。现在完成时用过去分词：have eaten。"),
    I("【不规则过去分词】写出 go 的过去分词：", ["gone"], "go → went → *gone*。have gone（已经去了，人还没回来）。"),
    I("【不规则过去分词】写出 write 的过去分词：", ["written"], "write → wrote → *written*。have written。"),
    I("【不规则过去分词】写出 take 的过去分词：", ["taken"], "take → took → *taken*。"),
    I("【不规则过去分词】写出 be 的过去分词：", ["been"], "be 的过去分词是 *been*（am/is/are/was/were → been）。have been。"),
    M("【不规则过去分词】把动词与它的过去分词配对：", ["do", "break", "speak", "give"], ["done", "broken", "spoken", "given"],
      "do→done、break→broken、speak→spoken、give→given，都是不规则过去分词。"),
    C("【不规则过去分词】drink 的过去分词是哪个？", ["drank", "drunk", "drinked", "drunken"], 1,
      "drink → drank（过去式）→ *drunk*（过去分词）。drank 是过去式，别混。"),
    J("【不规则过去分词】句子 *I have **wrote** three emails today.* 是正确的。", False,
      "现在完成时要用过去分词：write 的过去分词是 *written*：I have **written**。wrote 是过去式。"),
    C("【不规则过去分词】see 的过去分词是哪个？", ["saw", "seen", "seed", "sawn"], 1,
      "see → saw（过去式）→ *seen*（过去分词）。I have seen it。"),
]

adds = {"u3-l1": l1, "u3-l2": l2, "u3-l4": l4, "u3-l7": l7}
for lid, newqs in adds.items():
    les = by[lid]
    les["questions"].extend(newqs)
    for j, q in enumerate(les["questions"], 1):
        q["id"] = f"{lid}-q{j}"
    print(f"{lid} {les['title']}: +{len(newqs)} 题 → 共 {len(les['questions'])} 题")

pathlib.Path(F).write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
tot = sum(len(l["questions"]) for l in d["lessons"])
print("u3 现在共题数：", tot)
