# 英语课「词性精讲」拆分 + 非谓语动词 · 改写规范

把英语课 u2「词性精讲」里的**动词**从「动词与副词」中**单独拆出来**，并**新增非谓语动词**系列。风格、口吻、卡片格式必须和英语课现有关卡完全一致（不是新课的深度导入风）。

## 第一步：先读风格样板
先 Read /Users/yangyang/ai_projs/math/learn/_build/eng/_STYLE_U2.json （这是英语课词性单元现有的一关「动词与副词」）。**照它的语气、卡片粒度、中英混排、表格、斜体例句来写**。

排版约定：
- 英文例句用**斜体**：`*She **runs** fast.*`（句中强调词再 `**加粗**`）。
- 中文讲解为主，术语给中英对照（如 不定式 to-infinitive、动名词 gerund、分词 participle、非谓语动词 non-finite verb）。
- 对比类内容可用 markdown 表格（`|` 仅在表格行内用）。不要用 `$`、不要用代码围栏。

## 输入与内容源
- prompt 告诉你：负责哪几关、源章节路径、输出文件。
- 主源：Read /Users/yangyang/ai_projs/math/english-tutorial/part2-parts-of-speech/07-verbs-adverbs.md （含 7.1~7.3 动词分类/及物不及物/情态、7.4 非谓语入门、7.6~7.8 副词）。
- 非谓语部分源里只有 7.4「入门」，内容不够：**可按标准英语语法把不定式/动名词/分词讲全**（这些是公认语法，不是臆造），但要符合教学惯例、举正确的例子。

## 输出
- Write 到指定文件，**关卡数组**（按给定顺序），每关 `{ "title", "subtitle", "intro":[卡片], "questions":[题] }`，**不写 id**。合法 JSON。
- `title` 用关名（如 `动词`、`不定式`）；`subtitle` 一句话点主题。

## 各关要点（每关 6~8 卡）
- **动词**：动词分类（实义/系动词 linking/助动词 auxiliary/情态动词 modal）、及物 vs 不及物、情态动词语义辨析（can/may/must/should…）。含「易错点」卡。
- **副词**：副词六大类型、位置三区域规则、形容词 vs 副词同根易混对（hard/hardly、late/lately…）。含「易错点」卡。
- **非谓语动词总览**：什么是非谓语（不定式 to do、动名词 doing、分词 doing/done 三种形式）、**为什么它们不能单独作谓语**（用「为什么…？因为…」讲清）、在句中能充当什么成分（主/宾/定/状/补）。
- **不定式 to do**：构成（to + 动词原形；不带 to 的场合 make/let/see sb do、情态动词后）、作主语/宾语/宾补/定语/状语（目的）、典型用法。
- **动名词 doing**：构成、作主语/宾语、介词后必用动名词、只接动名词的动词（enjoy/finish/avoid/mind/suggest…）、**动名词 vs 不定式辨析**（remember/forget/stop/try to do vs doing 的意义差别——这张卡要深）。
- **分词（现在分词/过去分词作非谓语）**：现在分词 doing（主动/进行）vs 过去分词 done（被动/完成）、作定语、作状语、作补语、**-ing 形容词 vs -ed 形容词**（interesting/interested、boring/bored，主动让人 vs 被动感到——这张卡要深）。

## 题目（每关 9~11 道）
schema：`choice{options,answer:0基下标}` / `input{accept:[可接受写法]}` / `judge{answer:bool}` / `match{left,right}`。
- input：填非谓语形式 / 选搭配（如 enjoy 后填 doing 形式：`enjoy reading`）；accept 列全写法。每关 ≥2 道。
- choice：选正确形式 / 辨析（to do vs doing、-ing vs -ed 形容词）。
- judge：判断常见错误（如 *I enjoy to read.* 错，应 reading）。
- match：动词 ↔ 它接 to do 还是 doing、或 形式 ↔ 句中成分、或 -ing/-ed 形容词 ↔ 含义（**right[i] 必须与 left[i] 逐项对齐，别打乱**）。
均衡：choice ≤ 5；judge、match 各 ≥ 1；input ≥ 2。每题写 `explain`（可含斜体例句）。

## 自检
- 每关 6~8 卡、含「易错点/辨析」深度卡；英文例句斜体、强调词加粗；非谓语形式与搭配（哪些动词接 to do / doing）务必正确。
- 每关 9~11 题、类型均衡、每题 explain、答案/下标核对；match 逐项对齐。
- 合法 JSON、顺序对、不写 id、不用 `$`/代码围栏。
