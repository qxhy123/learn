# 英语课「词汇构建」扩充 · 改写规范

把英语课 u1 里原来压成一关的「词汇构建策略」扩成多关：**词汇构建策略(总览) / 词根 / 前缀 / 后缀 / 语义场 / 同义词辨析**。风格、口吻、卡片格式必须和英语课现有关卡完全一致。

## 第一步：先读风格样板
先 Read /Users/yangyang/ai_projs/math/learn/_build/eng/_STYLE_U1VOCAB.json （这是原来那关「词汇构建策略」）。**照它的语气、卡片粒度、中英混排、表格、斜体例句来写**。

排版约定：
- 英文单词/例句用**斜体**：`*reconstruct*`、`*She is **happy**.*`（强调处再 `**加粗**`）。
- 中文讲解为主，术语给中英对照（如 词根 root、前缀 prefix、后缀 suffix、语义场 semantic field、搭配 collocation）。
- 词根/词缀/同义词的对照很适合 markdown 表格（`|` 仅在表格行内用）。不要用 `$`、不要用代码围栏。

## 内容源
主源：Read /Users/yangyang/ai_projs/math/english-tutorial/part1-phonetics-vocabulary/03-vocabulary-building.md 。其中：
- §3.1 词汇量级与目标、§3.3 记忆科学、§3.4 语境记忆、§3.6 词频表 → 用于「总览」关。
- §3.2 词根词缀法（**50 个高频词根**、**30 个高频前后缀**、拆词猜义）→ 用于 词根 / 前缀 / 后缀 三关。
- §3.5 同义词辨析与词汇网络（**语义场理论**、**高频同义词辨析**、搭配）→ 用于 语义场 / 同义词辨析 两关。
忠实原文，举正确的词例（词根含义、词缀派生、同义词区别务必准确）。源里词例不够时可补标准常见词，但不臆造词源。

## 输出
- Write 到指定文件，**关卡数组**（按给定顺序），每关 `{ "title", "subtitle", "intro":[卡片], "questions":[题] }`，**不写 id**。合法 JSON。
- `title` 用关名；`subtitle` 一句话点主题。

## 各关要点（每关 6~8 卡）
- **词汇构建策略（总览）**：词汇量临界点与各阶段目标、“懂一个词”的多维度、构词法三大途径概览（派生/合成/转化）、记忆科学（间隔重复 spaced repetition、主动回忆 active recall、记忆宫殿）、词频表与学习优先级（GSL/AWL）。不展开具体词根词缀（那是后面几关）。
- **词根 Roots**：为什么词根高效、拆词猜义法、一批高频词根（如 *spect*=看、*port*=拿/搬、*dict*=说、* duc/duct*=引导、*ven/vent*=来、*scrib/script*=写、*vis/vid*=看、*tract*=拉…）每个配 2~3 个派生词。用表格列「词根—含义—例词」。
- **前缀 Prefixes**：按功能分组——否定（*un-/in-/im-/dis-/non-/mis-*）、方向位置（*re-/pre-/post-/sub-/inter-/trans-/ex-*）、数量（*mono-/bi-/multi-/semi-*）、程度（*over-/under-/super-*）。每个前缀给含义 + 例词。
- **后缀 Suffixes**：核心是“看词尾辨词性”——名词后缀（*-tion/-ment/-ness/-ity/-er/-ist/-ism*）、动词后缀（*-ize/-ify/-en*）、形容词后缀（*-able/-ful/-less/-ous/-al/-ive/-y*）、副词后缀（*-ly*）。用表格列「后缀—词性—例词」。
- **语义场 Semantic Fields**：语义场理论（一组相关词构成场，如“烹饪”场：boil/fry/roast/steam…）、词汇网络/聚类记忆、上下义词（hyponym）、搭配 collocation（make/do、heavy rain 等地道搭配 vs 中式英语）。
- **同义词辨析 Synonyms**：高频同义词组的细微区别 + 用法/语域，如 *big/large/huge/enormous*、*look/see/watch/glance*、*say/tell/speak/talk*、*small/little/tiny*、*happy/glad/pleased/delighted*、*begin/start/commence*。每组讲清区别并配例句。

## 题目（每关 9~11 道）
schema：`choice{options,answer:0基下标}` / `input{accept:[可接受写法]}` / `judge{answer:bool}` / `match{left,right}`。
- input：给词根/前缀/后缀问含义或派生词（如 “前缀 re- 的含义？”→ 再/重新；“port 词根派生一个词”→ import/export/transport 之一）；同义词填空选词。每关 ≥2 道。
- choice：选词义/词性/正确搭配/正确同义词。
- judge：判断词源或用法说法对错。
- match：**词根↔含义、前缀↔含义、后缀↔词性、同义词↔区别、单词↔语义场**（很合适；**right[i] 必须与 left[i] 逐项对齐，别打乱**）。
均衡：choice ≤ 5；judge、match 各 ≥ 1；input ≥ 2。每题写 `explain`（可含斜体例词）。

## 自检
- 每关 6~8 卡、含表格或深度辨析卡；英文词例斜体；词根含义/词缀派生/同义词区别务必正确。
- 每关 9~11 题、类型均衡、每题 explain、答案/下标核对；match 逐项对齐。
- 合法 JSON、顺序对、不写 id、不用 `$`/代码围栏。
