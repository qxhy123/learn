// 组装「日语」课程：_build/jpn/uNcK.json → public/courses/japanese/{course.json,units/uN.json}
// 用法：node _build/assemble-jpn.mjs   （日语 agent 直接产出 JSON，避开 TS 字符串转义问题）
import katex from 'katex'
import { promises as fs } from 'fs'
import { existsSync, readFileSync } from 'fs'
import path from 'path'

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..')
const SRC = path.join(ROOT, '_build/jpn')
const OUT = path.join(ROOT, 'public/courses/japanese')

// 9 单元元信息（每 part 一个单元，各 4 章）
const UNITS = [
  { id: 'u1', n: 4, title: '文字系统',     color: '#58cc02', icon: 'あ', blurb: '假名·五十音·浊拗音·汉字音训读' },
  { id: 'u2', n: 4, title: '基础语法',     color: '#1cb0f6', icon: '🧩', blurb: 'SOV语序·助词·动词活用·形容词' },
  { id: 'u3', n: 4, title: '词类精讲',     color: '#ce82ff', icon: '🏷️', blurb: '名词代词·动词·形容词副词·高级助词' },
  { id: 'u4', n: 4, title: '中级语法',     color: '#ff9600', icon: '🔧', blurb: 'て形·条件·被动使役·授受' },
  { id: 'u5', n: 4, title: '高级语法',     color: '#ff4b4b', icon: '🎌', blurb: '敬语·复句·情态·古典语法' },
  { id: 'u6', n: 4, title: '阅读理解',     color: '#0d9488', icon: '📖', blurb: '阅读策略·高级汉字·文体分析·文学阅读' },
  { id: 'u7', n: 4, title: '写作精通',     color: '#9333ea', icon: '✍️', blurb: '作文基础·文体·学术写作·商务写作' },
  { id: 'u8', n: 4, title: '口语与听力',   color: '#e11d48', icon: '🗣️', blurb: '发音声调·日常会话·正式发言·听力技巧' },
  { id: 'u9', n: 4, title: '高阶精通',     color: '#db2777', icon: '🎓', blurb: '拟声拟态·文化交际·翻译技巧·考试策略' },
]

// --- CJK in math → \text{} 包裹（英语课基本无 $...$，留作兜底） ---
const CJK = /[　-〿一-鿿＀-￯‘’“”]/
function wrapCJK(math) {
  let out = '', i = 0, depth = 0
  const textDepths = []
  while (i < math.length) {
    const m = /^\\(text|textbf|textit|textrm|textsf|texttt|mathrm|operatorname|mbox|hbox)\s*\{/.exec(math.slice(i))
    if (m) { out += m[0]; i += m[0].length; depth++; textDepths.push(depth); continue }
    const c = math[i]
    if (c === '{') { depth++; out += c; i++; continue }
    if (c === '}') {
      if (textDepths.length && textDepths[textDepths.length - 1] === depth) textDepths.pop()
      depth--; out += c; i++; continue
    }
    if (c === '\\') { const cm = /^\\[a-zA-Z]+|^\\./.exec(math.slice(i)); out += cm[0]; i += cm[0].length; continue }
    const inText = textDepths.length > 0
    if (!inText && CJK.test(c)) {
      let run = ''
      while (i < math.length && CJK.test(math[i]) && math[i] !== '$') { run += math[i]; i++ }
      out += '\\text{' + run + '}'
      continue
    }
    out += c; i++
  }
  return out
}
function cleanRich(s) {
  if (typeof s !== 'string') return s
  return s.replace(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g, (full, disp, inl) =>
    disp !== undefined ? '$$' + wrapCJK(disp) + '$$' : '$' + wrapCJK(inl) + '$'
  )
}

// --- 审计 ---
let warnCount = 0
const warnSamples = []
function auditMath(tex, where, display = false) {
  try {
    katex.renderToString(tex, { throwOnError: false, displayMode: display, strict: () => 'ignore' })
  } catch (e) {
    warnSamples.push(`[${where}] THROW: ${String(e).slice(0, 60)}`); warnCount++
  }
}
// 散落的单个 $（英文价格符号等）会被前端 $...$ 渲染误吞 —— 检测奇数个 $
let strayDollar = 0
const strayCount = (s) => (typeof s === 'string' ? (s.match(/\$/g) || []).length : 0)
function auditRich(s, where) {
  if (typeof s !== 'string') return
  if (strayCount(s) % 2 === 1) { strayDollar++; if (warnSamples.length < 20) warnSamples.push(`[${where}] 奇数个$: ${s.slice(0, 60)}`) }
  const re = /\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g; let m
  while ((m = re.exec(s))) { if (m[1] !== undefined) auditMath(m[1], where, true); else auditMath(m[2], where, false) }
}

const problems = []
function check(cond, msg) { if (!cond) problems.push(msg) }

// 兜底归一化（万一 agent 漂移）
function normalizeAltSchema(q, where) {
  const hasPrompt = typeof q.prompt === 'string' && q.prompt.trim()
  if (!hasPrompt && q.stem) { q.prompt = q.stem; delete q.stem }
  if (!(typeof q.prompt === 'string' && q.prompt.trim()) && q.question) { q.prompt = q.question; delete q.question }
  if (q.explanation && !q.explain) { q.explain = q.explanation; delete q.explanation }
  if (q.type === 'choice' && typeof q.answer === 'string' && Array.isArray(q.options)) {
    const idx = q.options.indexOf(q.answer)
    if (idx >= 0) q.answer = idx
    else if (/^[A-Z]$/.test(q.answer.trim()) && q.answer.trim().charCodeAt(0) - 65 < q.options.length) q.answer = q.answer.trim().charCodeAt(0) - 65
    else check(false, `${where} choice 文本答案不在选项中: ${q.answer}`)
  }
  if (q.type === 'input' && !Array.isArray(q.accept) && typeof q.answer === 'string') { q.accept = [q.answer]; delete q.answer }
  if (q.type === 'judge' && typeof q.answer !== 'boolean') {
    const a = String(q.answer).trim()
    if (['正确', '对', 'true', 'True', 'T', '√'].includes(a)) q.answer = true
    else if (['错误', '错', 'false', 'False', 'F', '×'].includes(a)) q.answer = false
    else check(false, `${where} judge 答案无法判定: ${q.answer}`)
  }
  if (q.type === 'match' && Array.isArray(q.pairs) && !q.left) {
    q.left = q.pairs.map((p) => p.left); q.right = q.pairs.map((p) => p.right); delete q.pairs
    if (Array.isArray(q.answer)) delete q.answer
  }
}

function loadLessons(file) {
  const arr = JSON.parse(readFileSync(file, 'utf8'))
  return arr
}

function processLesson(lesson, unitId, lessonSeq, where) {
  const lid = `${unitId}-l${lessonSeq}`
  for (const card of lesson.intro || []) {
    check(typeof card.body === 'string' && card.body.trim() !== '', `${where} 卡 body 空`)
    for (const k of ['title', 'body', 'tip']) if (card[k]) { card[k] = cleanRich(card[k]); auditRich(card[k], `${where}.${k}`) }
    if (card.formula) { card.formula = wrapCJK(card.formula); auditMath(card.formula, `${where}.formula`, true) }
    if (Array.isArray(card.steps)) card.steps = card.steps.map((s, i) => { const v = cleanRich(s); auditRich(v, `${where}.step${i}`); return v })
    if (card.reveal) {
      card.reveal.q = cleanRich(card.reveal.q); auditRich(card.reveal.q, `${where}.reveal.q`)
      card.reveal.a = cleanRich(card.reveal.a); auditRich(card.reveal.a, `${where}.reveal.a`)
    }
  }
  const qs = lesson.questions || []
  qs.forEach((q, qi) => {
    normalizeAltSchema(q, `${where} q${qi + 1}`)
    q.id = `${lid}-q${qi + 1}`
    check(['choice', 'input', 'judge', 'match'].includes(q.type), `${where} q${qi + 1} 非法 type=${q.type}`)
    check(typeof q.prompt === 'string' && q.prompt.trim() !== '', `${where} q${qi + 1} prompt 空`)
    q.prompt = cleanRich(q.prompt); auditRich(q.prompt, `${where}.q${qi + 1}.prompt`)
    if (q.explain) { q.explain = cleanRich(q.explain); auditRich(q.explain, `${where}.q${qi + 1}.explain`) }
    if (q.type === 'choice') {
      if (!Array.isArray(q.options) || q.options.length < 2) { check(false, `${where} q${qi + 1} choice options 缺: keys=${Object.keys(q).join(',')}`); return }
      check(Number.isInteger(q.answer) && q.answer >= 0 && q.answer < q.options.length, `${where} q${qi + 1} choice answer idx=${q.answer}`)
      q.options = q.options.map((o) => { const v = cleanRich(o); auditRich(v, `${where}.q${qi + 1}.opt`); return v })
    } else if (q.type === 'input') {
      check(Array.isArray(q.accept) && q.accept.length >= 1, `${where} q${qi + 1} input accept 缺: keys=${Object.keys(q).join(',')}`)
    } else if (q.type === 'judge') {
      check(typeof q.answer === 'boolean', `${where} q${qi + 1} judge answer bool: keys=${Object.keys(q).join(',')}`)
    } else if (q.type === 'match') {
      if (!Array.isArray(q.left) || !Array.isArray(q.right)) { check(false, `${where} q${qi + 1} match 缺 left/right: keys=${Object.keys(q).join(',')}`); return }
      check(q.left.length === q.right.length && q.left.length >= 2, `${where} q${qi + 1} match 左右不等长 ${q.left.length}/${q.right.length}`)
      q.left = q.left.map((o) => { const v = cleanRich(o); auditRich(v, `${where}.q${qi + 1}.L`); return v })
      q.right = q.right.map((o) => { const v = cleanRich(o); auditRich(v, `${where}.q${qi + 1}.R`); return v })
    }
  })
  lesson.id = lid
  return lesson
}

async function main() {
  await fs.mkdir(path.join(OUT, 'units'), { recursive: true })
  const manifestUnits = []
  let totalLessons = 0
  for (const u of UNITS) {
    const lessons = []
    for (let c = 1; c <= u.n; c++) {
      const file = path.join(SRC, `${u.id}c${c}.json`)
      if (!existsSync(file)) { check(false, `缺文件 ${u.id}c${c}.json`); continue }
      const arr = loadLessons(file)
      check(Array.isArray(arr) && arr.length === 1, `${u.id}c${c} 应恰好 1 关，实为 ${arr?.length}`)
      lessons.push(processLesson(arr[0], u.id, lessons.length + 1, `${u.id}c${c}`))
    }
    totalLessons += lessons.length
    await fs.writeFile(path.join(OUT, 'units', `${u.id}.json`), JSON.stringify({ id: u.id, lessons }), 'utf8')
    manifestUnits.push({ id: u.id, title: u.title, color: u.color, icon: u.icon, blurb: u.blurb, file: `courses/japanese/units/${u.id}.json` })
  }

  const manifest = {
    id: 'japanese',
    title: '日语闯关',
    subtitle: '像闯关一样学日语',
    brandMark: 'あ',
    brandName: 'Japanese Quest',
    color: '#ec4899',
    source: '来自《日语从零到高阶完全教程》',
    figureBase: 'courses/japanese/figures/',
    inputHint: "可填假名或罗马音；不区分大小写",
    config: { unlockAll: true },
    units: manifestUnits,
  }
  await fs.writeFile(path.join(OUT, 'course.json'), JSON.stringify(manifest, null, 1), 'utf8')

  console.log(`\n==== 日语组装完成 ====`)
  console.log(`单元 ${manifestUnits.length}，关卡 ${totalLessons}`)
  console.log(`KaTeX 异常：${warnCount}，散落$(奇数)：${strayDollar}`)
  if (warnSamples.length) console.log('样例:\n' + warnSamples.join('\n'))
  console.log(`结构问题：${problems.length}`)
  if (problems.length) console.log(problems.slice(0, 40).join('\n'))
}
main().catch((e) => { console.error('FATAL', e); process.exit(1) })
