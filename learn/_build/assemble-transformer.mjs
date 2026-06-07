// 组装「Transformer 闯关」课程：_build/tf/uN.json → public/courses/transformer/{course.json, units/uN.json}
// 用法：node _build/assemble-transformer.mjs   只组装存在源文件的单元。
import katex from 'katex'
import { promises as fs } from 'fs'
import { existsSync, readFileSync } from 'fs'
import path from 'path'

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..')
const SRC = path.join(ROOT, '_build/tf')
const OUT = path.join(ROOT, 'public/courses/transformer')

// 8 单元 = 教程 8 个 part
const UNITS = [
  { id: 'u1', title: '基础',         color: '#58cc02', icon: '🌱', blurb: '序列建模 · 注意力初步 · 位置编码' },
  { id: 'u2', title: '注意力机制',   color: '#1cb0f6', icon: '🎯', blurb: '自注意力 · 多头注意力 · 掩码注意力' },
  { id: 'u3', title: '架构',         color: '#ce82ff', icon: '🏗️', blurb: '编码器 · 解码器 · 完整 Transformer' },
  { id: 'u4', title: '训练',         color: '#ff9600', icon: '🏋️', blurb: '训练策略 · 优化 · 损失与评估' },
  { id: 'u5', title: '预训练模型',   color: '#ff4b4b', icon: '🧬', blurb: 'BERT · GPT · T5 · 现代 LLM' },
  { id: 'u6', title: '应用',         color: '#0d9488', icon: '🛠️', blurb: '微调 · PEFT · 下游任务' },
  { id: 'u7', title: '进阶',         color: '#9333ea', icon: '🔬', blurb: '高效 Transformer · 多模态 · 可解释性' },
  { id: 'u8', title: '工程',         color: '#e11d48', icon: '🚀', blurb: '推理优化 · 完整项目' },
]

const CJK = /[　-〿一-鿿＀-￯‘’“”]/
function wrapCJK(math) {
  let out = '', i = 0, depth = 0; const td = []
  while (i < math.length) {
    const m = /^\\(text|textbf|textit|textrm|textsf|texttt|mathrm|operatorname|mbox|hbox)\s*\{/.exec(math.slice(i))
    if (m) { out += m[0]; i += m[0].length; depth++; td.push(depth); continue }
    const c = math[i]
    if (c === '{') { depth++; out += c; i++; continue }
    if (c === '}') { if (td.length && td[td.length - 1] === depth) td.pop(); depth--; out += c; i++; continue }
    if (c === '\\') { const cm = /^\\[a-zA-Z]+|^\\./.exec(math.slice(i)); out += cm[0]; i += cm[0].length; continue }
    if (!td.length && CJK.test(c)) { let r = ''; while (i < math.length && CJK.test(math[i]) && math[i] !== '$') { r += math[i]; i++ } out += '\\text{' + r + '}'; continue }
    out += c; i++
  }
  return out
}
function collapseFenced(s) {
  if (typeof s !== 'string' || !s.includes('```')) return s
  return s.replace(/```[^\n]*\n[\s\S]*?\n```/g, (block) =>
    block.replace(/\t/g, '    ').replace(/\n[ \t]*\n+/g, '\n'))
}
function cleanRich(s) {
  if (typeof s !== 'string') return s
  s = collapseFenced(s)
  const fences = []
  s = s.replace(/```[^\n]*\n[\s\S]*?\n```/g, (b) => { fences.push(b); return `\uE000${fences.length - 1}\uE001` })
  s = s.replace(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g, (_, d, inl) => d !== undefined ? '$$' + wrapCJK(d) + '$$' : '$' + wrapCJK(inl) + '$')
  s = s.replace(/\uE000(\d+)\uE001/g, (_, i) => fences[+i])
  return s
}

let warnCount = 0, strayDollar = 0
const warnSamples = []
function auditMath(tex, where, display = false) {
  try { katex.renderToString(tex, { throwOnError: false, displayMode: display, strict: () => 'ignore' }) }
  catch (e) { if (warnSamples.length < 25) warnSamples.push(`[${where}] THROW: ${String(e).slice(0, 60)}`); warnCount++ }
}
function auditRich(s, where) {
  if (typeof s !== 'string') return
  const noFence = s.replace(/```[^\n]*\n[\s\S]*?\n```/g, '').replace(/`[^`]*`/g, '')
  if ((noFence.match(/\$/g) || []).length % 2 === 1) { strayDollar++; if (warnSamples.length < 25) warnSamples.push(`[${where}] 奇数个$: ${s.slice(0, 60)}`) }
  const re = /\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g; let m
  while ((m = re.exec(noFence))) { if (m[1] !== undefined) auditMath(m[1], where, true); else auditMath(m[2], where, false) }
}

const problems = []
function check(cond, msg) { if (!cond) problems.push(msg) }

function normalizeAltSchema(q, where) {
  const hasP = typeof q.prompt === 'string' && q.prompt.trim()
  if (!hasP && q.stem) { q.prompt = q.stem; delete q.stem }
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
    if (['正确', '对', 'true', 'True', 'T', '√', 'Yes', 'yes'].includes(a)) q.answer = true
    else if (['错误', '错', 'false', 'False', 'F', '×', 'No', 'no'].includes(a)) q.answer = false
    else check(false, `${where} judge 答案无法判定: ${q.answer}`)
  }
  if (q.type === 'match' && Array.isArray(q.pairs) && !q.left) {
    q.left = q.pairs.map((p) => p.left); q.right = q.pairs.map((p) => p.right); delete q.pairs
    if (Array.isArray(q.answer)) delete q.answer
  }
}

function processLesson(lesson, unitId, seq, where) {
  const lid = `${unitId}-l${seq}`
  check(typeof lesson.title === 'string' && lesson.title.trim() !== '', `${where} 缺 title`)
  for (const card of lesson.intro || []) {
    const hasContent = (typeof card.body === 'string' && card.body.trim() !== '') ||
      (Array.isArray(card.steps) && card.steps.length) || card.formula || card.image
    check(hasContent, `${where} 卡无内容`)
    for (const k of ['title', 'body', 'tip']) if (card[k]) { card[k] = cleanRich(card[k]); auditRich(card[k], `${where}.${k}`) }
    if (card.formula) { card.formula = wrapCJK(card.formula); auditMath(card.formula, `${where}.formula`, true) }
    if (Array.isArray(card.steps)) card.steps = card.steps.map((s, i) => { const v = cleanRich(s); auditRich(v, `${where}.step${i}`); return v })
    if (card.reveal) {
      card.reveal.q = cleanRich(card.reveal.q); auditRich(card.reveal.q, `${where}.reveal.q`)
      card.reveal.a = cleanRich(card.reveal.a); auditRich(card.reveal.a, `${where}.reveal.a`)
    }
  }
  const qs = lesson.questions || []
  check(qs.length >= 1, `${where} 无题目`)
  qs.forEach((q, qi) => {
    normalizeAltSchema(q, `${where} q${qi + 1}`)
    q.id = `${lid}-q${qi + 1}`
    check(['choice', 'input', 'judge', 'match'].includes(q.type), `${where} q${qi + 1} 非法 type=${q.type}`)
    check(typeof q.prompt === 'string' && q.prompt.trim() !== '', `${where} q${qi + 1} prompt 空`)
    q.prompt = cleanRich(q.prompt); auditRich(q.prompt, `${where}.q${qi + 1}.prompt`)
    if (q.explain) { q.explain = cleanRich(q.explain); auditRich(q.explain, `${where}.q${qi + 1}.explain`) }
    if (q.type === 'choice') {
      if (!Array.isArray(q.options) || q.options.length < 2) { check(false, `${where} q${qi + 1} choice options 缺`); return }
      check(Number.isInteger(q.answer) && q.answer >= 0 && q.answer < q.options.length, `${where} q${qi + 1} choice answer idx=${q.answer}`)
      q.options = q.options.map((o) => { const v = cleanRich(o); auditRich(v, `${where}.q${qi + 1}.opt`); return v })
    } else if (q.type === 'input') {
      check(Array.isArray(q.accept) && q.accept.length >= 1, `${where} q${qi + 1} input accept 缺`)
      if (Array.isArray(q.accept)) q.accept = q.accept.map(String)
    } else if (q.type === 'judge') {
      check(typeof q.answer === 'boolean', `${where} q${qi + 1} judge answer 非 bool`)
    } else if (q.type === 'match') {
      if (!Array.isArray(q.left) || !Array.isArray(q.right)) { check(false, `${where} q${qi + 1} match 缺 left/right`); return }
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
  let totalLessons = 0, builtUnits = 0
  for (const u of UNITS) {
    const file = path.join(SRC, `${u.id}.json`)
    if (!existsSync(file)) continue
    const arr = JSON.parse(readFileSync(file, 'utf8'))
    check(Array.isArray(arr) && arr.length >= 1, `${u.id} 源应为非空关卡数组`)
    const lessons = arr.map((les, i) => processLesson(les, u.id, i + 1, `${u.id}-l${i + 1}`))
    totalLessons += lessons.length; builtUnits++
    await fs.writeFile(path.join(OUT, 'units', `${u.id}.json`), JSON.stringify({ id: u.id, lessons }), 'utf8')
    manifestUnits.push({ id: u.id, title: u.title, color: u.color, icon: u.icon, blurb: u.blurb, file: `courses/transformer/units/${u.id}.json` })
  }

  const manifest = {
    id: 'transformer',
    title: 'Transformer 闯关',
    subtitle: '像闯关一样学 Transformer',
    brandMark: 'QKV',
    brandName: 'Transformer Quest',
    color: '#d97706',
    source: '来自《从零到高阶的 Transformer 教程》',
    figureBase: 'courses/transformer/figures/',
    inputHint: '可填术语、公式符号或数值（中英皆可）；空格大小写不敏感',
    config: { unlockAll: true },
    units: manifestUnits,
  }
  await fs.writeFile(path.join(OUT, 'course.json'), JSON.stringify(manifest, null, 1), 'utf8')

  console.log(`\n==== Transformer 组装完成 ====`)
  console.log(`单元 ${builtUnits}，关卡 ${totalLessons}`)
  console.log(`KaTeX 异常：${warnCount}，散落$(奇数)：${strayDollar}`)
  if (warnSamples.length) console.log('样例:\n' + warnSamples.join('\n'))
  console.log(`结构问题：${problems.length}`)
  if (problems.length) console.log(problems.slice(0, 50).join('\n'))
}
main().catch((e) => { console.error('FATAL', e); process.exit(1) })
