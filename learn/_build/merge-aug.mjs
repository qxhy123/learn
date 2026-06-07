// 把 agent 生成的新题合并进某门课的 units/*.json（去重 + 校验 + 重排题号）。
// 用法：node _build/merge-aug.mjs <courseId> <augDir>
//   augDir 下每个 uN.json 形如 { "<lessonId>": [ 新题... ], ... }
import { promises as fs } from 'fs'
import { existsSync, readFileSync, readdirSync, copyFileSync, mkdirSync } from 'fs'
import path from 'path'

const [courseId, augDirArg] = process.argv.slice(2)
if (!courseId || !augDirArg) { console.error('用法: node merge-aug.mjs <courseId> <augDir>'); process.exit(1) }
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..')
const UNITS_DIR = path.join(ROOT, 'public/courses', courseId, 'units')
const AUG = path.resolve(ROOT, augDirArg)

// --- CJK in math → \text{} ---
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
function cleanRich(s) {
  if (typeof s !== 'string') return s
  return s.replace(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g, (_, d, inl) => d !== undefined ? '$$' + wrapCJK(d) + '$$' : '$' + wrapCJK(inl) + '$')
}

// --- 兜底归一化（同 assembler）---
function normalizeAltSchema(q) {
  const hasP = typeof q.prompt === 'string' && q.prompt.trim()
  if (!hasP && q.stem) { q.prompt = q.stem; delete q.stem }
  if (!(typeof q.prompt === 'string' && q.prompt.trim()) && q.question) { q.prompt = q.question; delete q.question }
  if (q.explanation && !q.explain) { q.explain = q.explanation; delete q.explanation }
  if (q.type === 'choice' && typeof q.answer === 'string' && Array.isArray(q.options)) {
    const idx = q.options.indexOf(q.answer)
    if (idx >= 0) q.answer = idx
    else if (/^[A-Z]$/.test(q.answer.trim()) && q.answer.trim().charCodeAt(0) - 65 < q.options.length) q.answer = q.answer.trim().charCodeAt(0) - 65
  }
  if (q.type === 'input' && !Array.isArray(q.accept) && typeof q.answer === 'string') { q.accept = [q.answer]; delete q.answer }
  if (q.type === 'judge' && typeof q.answer !== 'boolean') {
    const a = String(q.answer).trim()
    if (['正确', '对', 'true', 'True', 'T', '√'].includes(a)) q.answer = true
    else if (['错误', '错', 'false', 'False', 'F', '×'].includes(a)) q.answer = false
  }
  if (q.type === 'match' && Array.isArray(q.pairs) && !q.left) {
    q.left = q.pairs.map((p) => p.left); q.right = q.pairs.map((p) => p.right); delete q.pairs
    if (Array.isArray(q.answer)) delete q.answer
  }
}

function valid(q) {
  if (!['choice', 'input', 'judge', 'match'].includes(q.type)) return false
  if (typeof q.prompt !== 'string' || !q.prompt.trim()) return false
  if (q.type === 'choice') return Array.isArray(q.options) && q.options.length >= 2 && Number.isInteger(q.answer) && q.answer >= 0 && q.answer < q.options.length
  if (q.type === 'input') return Array.isArray(q.accept) && q.accept.length >= 1 && q.accept.every((a) => typeof a === 'string' && a.length)
  if (q.type === 'judge') return typeof q.answer === 'boolean'
  if (q.type === 'match') return Array.isArray(q.left) && Array.isArray(q.right) && q.left.length === q.right.length && q.left.length >= 2
  return false
}

// 归一化题干用于去重：去空白/美元/标点，小写
const dedupKey = (s) => (s || '').toLowerCase().replace(/\$/g, '').replace(/[\s，。、；：！？,.;:!?()（）]/g, '')

function cleanQuestion(q) {
  for (const k of ['prompt', 'explain']) if (q[k]) q[k] = cleanRich(q[k])
  if (Array.isArray(q.options)) q.options = q.options.map(cleanRich)
  if (Array.isArray(q.left)) q.left = q.left.map(cleanRich)
  if (Array.isArray(q.right)) q.right = q.right.map(cleanRich)
  return q
}

async function main() {
  // 备份一次
  const bak = UNITS_DIR + '.bak'
  if (!existsSync(bak)) {
    mkdirSync(bak, { recursive: true })
    for (const f of readdirSync(UNITS_DIR)) copyFileSync(path.join(UNITS_DIR, f), path.join(bak, f))
    console.log('已备份原 units →', path.basename(bak))
  }

  // 1) 聚合 augDir 下所有文件 → lessonId -> 新题数组（支持一个单元拆成多文件、以及 <lessonId>.json 数组形式）
  const augMap = {}
  const push = (lid, arr) => { if (Array.isArray(arr) && arr.length) (augMap[lid] ||= []).push(...arr) }
  for (const f of readdirSync(AUG).filter((f) => f.endsWith('.json'))) {
    const data = JSON.parse(readFileSync(path.join(AUG, f), 'utf8'))
    if (Array.isArray(data)) push(f.replace(/\.json$/, ''), data) // 文件名即 lessonId
    else for (const [lid, arr] of Object.entries(data)) push(lid, arr) // { lessonId: [...] }
  }

  let added = 0, skippedDup = 0, skippedInvalid = 0, lessonsTouched = 0
  for (const uf of readdirSync(UNITS_DIR).filter((f) => f.endsWith('.json')).sort()) {
    const unit = JSON.parse(readFileSync(path.join(UNITS_DIR, uf), 'utf8'))
    let changed = false
    for (const lesson of unit.lessons) {
      const news = augMap[lesson.id]
      if (!Array.isArray(news) || !news.length) continue
      const seen = new Set(lesson.questions.map((q) => dedupKey(q.prompt)))
      const accepted = []
      for (const raw of news) {
        normalizeAltSchema(raw)
        if (!valid(raw)) { skippedInvalid++; continue }
        const key = dedupKey(raw.prompt)
        if (seen.has(key)) { skippedDup++; continue }
        seen.add(key)
        accepted.push(cleanQuestion(raw))
      }
      if (!accepted.length) continue
      lessonsTouched++
      changed = true
      lesson.questions = lesson.questions.concat(accepted)
      lesson.questions.forEach((q, i) => { q.id = `${lesson.id}-q${i + 1}` })
      added += accepted.length
    }
    if (changed) await fs.writeFile(path.join(UNITS_DIR, uf), JSON.stringify(unit), 'utf8')
  }
  console.log(`\n==== 合并完成 (${courseId}) ====`)
  console.log(`新增题 ${added}，跳过重复 ${skippedDup}，跳过非法 ${skippedInvalid}，涉及关卡 ${lessonsTouched}`)
}
main().catch((e) => { console.error('FATAL', e); process.exit(1) })
