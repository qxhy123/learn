// 组装「高中几何」课程：_build/hgeo/uNcK.ts → public/courses/senior-geometry/{course.json,units/uN.json}
// 用法：node _build/assemble-hgeo.mjs
import { build } from 'esbuild'
import katex from 'katex'
import { promises as fs } from 'fs'
import { existsSync } from 'fs'
import path from 'path'
import os from 'os'

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..')
const SRC = path.join(ROOT, '_build/hgeo')
const FIG_SRC = '/Users/yangyang/ai_projs/math/gaozhong_math/geometry/figures/svg'
const OUT = path.join(ROOT, 'public/courses/senior-geometry')

// 10 单元元信息（章数与 part 目录对应）
const UNITS = [
  { id: 'u1',  n: 4, title: '平面向量基础',   color: '#58cc02', icon: '➡️', blurb: '向量概念·加减·数乘·基底坐标' },
  { id: 'u2',  n: 4, title: '平面向量的应用', color: '#1cb0f6', icon: '🔻', blurb: '数量积·三角形·线圆·物理应用' },
  { id: 'u3',  n: 5, title: '直线与方程',     color: '#ce82ff', icon: '📏', blurb: '斜率·方程形式·位置关系·距离' },
  { id: 'u4',  n: 4, title: '圆与方程',       color: '#ff9600', icon: '⭕', blurb: '标准方程·直线与圆·两圆·应用' },
  { id: 'u5',  n: 4, title: '椭圆',           color: '#ff4b4b', icon: '🥚', blurb: '定义方程·几何性质·直线椭圆·应用' },
  { id: 'u6',  n: 4, title: '双曲线',         color: '#2b70c9', icon: '〽️', blurb: '定义方程·渐近线·直线双曲线·应用' },
  { id: 'u7',  n: 4, title: '抛物线',         color: '#00b4a0', icon: '🛰️', blurb: '定义方程·焦点准线·直线抛物线·应用' },
  { id: 'u8',  n: 6, title: '立体几何基础',   color: '#f5b800', icon: '🧊', blurb: '点线面·线面关系·面面关系·面积体积' },
  { id: 'u9',  n: 6, title: '空间向量与立体几何', color: '#9333ea', icon: '🧭', blurb: '空间向量·坐标·证平行垂直·求角与距离' },
  { id: 'u10', n: 6, title: '几何综合与压轴', color: '#db2777', icon: '🏆', blurb: '圆锥曲线综合·含参·轨迹·定点定值·立体综合' },
]

// --- CJK in math → \text{} 包裹 ---
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

// 清洗一段含 $...$ / $$...$$ 的富文本：每个 math 段内做 CJK 包裹
function cleanRich(s) {
  if (typeof s !== 'string') return s
  return s.replace(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g, (full, disp, inl) =>
    disp !== undefined ? '$$' + wrapCJK(disp) + '$$' : '$' + wrapCJK(inl) + '$'
  )
}
function cleanFormula(s) {
  if (typeof s !== 'string') return s
  return wrapCJK(s)
}

// --- KaTeX 审计：收集 warning ---
let warnCount = 0
const warnSamples = []
function auditMath(tex, where, display = false) {
  try {
    katex.renderToString(tex, {
      throwOnError: false, displayMode: display, strict: (code) => {
        if (code === 'unicodeTextInMathMode' || code === 'unknownSymbol') {
          warnCount++
          if (warnSamples.length < 20) warnSamples.push(`[${where}] ${code}: ${tex.slice(0, 60)}`)
          return 'ignore'
        }
        return 'ignore'
      },
    })
  } catch (e) {
    warnSamples.push(`[${where}] THROW: ${String(e).slice(0, 80)} :: ${tex.slice(0, 50)}`)
    warnCount++
  }
}
function auditRich(s, where) {
  if (typeof s !== 'string') return
  const re = /\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g; let m
  while ((m = re.exec(s))) {
    if (m[1] !== undefined) auditMath(m[1], where, true)
    else auditMath(m[2], where, false)
  }
}

const problems = []
function check(cond, msg) { if (!cond) problems.push(msg) }

// 把 u9/u10/u13 那批 agent 用的 stem/pairs/文本答案 schema 归一化为标准 schema
function normalizeAltSchema(q, where) {
  const hasPrompt = typeof q.prompt === 'string' && q.prompt.trim()
  if (!hasPrompt && q.stem) { q.prompt = q.stem; delete q.stem }
  if (!(typeof q.prompt === 'string' && q.prompt.trim()) && q.question) { q.prompt = q.question; delete q.question }
  if (q.explanation && !q.explain) { q.explain = q.explanation; delete q.explanation }
  if (q.type === 'choice' && typeof q.answer === 'string' && Array.isArray(q.options)) {
    const idx = q.options.indexOf(q.answer)
    if (idx >= 0) q.answer = idx
    else if (/^[A-Z]$/.test(q.answer.trim()) && q.answer.trim().charCodeAt(0) - 65 < q.options.length) {
      q.answer = q.answer.trim().charCodeAt(0) - 65 // 字母序号 A/B/C → 0/1/2
    } else check(false, `${where} choice 文本答案不在选项中: ${q.answer}`)
  }
  if (q.type === 'input' && !Array.isArray(q.accept)) {
    if (typeof q.answer === 'string') { q.accept = [q.answer]; delete q.answer }
  }
  if (q.type === 'judge' && typeof q.answer !== 'boolean') {
    const a = String(q.answer).trim()
    if (['正确', '对', 'true', '真', 'T', '√'].includes(a)) q.answer = true
    else if (['错误', '错', 'false', '假', 'F', '×'].includes(a)) q.answer = false
    else check(false, `${where} judge 答案无法判定: ${q.answer}`)
  }
  if (q.type === 'match' && Array.isArray(q.pairs) && !q.left) {
    q.left = q.pairs.map((p) => p.left)
    q.right = q.pairs.map((p) => p.right)
    delete q.pairs
    if (Array.isArray(q.answer)) delete q.answer // match 由 left/right 顺序对应，无需独立 answer
  }
}

async function loadLessons(file) {
  const res = await build({
    entryPoints: [file], bundle: false, write: false, format: 'esm', platform: 'neutral', logLevel: 'silent',
  })
  const code = res.outputFiles[0].text
  const dataUrl = 'data:text/javascript;base64,' + Buffer.from(code).toString('base64')
  const mod = await import(dataUrl)
  return mod.LESSONS
}

const usedFigs = new Set()

function processLesson(lesson, unitId, lessonSeq, where) {
  const lid = `${unitId}-l${lessonSeq}`
  // intro 卡
  for (const card of lesson.intro || []) {
    check(typeof card.body === 'string' && card.body.trim() !== '', `${where} 卡 body 空`)
    for (const k of ['title', 'body', 'tip']) if (card[k]) { card[k] = cleanRich(card[k]); auditRich(card[k], `${where}.${k}`) }
    if (card.formula) { card.formula = cleanFormula(card.formula); auditMath(card.formula, `${where}.formula`, true) }
    if (Array.isArray(card.steps)) card.steps = card.steps.map((s, i) => { const v = cleanRich(s); auditRich(v, `${where}.step${i}`); return v })
    if (card.reveal) {
      card.reveal.q = cleanRich(card.reveal.q); auditRich(card.reveal.q, `${where}.reveal.q`)
      card.reveal.a = cleanRich(card.reveal.a); auditRich(card.reveal.a, `${where}.reveal.a`)
    }
    if (card.image) usedFigs.add(card.image)
  }
  // questions
  const qs = lesson.questions || []
  qs.forEach((q, qi) => {
    normalizeAltSchema(q, `${where} q${qi + 1}`)
    q.id = `${lid}-q${qi + 1}`
    const ok = ['choice', 'input', 'judge', 'match'].includes(q.type)
    check(ok, `${where} q${qi + 1} 非法 type=${q.type}`)
    check(typeof q.prompt === 'string' && q.prompt.trim() !== '', `${where} q${qi + 1} prompt 空`)
    q.prompt = cleanRich(q.prompt); auditRich(q.prompt, `${where}.q${qi + 1}.prompt`)
    if (q.explain) { q.explain = cleanRich(q.explain); auditRich(q.explain, `${where}.q${qi + 1}.explain`) }
    if (q.type === 'choice') {
      if (!Array.isArray(q.options) || q.options.length < 2) { check(false, `${where} q${qi + 1} choice options 缺/少: keys=${Object.keys(q).join(',')}`); return }
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
  await fs.mkdir(path.join(OUT, 'figures'), { recursive: true })
  const manifestUnits = []
  let totalLessons = 0
  for (const u of UNITS) {
    const lessons = []
    for (let c = 1; c <= u.n; c++) {
      const file = path.join(SRC, `${u.id}c${c}.ts`)
      check(existsSync(file), `缺文件 ${u.id}c${c}.ts`)
      if (!existsSync(file)) continue
      const arr = await loadLessons(file)
      check(Array.isArray(arr) && arr.length === 1, `${u.id}c${c} LESSONS 应恰好 1 关，实为 ${arr?.length}`)
      const lesson = arr[0]
      lessons.push(processLesson(lesson, u.id, lessons.length + 1, `${u.id}c${c}`))
    }
    totalLessons += lessons.length
    const unitJson = { id: u.id, lessons }
    await fs.writeFile(path.join(OUT, 'units', `${u.id}.json`), JSON.stringify(unitJson), 'utf8')
    manifestUnits.push({ id: u.id, title: u.title, color: u.color, icon: u.icon, blurb: u.blurb, file: `courses/senior-geometry/units/${u.id}.json` })
  }

  // course.json 清单
  const manifest = {
    id: 'senior-geometry',
    title: '高中几何闯关',
    subtitle: '像闯关一样学高中几何',
    brandMark: '∡',
    brandName: 'Senior Geometry Quest',
    color: '#0d9488',
    source: '来自《高中几何教程》',
    figureBase: 'courses/senior-geometry/figures/',
    inputHint: "坐标/向量按题目格式填，数值支持分数与 √ 形式",
    config: { unlockAll: true },
    units: manifestUnits,
  }
  await fs.writeFile(path.join(OUT, 'course.json'), JSON.stringify(manifest, null, 1), 'utf8')

  // 拷贝引用到的图
  let copied = 0, missFig = 0
  for (const f of usedFigs) {
    const s = path.join(FIG_SRC, f)
    if (existsSync(s)) { await fs.copyFile(s, path.join(OUT, 'figures', f)); copied++ }
    else { problems.push(`缺图 ${f}`); missFig++ }
  }

  console.log(`\n==== 高中几何组装完成 ====`)
  console.log(`单元 ${manifestUnits.length}，关卡 ${totalLessons}，拷图 ${copied}（缺 ${missFig}）`)
  console.log(`KaTeX 警告/异常：${warnCount}`)
  if (warnSamples.length) console.log('样例:\n' + warnSamples.join('\n'))
  console.log(`结构问题：${problems.length}`)
  if (problems.length) console.log(problems.slice(0, 40).join('\n'))
}
main().catch((e) => { console.error('FATAL', e); process.exit(1) })
