// 修复 JSON 字符串值里未转义的内容双引号。
// 启发式：一个 " 是“结构性”引号当且仅当 它紧邻 JSON 语法符号
//   - 左侧最近非空白 ∈ { [ , :   （开引号）
//   - 或 右侧最近非空白 ∈ } ] , :  （闭引号）
// 否则视为内容引号 → 转义为 \"
// 已经是 \" 的（前面是反斜杠）保持不变。
import { readFileSync, writeFileSync } from 'fs'

const OPEN_BEFORE = new Set(['{', '[', ',', ':'])
const CLOSE_AFTER = new Set(['}', ']', ',', ':'])

function fix(text) {
  let out = ''
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (c !== '"') { out += c; continue }
    // 已转义的 \" —— 数一下前面连续反斜杠个数，奇数则已转义
    let bs = 0, j = i - 1
    while (j >= 0 && text[j] === '\\') { bs++; j-- }
    if (bs % 2 === 1) { out += c; continue }
    // 左侧最近非空白
    let p = i - 1
    while (p >= 0 && /\s/.test(text[p])) p--
    const prev = p >= 0 ? text[p] : ''
    // 右侧最近非空白
    let n = i + 1
    while (n < text.length && /\s/.test(text[n])) n++
    const next = n < text.length ? text[n] : ''
    const structural = OPEN_BEFORE.has(prev) || CLOSE_AFTER.has(next)
    out += structural ? c : '\\"'
  }
  return out
}

const files = process.argv.slice(2)
let fixed = 0, fail = 0
for (const f of files) {
  const orig = readFileSync(f, 'utf8')
  let ok = true
  try { JSON.parse(orig) } catch { ok = false }
  if (ok) { console.log(`OK(已合法) ${f}`); continue }
  const repaired = fix(orig)
  try {
    JSON.parse(repaired)
    writeFileSync(f, repaired)
    console.log(`FIXED ${f}`)
    fixed++
  } catch (e) {
    console.log(`STILL BAD ${f} :: ${String(e).slice(0, 90)}`)
    fail++
  }
}
console.log(`\n修复 ${fixed}，仍失败 ${fail}`)
