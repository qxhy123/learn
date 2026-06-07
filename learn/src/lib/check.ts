// 输入题的容错答案匹配。
// 目标：让 "√3/2"、"sqrt3/2"、"root3/2"、"(√3)/2" 等写法都能判对，
// 同时支持数值近似（如 0.866 ≈ √3/2）。

/** 把各种等价写法标准化成一个规范字符串，用于精确比较。 */
export function normalize(raw: string): string {
  let s = raw.trim().toLowerCase()
  if (s === '') return ''
  // 全角 → 半角常见符号
  s = s
    .replace(/[（(]/g, '(')
    .replace(/[）)]/g, ')')
    .replace(/[／]/g, '/')
    .replace(/[＋]/g, '+')
    .replace(/[－—–]/g, '-')
    .replace(/[×*∗]/g, '*')
    .replace(/[＝]/g, '=')
  // 去掉所有空白
  s = s.replace(/\s+/g, '')
  // 根号的各种写法统一成 √
  s = s
    .replace(/sqrt/g, '√')
    .replace(/root/g, '√')
    .replace(/\\sqrt/g, '√')
  // π 的写法
  s = s.replace(/\\?pi/g, 'π').replace(/[Π]/g, 'π')
  // \frac{a}{b} → a/b（简单情形）
  s = s.replace(/\\?frac\{([^{}]*)\}\{([^{}]*)\}/g, '$1/$2')
  // 去掉无歧义的外层括号包裹的根号项，如 (√3)/2 → √3/2
  s = s.replace(/\(√(\d+)\)/g, '√$1')
  // 去掉乘号 1* 之类冗余
  s = s.replace(/(^|[^0-9])1\*/g, '$1')
  // 统一 -0 → 0
  if (s === '-0') s = '0'
  return s
}

/** 尝试把答案解析为数值（用于近似比较）。支持 √、π、分数。 */
export function toNumber(raw: string): number | null {
  let s = normalize(raw)
  if (s === '') return null
  // 替换 √n → Math.sqrt(n)
  s = s.replace(/√(\d+(?:\.\d+)?)/g, (_m, n) => `Math.sqrt(${n})`)
  // √ 后跟括号
  s = s.replace(/√\(/g, 'Math.sqrt(')
  // π
  s = s.replace(/π/g, `(${Math.PI})`)
  // 仅允许安全字符，避免任意代码执行
  if (!/^[-+*/().0-9a-z_]*$/i.test(s.replace(/Math\.sqrt/g, ''))) {
    // 含字母（除 Math.sqrt 外）→ 不是纯数值
  }
  if (!/^[0-9+\-*/().\s]*$/.test(s.replace(/Math\.sqrt/g, ''))) return null
  try {
    // eslint-disable-next-line no-new-func
    const val = Function(`"use strict";return (${s})`)()
    return typeof val === 'number' && isFinite(val) ? val : null
  } catch {
    return null
  }
}

/** 判定用户输入是否匹配任一可接受答案。 */
export function matchesInput(user: string, accept: string[]): boolean {
  const u = normalize(user)
  if (u === '') return false
  for (const a of accept) {
    if (normalize(a) === u) return true
  }
  // 数值近似（容差 1e-2，足以覆盖 0.87 ≈ √3/2 这类）
  const un = toNumber(user)
  if (un !== null) {
    for (const a of accept) {
      const an = toNumber(a)
      if (an !== null && Math.abs(an - un) < 1e-2) return true
    }
  }
  return false
}

/** 今天的本地自然日 YYYY-MM-DD。 */
export function today(): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

/** 计算从 lastDay 到今天的关系：'same' | 'consecutive' | 'gap'。 */
export function dayRelation(lastDay: string | null): 'same' | 'consecutive' | 'gap' {
  if (!lastDay) return 'gap'
  const t = today()
  if (lastDay === t) return 'same'
  const d1 = new Date(lastDay + 'T00:00:00')
  const d2 = new Date(t + 'T00:00:00')
  const diff = Math.round((d2.getTime() - d1.getTime()) / 86400000)
  return diff === 1 ? 'consecutive' : 'gap'
}
