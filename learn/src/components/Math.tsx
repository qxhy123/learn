import katex from 'katex'
import { useMemo } from 'react'

/**
 * 渲染一段可能含 `行内代码`、$...$ 行内公式、**加粗** 的文本。
 * 先切出 `code` 段（内部不再解析公式/加粗），其余按公式+加粗渲染。
 */
export function RichText({ text }: { text: string }) {
  const segs = useMemo(() => splitCode(text), [text])
  return (
    <>
      {segs.map((s, i) =>
        s.code ? (
          <code key={i} className="md-code">{s.value}</code>
        ) : (
          <MathBold key={i} text={s.value} />
        )
      )}
    </>
  )
}

/** 把一段（已剔除 `code`）文本按 $...$ / $$...$$ 公式 + **加粗** 渲染。 */
function MathBold({ text }: { text: string }) {
  const parts = useMemo(() => splitMath(text), [text])
  return (
    <>
      {parts.map((p, i) =>
        p.kind === 'inline' ? (
          <span
            key={i}
            className="kx"
            dangerouslySetInnerHTML={{ __html: renderInline(p.value, false) }}
          />
        ) : p.kind === 'display' ? (
          <span
            key={i}
            className="kx kx-display"
            dangerouslySetInnerHTML={{ __html: renderInline(p.value, true) }}
          />
        ) : (
          <span key={i}>{renderBold(p.value)}</span>
        )
      )}
    </>
  )
}

/** 把文本按反引号 `code` 切成代码段与普通段。 */
function splitCode(text: string): { code: boolean; value: string }[] {
  const out: { code: boolean; value: string }[] = []
  const re = /`([^`]+)`/g
  let last = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(text))) {
    if (m.index > last) out.push({ code: false, value: text.slice(last, m.index) })
    out.push({ code: true, value: m[1] })
    last = m.index + m[0].length
  }
  if (last < text.length) out.push({ code: false, value: text.slice(last) })
  return out
}

/** 把非公式文本里的 **加粗** 与 *斜体* 渲染为 <strong> / <em>。
 *  斜体内容须以非空白起头（如 *I went*），避免误伤散文里的 `a * b` 这类乘号。 */
function renderBold(text: string) {
  const out: (string | JSX.Element)[] = []
  const re = /\*\*([^*]+)\*\*|\*([^\s*][^*]*?)\*/g
  let last = 0
  let m: RegExpExecArray | null
  let k = 0
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index))
    if (m[1] !== undefined) out.push(<strong key={k++}>{m[1]}</strong>)
    else out.push(<em key={k++}>{m[2]}</em>)
    last = m.index + m[0].length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

function renderInline(tex: string, display: boolean): string {
  try {
    return katex.renderToString(tex, {
      throwOnError: false,
      displayMode: display,
    })
  } catch {
    return tex
  }
}

/**
 * 渲染一段可能含 markdown 块级语法的文本：表格、无序/有序列表、段落（单换行→<br>）。
 * 行内仍复用 RichText（$...$ 公式 + **加粗**）。用于 body/tip/讲解/揭示等多行富文本。
 */
export function Markdown({ text }: { text: string }) {
  if (typeof text !== 'string') return null
  const blocks = text.split(/\n\n+/).filter((b) => b.trim() !== '')
  return (
    <>
      {blocks.map((b, i) => (
        <MdBlock key={i} text={b} />
      ))}
    </>
  )
}

function isSeparatorRow(line: string): boolean {
  const cells = line.trim().replace(/^\||\|$/g, '').split('|')
  return cells.length >= 1 && cells.every((c) => /^\s*:?-{2,}:?\s*$/.test(c))
}

function splitRow(line: string): string[] {
  return line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim())
}

function MdBlock({ text }: { text: string }) {
  const lines = text.split('\n')
  const nonEmpty = lines.filter((l) => l.trim() !== '')

  // 代码块：```lang ... ```（不含空行的单块）
  const trimmed = text.trim()
  if (trimmed.startsWith('```') && trimmed.endsWith('```') && trimmed.length > 6) {
    const inner = trimmed.replace(/^```[^\n]*\n?/, '').replace(/\n?```$/, '')
    return <pre className="md-pre"><code>{inner}</code></pre>
  }

  // 表格：第 1 行含 |，第 2 行是 ---|--- 分隔
  if (lines.length >= 2 && lines[0].includes('|') && isSeparatorRow(lines[1])) {
    const header = splitRow(lines[0])
    const body = lines.slice(2).filter((l) => l.trim() !== '').map(splitRow)
    return (
      <div className="md-table-wrap">
        <table className="md-table">
          <thead>
            <tr>{header.map((c, i) => <th key={i}><RichText text={c} /></th>)}</tr>
          </thead>
          <tbody>
            {body.map((r, ri) => (
              <tr key={ri}>{r.map((c, ci) => <td key={ci}><RichText text={c} /></td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  // 无序列表：所有非空行都以 - 或 * 开头
  if (nonEmpty.length > 0 && nonEmpty.every((l) => /^\s*[-*]\s+/.test(l))) {
    return (
      <ul className="md-list">
        {nonEmpty.map((l, i) => <li key={i}><RichText text={l.replace(/^\s*[-*]\s+/, '')} /></li>)}
      </ul>
    )
  }

  // 有序列表：所有非空行都以 1. 2. 开头
  if (nonEmpty.length > 0 && nonEmpty.every((l) => /^\s*\d+\.\s+/.test(l))) {
    return (
      <ol className="md-list">
        {nonEmpty.map((l, i) => <li key={i}><RichText text={l.replace(/^\s*\d+\.\s+/, '')} /></li>)}
      </ol>
    )
  }

  // 普通段落：单换行渲染为 <br>
  return (
    <p className="md-p">
      {lines.map((l, i) => (
        <span key={i}>
          {i > 0 && <br />}
          <RichText text={l} />
        </span>
      ))}
    </p>
  )
}

/** 居中大号公式（display 模式），用于学新知卡片的“关键公式”。 */
export function BlockMath({ tex }: { tex: string }) {
  let html = tex
  try {
    html = katex.renderToString(tex, { throwOnError: false, displayMode: true })
  } catch {
    /* keep raw */
  }
  return <div className="block-math" dangerouslySetInnerHTML={{ __html: html }} />
}

type MathPart = { kind: 'text' | 'inline' | 'display'; value: string }
function splitMath(text: string): MathPart[] {
  const out: MathPart[] = []
  // 先匹配 $$...$$ 显示公式，再匹配 $...$ 行内公式
  const re = /\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g
  let last = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(text))) {
    if (m.index > last) out.push({ kind: 'text', value: text.slice(last, m.index) })
    if (m[1] !== undefined) out.push({ kind: 'display', value: m[1] })
    else out.push({ kind: 'inline', value: m[2] })
    last = m.index + m[0].length
  }
  if (last < text.length) out.push({ kind: 'text', value: text.slice(last) })
  return out
}
