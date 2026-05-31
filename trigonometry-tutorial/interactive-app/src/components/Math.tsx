import katex from 'katex'
import { useMemo } from 'react'

/**
 * 渲染一段可能含 $...$ 行内公式的文本。
 * 例："求 $\\cos 60^\\circ$ 的值" → 文本 + KaTeX 公式混排。
 */
export function RichText({ text }: { text: string }) {
  const parts = useMemo(() => splitMath(text), [text])
  return (
    <>
      {parts.map((p, i) =>
        p.math ? (
          <span
            key={i}
            className="kx"
            dangerouslySetInnerHTML={{ __html: renderInline(p.value) }}
          />
        ) : (
          <span key={i}>{p.value}</span>
        )
      )}
    </>
  )
}

function renderInline(tex: string): string {
  try {
    return katex.renderToString(tex, {
      throwOnError: false,
      displayMode: false,
    })
  } catch {
    return tex
  }
}

function splitMath(text: string): { math: boolean; value: string }[] {
  const out: { math: boolean; value: string }[] = []
  const re = /\$([^$]+)\$/g
  let last = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(text))) {
    if (m.index > last) out.push({ math: false, value: text.slice(last, m.index) })
    out.push({ math: true, value: m[1] })
    last = m.index + m[0].length
  }
  if (last < text.length) out.push({ math: false, value: text.slice(last) })
  return out
}
