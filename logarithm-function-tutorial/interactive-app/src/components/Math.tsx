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
          <span key={i}>{renderBold(p.value)}</span>
        )
      )}
    </>
  )
}

/** 把非公式文本里的 **加粗** 渲染为 <strong>。 */
function renderBold(text: string) {
  const out: (string | JSX.Element)[] = []
  const re = /\*\*([^*]+)\*\*/g
  let last = 0
  let m: RegExpExecArray | null
  let k = 0
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index))
    out.push(<strong key={k++}>{m[1]}</strong>)
    last = m.index + m[0].length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
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
