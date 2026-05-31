import { useMemo, useState } from 'react'
import type { MatchQuestion as Q } from '../../types'
import { RichText } from '../Math'

interface Props {
  q: Q
  submitted: boolean
  onState: (ready: boolean, correct: boolean) => void
}

/** 基于题目 id 的稳定打乱，避免每次渲染顺序变化。 */
function shuffleIdx(n: number, seed: string): number[] {
  const arr = Array.from({ length: n }, (_, i) => i)
  let h = 0
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0
  for (let i = n - 1; i > 0; i--) {
    h = (h * 1103515245 + 12345) & 0x7fffffff
    const j = h % (i + 1)
    ;[arr[i], arr[j]] = [arr[j], arr[i]]
  }
  return arr
}

const PAIR_COLORS = ['#58cc02', '#1cb0f6', '#ff9600', '#ce82ff', '#ff4b4b', '#2b70c9']

export function MatchQuestion({ q, submitted, onState }: Props) {
  const rightOrder = useMemo(() => shuffleIdx(q.right.length, q.id), [q])
  // leftIndex -> rightOriginalIndex
  const [pairs, setPairs] = useState<Record<number, number>>({})
  const [activeLeft, setActiveLeft] = useState<number | null>(null)

  function emit(next: Record<number, number>) {
    const keys = Object.keys(next)
    const ready = keys.length === q.left.length
    const correct = ready && keys.every((k) => next[+k] === +k)
    onState(ready, correct)
  }

  function clickLeft(li: number) {
    if (submitted) return
    setActiveLeft(activeLeft === li ? null : li)
  }
  function clickRight(ri: number) {
    if (submitted || activeLeft === null) return
    const next: Record<number, number> = {}
    // 移除任何已指向 ri 的旧配对 & activeLeft 的旧配对
    for (const [k, v] of Object.entries(pairs)) {
      if (+k === activeLeft) continue
      if (v === ri) continue
      next[+k] = v
    }
    next[activeLeft] = ri
    setPairs(next)
    setActiveLeft(null)
    emit(next)
  }

  // 给每个 left 一个稳定的配色（按配对建立的顺序）
  const colorOf = (li: number) => {
    const order = Object.keys(pairs)
      .map(Number)
      .sort((a, b) => a - b)
    const idx = order.indexOf(li)
    return idx >= 0 ? PAIR_COLORS[idx % PAIR_COLORS.length] : undefined
  }
  const rightPairedLeft = (ri: number): number | null => {
    for (const [k, v] of Object.entries(pairs)) if (v === ri) return +k
    return null
  }

  return (
    <div className="match-wrap">
      <div className="match-col">
        {q.left.map((t, li) => {
          const c = colorOf(li)
          const classes = ['match-tile']
          if (activeLeft === li) classes.push('active')
          if (li in pairs) classes.push('paired')
          if (submitted) classes.push(pairs[li] === li ? 'correct' : 'wrong')
          return (
            <button
              key={li}
              className={classes.join(' ')}
              style={c ? { borderColor: c } : undefined}
              disabled={submitted}
              onClick={() => clickLeft(li)}
            >
              {c && <span className="dot" style={{ background: c }} />}
              <RichText text={t} />
            </button>
          )
        })}
      </div>
      <div className="match-col">
        {rightOrder.map((ri) => {
          const li = rightPairedLeft(ri)
          const c = li !== null ? colorOf(li) : undefined
          const classes = ['match-tile']
          if (li !== null) classes.push('paired')
          if (submitted && li !== null) classes.push(li === ri ? 'correct' : 'wrong')
          return (
            <button
              key={ri}
              className={classes.join(' ')}
              style={c ? { borderColor: c } : undefined}
              disabled={submitted}
              onClick={() => clickRight(ri)}
            >
              {c && <span className="dot" style={{ background: c }} />}
              <RichText text={q.right[ri]} />
            </button>
          )
        })}
      </div>
    </div>
  )
}
