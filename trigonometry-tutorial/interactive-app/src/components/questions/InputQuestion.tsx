import { useState } from 'react'
import type { InputQuestion as Q } from '../../types'
import { matchesInput } from '../../lib/check'

interface Props {
  q: Q
  submitted: boolean
  onState: (ready: boolean, correct: boolean) => void
}

export function InputQuestion({ q, submitted, onState }: Props) {
  const [val, setVal] = useState('')
  const correct = matchesInput(val, q.accept)
  const cls = ['text-input']
  if (submitted) cls.push(correct ? 'correct' : 'wrong')
  return (
    <div className="input-wrap">
      <input
        className={cls.join(' ')}
        value={val}
        disabled={submitted}
        placeholder={q.placeholder ?? '在此输入答案'}
        autoFocus
        onChange={(e) => {
          const v = e.target.value
          setVal(v)
          onState(v.trim() !== '', matchesInput(v, q.accept))
        }}
      />
      <div className="input-hint">支持 √3/2、sqrt3/2、0.866、π/6 等写法</div>
    </div>
  )
}
