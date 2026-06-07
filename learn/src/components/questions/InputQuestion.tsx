import { useState } from 'react'
import type { InputQuestion as Q } from '../../types'
import { matchesInput } from '../../lib/check'
import { useCourse } from '../../course'

interface Props {
  q: Q
  submitted: boolean
  onState: (ready: boolean, correct: boolean) => void
}

const DEFAULT_HINT = '输入答案后点击「检查」'

export function InputQuestion({ q, submitted, onState }: Props) {
  const course = useCourse()
  const hint = course.inputHint ?? DEFAULT_HINT
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
      <div className="input-hint">{hint}</div>
    </div>
  )
}
