import { useState } from 'react'
import type { JudgeQuestion as Q } from '../../types'

interface Props {
  q: Q
  submitted: boolean
  onState: (ready: boolean, correct: boolean) => void
}

export function JudgeQuestion({ q, submitted, onState }: Props) {
  const [sel, setSel] = useState<boolean | null>(null)
  const render = (value: boolean, label: string, emoji: string) => {
    const classes = ['judge-btn']
    if (sel === value) classes.push('selected')
    if (submitted) {
      if (value === q.answer) classes.push('correct')
      else if (sel === value) classes.push('wrong')
    }
    return (
      <button
        className={classes.join(' ')}
        disabled={submitted}
        onClick={() => {
          setSel(value)
          onState(true, value === q.answer)
        }}
      >
        <span className="judge-emoji">{emoji}</span>
        {label}
      </button>
    )
  }
  return (
    <div className="judge-row">
      {render(true, '正确', '✓')}
      {render(false, '错误', '✗')}
    </div>
  )
}
