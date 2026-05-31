import { useState } from 'react'
import type { ChoiceQuestion as Q } from '../../types'
import { RichText } from '../Math'

interface Props {
  q: Q
  submitted: boolean
  onState: (ready: boolean, correct: boolean) => void
}

export function ChoiceQuestion({ q, submitted, onState }: Props) {
  const [sel, setSel] = useState<number | null>(null)
  return (
    <div className="choice-grid">
      {q.options.map((opt, i) => {
        const classes = ['choice']
        if (sel === i) classes.push('selected')
        if (submitted) {
          if (i === q.answer) classes.push('correct')
          else if (sel === i) classes.push('wrong')
        }
        return (
          <button
            key={i}
            className={classes.join(' ')}
            disabled={submitted}
            onClick={() => {
              setSel(i)
              onState(true, i === q.answer)
            }}
          >
            <span className="choice-key">{String.fromCharCode(65 + i)}</span>
            <span className="choice-body">
              <RichText text={opt} />
            </span>
          </button>
        )
      })}
    </div>
  )
}
