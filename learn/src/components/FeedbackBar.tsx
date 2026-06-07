import { RichText, Markdown } from './Math'

interface Props {
  correct: boolean
  explain?: string
  /** 答错时给出的正确答案文本（可选，含 $...$） */
  correctAnswer?: string
}

export function FeedbackBar({ correct, explain, correctAnswer }: Props) {
  return (
    <div className={'feedback ' + (correct ? 'ok' : 'bad')}>
      <div className="feedback-title">
        <span className="feedback-icon">{correct ? '✓' : '✗'}</span>
        {correct ? '回答正确！' : '再想想～'}
      </div>
      {!correct && correctAnswer && (
        <div className="feedback-answer">
          正确答案：<RichText text={correctAnswer} />
        </div>
      )}
      {explain && (
        <div className="feedback-explain">
          <Markdown text={explain} />
        </div>
      )}
    </div>
  )
}
