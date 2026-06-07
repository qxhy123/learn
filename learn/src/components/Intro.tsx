import { useState } from 'react'
import type { IntroCard } from '../types'
import { RichText, BlockMath, Markdown } from './Math'
import { useCourse } from '../course'

interface Props {
  title: string
  color: string
  cards: IntroCard[]
  onStart: () => void
  onQuit: () => void
}

/** 学新知：练习前的概念讲解，逐张翻看，最后一张点“开始练习”。 */
export function Intro({ title, color, cards, onStart, onQuit }: Props) {
  const [i, setI] = useState(0)
  const card = cards[i]
  const last = i === cards.length - 1

  return (
    <div className="intro">
      <div className="intro-bar">
        <button className="quit" onClick={onQuit} aria-label="退出">
          ✕
        </button>
        <div className="intro-dots">
          {cards.map((_, k) => (
            <span key={k} className={'dot2 ' + (k <= i ? 'on' : '')} style={k <= i ? { background: color } : undefined} />
          ))}
        </div>
        <button className="intro-skip" onClick={onStart}>
          跳过
        </button>
      </div>

      <div className="intro-main">
        <div className="intro-badge" style={{ background: color }}>
          学新知 · <RichText text={title} />
        </div>

        <IntroCardView key={i} card={card} color={color} />
      </div>

      <div className="intro-foot">
        <button
          className="btn big ghost intro-back"
          disabled={i === 0}
          onClick={() => setI((k) => Math.max(0, k - 1))}
        >
          ← 上一步
        </button>
        <button
          className={'btn big ' + (last ? 'primary' : 'blue')}
          onClick={() => (last ? onStart() : setI(i + 1))}
        >
          {last ? '开始练习 →' : '继续'}
        </button>
      </div>
    </div>
  )
}

function IntroCardView({ card, color }: { card: IntroCard; color: string }) {
  const [shown, setShown] = useState(false)
  const course = useCourse()
  const figBase = import.meta.env.BASE_URL + course.figureBase
  return (
    <div className="intro-card">
      {card.title && (
        <h3 className="intro-title" style={{ color }}>
          <RichText text={card.title} />
        </h3>
      )}

      <div className="intro-body">
        <Markdown text={card.body} />
      </div>

      {card.image && (
        <figure className="intro-figure">
          <img src={figBase + card.image} alt={card.imageCaption ?? ''} loading="lazy" />
          {card.imageCaption && (
            <figcaption>
              <RichText text={card.imageCaption} />
            </figcaption>
          )}
        </figure>
      )}

      {card.formula && (
        <div className="intro-formula" style={{ borderColor: color }}>
          <BlockMath tex={card.formula} />
        </div>
      )}

      {card.steps && card.steps.length > 0 && (
        <ol className="intro-steps">
          {card.steps.map((s, i) => (
            <li key={i}>
              <span className="step-no" style={{ background: color }}>{i + 1}</span>
              <span className="step-body">
                <RichText text={s} />
              </span>
            </li>
          ))}
        </ol>
      )}

      {card.tip && (
        <div className="intro-tip">
          <span className="tip-icon">💡</span>
          <span>
            <Markdown text={card.tip} />
          </span>
        </div>
      )}

      {card.reveal && (
        <div className="intro-reveal">
          <div className="reveal-q">
            🤔 <RichText text={card.reveal.q} />
          </div>
          {shown ? (
            <div className="reveal-a" style={{ borderColor: color }}>
              <Markdown text={card.reveal.a} />
            </div>
          ) : (
            <button className="reveal-btn" onClick={() => setShown(true)}>
              点击揭示答案
            </button>
          )}
        </div>
      )}
    </div>
  )
}
