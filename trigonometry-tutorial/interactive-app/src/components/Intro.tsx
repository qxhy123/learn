import { useState } from 'react'
import type { IntroCard } from '../types'
import { RichText, BlockMath } from './Math'

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
          学新知 · {title}
        </div>

        <IntroCardView key={i} card={card} color={color} />
      </div>

      <div className="intro-foot">
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
  return (
    <div className="intro-card">
      {card.title && <h3 className="intro-title" style={{ color }}>{card.title}</h3>}
      <p className="intro-body">
        <RichText text={card.body} />
      </p>
      {card.formula && (
        <div className="intro-formula" style={{ borderColor: color }}>
          <BlockMath tex={card.formula} />
        </div>
      )}
      {card.reveal && (
        <div className="intro-reveal">
          <div className="reveal-q">
            🤔 <RichText text={card.reveal.q} />
          </div>
          {shown ? (
            <div className="reveal-a" style={{ borderColor: color }}>
              <RichText text={card.reveal.a} />
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
