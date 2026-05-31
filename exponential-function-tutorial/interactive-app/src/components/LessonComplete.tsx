import { useEffect } from 'react'
import confetti from 'canvas-confetti'

interface Props {
  xp: number
  accuracy: number
  maxCombo: number
  passed: boolean
  onDone: () => void
  onRetry: () => void
}

export function LessonComplete({ xp, accuracy, maxCombo, passed, onDone, onRetry }: Props) {
  useEffect(() => {
    if (!passed) return
    const end = Date.now() + 800
    const tick = () => {
      confetti({ particleCount: 4, angle: 60, spread: 70, origin: { x: 0 } })
      confetti({ particleCount: 4, angle: 120, spread: 70, origin: { x: 1 } })
      if (Date.now() < end) requestAnimationFrame(tick)
    }
    tick()
  }, [passed])

  if (!passed) {
    return (
      <div className="complete fail">
        <div className="complete-emoji">💔</div>
        <h2>生命用完啦</h2>
        <p className="complete-sub">别灰心，再来一次就更熟练了！</p>
        <button className="btn primary big" onClick={onRetry}>
          重试本课
        </button>
        <button className="btn ghost" onClick={onDone}>
          返回主页
        </button>
      </div>
    )
  }

  return (
    <div className="complete">
      <div className="complete-emoji">🎉</div>
      <h2>课程完成！</h2>
      <div className="complete-cards">
        <div className="ccard xp">
          <div className="ccard-label">获得经验</div>
          <div className="ccard-value">+{xp} ⭐</div>
        </div>
        <div className="ccard acc">
          <div className="ccard-label">正确率</div>
          <div className="ccard-value">{accuracy}%</div>
        </div>
        <div className="ccard combo">
          <div className="ccard-label">最高连击</div>
          <div className="ccard-value">{maxCombo}× 🔥</div>
        </div>
      </div>
      <button className="btn primary big" onClick={onDone}>
        继续
      </button>
    </div>
  )
}
