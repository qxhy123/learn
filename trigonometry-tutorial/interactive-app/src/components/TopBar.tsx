import { useProgress } from '../progress'

interface Props {
  /** 课程进行中传入剩余心数；主页传 null 表示不显示心 */
  hearts?: number | null
  maxHearts?: number
}

export function TopBar({ hearts = null, maxHearts = 5 }: Props) {
  const { progress } = useProgress()
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark">∿</span> Trig&nbsp;Quest
      </div>
      <div className="stats">
        <span className="stat streak" title="连续学习天数">
          🔥 {progress.streak}
        </span>
        <span className="stat xp" title="经验值">
          ⭐ {progress.xp}
        </span>
        {hearts !== null && (
          <span className="stat hearts" title="生命">
            {Array.from({ length: maxHearts }, (_, i) => (
              <span key={i} className={i < hearts ? 'heart full' : 'heart empty'}>
                {i < hearts ? '❤️' : '🤍'}
              </span>
            ))}
          </span>
        )}
      </div>
    </header>
  )
}
