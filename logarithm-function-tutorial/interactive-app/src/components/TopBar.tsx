import { useProgress } from '../progress'

interface Props {
  /** è¯¾ç¨è¿è¡ä¸­ä¼ å¥å©ä½å¿æ°ï¼ä¸»é¡µä¼  null è¡¨ç¤ºä¸æ¾ç¤ºå¿ */
  hearts?: number | null
  maxHearts?: number
}

export function TopBar({ hearts = null, maxHearts = 5 }: Props) {
  const { progress } = useProgress()
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark">㏒</span> Log&nbsp;Quest
      </div>
      <div className="stats">
        <span className="stat streak" title="è¿ç»­å­¦ä¹ å¤©æ°">
          ð¥ {progress.streak}
        </span>
        <span className="stat xp" title="ç»éªå¼">
          â­ {progress.xp}
        </span>
        {hearts !== null && (
          <span className="stat hearts" title="çå½">
            {Array.from({ length: maxHearts }, (_, i) => (
              <span key={i} className={i < hearts ? 'heart full' : 'heart empty'}>
                {i < hearts ? 'â¤ï¸' : 'ð¤'}
              </span>
            ))}
          </span>
        )}
      </div>
    </header>
  )
}
