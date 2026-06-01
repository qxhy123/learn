import { UNITS, LESSON_ORDER } from '../content/units'
import { useProgress } from '../progress'
import { CONFIG } from '../config'
import { RichText } from './Math'

interface Props {
  onStart: (lessonId: string) => void
}

export function SkillTree({ onStart }: Props) {
  const { progress } = useProgress()

  const isCompleted = (id: string) => Boolean(progress.completed[id])
  const isUnlocked = (id: string) => {
    if (CONFIG.unlockAll) return true
    const idx = LESSON_ORDER.indexOf(id)
    if (idx <= 0) return true
    return isCompleted(LESSON_ORDER[idx - 1])
  }

  const totalLessons = LESSON_ORDER.length
  const doneLessons = LESSON_ORDER.filter(isCompleted).length

  return (
    <div className="tree">
      <div className="tree-head">
        <h1>指数函数闯关</h1>
        <p className="tree-sub">
          像玩多邻国一样学指数函数 · 已完成 {doneLessons}/{totalLessons} 课
        </p>
        <div className="overall-bar">
          <div
            className="overall-fill"
            style={{ width: `${(doneLessons / totalLessons) * 100}%` }}
          />
        </div>
      </div>

      {UNITS.map((u, ui) => (
        <section key={u.id} className="unit" style={{ ['--unit' as string]: u.color }}>
          <div className="unit-banner" style={{ background: u.color }}>
            <span className="unit-icon">{u.icon}</span>
            <div className="unit-meta">
              <div className="unit-no">第 {ui + 1} 单元</div>
              <div className="unit-title">{u.title}</div>
              <div className="unit-blurb">
                <RichText text={u.blurb} />
              </div>
            </div>
          </div>

          <div className="path">
            {u.lessons.map((l, li) => {
              const completed = isCompleted(l.id)
              const unlocked = isUnlocked(l.id)
              const state = completed ? 'done' : unlocked ? 'open' : 'locked'
              const acc = progress.bestAccuracy[l.id]
              return (
                <div key={l.id} className={'node-row ' + (li % 2 ? 'right' : 'left')}>
                  <button
                    className={'node ' + state}
                    style={completed || unlocked ? { background: u.color } : undefined}
                    disabled={!unlocked}
                    onClick={() => unlocked && onStart(l.id)}
                  >
                    <span className="node-icon">
                      {completed ? '★' : unlocked ? li + 1 : '🔒'}
                    </span>
                  </button>
                  <div className="node-label">
                    <div className="node-title">{l.title}</div>
                    {l.subtitle && (
                      <div className="node-subtitle">
                        <RichText text={l.subtitle} />
                      </div>
                    )}
                    {completed && acc != null && (
                      <div className="node-acc">最佳正确率 {acc}%</div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </section>
      ))}

      <footer className="tree-foot">
        来自《从零到高阶的指数函数教程》
      </footer>
    </div>
  )
}
