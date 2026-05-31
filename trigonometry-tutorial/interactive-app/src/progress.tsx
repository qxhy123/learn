import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import type { Progress } from './types'
import { dayRelation, today } from './lib/check'

const KEY = 'trig-quest-progress-v1'

const empty: Progress = {
  xp: 0,
  streak: 0,
  lastActiveDay: null,
  completed: {},
  bestAccuracy: {},
}

function load(): Progress {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return empty
    return { ...empty, ...(JSON.parse(raw) as Progress) }
  } catch {
    return empty
  }
}

interface Ctx {
  progress: Progress
  /** 完成一课：累加 XP、更新 streak、记录完成与最佳正确率 */
  finishLesson: (lessonId: string, gainedXp: number, accuracy: number) => void
  reset: () => void
}

const ProgressContext = createContext<Ctx | null>(null)

export function ProgressProvider({ children }: { children: ReactNode }) {
  const [progress, setProgress] = useState<Progress>(load)

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(progress))
    } catch {
      /* ignore quota errors */
    }
  }, [progress])

  const finishLesson = useCallback(
    (lessonId: string, gainedXp: number, accuracy: number) => {
      setProgress((p) => {
        // streak 更新
        const rel = dayRelation(p.lastActiveDay)
        let streak = p.streak
        if (rel === 'gap') streak = 1
        else if (rel === 'consecutive') streak = p.streak + 1
        // 'same' → 不变
        if (streak === 0) streak = 1
        return {
          ...p,
          xp: p.xp + gainedXp,
          streak,
          lastActiveDay: today(),
          completed: { ...p.completed, [lessonId]: true },
          bestAccuracy: {
            ...p.bestAccuracy,
            [lessonId]: Math.max(p.bestAccuracy[lessonId] ?? 0, accuracy),
          },
        }
      })
    },
    []
  )

  const reset = useCallback(() => setProgress(empty), [])

  return (
    <ProgressContext.Provider value={{ progress, finishLesson, reset }}>
      {children}
    </ProgressContext.Provider>
  )
}

export function useProgress(): Ctx {
  const ctx = useContext(ProgressContext)
  if (!ctx) throw new Error('useProgress 必须在 ProgressProvider 内使用')
  return ctx
}
