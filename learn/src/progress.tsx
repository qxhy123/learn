import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import type { Progress } from './types'
import { dayRelation, today } from './lib/check'

/** 每门课独立存档：键名按 courseId 区分，互不干扰。 */
const keyOf = (courseId: string) => `mq-${courseId}-v1`

const empty: Progress = {
  xp: 0,
  streak: 0,
  lastActiveDay: null,
  completed: {},
  bestAccuracy: {},
}

/** 不挂载 Provider 也能读某门课的存档（首页用来显示进度）。 */
export function readProgress(courseId: string): Progress {
  try {
    const raw = localStorage.getItem(keyOf(courseId))
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

export function ProgressProvider({ courseId, children }: { courseId: string; children: ReactNode }) {
  const [progress, setProgress] = useState<Progress>(() => readProgress(courseId))

  useEffect(() => {
    try {
      localStorage.setItem(keyOf(courseId), JSON.stringify(progress))
    } catch {
      /* ignore quota errors */
    }
  }, [courseId, progress])

  const finishLesson = useCallback(
    (lessonId: string, gainedXp: number, accuracy: number) => {
      setProgress((p) => {
        const rel = dayRelation(p.lastActiveDay)
        let streak = p.streak
        if (rel === 'gap') streak = 1
        else if (rel === 'consecutive') streak = p.streak + 1
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
