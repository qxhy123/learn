import { createContext, useContext } from 'react'
import type { ReactNode } from 'react'
import type { Course } from './types'

const CourseContext = createContext<Course | null>(null)

export function CourseProvider({ course, children }: { course: Course; children: ReactNode }) {
  return <CourseContext.Provider value={course}>{children}</CourseContext.Provider>
}

export function useCourse(): Course {
  const c = useContext(CourseContext)
  if (!c) throw new Error('useCourse 必须在 CourseProvider 内使用')
  return c
}

/** 课程内所有课的全局顺序（用于顺序解锁判定）。 */
export function lessonOrder(course: Course): string[] {
  return course.units.flatMap((u) => u.lessons.map((l) => l.id))
}
