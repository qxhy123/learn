import { useEffect, useState } from 'react'
import type { Course } from './types'
import { loadCourse } from './loader'
import { CourseProvider } from './course'
import { ProgressProvider } from './progress'
import { TopBar } from './components/TopBar'
import { SkillTree } from './components/SkillTree'
import { Lesson } from './components/Lesson'

interface Props {
  courseId: string
  lessonId?: string
  onHome: () => void
  onOpenLesson: (lessonId: string) => void
  onExitLesson: () => void
}

// 已加载课程的简单内存缓存，切换课程/课时不重复 fetch。
const cache = new Map<string, Course>()

export function CourseApp({ courseId, lessonId, onHome, onOpenLesson, onExitLesson }: Props) {
  const [course, setCourse] = useState<Course | null>(() => cache.get(courseId) ?? null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (cache.has(courseId)) {
      setCourse(cache.get(courseId)!)
      return
    }
    let alive = true
    setCourse(null)
    setError(null)
    loadCourse(courseId)
      .then((c) => {
        cache.set(courseId, c)
        if (alive) setCourse(c)
      })
      .catch((e) => alive && setError(String(e.message ?? e)))
    return () => {
      alive = false
    }
  }, [courseId])

  if (error) {
    return (
      <div className="course-status">
        <p className="status-emoji">😕</p>
        <p>{error}</p>
        <button className="btn primary" onClick={onHome}>
          返回首页
        </button>
      </div>
    )
  }

  if (!course) {
    return (
      <div className="course-status">
        <p className="status-emoji">📚</p>
        <p>正在加载课程…</p>
      </div>
    )
  }

  return (
    <CourseProvider course={course}>
      <ProgressProvider courseId={courseId}>
        {lessonId ? (
          <Lesson lessonId={lessonId} onExit={onExitLesson} />
        ) : (
          <div className="home">
            <TopBar hearts={null} />
            <SkillTree onStart={onOpenLesson} onHome={onHome} />
          </div>
        )}
      </ProgressProvider>
    </CourseProvider>
  )
}
