import { useEffect, useState } from 'react'
import type { CourseIndex, CourseMeta } from '../types'
import { loadIndex } from '../loader'
import { readProgress } from '../progress'
import { RichText } from '../components/Math'

interface Props {
  onOpen: (courseId: string) => void
}

export function Home({ onOpen }: Props) {
  const [index, setIndex] = useState<CourseIndex | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    loadIndex()
      .then((idx) => alive && setIndex(idx))
      .catch((e) => alive && setError(String(e.message ?? e)))
    return () => {
      alive = false
    }
  }, [])

  return (
    <div className="landing">
      <header className="landing-head">
        <h1 className="landing-title">
          <span className="landing-logo">∑</span> 数学闯关
        </h1>
        <p className="landing-sub">一关一关把知识学透 · 选一门开始</p>
      </header>

      {error && (
        <div className="course-status">
          <p className="status-emoji">😕</p>
          <p>{error}</p>
        </div>
      )}

      {!index && !error && (
        <div className="course-status">
          <p className="status-emoji">📚</p>
          <p>正在加载课程列表…</p>
        </div>
      )}

      {index && (
        <div className="course-grid">
          {index.courses.map((c) => (
            <CourseCard key={c.id} meta={c} onOpen={() => onOpen(c.id)} />
          ))}
        </div>
      )}

      <footer className="landing-foot">数学闯关 · Math Quest</footer>
    </div>
  )
}

function CourseCard({ meta, onOpen }: { meta: CourseMeta; onOpen: () => void }) {
  const prog = readProgress(meta.id)
  const done = Object.keys(prog.completed).length
  const pct = meta.lessonCount ? Math.round((done / meta.lessonCount) * 100) : 0
  return (
    <button className="course-card" style={{ ['--c' as string]: meta.color }} onClick={onOpen}>
      <div className="course-card-top">
        <span className="course-icon" style={{ background: meta.color }}>
          {meta.icon}
        </span>
        <div className="course-mark">{meta.brandMark}</div>
      </div>
      <div className="course-name">{meta.title}</div>
      <div className="course-blurb">
        <RichText text={meta.blurb} />
      </div>
      <div className="course-meta">
        {meta.unitCount} 单元 · {meta.lessonCount} 课
      </div>
      <div className="course-bar">
        <div className="course-bar-fill" style={{ width: `${pct}%`, background: meta.color }} />
      </div>
      <div className="course-progress-text">
        {done > 0 ? `已完成 ${done}/${meta.lessonCount} 课（${pct}%）` : '开始学习 →'}
      </div>
    </button>
  )
}
