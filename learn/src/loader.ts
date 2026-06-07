import type { Course, CourseIndex, Unit } from './types'

const BASE = import.meta.env.BASE_URL

async function fetchJson<T>(path: string, what: string): Promise<T> {
  const res = await fetch(BASE + path, { cache: 'no-cache' })
  if (!res.ok) throw new Error(`无法加载${what}（HTTP ${res.status}）`)
  return res.json() as Promise<T>
}

/** 读取课程注册表 public/courses/index.json。 */
export async function loadIndex(): Promise<CourseIndex> {
  return fetchJson<CourseIndex>('courses/index.json', '课程列表')
}

/** 清单里的单元引用：单元元信息 + 指向 units/<id>.json 的相对路径。 */
interface UnitRef {
  id: string
  title: string
  color: string
  icon: string
  blurb: string
  file: string
}
/** course.json 现在是「清单」：课程级字段 + units 引用（内容按单元拆到 units/*.json）。 */
interface Manifest extends Omit<Course, 'units'> {
  units: UnitRef[]
}

/**
 * 约定每门课的清单在 public/courses/<id>/course.json，
 * 各单元内容拆在 public/courses/<id>/units/<uid>.json（{ id, lessons }）。
 * 先读清单，再并行拉各单元文件，组装成完整 Course。
 */
export async function loadCourse(courseId: string): Promise<Course> {
  const man = await fetchJson<Manifest>(`courses/${courseId}/course.json`, `课程「${courseId}」`)
  const units: Unit[] = await Promise.all(
    man.units.map(async (ref) => {
      const uf = await fetchJson<{ lessons: Unit['lessons'] }>(ref.file, `单元「${ref.id}」`)
      return { id: ref.id, title: ref.title, color: ref.color, icon: ref.icon, blurb: ref.blurb, lessons: uf.lessons }
    })
  )
  const { units: _refs, ...meta } = man
  return { ...meta, units }
}
