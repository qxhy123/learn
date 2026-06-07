import { useEffect, useState } from 'react'
import { Home } from './pages/Home'
import { CourseApp } from './CourseApp'

type Route =
  | { name: 'home' }
  | { name: 'course'; id: string; lesson?: string }

// 哈希路由：
//   #/                          首页（课程选择）
//   #/course/<id>               某课程的关卡树
//   #/course/<id>/lesson/<lid>  某课程内的一课
function routeFromHash(): Route {
  const h = location.hash.replace(/^#\/?/, '')
  const m = h.match(/^course\/([^/]+)(?:\/lesson\/([\w-]+))?$/)
  if (m) return { name: 'course', id: decodeURIComponent(m[1]), lesson: m[2] }
  return { name: 'home' }
}

function hashOf(r: Route): string {
  if (r.name === 'home') return '#/'
  const base = `#/course/${encodeURIComponent(r.id)}`
  return r.lesson ? `${base}/lesson/${r.lesson}` : base
}

export default function App() {
  const [route, setRoute] = useState<Route>(routeFromHash)

  useEffect(() => {
    const on = () => setRoute(routeFromHash())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  const go = (r: Route) => {
    const next = hashOf(r)
    if (location.hash !== next) location.hash = next
    setRoute(r)
    window.scrollTo(0, 0)
  }

  if (route.name === 'course') {
    return (
      <CourseApp
        courseId={route.id}
        lessonId={route.lesson}
        onHome={() => go({ name: 'home' })}
        onOpenLesson={(lid) => go({ name: 'course', id: route.id, lesson: lid })}
        onExitLesson={() => go({ name: 'course', id: route.id })}
      />
    )
  }

  return <Home onOpen={(id) => go({ name: 'course', id })} />
}
