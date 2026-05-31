import { useEffect, useState } from 'react'
import { ProgressProvider } from './progress'
import { TopBar } from './components/TopBar'
import { SkillTree } from './components/SkillTree'
import { Lesson } from './components/Lesson'

type Route = { name: 'home' } | { name: 'lesson'; id: string }

function routeFromHash(): Route {
  const m = location.hash.match(/lesson=([\w-]+)/)
  return m ? { name: 'lesson', id: m[1] } : { name: 'home' }
}

function Shell() {
  const [route, setRoute] = useState<Route>(routeFromHash)

  // 支持刷新 / 分享深链接：#lesson=u1-l1
  useEffect(() => {
    const on = () => setRoute(routeFromHash())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  const go = (r: Route) => {
    location.hash = r.name === 'lesson' ? `lesson=${r.id}` : ''
    setRoute(r)
  }

  if (route.name === 'lesson') {
    return <Lesson lessonId={route.id} onExit={() => go({ name: 'home' })} />
  }
  return (
    <div className="home">
      <TopBar hearts={null} />
      <SkillTree onStart={(id) => go({ name: 'lesson', id })} />
    </div>
  )
}

export default function App() {
  return (
    <ProgressProvider>
      <Shell />
    </ProgressProvider>
  )
}
