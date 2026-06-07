import { useMemo, useState } from 'react'
import { useProgress } from '../progress'
import { useCourse } from '../course'
import type { Question } from '../types'
import { shuffle } from '../lib/shuffle'
import { RichText } from './Math'
import { Intro } from './Intro'
import { TopBar } from './TopBar'
import { FeedbackBar } from './FeedbackBar'
import { LessonComplete } from './LessonComplete'
import { ChoiceQuestion } from './questions/ChoiceQuestion'
import { InputQuestion } from './questions/InputQuestion'
import { JudgeQuestion } from './questions/JudgeQuestion'
import { MatchQuestion } from './questions/MatchQuestion'

const MAX_HEARTS = 5
const BASE_XP = 10

interface Props {
  lessonId: string
  onExit: () => void
}

export function Lesson({ lessonId, onExit }: Props) {
  const { finishLesson } = useProgress()
  const course = useCourse()
  const unit = useMemo(
    () => course.units.find((u) => u.lessons.some((l) => l.id === lessonId))!,
    [course, lessonId]
  )
  const lesson = useMemo(() => unit.lessons.find((l) => l.id === lessonId)!, [unit, lessonId])
  // 每次进入本课时动态打乱：题目顺序 + 选择题选项顺序（配对题右列在 MatchQuestion 内打乱）
  const questions = useMemo(() => prepareQuestions(lesson.questions), [lesson])
  const total = questions.length
  const introCards = lesson.intro

  // 阶段：先“学新知”（若有讲解卡片），再进入练习
  const [stage, setStage] = useState<'intro' | 'quiz'>(
    introCards && introCards.length ? 'intro' : 'quiz'
  )

  const [index, setIndex] = useState(0)
  const [hearts, setHearts] = useState(MAX_HEARTS)
  const [phase, setPhase] = useState<'answering' | 'feedback'>('answering')
  const [answer, setAnswer] = useState({ ready: false, correct: false })
  const [xp, setXp] = useState(0)
  const [combo, setCombo] = useState(0)
  const [maxCombo, setMaxCombo] = useState(0)
  const [correctCount, setCorrectCount] = useState(0)
  const [done, setDone] = useState<null | { passed: boolean }>(null)

  const q = questions[index]

  if (stage === 'intro' && introCards && introCards.length) {
    return (
      <Intro
        title={lesson.title}
        color={unit.color}
        cards={introCards}
        onStart={() => setStage('quiz')}
        onQuit={onExit}
      />
    )
  }

  if (done) {
    const accuracy = Math.round((correctCount / total) * 100)
    return (
      <LessonComplete
        xp={xp}
        accuracy={accuracy}
        maxCombo={maxCombo}
        passed={done.passed}
        onDone={onExit}
        onRetry={() => window.location.reload()}
      />
    )
  }

  function check() {
    const correct = answer.correct
    if (correct) {
      const nc = combo + 1
      const gained = BASE_XP + Math.min(nc - 1, 5) * 2
      setXp((x) => x + gained)
      setCombo(nc)
      setMaxCombo((m) => Math.max(m, nc))
      setCorrectCount((c) => c + 1)
    } else {
      setCombo(0)
      setHearts((h) => h - 1)
    }
    setPhase('feedback')
  }

  function next() {
    const lastCorrect = answer.correct
    const heartsLeft = hearts // already decremented in check() for wrong
    if (!lastCorrect && heartsLeft <= 0) {
      setDone({ passed: false })
      return
    }
    if (index + 1 >= total) {
      const accuracy = Math.round(((correctCount) / total) * 100)
      finishLesson(lessonId, xp, accuracy)
      setDone({ passed: true })
      return
    }
    setIndex((i) => i + 1)
    setAnswer({ ready: false, correct: false })
    setPhase('answering')
  }

  const correctAnswerText = describeAnswer(q)
  const submitted = phase === 'feedback'

  return (
    <div className="lesson">
      <TopBar hearts={hearts} maxHearts={MAX_HEARTS} />
      <div className="lesson-bar-row">
        <button className="quit" onClick={onExit} aria-label="退出">
          ✕
        </button>
        <div className="progress">
          <div className="progress-fill" style={{ width: `${(index / total) * 100}%` }} />
        </div>
        <div className="combo-badge" data-on={combo >= 2}>
          {combo >= 2 ? `连击 ${combo}× 🔥` : ''}
        </div>
      </div>

      <main className="lesson-main">
        <div className="qtype-tag">{typeLabel(q.type)}</div>
        <h2 className="prompt">
          <RichText text={q.prompt} />
        </h2>

        <div key={q.id} className="qbody">
          {q.type === 'choice' && (
            <ChoiceQuestion q={q} submitted={submitted} onState={(r, c) => setAnswer({ ready: r, correct: c })} />
          )}
          {q.type === 'input' && (
            <InputQuestion q={q} submitted={submitted} onState={(r, c) => setAnswer({ ready: r, correct: c })} />
          )}
          {q.type === 'judge' && (
            <JudgeQuestion q={q} submitted={submitted} onState={(r, c) => setAnswer({ ready: r, correct: c })} />
          )}
          {q.type === 'match' && (
            <MatchQuestion q={q} submitted={submitted} onState={(r, c) => setAnswer({ ready: r, correct: c })} />
          )}
        </div>
      </main>

      <div className={'lesson-foot ' + (submitted ? (answer.correct ? 'ok' : 'bad') : '')}>
        {submitted && (
          <FeedbackBar
            correct={answer.correct}
            explain={q.explain}
            correctAnswer={answer.correct ? undefined : correctAnswerText}
          />
        )}
        <div className="foot-action">
          {!submitted ? (
            <button className="btn primary big" disabled={!answer.ready} onClick={check}>
              检查
            </button>
          ) : (
            <button
              className={'btn big ' + (answer.correct ? 'primary' : 'danger')}
              onClick={next}
            >
              继续
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

/** 进入课程时生成一份动态副本：打乱题目顺序，并打乱每道选择题的选项（重映射正确下标）。 */
function prepareQuestions(qs: Question[]): Question[] {
  return shuffle(qs).map((q) => {
    if (q.type === 'choice') {
      const order = shuffle(q.options.map((_, i) => i))
      return {
        ...q,
        options: order.map((i) => q.options[i]),
        answer: order.indexOf(q.answer),
      }
    }
    if (q.type === 'match') {
      // 返回浅拷贝，使 MatchQuestion 的 useMemo 在每次进入时重新打乱右列
      return { ...q }
    }
    return q
  })
}

function typeLabel(t: Question['type']): string {
  return { choice: '选择题', input: '填空题', judge: '判断题', match: '配对题' }[t]
}

function describeAnswer(q: Question): string {
  switch (q.type) {
    case 'choice':
      return q.options[q.answer]
    case 'input':
      return q.accept[0]
    case 'judge':
      return q.answer ? '正确' : '错误'
    case 'match':
      return q.left.map((l, i) => `${l} ↔ ${q.right[i]}`).join('，')
  }
}
