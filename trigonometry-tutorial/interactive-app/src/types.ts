// ===== 内容模型：单元 → 课程 → 题目 =====

export type QuestionType = 'choice' | 'input' | 'judge' | 'match'

interface BaseQuestion {
  id: string
  type: QuestionType
  /** 题干（支持用 $...$ 写 KaTeX 公式） */
  prompt: string
  /** 答对后展示的讲解（支持 $...$） */
  explain?: string
}

/** 选择题：单选 */
export interface ChoiceQuestion extends BaseQuestion {
  type: 'choice'
  options: string[]
  /** 正确选项下标 */
  answer: number
}

/** 输入题：键入答案，容错匹配 */
export interface InputQuestion extends BaseQuestion {
  type: 'input'
  /** 可接受的答案（任一匹配即对），会做标准化容错比较 */
  accept: string[]
  placeholder?: string
}

/** 判断题：对 / 错 */
export interface JudgeQuestion extends BaseQuestion {
  type: 'judge'
  /** 命题是否为真 */
  answer: boolean
}

/** 配对题：左右两列一一对应 */
export interface MatchQuestion extends BaseQuestion {
  type: 'match'
  /** 左列（如角） */
  left: string[]
  /** 右列（如函数值），right[i] 与 left[i] 配对 */
  right: string[]
}

export type Question =
  | ChoiceQuestion
  | InputQuestion
  | JudgeQuestion
  | MatchQuestion

/** 学新知环节的一张讲解卡片（练习前的概念引入）。 */
export interface IntroCard {
  /** 卡片小标题，如“概念”“关键公式”“看个例子”“易错点” */
  title?: string
  /** 讲解正文，可含行内公式 $...$ */
  body: string
  /** 需要重点突出的公式（居中大号显示，写纯 LaTeX，不含 $） */
  formula?: string
  /** 互动：先抛出一个小问题，点击“看答案”再揭示，引导思考 */
  reveal?: { q: string; a: string }
}

export interface Lesson {
  id: string
  title: string
  /** 一句话副标题 */
  subtitle?: string
  /** 学新知：练习前的讲解卡片（按顺序翻看） */
  intro?: IntroCard[]
  questions: Question[]
}

export interface Unit {
  id: string
  title: string
  /** 单元主题色（多邻国式每单元一色） */
  color: string
  icon: string
  blurb: string
  lessons: Lesson[]
}

// ===== 进度 / 存档 =====

export interface Progress {
  /** 累计经验值 */
  xp: number
  /** 连续学习天数 */
  streak: number
  /** 最近一次学习的自然日（YYYY-MM-DD），用于 streak 判定 */
  lastActiveDay: string | null
  /** 已完成课程 id 集合 */
  completed: Record<string, true>
  /** 每课最佳正确率（0-100） */
  bestAccuracy: Record<string, number>
}
