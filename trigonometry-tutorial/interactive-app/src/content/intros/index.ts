import type { IntroCard } from '../../types'
import { INTRO1 } from './intro1'
import { INTRO2 } from './intro2'
import { INTRO3 } from './intro3'
import { INTRO4 } from './intro4'
import { INTRO5 } from './intro5'
import { INTRO6 } from './intro6'
import { INTRO7 } from './intro7'
import { INTRO8 } from './intro8'

// 汇聚全 24 章各课的“学新知”讲解卡片，按 lesson id 索引。
export const INTROS: Record<string, IntroCard[]> = {
  ...INTRO1,
  ...INTRO2,
  ...INTRO3,
  ...INTRO4,
  ...INTRO5,
  ...INTRO6,
  ...INTRO7,
  ...INTRO8,
}
