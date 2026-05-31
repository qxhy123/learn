import type { Unit } from '../types'
import { PART1_UNITS } from './parts/part1'
import { PART2_UNITS } from './parts/part2'
import { PART3_UNITS } from './parts/part3'
import { PART4_UNITS } from './parts/part4'
import { PART5_UNITS } from './parts/part5'
import { PART6_UNITS } from './parts/part6'
import { PART7_UNITS } from './parts/part7'
import { PART8_UNITS } from './parts/part8'

// 内容取自《从零到高阶的三角函数教程》全 24 章（part1–part8）。
// 每章 → 一个单元（unit），共 24 个单元；题库完全数据驱动。
export const UNITS: Unit[] = [
  ...PART1_UNITS,
  ...PART2_UNITS,
  ...PART3_UNITS,
  ...PART4_UNITS,
  ...PART5_UNITS,
  ...PART6_UNITS,
  ...PART7_UNITS,
  ...PART8_UNITS,
]

/** 课程在全局的顺序（用于解锁判定）。 */
export const LESSON_ORDER: string[] = UNITS.flatMap((u) => u.lessons.map((l) => l.id))
