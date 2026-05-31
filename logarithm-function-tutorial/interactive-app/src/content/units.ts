import type { Unit } from '../types'
import { PART1_UNITS } from './parts/part1'
import { PART2_UNITS } from './parts/part2'
import { PART3_UNITS } from './parts/part3'
import { PART4_UNITS } from './parts/part4'
import { PART5_UNITS } from './parts/part5'
import { PART6_UNITS } from './parts/part6'
// 内容取自《从零到高阶的对数函数教程》全 20 章（part1–part6）。每章 → 一个单元。
export const UNITS: Unit[] = [ ...PART1_UNITS, ...PART2_UNITS, ...PART3_UNITS, ...PART4_UNITS, ...PART5_UNITS, ...PART6_UNITS ]
export const LESSON_ORDER: string[] = UNITS.flatMap((u) => u.lessons.map((l) => l.id))
