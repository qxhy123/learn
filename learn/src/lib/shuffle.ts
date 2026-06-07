/** Fisher–Yates 洗牌，返回新数组（不改原数组）。每次调用结果随机。 */
export function shuffle<T>(arr: readonly T[]): T[] {
  const a = arr.slice()
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

/** 返回 0..n-1 的一个随机排列。 */
export function shuffledIndices(n: number): number[] {
  return shuffle(Array.from({ length: n }, (_, i) => i))
}
