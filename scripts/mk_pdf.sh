#!/usr/bin/env bash
set -e
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"
TUT_DIR="$1"; OUT_PDF="$2"; TITLE="${3:-$(basename "$TUT_DIR")}"
[[ ! -d "$TUT_DIR" ]] && { echo "❌ $TUT_DIR not found" >&2; exit 1; }
WORK=$(mktemp -d); trap "rm -rf $WORK" EXIT
cd "$TUT_DIR"; TUT_ABS=$(pwd)

FILES=()
[[ -f README.md ]] && FILES+=("README.md")
[[ -f 00-preface.md ]] && FILES+=("00-preface.md")
for d in $(ls -d part*/ 2>/dev/null | sort); do
  for f in $(ls "$d"*.md 2>/dev/null | sort); do
    base=$(basename "$f" .md)
    [[ "$base" == *-fusion || "$base" == *-rewrite ]] && continue
    FILES+=("$f")
  done
done
for d in thinking-toolkit/ docs/categories/; do
  [[ -d "$d" ]] || continue
  for f in $(ls "$d"*.md 2>/dev/null | sort); do FILES+=("$f"); done
done
[[ -d "appendix" ]] && for f in $(ls appendix/*.md 2>/dev/null | sort); do FILES+=("$f"); done
echo "→ ${#FILES[@]} md 文件"

# SVG → PDF
SVG_LIST=$(grep -rohE '!\[[^]]*\]\([^)]+\.svg\)' "${FILES[@]}" 2>/dev/null | sed -E 's/.*\(([^)]+\.svg)\).*/\1/' | sort -u)
SVG_CNT=0
for rel in $SVG_LIST; do
  actual="$rel"
  while [[ "$actual" == ../* ]]; do actual="${actual#../}"; done
  if [[ -f "$TUT_ABS/$actual" ]]; then
    pdf_target="$TUT_ABS/${actual%.svg}.pdf"
    if [[ ! -f "$pdf_target" || "$TUT_ABS/$actual" -nt "$pdf_target" ]]; then
      rsvg-convert -f pdf "$TUT_ABS/$actual" -o "$pdf_target" 2>/dev/null && SVG_CNT=$((SVG_CNT+1))
    fi
  fi
done
echo "→ SVG 新转 $SVG_CNT"

# 合并 + 替换 SVG 引用 + 修公式里的 |
COMBINED="$WORK/combined.md"
> "$COMBINED"
for f in "${FILES[@]}"; do
  echo "" >> "$COMBINED"
  cat "$f" >> "$COMBINED"
  echo "" >> "$COMBINED"
done

# Python 后处理：替换 .svg 为 PDF + 修公式里的 | → \vert
TUT_ABS="$TUT_ABS" python3 - "$COMBINED" <<'PY'
import re, os, sys
TUT = os.environ['TUT_ABS']
path = sys.argv[1]
text = open(path).read()

# 1) 替换 SVG 引用为绝对 PDF 路径
def fix_svg(m):
    alt, p = m.group(1), m.group(2)
    rel = p
    while rel.startswith('../'): rel = rel[3:]
    while rel.startswith('./'): rel = rel[2:]
    pdf_abs = os.path.join(TUT, rel[:-4] + '.pdf')
    return f'![{alt}]({pdf_abs})' if os.path.exists(pdf_abs) else f'_[图缺失: {os.path.basename(p)}]_'
text = re.sub(r'!\[([^\]]*)\]\(([^)]+\.svg)\)', fix_svg, text)

# 2) 修行内公式 $...$ 中的裸 | → \vert（保留 \| 范数）
def fix_inline_math(m):
    body = m.group(1)
    # 先把 \begin{array}{...} 列对齐参数 mask 起来（保留原样不替换）
    placeholders = []
    def mask(mm):
        placeholders.append(mm.group(0))
        return f'\x00ARR{len(placeholders)-1}\x00'
    body_masked = re.sub(r'\\begin\{array\}\{[^}]+\}', mask, body)
    # 逐字符替换 | → \vert，保留 \|
    out = []
    i = 0
    while i < len(body_masked):
        if body_masked[i] == '|':
            if i > 0 and body_masked[i-1] == '\\':
                out.append('|')
            else:
                out.append('\\vert ')
            i += 1
        else:
            out.append(body_masked[i])
            i += 1
    result = ''.join(out).rstrip(' ')
    # 还原 array 列对齐
    for k, ph in enumerate(placeholders):
        result = result.replace(f'\x00ARR{k}\x00', ph)
    return '$' + result + '$'

# 注意：只替换单 $...$ 不替换 $$...$$
# pattern 匹配 $X$（X 不含 $ 和换行）
text = re.sub(r'(?<!\$)\$([^$\n]+?)\$(?!\$)', fix_inline_math, text)

open(path, 'w').write(text)
print(f'后处理完成: {path}')
PY

REMAINING=$(grep -c '\.svg)' "$COMBINED" || echo 0)
echo "→ 剩余 SVG 引用: $REMAINING"

# 让 ★ ☆ 等符号走 CJK 字体（Songti SC 含这些字形，否则会被路由到缺字形的 Times 而丢失）
cat > "$WORK/symfix.tex" <<'TEX'
\xeCJKDeclareCharClass{CJK}{"2605, "2606, "25C6, "25CF, "2713, "2714, "2717, "2718, "2260, "2261, "2080 -> "2089, "2070 -> "2079, "00B2, "00B3, "00B9}
TEX

pandoc -s --toc --toc-depth=2 \
  --pdf-engine=xelatex --from=markdown-yaml_metadata_block-bracketed_spans-link_attributes-fenced_code_attributes-header_attributes \
  -V documentclass=ctexart \
  -V geometry:margin=2cm \
  -V title="$TITLE" \
  -V CJKmainfont="Songti SC" \
  -V mainfont="Times New Roman" \
  --include-in-header="$WORK/symfix.tex" \
  -o "$OUT_PDF" \
  "$COMBINED" 2>&1 | tail -5

if [[ -f "$OUT_PDF" ]]; then
  SIZE=$(stat -f%z "$OUT_PDF" 2>/dev/null || stat -c%s "$OUT_PDF")
  echo "✓ $OUT_PDF ($(echo "scale=1; $SIZE/1024/1024" | bc) MB)"
else
  echo "❌ 失败"; exit 1
fi
