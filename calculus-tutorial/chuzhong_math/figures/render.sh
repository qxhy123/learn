#!/usr/bin/env bash
# 批量渲染 figures/src/{tikz,asy}/*.{tex,asy} → figures/svg/*.svg
#
# 依赖：pdflatex (MacTeX), asy (asymptote), pdf2svg
# 用法：在 chuzhong_math/figures 目录下运行 ./render.sh
#       或指定单个文件：./render.sh src/asy/handshake.asy

set -euo pipefail

# 把 MacTeX 加入 PATH（如尚未存在）
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_TIKZ="$DIR/src/tikz"
SRC_ASY="$DIR/src/asy"
OUT="$DIR/svg"
mkdir -p "$OUT"

render_tikz() {
    local tex="$1"
    local name
    name="$(basename "$tex" .tex)"
    local tmpdir
    tmpdir="$(mktemp -d)"
    echo "  TikZ: $name"
    cp "$tex" "$tmpdir/$name.tex"
    (cd "$tmpdir" && pdflatex -interaction=nonstopmode "$name.tex" >/dev/null 2>&1) || {
        echo "    ✗ pdflatex failed for $name"
        rm -rf "$tmpdir"
        return 1
    }
    pdf2svg "$tmpdir/$name.pdf" "$OUT/$name.svg"
    rm -rf "$tmpdir"
    echo "    ✓ $OUT/$name.svg"
}

render_asy() {
    local src="$1"
    local name
    name="$(basename "$src" .asy)"
    echo "  Asy:  $name"
    asy -f svg -tex pdflatex -o "$OUT/$name.svg" "$src" 2>&1 | grep -v "^$" || true
    if [[ -f "$OUT/$name.svg" ]]; then
        echo "    ✓ $OUT/$name.svg"
    else
        echo "    ✗ asy failed for $name"
        return 1
    fi
}

# 主流程
if [[ $# -gt 0 ]]; then
    # 渲染指定文件
    for f in "$@"; do
        case "$f" in
            *.tex) render_tikz "$f" ;;
            *.asy) render_asy "$f" ;;
            *)     echo "skip: $f" ;;
        esac
    done
else
    # 批量渲染
    echo "→ TikZ 源文件 ($SRC_TIKZ):"
    if compgen -G "$SRC_TIKZ/*.tex" >/dev/null; then
        for tex in "$SRC_TIKZ"/*.tex; do render_tikz "$tex" || true; done
    else
        echo "  (无 .tex 文件)"
    fi

    echo "→ Asymptote 源文件 ($SRC_ASY):"
    if compgen -G "$SRC_ASY/*.asy" >/dev/null; then
        for asy in "$SRC_ASY"/*.asy; do render_asy "$asy" || true; done
    else
        echo "  (无 .asy 文件)"
    fi
fi

echo "完成。SVG 输出目录: $OUT"
