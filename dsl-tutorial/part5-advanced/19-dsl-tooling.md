# 第19章：DSL工具链——LSP与IDE支持

## 核心思维模型

> 一门没有IDE支持的DSL，就像一本没有目录、没有索引、没有高亮的技术手册——理论上可用，实际上痛苦。**语言工具链把DSL从"勉强可用"变成"用户爱用"。**

---

## 19.1 现代IDE支持的层次

```
层次1：语法高亮（Syntax Highlighting）
  - 关键字、字符串、注释用不同颜色显示
  - 实现工具：TextMate文法、tree-sitter

层次2：错误提示（Error Reporting）
  - 红色波浪线标注错误位置
  - 实现工具：Language Server Protocol（LSP）

层次3：自动补全（Code Completion）
  - 输入时提示关键字、表名、字段名
  - 实现工具：LSP（completionProvider）

层次4：悬停文档（Hover Documentation）
  - 鼠标悬停在关键字/表名上，显示文档
  - 实现工具：LSP（hoverProvider）

层次5：跳转定义（Go to Definition）
  - 跳转到表/字段的定义位置
  - 实现工具：LSP（definitionProvider）

层次6：重构（Refactoring）
  - 重命名字段、提取表达式等
  - 实现工具：LSP（renameProvider）
```

---

## 19.2 Language Server Protocol（LSP）简介

LSP 是微软于2016年提出的**编辑器-语言服务器通信协议**，彻底改变了语言工具链的建设方式：

**LSP之前**：每种IDE都需要为每种语言实现独立的插件
```
IntelliJ + Java    = IntelliJ的Java支持
Eclipse + Java     = Eclipse的Java支持
VSCode + Java      = VSCode的Java支持
── 需要 M×N 个实现 ──
```

**LSP之后**：语言服务器只实现一次，所有支持LSP的编辑器自动获得支持
```
QueryLang Server ←→ VSCode
QueryLang Server ←→ Neovim
QueryLang Server ←→ Emacs
── 只需要 M+N 个实现 ──
```

### LSP通信模型

```
VSCode（客户端）                  QueryLang Server（服务端）
       │                                  │
       │── textDocument/completion ──────→│
       │                                  │
       │← CompletionList ────────────────│
       │                                  │
       │── textDocument/hover ───────────→│
       │                                  │
       │← Hover {contents: "..."} ───────│
```

---

## 19.3 实现 QueryLang LSP 服务器

使用Python的`pygls`库（Python Generic Language Server）：

```python
# querylang_server.py
# 安装：pip install pygls

from pygls.server import LanguageServer
from lsprotocol.types import (
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_COMPLETION,
    TEXT_DOCUMENT_HOVER,
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    CompletionParams,
    Diagnostic as LSDiagnostic,
    DiagnosticSeverity,
    DidChangeTextDocumentParams,
    DidOpenTextDocumentParams,
    Hover,
    HoverParams,
    MarkupContent,
    MarkupKind,
    Position,
    Range,
)

# 初始化语言服务器
server = LanguageServer("querylang-server", "v1.0")

# Schema定义（实际项目中从配置或数据库读取）
SCHEMA = {
    "users": {
        "id": "int",
        "name": "string",
        "age": "int",
        "status": "string",
        "email": "string",
        "created_at": "datetime",
    },
    "orders": {
        "id": "int",
        "user_id": "int",
        "amount": "float",
        "status": "string",
        "created_at": "datetime",
    }
}

# ─── 诊断（错误提示）──────────────────────────────────────────

def validate_document(source: str) -> list[LSDiagnostic]:
    """解析并验证文档，返回LSP诊断列表"""
    diagnostics = []
    
    try:
        from querylang.lexer import Lexer
        from querylang.parser import Parser
        from querylang.semantic import SemanticAnalyzer, SemanticError
        
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        
        analyzer = SemanticAnalyzer(schema=SCHEMA)
        try:
            analyzer.analyze(ast)
        except SemanticError as e:
            line = (getattr(e, 'line', 1) or 1) - 1
            col = (getattr(e, 'column', 1) or 1) - 1
            diagnostics.append(LSDiagnostic(
                range=Range(
                    start=Position(line=line, character=col),
                    end=Position(line=line, character=col + 10)
                ),
                message=str(e),
                severity=DiagnosticSeverity.Error,
                source="querylang"
            ))
    
    except Exception as e:
        # 所有解析错误转为诊断
        line_num = getattr(e, 'line', 1) or 1
        col_num = getattr(e, 'column', 1) or 1
        
        diagnostics.append(LSDiagnostic(
            range=Range(
                start=Position(line=line_num - 1, character=col_num - 1),
                end=Position(line=line_num - 1, character=col_num + 5)
            ),
            message=str(e),
            severity=DiagnosticSeverity.Error,
            source="querylang"
        ))
    
    return diagnostics

@server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls: LanguageServer, params: DidOpenTextDocumentParams):
    """文档打开时触发验证"""
    doc = params.text_document
    diagnostics = validate_document(doc.text)
    ls.publish_diagnostics(doc.uri, diagnostics)

@server.feature(TEXT_DOCUMENT_DID_CHANGE)
async def did_change(ls: LanguageServer, params: DidChangeTextDocumentParams):
    """文档修改时触发验证"""
    doc = params.text_document
    source = params.content_changes[-1].text
    diagnostics = validate_document(source)
    ls.publish_diagnostics(doc.uri, diagnostics)

# ─── 自动补全 ──────────────────────────────────────────────────

KEYWORDS = ["FROM", "WHERE", "SELECT", "ORDER", "BY", "LIMIT", "AND", "OR", "ASC", "DESC"]

@server.feature(TEXT_DOCUMENT_COMPLETION)
async def completions(ls: LanguageServer, params: CompletionParams):
    """提供自动补全"""
    items = []
    doc = ls.workspace.get_document(params.text_document.uri)
    source = doc.source
    
    # 分析当前上下文
    context = _analyze_completion_context(source, params.position)
    
    if context == "after_from":
        # FROM之后：补全表名
        for table_name, fields in SCHEMA.items():
            items.append(CompletionItem(
                label=table_name,
                kind=CompletionItemKind.Class,
                detail=f"表（{len(fields)}个字段）",
                documentation=MarkupContent(
                    kind=MarkupKind.Markdown,
                    value=f"字段：{', '.join(fields.keys())}"
                )
            ))
    
    elif context == "after_where" or context == "field_position":
        # WHERE之后：补全字段名
        current_table = _extract_current_table(source)
        if current_table and current_table in SCHEMA:
            for field_name, field_type in SCHEMA[current_table].items():
                items.append(CompletionItem(
                    label=field_name,
                    kind=CompletionItemKind.Field,
                    detail=f"类型: {field_type}",
                    documentation=MarkupContent(
                        kind=MarkupKind.PlainText,
                        value=f"{current_table}.{field_name} ({field_type})"
                    )
                ))
    
    elif context == "keyword_position":
        # 关键字位置：补全关键字
        for kw in KEYWORDS:
            items.append(CompletionItem(
                label=kw,
                kind=CompletionItemKind.Keyword,
                detail="QueryLang关键字"
            ))
    
    elif context == "operator_position":
        # 运算符位置
        for op in [">", "<", ">=", "<=", "=", "!="]:
            items.append(CompletionItem(
                label=op,
                kind=CompletionItemKind.Operator
            ))
    
    return CompletionList(is_incomplete=False, items=items)

def _analyze_completion_context(source: str, position) -> str:
    """分析光标位置的上下文，确定补全类型"""
    lines = source.split('\n')
    current_line = lines[position.line] if position.line < len(lines) else ""
    text_before = current_line[:position.character].upper().strip()
    
    # 简单的启发式上下文分析
    if text_before.endswith("FROM"):
        return "after_from"
    if text_before.endswith("WHERE") or text_before.endswith("AND") or text_before.endswith("OR"):
        return "after_where"
    if any(text_before.endswith(op) for op in [">", "<", ">=", "<=", "=", "!="]):
        return "value_position"
    
    # 检查当前行是否以FROM开头（可能在补全表名的中途）
    full_source_upper = source.upper()
    lines_before = '\n'.join(lines[:position.line + 1])
    
    return "keyword_position"

def _extract_current_table(source: str) -> str | None:
    """从源码中提取当前查询的表名"""
    import re
    match = re.search(r'FROM\s+(\w+)', source, re.IGNORECASE)
    return match.group(1) if match else None

# ─── 悬停文档 ──────────────────────────────────────────────────

KEYWORD_DOCS = {
    "FROM": "指定查询的数据来源表\n\n语法：`FROM <表名>`\n\n示例：\n```\nFROM users\n```",
    "WHERE": "过滤条件子句，只返回满足条件的行\n\n语法：`WHERE <条件> [AND|OR <条件>]*`",
    "SELECT": "指定返回的字段\n\n语法：`SELECT * | <字段名> [, <字段名>]*`",
    "ORDER": "与BY配合使用，指定排序\n\n语法：`ORDER BY <字段名> [ASC|DESC]`",
    "LIMIT": "限制返回的最大行数\n\n语法：`LIMIT <正整数>`",
    "AND": "逻辑与，两个条件都必须为真",
    "OR": "逻辑或，至少一个条件为真",
}

@server.feature(TEXT_DOCUMENT_HOVER)
async def hover(ls: LanguageServer, params: HoverParams) -> Hover | None:
    """悬停时显示文档"""
    doc = ls.workspace.get_document(params.text_document.uri)
    source = doc.source
    
    word = _get_word_at_position(source, params.position)
    if not word:
        return None
    
    word_upper = word.upper()
    
    # 关键字文档
    if word_upper in KEYWORD_DOCS:
        return Hover(
            contents=MarkupContent(
                kind=MarkupKind.Markdown,
                value=f"**{word_upper}**\n\n{KEYWORD_DOCS[word_upper]}"
            )
        )
    
    # 表名文档
    if word in SCHEMA:
        fields = SCHEMA[word]
        field_list = "\n".join(f"- `{k}`: {v}" for k, v in fields.items())
        return Hover(
            contents=MarkupContent(
                kind=MarkupKind.Markdown,
                value=f"**表：{word}**\n\n字段：\n{field_list}"
            )
        )
    
    # 字段名文档（检查所有表）
    for table_name, fields in SCHEMA.items():
        if word in fields:
            return Hover(
                contents=MarkupContent(
                    kind=MarkupKind.Markdown,
                    value=f"**{word}**\n\n表 `{table_name}` 的字段\n类型：`{fields[word]}`"
                )
            )
    
    return None

def _get_word_at_position(source: str, position) -> str | None:
    """获取光标位置的单词"""
    lines = source.split('\n')
    if position.line >= len(lines):
        return None
    
    line = lines[position.line]
    col = position.character
    
    # 向左扩展
    start = col
    while start > 0 and (line[start-1].isalnum() or line[start-1] == '_'):
        start -= 1
    
    # 向右扩展
    end = col
    while end < len(line) and (line[end].isalnum() or line[end] == '_'):
        end += 1
    
    word = line[start:end]
    return word if word else None


if __name__ == "__main__":
    server.start_io()
```

---

## 19.4 tree-sitter 语法高亮

tree-sitter是现代语法高亮的标准方案，支持增量解析：

```javascript
// grammar.js - QueryLang的tree-sitter文法
// 安装：npm install tree-sitter-cli

module.exports = grammar({
  name: 'querylang',
  
  extras: $ => [
    /\s/,              // 跳过空白
    $.comment,         // 跳过注释
  ],
  
  rules: {
    // 入口规则
    source_file: $ => $.query,
    
    query: $ => seq(
      $.from_clause,
      optional($.where_clause),
      optional($.select_clause),
      optional($.order_clause),
      optional($.limit_clause),
    ),
    
    from_clause: $ => seq(
      field('keyword', /[Ff][Rr][Oo][Mm]/),
      field('table', $.identifier),
    ),
    
    where_clause: $ => seq(
      field('keyword', /[Ww][Hh][Ee][Rr][Ee]/),
      field('condition', $.condition),
    ),
    
    condition: $ => choice(
      $.binary_condition,
      $.comparison,
    ),
    
    binary_condition: $ => seq(
      $.condition,
      field('operator', choice('AND', 'OR', 'and', 'or')),
      $.condition,
    ),
    
    comparison: $ => seq(
      field('field', $.identifier),
      field('operator', $.compare_op),
      field('value', $.value),
    ),
    
    compare_op: $ => choice('>', '<', '>=', '<=', '=', '!='),
    
    value: $ => choice(
      $.integer,
      $.float,
      $.string,
      $.boolean,
    ),
    
    select_clause: $ => seq(
      /[Ss][Ee][Ll][Ee][Cc][Tt]/,
      choice('*', commaSep1($.identifier)),
    ),
    
    order_clause: $ => seq(
      /[Oo][Rr][Dd][Ee][Rr]/,
      /[Bb][Yy]/,
      $.identifier,
      optional(choice('ASC', 'DESC', 'asc', 'desc')),
    ),
    
    limit_clause: $ => seq(
      /[Ll][Ii][Mm][Ii][Tt]/,
      $.integer,
    ),
    
    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,
    integer: $ => /\d+/,
    float: $ => /\d+\.\d+/,
    string: $ => /"[^"]*"/,
    boolean: $ => choice('true', 'false', 'TRUE', 'FALSE'),
    comment: $ => /--[^\n]*/,
  }
});

function commaSep1(rule) {
  return seq(rule, repeat(seq(',', rule)));
}
```

配套的TextMate语法高亮（用于没有tree-sitter的编辑器）：

```json
{
  "name": "QueryLang",
  "scopeName": "source.querylang",
  "fileTypes": ["ql", "querylang"],
  "patterns": [
    {
      "name": "keyword.control.querylang",
      "match": "\\b(?:FROM|WHERE|SELECT|ORDER|BY|LIMIT|AND|OR|ASC|DESC)\\b",
      "captures": {
        "0": {"name": "keyword.control.querylang"}
      }
    },
    {
      "name": "string.quoted.double.querylang",
      "begin": "\"",
      "end": "\"",
      "patterns": [
        {"name": "constant.character.escape.querylang", "match": "\\\\."}
      ]
    },
    {
      "name": "constant.numeric.querylang",
      "match": "\\b\\d+(\\.\\d+)?\\b"
    },
    {
      "name": "constant.language.boolean.querylang",
      "match": "\\b(?:true|false|TRUE|FALSE)\\b"
    },
    {
      "name": "comment.line.double-dash.querylang",
      "match": "--.*$"
    },
    {
      "name": "keyword.operator.querylang",
      "match": ">=|<=|!=|>|<|="
    }
  ]
}
```

---

## 19.5 VSCode插件结构

```
querylang-vscode/
├── package.json          # 插件元数据和贡献点
├── src/
│   └── extension.ts      # 插件入口
├── syntaxes/
│   └── querylang.tmGrammar.json  # 语法高亮
└── language-configuration.json   # 括号匹配等
```

```json
// package.json（关键部分）
{
  "name": "querylang",
  "displayName": "QueryLang",
  "description": "QueryLang语言支持",
  "version": "1.0.0",
  "engines": {"vscode": "^1.75.0"},
  "categories": ["Programming Languages"],
  "activationEvents": ["onLanguage:querylang"],
  "contributes": {
    "languages": [{
      "id": "querylang",
      "aliases": ["QueryLang", "ql"],
      "extensions": [".ql"],
      "configuration": "./language-configuration.json"
    }],
    "grammars": [{
      "language": "querylang",
      "scopeName": "source.querylang",
      "path": "./syntaxes/querylang.tmGrammar.json"
    }],
    "configuration": {
      "title": "QueryLang",
      "properties": {
        "querylang.serverPath": {
          "type": "string",
          "description": "QueryLang语言服务器路径"
        }
      }
    }
  }
}
```

```typescript
// src/extension.ts
import * as vscode from 'vscode';
import { LanguageClient, ServerOptions, TransportKind } from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
  const serverModule = context.asAbsolutePath('server/querylang_server.py');
  
  const serverOptions: ServerOptions = {
    run: { command: 'python', args: [serverModule], transport: TransportKind.stdio },
    debug: { command: 'python', args: [serverModule, '--debug'], transport: TransportKind.stdio }
  };
  
  client = new LanguageClient(
    'querylang',
    'QueryLang Language Server',
    serverOptions,
    { documentSelector: [{ scheme: 'file', language: 'querylang' }] }
  );
  
  client.start();
  
  // 注册自定义命令
  context.subscriptions.push(
    vscode.commands.registerCommand('querylang.runQuery', runCurrentQuery)
  );
}

async function runCurrentQuery() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return;
  
  const source = editor.document.getText();
  // 调用QueryLang引擎执行并显示结果
  const result = await executeQueryLang(source);
  const panel = vscode.window.createWebviewPanel('queryResult', '查询结果', vscode.ViewColumn.Beside, {});
  panel.webview.html = renderResultTable(result);
}

export function deactivate() {
  if (client) client.stop();
}
```

---

## 小结

| 工具层 | 技术方案 | 难度 |
|--------|---------|------|
| 语法高亮 | TextMate文法 / tree-sitter | ★★ |
| 错误提示 | LSP（publishDiagnostics）| ★★★ |
| 自动补全 | LSP（completionProvider）| ★★★ |
| 悬停文档 | LSP（hoverProvider）| ★★★ |
| 跳转定义 | LSP（definitionProvider）| ★★★★ |
| 重构 | LSP（renameProvider）| ★★★★★ |

---

**上一章**：[错误报告与诊断](./18-error-handling.md)
**下一章**：[真实世界DSL案例解析](./20-real-world-dsls.md)
