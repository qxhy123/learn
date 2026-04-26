# Linux / AI Infra / Python Engineering Tutorials

本目录包含统一 HTML 风格的中文深度教程。

- 入口：`index.html`
- 教程目录：`tutorials/`
- 共享规范：`specs/tutorial-style-and-content.md`
- 当前纳入统一索引/验证的教程页数：92
- 旧版/非统一规范页面清单：`.omx/tasks/excluded-legacy-tutorial-pages.txt`

每个纳入索引的教程都是 standalone HTML：内嵌 CSS/SVG，不依赖外部 JS/CSS/图片。参考链接可以出现在正文/参考资料中，但页面渲染不依赖网络资源。

## 批次说明

- 第一批：常规 Linux command。
- 第二批：AI Infra 工程师命令清单，执行约束为“一个命令一个 agent，并发 5”。
- 第三批：`Python asyncio`、`Pydantic v2`、`mypy`、`FastAPI`，每个主题一个独立 agent。
- 第四批：`Python type`、`ruff`、`isort`、`ry`，每个主题一个独立 agent。
- 第五批：`ty`，Astral Python 类型检查器/语言服务器教程。
- 已运行统一教程集合结构检查：HTML parser、TOC anchor、资源依赖、SVG 数量、章节数量。
