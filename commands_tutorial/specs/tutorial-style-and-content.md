# Tutorial Style and Content Standard

Use `tutorials/sed.html` as the sample page and maintain the same single-file HTML style.

## Required Page Anatomy

- HTML language: `zh-CN`.
- Embedded `<style>` copied/adapted from the sample: CSS variables, `.page`, `.hero`, `.toc`, `.section`, `.figure`, `.refbox`, cards, notes, warnings, tables, and responsive grid rules.
- Hero with title, one-paragraph summary, chips, one concise mental-model note, and one safety/practice note.
- TOC with anchors for all major sections.
- At least 15 substantive numbered sections.
- At least 4 inline SVG diagrams: overall model, one core workflow, one selection/comparison/use-case diagram, and one troubleshooting/safety diagram.
- Code examples must be copyable and realistic.
- Destructive operations must show dry-run or confirmation first.
- References section and footer.

## Tone

Chinese, practical, concept-first, production-aware. Explain the command’s mental model before listing options.

## Agent Boundaries

Each command agent writes only `tutorials/<command>.html` and must not edit shared files.
