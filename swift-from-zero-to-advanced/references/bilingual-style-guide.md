# Bilingual Style Guide

## Core Model

This course is strongly bilingual, but it is not written as full paragraph-by-
paragraph translation.

Chinese carries:

- explanations
- reasoning
- pitfalls
- design tradeoffs
- exercise guidance

English carries:

- code
- API names
- type names
- canonical technical terms

## First-Use Term Rule

Introduce key concepts in bilingual first-use form:

- `Value semantics（值语义）`
- `Optional binding（可选值绑定）`
- `Pattern matching（模式匹配）`

After first use, whichever side is more natural in context may lead.

## English Recap

Every full Part 1 chapter should end with a short `English Recap` section.
It is not a translation of the whole chapter.
It is a compact technical summary of the chapter's rules, vocabulary, and
engineering takeaways.

## Term Stability

- use one English term for one concept
- use one Chinese translation for one concept unless there is a strong reason
  to change it
- update the glossary before inventing a new translation locally

## Non-Rules

The course should not:

- transliterate Chinese into romanization
- duplicate the full body text in two languages
- translate code, API names, or symbol names into Chinese
- interrupt the teaching flow with glossary spam in every paragraph
