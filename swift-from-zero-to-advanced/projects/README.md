# Projects

This directory contains the long-running project spines used by the course.

## CLI / Package Spine

- Part 1: `TaskCLI Lite`
- Part 2: `TaskCore + TaskCLI`
- Part 4: `TaskCLI Pro`

The CLI line teaches Swift as a language first, then as a modular engineering
toolchain.

## Apple App Spine

- Part 3: `TaskFlow`
- Part 4: advanced `TaskFlow` hardening

The app line exists only after the reader has enough shared Swift foundation to
focus on UI, state, data flow, and persistence without relearning the domain.

## Why The Domain Stays The Same

The course keeps one task-management domain on purpose.

Readers should spend their energy on Swift semantics, code shape, and design
tradeoffs, not on re-learning a new business problem every part.

## Part 1 Constraint

Part 1 keeps the starter package intentionally small.
It should look real, but it should not import the architecture that belongs to
Part 2.
