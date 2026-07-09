# ScholarFlow Project Charter

This file acts as the standing checklist for all future work in this repository. Before making changes or giving implementation advice, always read and follow these points.

## Always follow
- Keep the project goal centered on a **generic, domain-agnostic literature collection Agent**.
- Prefer a **hybrid acquisition path**: official website first, API fallback.
- Remove any hard-coded bias toward a specific domain unless the user explicitly asks for it.
- Preserve a small number of **configuration points** so the project can be retargeted to a new domain with minimal changes.
- Treat the frontend as a **modern, high-end scientific workbench**; favor clarity, polish, and a technology-style visual language.
- Keep `dev_log.md` as the single source of truth for development history.
- Avoid drift: if a request conflicts with this charter, call it out and confirm before proceeding.

## Before each task
- Re-read this charter.
- Verify the requested work still matches the generic Agent direction.
- Check whether the task changes architecture, data flow, UI style, or logging rules.
- If it does, update the charter or ask for confirmation first.

## Success criteria
- The codebase stays domain-neutral.
- The acquisition pipeline remains extensible.
- The UI feels polished and professional.
- Logging stays consolidated and easy to review.
