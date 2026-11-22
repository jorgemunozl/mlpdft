---
status: active
date: 2026-11-20
owner: jorge
---

# Research documentation

This tree holds the project's durable research record. It is organised by
**document type**, not by phase. A phase is a label in the front matter.

## Rules

1. **One canonical document per question.** If a document is replaced, move it
   to `archive/` with a banner and `superseded_by:`. Do not rewrite it in place.
2. **Decisions change on the record.** Change an accepted decision by writing
   a new record that says `supersedes: ADR-000N`, or, with the owner's
   approval, by revising it in place: add `revised:` to the front matter and a
   dated note at the top that says what changed and where the original text is
   in git history.
3. **Results live with the run.** `experiments/<exp>/RESULTS.md` summarises and
   links; the numbers, figures and logs stay in `experiments/<exp>/runs/<date>/`.
4. **Every document starts with front matter:**

```yaml
---
status: active        # draft | active | superseded
date: 2026-11-20
phase: 2
owner: jorge
supersedes: —         # optional path or ADR id
superseded_by: —      # required when status is superseded
---
```

## Where things go

| Folder | Holds | Examples |
|---|---|---|
| [`decisions/`](decisions/) | accepted design decisions, numbered | why four heads, why MHC is outside the AL loop |
| [`literature/`](literature/) | the review and source notes | `REVIEW.md` |
| [`methods/`](methods/) | reusable method and diagnostic notes | autocorrelation and frame stride |
| [`datasets/`](datasets/) | one folder per dataset: curation, manifests, rebuild steps | `phase1-audit/`, `interim-2env/`, `final-3env/` |
| [`experiments/`](experiments/) | one folder per experiment: design, protocol, results, runs | `exp-03-mhc/` |
| [`archive/`](archive/) | superseded documents, frozen with a banner | early protocol drafts |

## Start here

- [`STATUS.md`](STATUS.md) — what is settled, what is open, what is next.
- [`CHARTER.md`](CHARTER.md) — the question, scope and claims.
- [`ROADMAP.md`](ROADMAP.md) — phases and gates.

## Adding a document

1. Decide its type from the table above. If it is a decision, use
   [`decisions/TEMPLATE.md`](decisions/TEMPLATE.md) and take the next free
   number; never reuse one.
2. Put it in the matching folder with an `UPPER_SNAKE.md` name, except
   decisions (`ADR-000N-slug.md`) and archived copies (`YYYY-MM-DD-slug.md`).
3. Add the front matter and link it from the section that needs it — no
   orphan documents.
4. If it replaces an older document, move that document to `archive/` and set
   `superseded_by:` on it.
