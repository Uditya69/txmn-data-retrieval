# AGENTS.md (root)

Instructions for coding agents working at the top of this workspace. If you're already
inside `BE/` or `FE/`, use that submodule's own `AGENTS.md`/`CLAUDE.md`/`README.md`
instead — it's authoritative for anything about the actual product/code. This file only
covers the structure and git lifecycle at this root level.

## Repo

This root (`github.com/Uditya69/txmn-data-retrieval`, branch `dev`) is a **submodule
workspace**, not a source repo. `BE/` and `FE/` are separate git repositories (see
`.gitmodules`), each with their own remote, branch, and history:

- `BE/` → `bitbucket.org/taxmannrevampteam/ai-intent-search`, branch `dev` — Python/uv backend
- `FE/` → `bitbucket.org/taxmannrevampteam/ai-intent-search-frontend`, branch `master` — TypeScript/Vite frontend

Bitbucket is authoritative for real work on either half. This root repo only pins which
`BE`/`FE` commit belongs together as one combined snapshot.

## Hard rule

**Never commit source files at this root.** A `git status` here should only ever show a
`BE`/`FE` pointer change, or an edit to this file/`CLAUDE.md`/`Makefile`. If you're
about to edit actual product code and you're sitting at the root, `cd BE` or `cd FE`
first — that submodule's own git history, remote, and rules apply from there on, as if
it were the only repo checked out.

## Workflow

1. `cd BE` (or `FE`) → do the work → commit → `git push` (goes to Bitbucket, that
   submodule's own `origin`). Standard single-repo workflow, nothing special.
2. Back at root: `make bump-be` (or `make bump-fe`) to record the new commit as this
   workspace's pinned pointer, and push the root repo to GitHub.
3. To pull in already-pushed Bitbucket work without making changes of your own:
   `make sync`.

See root `CLAUDE.md` for the full explanation and `make status` for a combined view of
all three repos (root + BE + FE) at once.

## Commands

```bash
git submodule update --init --recursive   # after a fresh clone of this root repo
make status                               # root + BE + FE status in one shot
make bump-be / make bump-fe               # record a pushed BE/FE commit as this workspace's pointer
make sync                                 # pull BE/FE to their tracked branch tips + bump both pointers
```

No tests, no dev server, no build runs from this root — `cd BE` or `cd FE` for those,
each has its own toolchain (`uv` vs `npm`).
