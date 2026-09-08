# CLAUDE.md (root)

Guidance for Claude Code sessions working at the top of this workspace. If you're
already inside `BE/` or `FE/`, that submodule's own `CLAUDE.md`/`AGENTS.md`/`README.md`
takes precedence for anything project-specific - this file only covers the workspace
structure and git lifecycle that live at this root level.

## What this is

`retrieval-system` is a **submodule workspace**, not a single repo. It exists so a
human or an agent working across both halves of this product has one directory to
`cd` into, one place to see both READMEs, and one shared Makefile - nothing more.

```
retrieval-system/            <- this repo (github.com/Uditya69/txmn-data-retrieval, branch dev)
├── BE/                      <- separate repo: bitbucket ai-intent-search, branch dev
└── FE/                      <- separate repo: bitbucket ai-intent-search-frontend, branch master
```

`BE` and `FE` are real git submodules (see `.gitmodules`) - each is its own independent
git repository with its own full history, its own remote, its own branch. This root
repo does not contain their source code; it only records, per submodule, *which commit*
of `BE`/`FE` this snapshot of the workspace points at (a "gitlink" - visible as a
special entry in `git status`/`git diff`, not a real directory of tracked files).

## Where things actually live (this is the part that matters)

- **Bitbucket is authoritative for day-to-day work.** `BE`'s origin
  (`bitbucket.org/taxmannrevampteam/ai-intent-search`, branch `dev`) and `FE`'s origin
  (`bitbucket.org/taxmannrevampteam/ai-intent-search-frontend`, branch `master`) are
  where real commits, pushes, and PRs happen. Nothing about adding these as submodules
  changes that - `BE`/`FE` still push to Bitbucket exactly as they did before this
  workspace existed.
- **This root repo on GitHub is a combined snapshot/mirror**, not a place either
  submodule's source code gets edited or committed directly. Its only job is to pin a
  `BE` commit + an `FE` commit together as "this is what the combined product looked
  like at this point," plus hold this Makefile/CLAUDE.md/AGENTS.md.

## Git lifecycle - the one rule that avoids all the confusion

**Never commit source files at this root level.** If `git status` at the root shows
anything other than a `BE`/`FE` pointer change (or edits to this Makefile/CLAUDE.md/
AGENTS.md), you're almost certainly `cd`'d into the wrong place - go into `BE/` or
`FE/` first.

The actual flow:

1. **Do the real work inside `BE/` or `FE/`.** `cd BE` (or `FE`), branch, edit, test,
   commit, push - all exactly as if this were the only repo you had cloned. Use that
   submodule's own remote (`origin`, Bitbucket) for every push. `BE`'s and `FE`'s own
   `CLAUDE.md`/`AGENTS.md` govern how to do this work - read those, not this file, for
   anything about the actual product/code.
2. **Once pushed to Bitbucket, come back to the root** and record the pointer bump:
   `make bump-be` (or `make bump-fe`) - this stages the submodule gitlink, commits
   `"chore: bump BE submodule pointer"`, and pushes the root repo to
   `github.com/Uditya69/txmn-data-retrieval` (`origin`, branch `dev`).
3. **To pick up someone else's already-pushed Bitbucket work** without having made any
   local changes of your own, `make sync` (or `git submodule update --remote --merge`
   then `make bump-be`/`bump-fe`).

A fresh clone of this root repo does **not** bring `BE`/`FE`'s files with it - run
`git submodule update --init --recursive` (or clone with `--recurse-submodules`) first.

## Don't run tests or dev servers from here

There is no unified toolchain at this root - `BE` is a `uv` workspace (Python), `FE` is
an `npm`/Vite project (TypeScript). `cd BE` or `cd FE` before running anything;
`make status` at the root is the only command meant to be run from here directly for
day-to-day use.
