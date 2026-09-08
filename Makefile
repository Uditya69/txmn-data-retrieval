\
# Root-level convenience targets for the BE/FE submodule workspace.
# See CLAUDE.md / AGENTS.md at this same root for the full git lifecycle
# explanation - this Makefile just wraps the commands described there.

.PHONY: status update-submodules bump-be bump-fe sync push-root clone-check dev dev-be dev-fe

# Status of the root repo plus both submodules in one shot.
status:
	@echo "=== root ==="
	@git status --short --branch
	@echo "=== BE (bitbucket ai-intent-search, dev) ==="
	@cd BE && git status --short --branch
	@echo "=== FE (bitbucket ai-intent-search-frontend, master) ==="
	@cd FE && git status --short --branch

# Pull the latest commit each submodule's tracked branch (dev for BE, master
# for FE) points to on Bitbucket. Does NOT change what the root repo has
# pinned - run `make bump-be`/`make bump-fe` after this to record the update.
update-submodules:
	git submodule update --remote --merge

# Stage+commit+push just the BE submodule's pointer update in the root repo.
# Run this AFTER you've committed and pushed real work inside BE/ to its own
# bitbucket origin - this only bumps which BE commit the root repo points at.
bump-be:
	git add BE
	git commit -m "chore: bump BE submodule pointer"
	git push origin dev

# Same as bump-be, for the FE submodule.
bump-fe:
	git add FE
	git commit -m "chore: bump FE submodule pointer"
	git push origin dev

# Pulls both submodules to their tracked branch tips and bumps both pointers
# in one commit - use when you just want the root repo caught up to whatever
# BE/FE's teams have already pushed, with no root-level work of your own.
sync:
	git submodule update --remote --merge
	git add BE FE
	git commit -m "chore: sync BE/FE submodule pointers" --allow-empty
	git push origin dev

push-root:
	git push origin dev

# --- Local dev servers, hot-reload ------------------------------------------
# CLAUDE.md at this root says not to run dev servers from here (no unified
# toolchain, BE is uv/Python, FE is npm/Vite) - these targets are just thin
# `cd`+run wrappers, same spirit as bump-be/bump-fe above, for convenience
# when you want both sides up with one command. Real per-project dev docs
# still live in BE/ and FE/'s own CLAUDE.md/README.

# retrieval-api with uvicorn --reload (model-gateway runs in-process, no
# separate command needed). BE/scripts/dev.sh also starts this but additionally
# tries to run a `packages/web` frontend that no longer exists there (it moved
# to FE/ in the BE/FE split) - use dev-be/dev-fe/dev here instead of that script.
dev-be:
	cd BE && uv run uvicorn retrieval_api.main:app --reload \
		--reload-dir packages/retrieval-api/src --reload-dir packages/common/src \
		--reload-dir packages/model-gateway/src --port 8010

# Vite dev server with HMR. Installs node_modules first if missing.
dev-fe:
	cd FE && [ -d node_modules ] || npm install
	cd FE && npm run dev

# Both at once. Ctrl-C stops both (trap kills the background FE job, then the
# foreground BE job exits with it).
dev:
	@trap 'kill %1 2>/dev/null' EXIT INT TERM; \
	$(MAKE) dev-fe & \
	$(MAKE) dev-be
