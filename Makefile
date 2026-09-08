\
# Root-level convenience targets for the BE/FE submodule workspace.
# See CLAUDE.md / AGENTS.md at this same root for the full git lifecycle
# explanation - this Makefile just wraps the commands described there.

.PHONY: status update-submodules bump-be bump-fe sync push-root clone-check

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
