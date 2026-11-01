# Contributing

Use `feat/*` branches and pull requests targeting `dev`. `dev` is the integration branch; `main` is stable. Release via a reviewed `dev` → `main` pull request. Never commit or push directly to either branch. Use Conventional Commits; keep changes small and reversible.

Run CPU compilation and unit tests documented in README. Validate `docker compose config --quiet`. Explain checks actually executed and checks unavailable. GPU inference needs explicit non-commercial opt-in and separate real-hardware smoke tests; CPU CI is not GPU coverage.

Preserve authenticated artifact downloads, input allowlists, bounded queues, environment separation, persistent volumes, and secret boundaries. Do not add arbitrary URL/path inputs. Do not commit `.env`, `.secrets/`, model weights, generated audio, or downloaded checkpoints. Review dependency changes for legacy AudioCraft/Torch compatibility and licensing; retain upstream notices.

Changes to database schemas need backward-compatibility and rollback review. Never delete volumes or stop unrelated GPU services as part of setup or tests. Public access and commercial model providers are outside the current scope.
