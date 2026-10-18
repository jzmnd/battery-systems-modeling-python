.PHONY: format
format:
	@uv run ruff format

.PHONY: format
check:
	@uv run ruff check
