.PHONY: format
format:
	@uv run ruff format

.PHONY: format
check:
	@uv run ruff check

.PHONY: test
test:
	@uv run pytest
