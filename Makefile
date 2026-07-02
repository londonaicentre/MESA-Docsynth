test:
	uv run mypy src/
	uv run pytest tests

prettier:
	uv run ruff check --fix . && uv run ruff format .