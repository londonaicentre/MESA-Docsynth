test:
	uv run ty check tests/ src/
	uv run pytest tests --cov --cov-fail-under=80

prettier:
	uv run ruff check --fix . && uv run ruff format .