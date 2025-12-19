test:
	mypy src/
	pytest tests

prettier:
	ruff check --fix . && ruff format .