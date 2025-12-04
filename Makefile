test:
	mypy src/

prettier:
	ruff check --fix . && ruff format .