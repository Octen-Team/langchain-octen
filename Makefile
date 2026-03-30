.PHONY: test integration_tests lint format all

test:
	poetry run pytest tests/unit_tests -v

integration_tests:
	poetry run pytest tests/integration_tests -v

lint:
	poetry run ruff check langchain_octen tests
	poetry run mypy langchain_octen

format:
	poetry run ruff format langchain_octen tests
	poetry run ruff check --fix langchain_octen tests

all: format lint test
