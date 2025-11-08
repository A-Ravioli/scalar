.PHONY: dev test lint format migrate

dev:
	@echo "Starting development services..."
	@docker-compose up -d

test:
	pytest

lint:
	black --check .
	flake8 .
	mypy libs/ services/ --ignore-missing-imports

format:
	black .
	isort .

migrate:
	@echo "Running migrations..."
	supabase db reset

install:
	pip install -r requirements.txt

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

