.PHONY: install train evaluate serve test clean docker-build docker-run

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

serve:
	cd Flask && python app.py

test:
	python scripts/test_model.py
	python scripts/test_flask.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d
