.PHONY: install train evaluate serve clean

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

serve:
	cd Flask && python app.py

preprocess:
	python scripts/preprocess.py

validate:
	python scripts/validate_dataset.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
