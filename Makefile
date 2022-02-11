install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C *.py

deploy:
	uvicorn user_app:app 

test:
	python -m pytest -vv --cov=hello test_local.py

all: install format lint test deploy 