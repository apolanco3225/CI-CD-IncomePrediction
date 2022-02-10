install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C *.py

deploy:
	uvicorn app.main:app 

test:
	python -m pytest -vv --cov=hello test_hello.py

all: install format lint deploy test