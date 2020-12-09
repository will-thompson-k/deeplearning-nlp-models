
# lint the project
lint:
	python -m pylint nlpmodels --rcfile=.pylintrc
# run full set of tests
test:
	python -m pytest
# run all the tests - regression tests in test_trainer (much faster).
test_light:
	python -m pytest -k "not test_trainer"