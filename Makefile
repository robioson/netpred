train:
	poetry run python -m netpred

analyse:
	poetry run jupyter notebook --ip=0.0.0.0 --notebook-dir=notebooks
