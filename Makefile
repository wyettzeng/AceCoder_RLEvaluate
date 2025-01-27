format:
	isort --skip-glob="ENV/*" .
	black --exclude '/ENV/*/|cache/*/' .

analyze:
	python analyze_rl_results.py