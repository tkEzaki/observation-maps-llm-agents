PYTHON ?= python3
.PHONY: restore verify figures check
restore:
	$(PYTHON) -B tools/restore_frozen_results.py
verify:
	$(PYTHON) -B tools/verify_release.py
figures:
	$(PYTHON) -B tools/build_current_figures.py
check:
	$(PYTHON) -B tools/check_repo_hygiene.py
