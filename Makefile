# Reproduction entry points.  `make help` lists them.
PYTHON ?= python3

.PHONY: help restore verify figures check clean-outputs

help:
	@echo "make restore   copy results_frozen/ into the analysis tree (do this first)"
	@echo "make figures   rebuild every manuscript figure from the restored artefacts"
	@echo "make verify    diff the analysis tree against results_frozen/"
	@echo "make check     scan for secrets, personal paths and oversized files"
	@echo "make clean-outputs  delete generated outputs, keeping results_frozen/ intact"

restore:
	$(PYTHON) tools/restore_frozen_results.py

figures:
	$(PYTHON) tools/build_all_figures.py

verify:
	$(PYTHON) tools/verify_frozen_results.py --quiet

check:
	$(PYTHON) tools/check_repo_hygiene.py

clean-outputs:
	$(PYTHON) tools/restore_frozen_results.py --dry-run
	@echo "(remove the listed paths by hand, or re-clone; results_frozen/ is never touched)"
