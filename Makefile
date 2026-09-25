PY ?= python
.PHONY: test demo data reproduce report check
test:        ## ledger and timing tests (offline)
	$(PY) -m pytest -q
demo:        ## synthetic panel through the same ledger, no network
	$(PY) scripts/demo.py
data:        ## public Binance archives (checksum-verified) and the hourly panel
	$(PY) scripts/download.py
	$(PY) scripts/build_panel.py
reproduce:   ## development selection, final evaluation, transfer test, report
	$(PY) scripts/run_study.py dev
	$(PY) scripts/run_study.py final --i-understand-this-is-the-final-evaluation
	$(PY) scripts/transfer_test.py
	$(PY) scripts/family_significance.py
	$(PY) scripts/make_report.py
report:
	$(PY) scripts/make_report.py
check: test
	$(PY) -m ruff check src scripts tests
	$(PY) scripts/check_claims.py
data-positioning:     ## Generation 2: Binance daily positioning metrics (63,837 archives, 0.7 GB, reduced to one table)
	$(PY) scripts/download_metrics.py
reproduce-positioning:  ## Generation 2 development selection and final evaluation
	$(PY) scripts/run_positioning.py dev
	$(PY) scripts/run_positioning.py final --i-understand-this-is-the-final-evaluation
	$(PY) scripts/capacity.py
	$(PY) scripts/make_report.py
