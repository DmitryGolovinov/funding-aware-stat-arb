"""The positioning snapshot for a 00:00 UTC decision uses only rows stamped before 00:00.

Most daily metrics files run 00:00-23:55, but some end with a row stamped at or just after the
next midnight (for example SXPUSDT-metrics-2024-06-10.zip: 00:05:00 ... 23:55:00, 00:00:01).
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_metrics import reduce_day  # noqa: E402

COLS = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio", "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio"]  # header of the Binance daily metrics archives


def archive(tmp_path: Path, rows: list[tuple[str, float, float]]) -> Path:
    """rows: (create_time, all-account ratio, taker ratio); other fields are filler."""
    df = pd.DataFrame([[t, "XUSDT", 1.0, 100.0 * a, 1.0, 1.5 * a, a, k] for t, a, k in rows], columns=COLS)
    p = tmp_path / "XUSDT-metrics-2024-06-10.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("XUSDT-metrics-2024-06-10.csv", df.to_csv(index=False))
    return p


DECISION = pd.Timestamp("2024-06-11 00:00:00")


def test_rows_stamped_at_or_after_the_decision_are_ignored(tmp_path):
    p = archive(tmp_path, [("2024-06-10 23:55:00", 2.0, 0.5), ("2024-06-11 00:00:01", 9.0, 8.0),
                           ("2024-06-10 23:50:00", 1.0, 2.0), ("2024-06-11 00:00:00", 7.0, 6.0)])
    r = reduce_day(p, before=DECISION)
    assert r["last_row_utc"] == "2024-06-10 23:55:00"
    assert r["all_ratio"] == 2.0 and r["top_pos_ratio"] == 3.0 and r["oi_value"] == 200.0
    assert np.isclose(r["taker_log_mean"], np.mean(np.log([0.5, 2.0])))


def test_a_file_without_rows_before_the_decision_gives_no_snapshot(tmp_path):
    p = archive(tmp_path, [("2024-06-11 00:00:00", 7.0, 6.0), ("2024-06-11 00:00:01", 9.0, 8.0)])
    assert reduce_day(p, before=DECISION) is None
