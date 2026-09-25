"""Offline demo: a synthetic 4-coin panel with funding, one funding long/short rebalance per day
through the same ledger. Writes results/demo/demo.csv (never over real results)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fasa import engine, evaluate

rng = np.random.default_rng(0)
idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="UTC")
cols = ["AAA", "BBB", "CCC", "DDD"]
px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.004, (len(idx), 4)), axis=0)), index=idx, columns=cols)
rates = [0.0003, 0.0001, -0.0001, 0.0002]  # per 8 h; AAA and DDD crowded long
f = pd.DataFrame([(t, c, r) for t in idx[idx.hour % 8 == 0] for c, r in zip(cols, rates, strict=True)], columns=["time", "symbol", "rate"])
w = np.array([-0.25, 0.25, 0.25, -0.25])  # short high funding, long low funding
targets = {t: w for t in idx[idx.hour == 0]}
led = engine.run(px, px, f, targets, engine.Costs(np.full(4, 5e-4), np.full(4, 5e-4)), band=0.02)["ledger"]
d = evaluate.daily_returns(led)
out = ROOT / "results" / "demo"
out.mkdir(parents=True, exist_ok=True)
d.to_csv(out / "demo.csv")
print(f"demo: {len(d)} days, net {100 * d['ret'].sum():+.2f}%, funding {100 * d['funding'].sum():+.2f}%, "
      f"fees {100 * (d['fee'].sum() + d['slip'].sum()):+.2f}%")
