"""Capacity of the Generation-2 primary (CROWD_72h) on the final period, 2024-01..2026-08.

For capital A, every rebalance's traded notional per coin is |target weight - current weight| x A
(from the ledger's own weights). Participation = traded notional / the coin's quote volume over
the 24 hours before the fill. Extra market impact beyond the modeled 5 bps slippage follows the
square-root law, cost = Y x sigma_daily x sqrt(participation) per unit traded, with Y = 1 (the
prefactor is of order one in Toth et al. 2011, Physical Review X 1, 021006, and in Donier and
Bonart 2015, Market Microstructure and Liquidity 1(2), for Bitcoin), charged on every one-way
trade. Reported: participation quantiles and the net return after this extra cost, by capital. A
sensitivity scenario under an assumed impact model, not measured capacity; nothing is re-selected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_positioning as rp
import run_study

from fasa import engine, evaluate, manifest

CAPITAL = [1e6, 5e6, 10e6, 25e6, 50e6, 100e6]
Y = 1.0


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "alpha_protocol.yaml").read_text())
    pcfg = yaml.safe_load((ROOT / "configs" / "positioning_protocol.yaml").read_text())
    S = run_study.setup(cfg)
    lo, hi = (pd.Timestamp(x, tz="UTC") for x in pcfg["calendar"]["final"])
    sel = json.loads((ROOT / "results" / "positioning_dev" / "selection.json").read_text())
    sig, ev = sel["primary"].rsplit("_", 1)
    sc = rp.scores(S)
    tg = rp.targets(sc[sig], S, ev, lo, hi)
    p = S["panel"]["perp"]
    res = engine.run(p["open"], p["close"], S["panel"]["funding"], tg, S["ls_costs"], delay=1, band=0.02,
                     start=lo, end=hi + pd.Timedelta(hours=23))
    led = res["ledger"]
    base = evaluate.daily_returns(led)
    qv24 = p["qv"].rolling(24, min_periods=12).sum().shift(1)  # quote volume in the 24 h before each hour
    sig_d = S["feat"]["ret"].rolling(30, min_periods=20).std()  # daily volatility known at the decision
    # weights before and after each fill from the ledger-consistent reconstruction
    O = p["open"]
    rows = []
    w_cur = pd.Series(0.0, index=O.columns)
    for t, w in sorted(tg.items()):
        fill = t + pd.Timedelta(hours=1)
        w_new = pd.Series(w, index=O.columns)
        dw = (w_new - w_cur).where((w_new - w_cur).abs() > 0.02, 0.0)
        for s in dw.index[dw != 0]:
            rows.append({"time": fill, "symbol": s, "dw": abs(float(dw[s])), "qv24": float(qv24.loc[fill, s]),
                         "sigma": float(sig_d.loc[t.normalize(), s]) if t.normalize() in sig_d.index else np.nan})
        w_cur = w_cur.where(dw == 0, w_new)
    tr = pd.DataFrame(rows).dropna()
    days = len(base)
    out = {"primary": sel["primary"], "trades": len(tr), "by_capital": {},
           "scenario": {"model": "square-root impact, cost per unit traded = Y x sigma_daily x sqrt(participation)",
                        "Y": Y, "sigma_daily": "std of daily returns over the 30 days before the decision",
                        "participation": "one-way traded notional / quote volume in the 24 h before the fill",
                        "charged": "on every one-way trade (entries, exits and resizing)",
                        "capital_denominator": "the scenario capital A (returns are per year of A)",
                        "overlap": "additive to the modeled 5 bps slippage, so it double-counts part of the "
                                   "spread cost (conservative)",
                        "status": "sensitivity scenario under an assumed impact model, not measured capacity"}}
    for A in CAPITAL:
        part = tr["dw"] * A / tr["qv24"]
        extra = (Y * tr["sigma"] * np.sqrt(part) * tr["dw"]).sum()  # fraction of capital lost, whole period
        ann_extra = float(extra / days * 365)
        net = float(base["ret"].mean() * 365 - ann_extra)
        out["by_capital"][f"{A:.0f}"] = {
            "participation_median": float(part.median()), "participation_p95": float(part.quantile(0.95)),
            "participation_max": float(part.max()), "extra_impact_ann": ann_extra, "net_return_ann": net,
        }
        print(f"${A / 1e6:>5.0f}M: participation median {100 * part.median():.3f}% p95 {100 * part.quantile(0.95):.3f}%"
              f"  extra impact {100 * ann_extra:.2f}%/yr  net {100 * net:+.1f}%/yr")
    out["net_return_ann_modeled_costs"] = float(base["ret"].mean() * 365)
    res_dir = ROOT / "results" / "capacity"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "summary.json").write_text(json.dumps(out, indent=1))
    manifest.write(res_dir / "manifest.json", {
        "study": "funding-aware-stat-arb / capacity of the Generation-2 primary", "command": "python scripts/capacity.py",
        "source_tree_sha256": manifest.tree_hash(ROOT),
        "protocol_sha256": manifest.sha256_file(ROOT / "configs" / "positioning_protocol.yaml"),
        "metrics_daily_sha256": manifest.sha256_file(ROOT / "data" / "metrics_daily.parquet")}, [res_dir / "summary.json"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
