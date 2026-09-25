"""Family-level significance of the registered funding long/short family (no selection).

The 12 family-B candidates were registered before any outcome. Their equal-weight average over
2021-01..2026-08 (development plus final, net of all costs) involves no choice among them. Tests:
mean return (Newey-West t, 10 lags) and alpha (HAC, 10 lags) against two factor sets, for the
whole period and each half:

* two-factor: the equal-weight market and 21-day momentum of the same universe;
* seven-factor: the eight below without carry;
* eight-factor: the equal-weight market, BTC, the S&P 500 (FRED SP500), 21-day momentum
  (the horizon of Liu, Tsyvinski and Wu's CMOM), 1-day reversal, low volatility (30-day),
  illiquidity (trailing 30-day quote volume; not a size factor, since market caps are not in the
  public archives) and the spot/perpetual carry trade (BTC, unlevered, net of costs).

Cross-sectional factors are zero-cost, dollar-neutral terciles (long 5, short 5 of the 15
eligible coins, equal weight), formed at each 00:00 decision from information before it and held
one day (price returns). The funding long/short factor is excluded: it is this family.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_study  # noqa: E402

from fasa import evaluate, manifest  # noqa: E402


def nw_t(r: pd.Series, lags: int = 10) -> float:
    x = r.dropna().to_numpy()
    e = x - x.mean()
    s = e @ e / len(x)
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / len(x)
    return float(x.mean() / np.sqrt(s / len(x)))


def ls_factor(score: pd.DataFrame, elig: pd.DataFrame, ret: pd.DataFrame, k: int = 5) -> pd.Series:
    """Long the k highest scores, short the k lowest (weights +-1/k), held from day d to d + 1."""
    out = {}
    days = score.index
    for i, d in enumerate(days[:-1]):
        s = score.loc[d].where(elig.loc[d]).dropna()
        if len(s) < 2 * k:
            continue
        o = s.sort_values()
        nd = days[i + 1]
        r = ret.loc[nd]
        out[nd] = float(r.reindex(o.index[-k:]).fillna(0).mean() - r.reindex(o.index[:k]).fillna(0).mean())
    return pd.Series(out)


def sp500(index: pd.DatetimeIndex) -> pd.Series:
    """S&P 500 daily return from FRED (public CSV), zero on days without a close."""
    path = ROOT / "data" / "sp500.parquet"
    if not path.exists():
        u = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500&cosd=2020-06-01&coed=2026-09-01"
        import io

        import requests

        x = pd.read_csv(io.StringIO(requests.get(u, timeout=60).text))
        x = x.set_index(x.columns[0])[x.columns[1]].apply(pd.to_numeric, errors="coerce").dropna()
        x.index = pd.to_datetime(x.index)
        x.to_frame("SP500").to_parquet(path)
    lvl = pd.read_parquet(path)["SP500"]
    lvl.index = pd.to_datetime(lvl.index).tz_localize("UTC")
    # a close on day t is known before 00:00 of day t+1 (UTC): return credited to day t+1
    r = lvl.pct_change()
    r.index = r.index + pd.Timedelta(days=1)
    return r.reindex(index).fillna(0.0)


def factor_set(S: dict, lo, hi) -> pd.DataFrame:
    feat, elig = S["feat"], S["elig"]
    ret = feat["ret"]
    two = run_study.factors(S, lo, hi)
    vol30 = ret.rolling(30, min_periods=20).std()
    f = pd.DataFrame({
        "MKT": two["MKT"],
        "BTC": ret["BTCUSDT"],
        "SPX": sp500(feat["days"]),
        "MOM": ls_factor((1 + ret).rolling(21).apply(np.prod, raw=True) - 1, elig, ret),
        "REV": ls_factor(-ret, elig, ret),
        "LOWVOL": ls_factor(-vol30, elig, ret),
        "ILLIQ": ls_factor(-feat["vol30"], elig, ret),
    })
    carry = pd.read_csv(ROOT / "results" / "dev" / "daily_returns.csv", index_col=0, parse_dates=True)["A_carry_BTC_static"]
    carry2 = pd.read_csv(ROOT / "results" / "final" / "daily_returns.csv", index_col=0, parse_dates=True)["A_carry_BTC_static"]
    c = pd.concat([carry, carry2])
    c.index = pd.to_datetime(c.index, utc=True)
    f["CARRY"] = c
    f.index = pd.to_datetime(f.index, utc=True)
    return f.loc[lo:hi]


def main() -> int:
    tic = time.perf_counter()
    cfg = yaml.safe_load((ROOT / "configs" / "alpha_protocol.yaml").read_text())
    S = run_study.setup(cfg)
    lo, hi = pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-08-31", tz="UTC")
    fac = run_study.factors(S, lo, hi)[["MKT", "MOM"]]
    fac8 = factor_set(S, lo, hi)
    dev = pd.read_csv(ROOT / "results" / "dev" / "daily_returns.csv", index_col=0, parse_dates=True)
    fin = pd.read_csv(ROOT / "results" / "final" / "daily_returns.csv", index_col=0, parse_dates=True)
    cols = [c for c in dev.columns if c.startswith("B_fund")]
    fam = pd.concat([dev[cols], fin[cols]]).mean(axis=1).dropna()
    fam.index = pd.to_datetime(fam.index, utc=True)
    fac.index = pd.to_datetime(fac.index, utc=True)
    out = {"configs": cols, "periods": {}}
    for name, a, b in (("2021-2026", "2021-01-01", "2026-08-31"), ("2021-2023", "2021-01-01", "2023-12-31"),
                       ("2024-2026", "2024-01-01", "2026-08-31")):
        y = fam.loc[a:b]
        reg = evaluate.ols_hac(y, fac, lags=10)
        r8 = evaluate.ols_hac(y, fac8, lags=10)
        r7 = evaluate.ols_hac(y, fac8.drop(columns=["CARRY"]), lags=10)
        out["periods"][name] = {"days": len(y), "ann_return": float(y.mean() * 365), "t_mean": nw_t(y),
                                "sharpe": float(y.mean() / y.std() * np.sqrt(365)), "alpha_ann": reg["alpha_ann"],
                                "alpha_t": reg["alpha_t"], "MKT": reg["MKT_coef"], "MOM": reg["MOM_coef"],
                                "MOM_t": reg["MOM_t"],
                                "seven_factor_no_carry": dict(r7.items()),
                                "eight_factor": dict(r8.items())}
    res = ROOT / "results" / "family"
    res.mkdir(parents=True, exist_ok=True)
    fam.to_frame("B_family_mean").to_csv(res / "daily.csv")
    fac8.to_csv(res / "factors.csv")
    out["factor_correlations"] = fac8.corrwith(fam.reindex(fac8.index)).to_dict()
    (res / "summary.json").write_text(json.dumps(out, indent=1))
    manifest.write(res / "manifest.json", {  # outputs: daily family returns, factors, summary
        "study": "funding-aware-stat-arb / family-level significance", "command": "python scripts/family_significance.py",
        "inputs": {"sp500_fred_sha256": manifest.sha256_file(ROOT / "data" / "sp500.parquet")},
        "source_tree_sha256": manifest.tree_hash(ROOT),
        "protocol_sha256": manifest.sha256_file(ROOT / "configs" / "alpha_protocol.yaml"),
        "runtime_seconds": time.perf_counter() - tic}, [res / "daily.csv", res / "factors.csv", res / "summary.json"])
    for k, v in out["periods"].items():
        e = v["eight_factor"]
        print(k, "2F alpha", round(v["alpha_ann"], 3), "t", round(v["alpha_t"], 2), "| 8F alpha", round(e["alpha_ann"], 3),
              "t", round(e["alpha_t"], 2), "R2", round(e["r2"], 2), {f: (round(e[f + "_coef"], 2), round(e[f + "_t"], 1))
              for f in ("MKT", "BTC", "SPX", "MOM", "REV", "LOWVOL", "ILLIQ", "CARRY")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
