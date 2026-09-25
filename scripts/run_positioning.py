"""Generation 2: trader-positioning signals (configs/positioning_protocol.yaml).

    python scripts/run_positioning.py dev
    python scripts/run_positioning.py final --i-understand-this-is-the-final-evaluation

Same ledger, costs, universe, portfolio construction and limits as Generation 1. The final mode
evaluates every candidate once and attributes the development-selected primary to nine factors:
the eight of scripts/family_significance.py plus the 3-day funding long/short.
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
import family_significance as fs
import run_study

from fasa import engine, evaluate, manifest, strategies

SIGNALS = ["OI", "TOP", "CROWD", "GAP", "FLOW_cont", "FLOW_rev", "COMPOSITE"]


def zs(x: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    return strategies.zscore(x.where(elig))


def scores(S: dict) -> dict[str, pd.DataFrame]:
    feat, elig = S["feat"], S["elig"]
    m = pd.read_parquet(ROOT / "data" / "metrics_daily.parquet")
    wide = {c: m.pivot_table(index="date", columns="symbol", values=c).reindex(index=feat["days"], columns=S["syms"])
            for c in ("oi_value", "top_pos_ratio", "all_ratio", "taker_log_mean")}
    lp = np.log(feat["px"])
    loi = np.log(wide["oi_value"].where(wide["oi_value"] > 0))
    raw = {
        "OI": -((loi - loi.shift(7)) - (lp - lp.shift(7))),
        "TOP": np.log(wide["top_pos_ratio"].where(wide["top_pos_ratio"] > 0)),
        "CROWD": -np.log(wide["all_ratio"].where(wide["all_ratio"] > 0)),
    }
    raw["GAP"] = raw["TOP"] + raw["CROWD"]  # log top ratio - log all-account ratio
    flow = wide["taker_log_mean"].rolling(3, min_periods=3).mean()
    raw["FLOW_cont"], raw["FLOW_rev"] = flow, -flow
    out = {k: zs(v, elig) for k, v in raw.items()}
    out["COMPOSITE"] = (out["OI"] + out["CROWD"] + out["GAP"]) / 3
    return out


def targets(sc: pd.DataFrame, S: dict, every: str, lo, hi) -> dict:
    days = [d for d in S["feat"]["days"] if lo <= d <= hi]
    days = days if every == "24h" else days[::3]
    return {d: strategies.long_short(sc.loc[d].where(S["elig"].loc[d]), S["beta"].loc[d]).to_numpy() for d in days}


def run(sc, S, every, lo, hi, cost_mult=1.0, delay=1) -> pd.DataFrame:
    p = S["panel"]["perp"]
    costs = engine.Costs(S["ls_costs"].fee * cost_mult, S["ls_costs"].slip * cost_mult)
    led = engine.run(p["open"], p["close"], S["panel"]["funding"], targets(sc, S, every, lo, hi), costs,
                     delay=delay, band=0.02, start=lo, end=hi + pd.Timedelta(hours=23))["ledger"]
    return evaluate.daily_returns(led)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    tic = time.perf_counter()
    cfg = yaml.safe_load((ROOT / "configs" / "alpha_protocol.yaml").read_text())
    pcfg = yaml.safe_load((ROOT / "configs" / "positioning_protocol.yaml").read_text())
    S = run_study.setup(cfg)
    rf = evaluate.rf_daily(S["dtb3"], S["feat"]["days"])
    sc = scores(S)
    cands = [(s, e) for s in SIGNALS for e in ("24h", "72h")]
    out = ROOT / "results" / f"positioning_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    if mode == "dev":
        lo, hi = (pd.Timestamp(x, tz="UTC") for x in pcfg["calendar"]["development"])
        rows, dr = [], {}
        for s, e in cands:
            d = run(sc[s], S, e, lo, hi)
            dr[f"{s}_{e}"] = d["ret"]
            f = [evaluate.metrics(d.loc[a:b], rf) for a, b in (("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31"))]
            sh = np.array([x["sharpe"] for x in f])
            full = evaluate.metrics(d, rf)
            rows.append({"id": f"{s}_{e}", "score": float(sh.mean() - 0.5 * sh.std()),
                         "folds_positive": int(sum(x["ann_return"] > 0 for x in f)),
                         "sharpe_2022": sh[0], "sharpe_2023": sh[1], **{f"full_{k}": v for k, v in full.items()}})
            print(f"{s}_{e:<4} score {rows[-1]['score']:+.2f} folds+ {rows[-1]['folds_positive']} Sharpe {full['sharpe']:+.2f} ann {full['ann_return']:+.3f}", flush=True)
        tab = pd.DataFrame(rows)
        tab.to_csv(out / "candidates.csv", index=False)
        pd.DataFrame(dr).to_csv(out / "daily_returns.csv")
        ok = tab[tab["folds_positive"] == 2].sort_values("score", ascending=False)
        prim = (ok if len(ok) else tab.sort_values("score", ascending=False)).iloc[0]["id"]
        sel = {"primary": prim, "met_requirement": bool(len(ok)),
               "protocol_sha256": manifest.sha256_file(ROOT / "configs" / "positioning_protocol.yaml"),
               "written_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
        (out / "selection.json").write_text(json.dumps(sel, indent=1))
        print(json.dumps(sel, indent=1))
        files = [out / "candidates.csv", out / "daily_returns.csv", out / "selection.json"]
    else:
        if "--i-understand-this-is-the-final-evaluation" not in sys.argv:
            sys.exit("final evaluation needs --i-understand-this-is-the-final-evaluation")
        sel = json.loads((ROOT / "results" / "positioning_dev" / "selection.json").read_text())
        lo, hi = (pd.Timestamp(x, tz="UTC") for x in pcfg["calendar"]["final"])
        rows, dr = [], {}
        for s, e in cands:
            d = run(sc[s], S, e, lo, hi)
            dr[f"{s}_{e}"] = d
            m = evaluate.metrics(d, rf)
            m["mean_ci"] = evaluate.ci_mean(d["ret"])
            rows.append({"id": f"{s}_{e}", **m})
            print(f"{s}_{e:<4} Sharpe {m['sharpe']:+.2f} ann {m['ann_return']:+.3f}", flush=True)
        tab = pd.DataFrame(rows)
        tab.to_csv(out / "candidates.csv", index=False)
        pd.DataFrame({k: v["ret"] for k, v in dr.items()}).to_csv(out / "daily_returns.csv")
        P = sel["primary"]
        sig, ev = P.rsplit("_", 1)
        fac = fs.factor_set(S, lo, hi)
        fund = run_study.factors(S, lo, hi)["FUND"]
        fund.index = pd.to_datetime(fund.index, utc=True)
        fac["FUND"] = fund
        y = dr[P]["ret"]
        att9 = evaluate.ols_hac(y, fac, lags=10)
        att8 = evaluate.ols_hac(y, fac.drop(columns=["FUND"]), lags=10)
        stress = {"costs_x2": evaluate.metrics(run(sc[sig], S, ev, lo, hi, cost_mult=2.0), rf),
                  "delay_2h": evaluate.metrics(run(sc[sig], S, ev, lo, hi, delay=2), rf),
                  # the positioning snapshot one full day older (bounds an unverified publication delay)
                  "positioning_lag_1d": evaluate.metrics(run(sc[sig].shift(1), S, ev, lo, hi), rf)}
        q = y.groupby(y.index.tz_localize(None).to_period("Q")).sum()
        summ = {"selection": sel, "primary_metrics": tab.set_index("id").loc[P].to_dict(),
                "attribution_9f": att9, "attribution_8f": att8, "stresses": stress,
                "quarters_positive": int((q > 0).sum()), "quarters": len(q),
                "rank_by_sharpe": int(tab.sort_values("sharpe", ascending=False)["id"].tolist().index(P) + 1)}
        (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
        dr[P].to_csv(out / f"daily_{P}.csv")
        fac.to_csv(out / "factors.csv")
        print(json.dumps({"primary": P, "sharpe": summ["primary_metrics"]["sharpe"],
                          "ann": summ["primary_metrics"]["ann_return"], "ci": summ["primary_metrics"]["mean_ci"],
                          "alpha9": att9["alpha_ann"], "t9": att9["alpha_t"]}, indent=1, default=str))
        files = [out / "candidates.csv", out / "daily_returns.csv", out / "summary.json", out / "factors.csv"]
    manifest.write(out / "manifest.json", {
        "study": "funding-aware-stat-arb / Generation 2 positioning", "mode": mode,
        "command": f"python scripts/run_positioning.py {mode}" + (" --i-understand-this-is-the-final-evaluation" if mode == "final" else ""),
        "source_tree_sha256": manifest.tree_hash(ROOT),
        "protocol_sha256": manifest.sha256_file(ROOT / "configs" / "positioning_protocol.yaml"),
        "metrics_ledger_sha256": manifest.sha256_file(ROOT / "data" / "download_ledger_metrics.json"),
        "metrics_daily_sha256": manifest.sha256_file(ROOT / "data" / "metrics_daily.parquet"),
        "runtime_seconds": time.perf_counter() - tic}, files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
