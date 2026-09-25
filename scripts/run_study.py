"""Run the registered candidates (configs/alpha_protocol.yaml).

    python scripts/run_study.py dev      # all 36 candidates on 2021-2023, fold scores, selection
    python scripts/run_study.py final --i-understand-this-is-the-final-evaluation

The final mode reads the frozen selection written by the development run and evaluates every
candidate once on 2024-01..2026-08 (the development-selected primary is the reported one).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fasa import engine, evaluate, manifest, ml, strategies
from fasa.panel import load
from fasa.signals import daily, eligible, residual


def candidates() -> list[dict]:
    out = []
    for a in ("BTC", "ETH", "BTCETH"):
        for rule in ("static", "cond"):
            out.append({"id": f"A_carry_{a}_{rule}", "family": "A", "assets": a, "rule": rule})
    for L in (1, 3, 7, 14):
        for ev in ("8h", "24h", "72h"):
            out.append({"id": f"B_fund_L{L}_{ev}", "family": "B", "L": L, "every": ev})
    for H in (1, 3, 7, 21):
        for mech in ("reversal", "momentum"):
            for wf in (False, True):
                out.append({"id": f"C_res_H{H}_{mech[:3]}_{'fund' if wf else 'nofund'}", "family": "C",
                            "H": H, "mech": mech, "funding": wf})
    for wf in (False, True):
        out.append({"id": f"C_lgbm_{'fund' if wf else 'nofund'}", "family": "C", "ml": True, "funding": wf})
    return out


def setup(cfg: dict) -> dict:
    panel = load(ROOT / "data" / "panel")
    feat = daily(panel)
    elig = eligible(feat)
    _, beta, mkt = residual(feat["ret"], elig, 1)
    dtb3 = pd.read_parquet(ROOT / "data" / "dtb3.parquet")["DTB3"]
    perp = panel["perp"]
    syms = list(perp["open"].columns)
    c = cfg["execution"]["costs_bps_per_side"]
    slip = np.array([c["slippage_btc_eth"] if s in ("BTCUSDT", "ETHUSDT") else c["slippage_other"] for s in syms]) / 1e4
    ls_costs = engine.Costs(fee=np.full(len(syms), c["perp_taker_fee"] / 1e4), slip=slip)
    carry_cols = ["spot_BTC", "spot_ETH", "perp_BTC", "perp_ETH"]
    co = pd.concat([panel["spot"]["open"][["BTCUSDT", "ETHUSDT"]], perp["open"][["BTCUSDT", "ETHUSDT"]]], axis=1)
    cc = pd.concat([panel["spot"]["close"][["BTCUSDT", "ETHUSDT"]], perp["close"][["BTCUSDT", "ETHUSDT"]]], axis=1)
    co.columns = cc.columns = carry_cols
    fund = panel["funding"].copy()
    fund_carry = fund[fund["symbol"].isin(["BTCUSDT", "ETHUSDT"])].copy()
    fund_carry["symbol"] = "perp_" + fund_carry["symbol"].str.replace("USDT", "")
    carry_costs = engine.Costs(
        fee=np.array([c["spot_taker_fee"], c["spot_taker_fee"], c["perp_taker_fee"], c["perp_taker_fee"]]) / 1e4,
        slip=np.full(4, c["slippage_btc_eth"] / 1e4),
    )
    return dict(panel=panel, feat=feat, elig=elig, beta=beta, mkt=mkt, dtb3=dtb3, syms=syms,
                ls_costs=ls_costs, carry=(co, cc, fund_carry, carry_costs, carry_cols))


def targets_for(c: dict, S: dict, window: tuple[pd.Timestamp, pd.Timestamp]) -> dict:
    lo, hi = window
    if c["family"] == "A":
        assets = ["BTC", "ETH"] if c["assets"] == "BTCETH" else [c["assets"]]
        t = strategies.carry(S["panel"], S["feat"], assets, c["rule"] == "cond", S["carry"][4])
    elif c["family"] == "B":
        t = strategies.funding_xs(S["panel"], S["feat"], S["elig"], c["L"], c["every"], S["beta"])
    elif c.get("ml"):
        sc = ml.lgbm_scores(S["panel"], S["feat"], S["elig"], c["funding"], lo, hi)
        t = {}
        for d in sc.index:
            e = S["elig"].loc[d]
            s = sc.loc[d].reindex(S["syms"]).where(e)
            t[d] = strategies.long_short(s, S["beta"].loc[d]).to_numpy()
    else:
        t = strategies.residual_funding(S["panel"], S["feat"], S["elig"], c["H"], c["mech"], c["funding"])
    return {k: v for k, v in t.items() if lo <= k <= hi}


def run_one(c: dict, S: dict, window, cost_mult: float = 1.0, delay: int = 1, leg_lag: int = 0) -> pd.DataFrame:
    lo, hi = window
    tg = targets_for(c, S, window)
    end = hi + pd.Timedelta(hours=23)
    if c["family"] == "A":
        co, cc, fc, costs, cols = S["carry"]
        costs = engine.Costs(costs.fee * cost_mult, costs.slip * cost_mult)
        res = engine.run(co, cc, fc, tg, costs, delay=delay, band=0.05, perp_mask=np.array([False, False, True, True]),
                         start=lo, end=end, pairs=[(0, 2), (1, 3)], leg_lag=leg_lag)
    else:
        costs = engine.Costs(S["ls_costs"].fee * cost_mult, S["ls_costs"].slip * cost_mult)
        p = S["panel"]["perp"]
        res = engine.run(p["open"], p["close"], S["panel"]["funding"], tg, costs, delay=delay, band=0.02,
                         start=lo, end=end)
    return evaluate.daily_returns(res["ledger"])


def fold_table(d: pd.DataFrame, rf: pd.Series, folds: list[tuple[str, str]]) -> list[dict]:
    out = []
    for a, b in folds:
        m = evaluate.metrics(d.loc[a:b], rf)
        m["fold"] = a[:4]
        out.append(m)
    return out


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    cfg = yaml.safe_load((ROOT / "configs" / "alpha_protocol.yaml").read_text())
    tic = time.perf_counter()
    S = setup(cfg)
    rf = evaluate.rf_daily(S["dtb3"], S["feat"]["days"])
    out = ROOT / "results" / mode
    out.mkdir(parents=True, exist_ok=True)
    cands = candidates()
    if mode == "dev":
        lo, hi = (pd.Timestamp(x, tz="UTC") for x in cfg["calendar"]["development"])
        folds = [("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31")]
        rows, daily_all = [], {}
        for c in cands:
            d = run_one(c, S, (lo, hi))
            daily_all[c["id"]] = d["ret"]
            ft = fold_table(d, rf, folds)
            sh = np.array([f["sharpe"] for f in ft])
            pos = int(sum(f["ann_return"] > 0 for f in ft))
            full = evaluate.metrics(d, rf)
            rows.append({"id": c["id"], "family": c["family"], "score": float(np.nanmean(sh) - 0.5 * np.nanstd(sh)),
                         "folds_positive": pos, **{f"sharpe_{f['fold']}": f["sharpe"] for f in ft},
                         **{f"full_{k}": v for k, v in full.items()}})
            print(f"{c['id']:<28} score {rows[-1]['score']:+.2f}  folds+ {pos}  full Sharpe {full['sharpe']:+.2f}  ann {full['ann_return']:+.3f}", flush=True)
        tab = pd.DataFrame(rows)
        tab.to_csv(out / "candidates.csv", index=False)
        pd.DataFrame(daily_all).to_csv(out / "daily_returns.csv")
        cfund = tab[(tab["family"] == "C") & tab["id"].str.endswith("_fund")].sort_values("score", ascending=False)
        ok = cfund[cfund["folds_positive"] >= 2]
        primary = (ok if len(ok) else cfund).iloc[0]["id"]
        comp_nofund = primary.replace("_fund", "_nofund")
        pc = next(c for c in cands if c["id"] == primary)
        comp_b = "B_fund_L3_24h"
        best = {f: tab[tab["family"] == f].sort_values("score", ascending=False).iloc[0]["id"] for f in ("A", "B")}
        sel = {"primary": primary, "primary_met_requirement": bool(len(ok) > 0),
               "comparator_same_model_without_funding": comp_nofund, "comparator_funding_only": comp_b,
               "best_A": best["A"], "best_B": best["B"], "primary_config": pc,
               "protocol_sha256": hashlib.sha256((ROOT / "configs" / "alpha_protocol.yaml").read_bytes()).hexdigest(),
               "written_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
        (out / "selection.json").write_text(json.dumps(sel, indent=1, default=str))
        print(json.dumps(sel, indent=1, default=str))
        write_manifest(out, "dev", cfg, [out / "candidates.csv", out / "daily_returns.csv", out / "selection.json"], tic)
    else:
        if "--i-understand-this-is-the-final-evaluation" not in sys.argv:
            sys.exit("final evaluation needs --i-understand-this-is-the-final-evaluation")
        sel = json.loads((ROOT / "results" / "dev" / "selection.json").read_text())
        lo, hi = (pd.Timestamp(x, tz="UTC") for x in cfg["calendar"]["final"])
        rows, daily_all = [], {}
        for c in cands:
            d = run_one(c, S, (lo, hi))
            daily_all[c["id"]] = d
            m = evaluate.metrics(d, rf)
            m["mean_ci"] = evaluate.ci_mean(d["ret"])
            rows.append({"id": c["id"], "family": c["family"], **m})
            print(f"{c['id']:<28} Sharpe {m['sharpe']:+.2f}  ann {m['ann_return']:+.3f}", flush=True)
        tab = pd.DataFrame(rows)
        tab.to_csv(out / "candidates.csv", index=False)
        pd.DataFrame({k: v["ret"] for k, v in daily_all.items()}).to_csv(out / "daily_returns.csv")
        for k in (sel["primary"], sel["comparator_same_model_without_funding"], sel["comparator_funding_only"],
                  sel["best_A"], sel["best_B"]):
            daily_all[k].to_csv(out / f"daily_{k}.csv")
        # paired increments and stresses for the primary and the benchmarks
        P = daily_all[sel["primary"]]["ret"]
        inc = {}
        for name in ("comparator_same_model_without_funding", "comparator_funding_only"):
            Q = daily_all[sel[name]]["ret"]
            diff = (P - Q).dropna()
            inc[name] = {"id": sel[name], "ann_diff": evaluate.ci_mean(diff)}
        stress = {}
        for k in {sel["primary"], sel["best_A"], sel["best_B"]}:
            c = next(x for x in cands if x["id"] == k)
            stress[k] = {
                "costs_x2": evaluate.metrics(run_one(c, S, (lo, hi), cost_mult=2.0), rf),
                "delay_2h": evaluate.metrics(run_one(c, S, (lo, hi), delay=2), rf),
            }
            if c["family"] == "A":
                stress[k]["leg_lag_1h"] = evaluate.metrics(run_one(c, S, (lo, hi), leg_lag=1), rf)
        # attribution: equal-weight market, 21-day momentum and 3-day funding long/short (zero cost)
        fac = factors(S, lo, hi)
        attr = {k: evaluate.ols_hac(daily_all[k]["ret"], fac) for k in {sel["primary"], sel["best_A"], sel["best_B"]}}
        rank = int(tab.sort_values("sharpe", ascending=False)["id"].tolist().index(sel["primary"]) + 1)
        summ = {"selection": sel, "increments": inc, "stresses": stress, "attribution": attr,
                "primary_final_rank_by_sharpe": rank, "candidates": len(cands),
                "runtime_seconds": time.perf_counter() - tic}
        (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
        fac.to_csv(out / "factors.csv")
        write_manifest(out, "final", cfg, sorted(out.glob("*.csv")) + [out / "summary.json"], tic)
        print(json.dumps({"primary": sel["primary"], "increments": inc, "rank": rank}, indent=1, default=str))
    return 0


def write_manifest(out: Path, mode: str, cfg: dict, outputs: list[Path], tic: float) -> None:
    manifest.write(out / "manifest.json", {
        "study": "funding-aware-stat-arb / Generation 1",
        "mode": mode,
        "command": f"python scripts/run_study.py {mode}" + (" --i-understand-this-is-the-final-evaluation" if mode == "final" else ""),
        "source_tree_sha256": manifest.tree_hash(ROOT),
        "protocol_sha256": manifest.sha256_file(ROOT / "configs" / "alpha_protocol.yaml"),
        "data_ledger_sha256": manifest.sha256_file(ROOT / "data" / "download_ledger.json"),
        "pool_sha256": manifest.sha256_file(ROOT / "data" / "pool.json"),
        "window": cfg["calendar"]["development" if mode == "dev" else "final"],
        "runtime_seconds": time.perf_counter() - tic,
    }, outputs)


def factors(S: dict, lo, hi) -> pd.DataFrame:
    """Daily zero-cost factor returns inside the same universe: EW market; long/short tercile by
    21-day return (momentum); long low / short high 3-day funding, funding included."""
    from fasa.signals import funding_per_day

    feat, elig = S["feat"], S["elig"]
    ret = feat["ret"]
    fund_day = funding_per_day(S["panel"], feat["days"] + pd.Timedelta(days=1), 1).shift(0)
    fund_day.index = feat["days"]  # funding a long paid over day d (known at d+1)
    mkt = ret.where(elig.shift(1)).mean(axis=1)
    mom = (1 + ret).rolling(21).apply(np.prod, raw=True) - 1
    f3 = funding_per_day(S["panel"], feat["days"], 3)
    out = {"MKT": mkt}
    for name, score in (("MOM", mom), ("FUND", -f3)):
        r = []
        for i, d in enumerate(feat["days"][:-1]):
            w = strategies.long_short(score.loc[d].where(elig.loc[d]), S["beta"].loc[d], w_cap=1.0, net_cap=1.0)
            nd = feat["days"][i + 1]
            gain = (w * ret.loc[nd].fillna(0.0)).sum() - (w * fund_day.loc[d].fillna(0.0)).sum()
            r.append((nd, gain))
        out[name] = pd.Series(dict(r))
    f = pd.DataFrame(out)
    f.index = pd.to_datetime(f.index)
    return f.loc[lo:hi]


if __name__ == "__main__":
    sys.exit(main())
