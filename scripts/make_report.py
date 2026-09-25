"""Report tables and figure from results/dev and results/final (writes reports/results.md)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, REP = ROOT / "results", ROOT / "reports"


def pct(x):
    return f"{100 * x:+.1f}%"


def ci(t):
    m, lo, hi = (float(v) for v in str(t).strip("()").split(","))
    return f"{pct(m)} ({pct(lo)} to {pct(hi)})"


def family_section() -> str:
    p = RES / "family" / "summary.json"
    if not p.exists():
        return "Not run."
    d = json.loads(p.read_text())
    rows = [{"period": k, "days": v["days"], "net return": pct(v["ann_return"]), "t (mean)": round(v["t_mean"], 2),
             "Sharpe": round(v["sharpe"], 2),
             "alpha, 2 factors (t)": f"{pct(v['alpha_ann'])} ({v['alpha_t']:.2f})",
             "alpha, 7 factors (t)": f"{pct(v['seven_factor_no_carry']['alpha_ann'])} ({v['seven_factor_no_carry']['alpha_t']:.2f})",
             "alpha, 8 factors incl. carry (t)": f"{pct(v['eight_factor']['alpha_ann'])} ({v['eight_factor']['alpha_t']:.2f})"}
            for k, v in d["periods"].items()]
    e = d["periods"]["2021-2026"]["eight_factor"]
    load = ", ".join(f"{f} {e[f + '_coef']:+.2f} (t {e[f + '_t']:.1f})"
                     for f in ("MKT", "BTC", "SPX", "MOM", "REV", "LOWVOL", "ILLIQ", "CARRY"))
    return pd.DataFrame(rows).to_markdown(index=False) + (
        "\n\n2 factors: equal-weight market and 21-day momentum. 7 factors: market, BTC, S&P 500, "
        "momentum, 1-day reversal, low volatility, illiquidity (trailing volume; not a size factor). "
        "8 factors: the seven plus the BTC spot/perpetual carry trade. Newey-West (10 lags) for the "
        "mean and HAC (10 lags) for the regressions; the funding long/short factor is excluded "
        f"because it is this family. Eight-factor loadings, 2021-2026: {load}."
    )


def positioning_section() -> str:
    p = RES / "positioning_final" / "summary.json"
    if not p.exists():
        return "Not run."
    s = json.loads(p.read_text())
    dev = pd.read_csv(RES / "positioning_dev" / "candidates.csv").set_index("id")
    fin = pd.read_csv(RES / "positioning_final" / "candidates.csv").set_index("id")
    P, m, a = s["selection"]["primary"], s["primary_metrics"], s["attribution_9f"]
    tab = pd.DataFrame({"dev score": dev["score"], "dev folds positive": dev["folds_positive"],
                        "dev Sharpe": dev["full_sharpe"], "final Sharpe": fin["sharpe"],
                        "final return": fin["ann_return"].map(pct)}).round(2)
    load = ", ".join(f"{f} {a[f + '_coef']:+.2f} (t {a[f + '_t']:.1f})"
                     for f in ("MKT", "BTC", "SPX", "MOM", "REV", "LOWVOL", "ILLIQ", "CARRY", "FUND"))
    st = "; ".join(f"{k}: {pct(v['ann_return'])} (Sharpe {v['sharpe']:.2f})" for k, v in s["stresses"].items())
    return (
        f"Protocol `configs/positioning_protocol.yaml` (sha256 `{s['selection']['protocol_sha256'][:12]}`), frozen "
        "before any positioning file was downloaded. Development-selected primary: "
        f"**{P}**{'' if s['selection']['met_requirement'] else ' (flagged: it did not meet the requirement of both development folds positive)'}. "
        f"Final 2024-01..2026-08: net {pct(m['ann_return'])} a year ({pct(m['ann_excess'])} over the T-bill), "
        f"Sharpe {m['sharpe']:.2f}, max drawdown {pct(m['max_drawdown'])}; price {pct(m['price_ann'])}, funding "
        f"{pct(m['funding_ann'])}, fees {pct(m['fee_ann'])}, slippage {pct(m['slip_ann'])}; mean return 95% CI "
        f"{ci(tuple(m['mean_ci']))}; {s['quarters_positive']} of {s['quarters']} quarters positive; rank "
        f"{s['rank_by_sharpe']} of 14 in the final. Nine-factor alpha {pct(a['alpha_ann'])} a year (HAC t "
        f"{a['alpha_t']:.2f}, R2 {a['r2']:.2f}); loadings: {load}. Stresses: {st}.\n\n"
        + tab.to_markdown()
    )


def capacity_section() -> str:
    p = RES / "capacity" / "summary.json"
    if not p.exists():
        return "Not run."
    c = json.loads(p.read_text())
    rows = [{"capital": f"${float(k) / 1e6:.0f}M", "median participation": f"{100 * v['participation_median']:.3f}%",
             "95th pct participation": f"{100 * v['participation_p95']:.2f}%",
             "extra impact / yr": pct(-v["extra_impact_ann"]), "net return / yr": pct(v["net_return_ann"])}
            for k, v in c["by_capital"].items()]
    sc = c["scenario"]
    return pd.DataFrame(rows).to_markdown(index=False) + (
        f"\n\nA sensitivity scenario for {c['primary']} on the final period under an assumed impact model, "
        f"not measured capacity. Model: {sc['model']}, Y = {sc['Y']:.0f}; daily volatility: {sc['sigma_daily']}; "
        f"participation: {sc['participation']}; charged {sc['charged']}; returns per year of {sc['capital_denominator']}; "
        f"{sc['overlap']}. The nine-factor alpha estimate refers to the modeled-cost result, not to these scenarios."
    )


def transfer_section() -> str:
    p = RES / "transfer" / "summary.json"
    if not p.exists():
        return "Not run."
    t = json.loads(p.read_text())
    m, a = t["metrics"], t["attribution_mkt_mom"]
    return (
        f"Rule {t['rule']} (14-day funding lookback, rebalanced every 72 h), frozen before the run, on "
        f"{t['universe_size']} USD-M perpetuals outside the study's pool, 2024-01..2026-08: net return "
        f"{pct(m['ann_return'])} a year ({pct(m['ann_excess'])} over the T-bill), Sharpe {m['sharpe']:.2f}, "
        f"volatility {100 * m['ann_vol']:.1f}%, max drawdown {pct(m['max_drawdown'])}; price {pct(m['price_ann'])}, "
        f"funding {pct(m['funding_ann'])}, fees {pct(m['fee_ann'])}, slippage {pct(m['slip_ann'])}; "
        f"{t['quarters_positive']} of {t['quarters']} quarters positive. Mean return 95% CI "
        f"{ci(tuple(m['mean_ci']))}. Relative to the new universe's market and momentum factors: alpha "
        f"{pct(a['alpha_ann'])} a year (t {a['alpha_t']:.2f}), momentum loading {a['MOM_coef']:.2f}."
    )


def main() -> int:
    dev = pd.read_csv(RES / "dev" / "candidates.csv").set_index("id")
    fin = pd.read_csv(RES / "final" / "candidates.csv").set_index("id")
    s = json.loads((RES / "final" / "summary.json").read_text())
    sel = s["selection"]
    P, N, B, A, BB = (sel["primary"], sel["comparator_same_model_without_funding"], sel["comparator_funding_only"],
                      sel["best_A"], sel["best_B"])
    key = [P, N, B, A, BB]
    lab = {P: "Primary: residual momentum 21 d + funding", N: "Same model without funding", B: "Funding only (3 d, daily)",
           A: "Carry: long spot / short perp, BTC", BB: "Funding only, best in development (3 d, 72 h)"}
    rows = []
    for k in key:
        f, d = fin.loc[k], dev.loc[k]
        rows.append({"strategy": lab[k], "id": k, "dev Sharpe": d["full_sharpe"], "dev return": pct(d["full_ann_return"]),
                     "final Sharpe": f["sharpe"], "final return": pct(f["ann_return"]),
                     "final excess over T-bill": pct(f["ann_excess"]), "final mean return, 95% CI": ci(f["mean_ci"]),
                     "max drawdown": pct(f["max_drawdown"])})
    head = pd.DataFrame(rows)
    dec = pd.DataFrame({lab[k]: {"price": pct(fin.loc[k, "price_ann"]), "funding": pct(fin.loc[k, "funding_ann"]),
                                 "fees": pct(fin.loc[k, "fee_ann"]), "slippage": pct(fin.loc[k, "slip_ann"]),
                                 "net": pct(fin.loc[k, "ann_return"]), "turnover x/yr": f"{fin.loc[k, 'turnover_ann']:.0f}",
                                 "mean gross": f"{fin.loc[k, 'gross_mean']:.2f}", "mean net notional": f"{fin.loc[k, 'net_mean']:+.2f}"}
                        for k in key}).T
    inc = pd.DataFrame([{"comparison": f"{P} minus {v['id']}", "annualized difference, 95% CI": ci(tuple(v["ann_diff"]))}
                        for v in s["increments"].values()])
    st = pd.DataFrame([{"strategy": k, "scenario": n, "return": pct(m["ann_return"]), "Sharpe": round(m["sharpe"], 2)}
                       for k, v in s["stresses"].items() for n, m in v.items()])
    at = pd.DataFrame([{"strategy": k, "alpha/yr": pct(v["alpha_ann"]), "alpha t": round(v["alpha_t"], 2),
                        "MKT": round(v["MKT_coef"], 2), "MOM": round(v["MOM_coef"], 2), "MOM t": round(v["MOM_t"], 1),
                        "FUND": round(v["FUND_coef"], 2), "FUND t": round(v["FUND_t"], 1), "R2": round(v["r2"], 2)}
                       for k, v in s["attribution"].items()])
    devt = dev[["family", "score", "folds_positive", "sharpe_2021", "sharpe_2022", "sharpe_2023", "full_sharpe",
                "full_ann_return"]].round(2)
    finall = fin[["family", "sharpe", "ann_return", "ann_excess", "max_drawdown", "turnover_ann"]].round(3)
    # figure: cumulative net return of the key strategies, development and final
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150, sharey=False)
    for ax, (name, path) in zip(axes, (("Development 2021-2023", RES / "dev" / "daily_returns.csv"),
                                       ("Final 2024-01..2026-08", RES / "final" / "daily_returns.csv")), strict=True):
        dr = pd.read_csv(path, index_col=0, parse_dates=True)
        for k, c in zip(key[:4], ("#2a78d6", "#8a8984", "#eb6834", "#1baf7a"), strict=True):
            ax.plot((1 + dr[k].fillna(0)).cumprod(), color=c, lw=1.3, label=lab[k])
        ax.set_title(name, fontsize=10, loc="left")
        ax.grid(color="#e6e5e0", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Growth of 1, net of costs and funding")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    (REP / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REP / "figures" / "cumulative.png")
    plt.close(fig)
    lines = [
        "# Results: funding-aware relative value (Generation 1)",
        "",
        "Generated by `scripts/make_report.py` from `results/dev/` and `results/final/`. Protocol: "
        f"`configs/alpha_protocol.yaml` (sha256 `{sel['protocol_sha256'][:12]}`), written before any outcome. "
        f"{s['candidates']} registered candidates were run on development (2021-2023); the primary was selected there "
        "and every candidate was then evaluated once on 2024-01..2026-08. Net of taker fees, slippage, realized "
        "funding and delisting exits; returns on total capital; Sharpe of excess returns over the 3-month T-bill.",
        "",
        "## Headline",
        "",
        head.to_markdown(index=False, floatfmt=".2f"),
        "",
        f"The development-selected primary ranked {s['primary_final_rank_by_sharpe']} of {s['candidates']} in the final.",
        "",
        "![cumulative net returns](figures/cumulative.png)",
        "",
        "## Where the money came from (final, per year of capital)",
        "",
        dec.to_markdown(),
        "",
        "## Increment of the primary (paired date-block bootstrap, 20-day blocks)",
        "",
        inc.to_markdown(index=False),
        "",
        "## Stresses (final)",
        "",
        st.to_markdown(index=False),
        "",
        "## Attribution (final; daily returns on zero-cost factors built in the same universe, HAC t)",
        "",
        "MKT: equal-weight market; MOM: 21-day return long/short terciles; FUND: long low / short high "
        "3-day funding, funding included. The intercept is alpha relative to these three factors only.",
        "",
        at.to_markdown(index=False),
        "",
        "## Development: every candidate (fold Sharpe ratios; selection score = mean - 0.5 sd)",
        "",
        devt.to_markdown(),
        "",
        "## Family-level significance (all 12 registered funding long/short configurations, averaged; no selection)",
        "",
        family_section(),
        "",
        "## Generation 2: trader positioning (open interest, long/short ratios, taker flow)",
        "",
        positioning_section(),
        "",
        "## Capacity of the Generation-2 primary",
        "",
        capacity_section(),
        "",
        "## Transfer test on new coins (registered after the final; docs/protocol_amendments.md)",
        "",
        transfer_section(),
        "",
        "## Final: every candidate (evaluated once after selection; not a basis for choosing)",
        "",
        finall.to_markdown(),
        "",
    ]
    (REP / "results.md").write_text("\n".join(lines))
    print("wrote reports/results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
