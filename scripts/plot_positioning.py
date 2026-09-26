"""Growth of 1 and drawdown of the against-the-crowd positioning strategy (CROWD_72h), net of all
costs: selection period 2022-2023 and held-out period 2024-01..2026-08, from saved daily returns.

    python scripts/plot_positioning.py   ->  reports/figures/positioning.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main() -> None:
    dev = pd.read_csv(RES / "positioning_dev" / "daily_returns.csv", index_col=0, parse_dates=True)["CROWD_72h"]
    fin = pd.read_csv(RES / "positioning_final" / "daily_CROWD_72h.csv", index_col=0, parse_dates=True)["ret"]
    ret = pd.concat([dev, fin]).sort_index()
    ret.index = ret.index.tz_localize(None)
    wealth = (1 + ret).cumprod()
    dd = wealth / wealth.cummax() - 1
    split = pd.Timestamp("2024-01-01")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5.2), dpi=150, sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    for a in (ax, ax2):
        a.axvspan(split, ret.index[-1], color="0.93", zorder=0)
        a.axvline(split, color="0.4", lw=0.8, ls="--")
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.3)
    ax.plot(wealth.index, wealth, color="#1f77b4", lw=1.4)
    ax.axhline(1, color="0.5", lw=0.6)
    ax.set_ylabel("Growth of 1, net")
    ax.set_title("Against the crowd: short the 5 most net-long perpetuals, long the 5 most net-short\n"
                 "net of fees, slippage and funding; rebalanced every 72 h", loc="left", fontsize=10)
    top = ax.get_ylim()[1]
    ax.text(pd.Timestamp("2022-01-20"), top * 0.97, "selection period", va="top", fontsize=9, color="0.3")
    ax.text(split + pd.Timedelta(days=20), top * 0.97, "held out", va="top", fontsize=9, color="0.3")
    ax2.fill_between(dd.index, dd, 0, color="#d62728", alpha=0.35, lw=0)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "positioning.png"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
