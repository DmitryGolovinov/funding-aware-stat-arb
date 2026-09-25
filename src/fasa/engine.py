"""Self-financing hourly ledger for linear USDT perpetuals and spot.

Convention (one USDT account): buying q units at price P moves q*P out of cash into the
position; equity = cash + sum_i q_i * P_i marked at each hour's open. For a linear perpetual this
equals collateral plus unrealized P&L. Cash earns nothing here (excess returns over the T-bill
rate are computed separately). Per hour, in this order:

1. mark the open (prices carried forward over missing bars; a symbol with no later bar is
   delisted: its position is closed at its last close, with costs);
2. apply every funding settlement in [t, t+1h) to the position held at that time: settlements
   are stamped at the start of the hour (hh:00:00.00x), before an order at this open can fill, so
   the pre-trade position pays or receives: cash -= q * price * rate (a long pays when the rate is
   positive); the price is this hour's open;
3. execute orders scheduled for this hour at the open, paying fee + slippage on traded notional.

Orders come from ``targets``: weights set at decision hours from information strictly before
the decision and executed ``delay`` hours later. A name is traded only when its weight moves by
more than ``band``. Execution is modeled at the bar open, not observed fills.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Costs:
    fee: np.ndarray  # per-side fee, fraction of notional, per instrument
    slip: np.ndarray  # per-side slippage, fraction of notional

    @property
    def total(self) -> np.ndarray:
        return self.fee + self.slip


def run(
    open_: pd.DataFrame,
    close: pd.DataFrame,
    funding: pd.DataFrame,
    targets: dict[pd.Timestamp, np.ndarray],
    costs: Costs,
    delay: int = 1,
    band: float = 0.0,
    capital: float = 1_000_000.0,
    perp_mask: np.ndarray | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    pairs: list[tuple[int, int]] = (),
    leg_lag: int = 0,
) -> dict:
    """``pairs``: (spot column, perp column) carry legs held in matched units (perp = -spot);
    both legs trade whenever either would. ``leg_lag``: the perp leg of a pair fills this many
    hours after the spot leg (legging stress)."""
    idx = open_.index
    t0 = idx.searchsorted(start) if start is not None else 0
    t1 = idx.searchsorted(end, side="right") if end is not None else len(idx)
    O = open_.to_numpy(float)
    C = close.to_numpy(float)
    n = O.shape[1]
    perp = np.ones(n, bool) if perp_mask is None else perp_mask
    last_bar = np.array([np.max(np.flatnonzero(~np.isnan(O[:, j]))) if (~np.isnan(O[:, j])).any() else -1 for j in range(n)])
    col = {s: j for j, s in enumerate(open_.columns)}
    fund = funding[funding["symbol"].isin(open_.columns)]
    fh = ((fund["time"] - idx[0]) // pd.Timedelta(hours=1)).to_numpy()
    fsym = fund["symbol"].map(col).to_numpy()
    frate = fund["rate"].to_numpy(float)
    by_hour: dict[int, list[tuple[int, float]]] = {}
    for h, j, r in zip(fh, fsym, frate, strict=True):
        by_hour.setdefault(int(h), []).append((int(j), float(r)))
    orders = {idx.searchsorted(t) + delay: w for t, w in targets.items()}
    q = np.zeros(n)
    pending: dict[int, list[tuple[int, float]]] = {}
    cash = capital
    px = np.full(n, np.nan)
    rows = []
    for t in range(t0, t1):
        ok = ~np.isnan(O[t])
        px = np.where(ok, O[t], px)
        comp = {"price": 0.0, "fee": 0.0, "slip": 0.0, "funding": 0.0, "traded": 0.0}
        # delisting: no bar at or after t for a held symbol -> close at its last close
        gone = (last_bar < t) & (q != 0)
        for j in np.flatnonzero(gone):
            p = C[last_bar[j], j]
            cash += q[j] * p - abs(q[j]) * p * costs.total[j]
            comp["traded"] += abs(q[j]) * p
            comp["fee"] -= abs(q[j]) * p * costs.fee[j]
            comp["slip"] -= abs(q[j]) * p * costs.slip[j]
            q[j] = 0.0
        for j, r in by_hour.get(t, ()):
            if perp[j] and q[j] != 0 and not np.isnan(px[j]):
                f = -q[j] * px[j] * r
                cash += f
                comp["funding"] += f
        eq = cash + np.nansum(q * px)
        if t in orders:
            w = orders[t]
            cur = np.where(np.isnan(px), 0.0, q * px) / eq if eq > 0 else np.zeros(n)
            trade = ok & (np.abs(w - cur) > band) & (last_bar >= t)
            tq = np.where(trade, w * eq / np.where(ok, O[t], 1.0) - q, 0.0)
            for sj, pj in pairs:
                if (trade[sj] or trade[pj]) and ok[sj] and ok[pj]:
                    tq[sj] = w[sj] * eq / O[t, sj] - q[sj]
                    target_perp = -(q[sj] + tq[sj])  # matched units
                    if leg_lag:
                        pending.setdefault(t + leg_lag, []).append((pj, target_perp))
                        tq[pj] = 0.0
                    else:
                        tq[pj] = target_perp - q[pj]
            notional = np.abs(tq) * np.where(ok, O[t], 0.0)
            cash -= float(np.sum(tq * np.where(ok, O[t], 0.0))) + float(np.sum(notional * costs.total))
            comp["traded"] += float(np.sum(notional))
            comp["fee"] -= float(np.sum(notional * costs.fee))
            comp["slip"] -= float(np.sum(notional * costs.slip))
            q = q + tq
        for pj, target in pending.pop(t, ()):
            if ok[pj]:
                dq = target - q[pj]
                cash -= dq * O[t, pj] + abs(dq) * O[t, pj] * costs.total[pj]
                comp["traded"] += abs(dq) * O[t, pj]
                comp["fee"] -= abs(dq) * O[t, pj] * costs.fee[pj]
                comp["slip"] -= abs(dq) * O[t, pj] * costs.slip[pj]
                q[pj] = target
        eq_after = cash + np.nansum(q * px)
        rows.append(
            (idx[t], eq_after, comp["fee"], comp["slip"], comp["funding"], comp["traded"],
             np.nansum(np.abs(q * px)), np.nansum(q * px))
        )  # fmt: skip
    cols = ["time", "equity", "fee", "slip", "funding", "traded", "gross", "net"]
    led = pd.DataFrame(rows, columns=cols).set_index("time")
    # price P&L = equity change not explained by costs and funding (marks between opens)
    led["pnl"] = led["equity"].diff().fillna(led["equity"].iloc[0] - capital)
    led["price"] = led["pnl"] - led["fee"] - led["slip"] - led["funding"]
    return {"ledger": led, "final_q": q}
