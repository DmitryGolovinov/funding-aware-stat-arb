"""Target weights for the three registered families (configs/alpha_protocol.yaml)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import funding_per_day, residual


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)


def long_short(score: pd.Series, beta: pd.Series, k: int = 5, gross: float = 1.0,
               net_cap: float = 0.25, w_cap: float = 0.125) -> pd.Series:
    """Long the k highest scores, short the k lowest, equal weight within legs; legs rescaled to
    zero estimated beta subject to |net| <= net_cap and |w_i| <= w_cap."""
    s = score.dropna()
    w = pd.Series(0.0, index=score.index)
    if len(s) < 2 * k:
        return w
    order = s.sort_values()
    lo, hi = order.index[:k], order.index[-k:]
    bl = float(beta.reindex(hi).fillna(1.0).mean())
    bs = float(beta.reindex(lo).fillna(1.0).mean())
    a = b = 1.0  # leg multipliers of 0.5 * gross each
    if bl > 0 and bs > 0:
        b = 2 * bl / (bl + bs)
        a = 2 - b
    net = 0.5 * gross * (a - b)
    if abs(net) > net_cap:
        a = 1 + np.sign(net) * net_cap / (0.5 * gross) / 2 * 1
        b = 2 - a
    w[hi] = min(0.5 * gross * a / k, w_cap)
    w[lo] = -min(0.5 * gross * b / k, w_cap)
    return w


def decision_times(days: pd.DatetimeIndex, every: str) -> pd.DatetimeIndex:
    if every == "24h":
        return days
    if every == "72h":
        return days[::3]
    if every == "8h":
        return pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in (0, 8, 16)])
    raise ValueError(every)


def funding_xs(panel, feat, elig, L: int, every: str, beta: pd.DataFrame) -> dict:
    """Family B: long the lowest trailing funding, short the highest (receive funding)."""
    times = decision_times(feat["days"], every)
    fund = funding_per_day(panel, times, L)
    out = {}
    for t in times:
        d = t.normalize()
        if d not in elig.index:
            continue
        e = elig.loc[d]
        sc = (-fund.loc[t]).where(e)
        out[t] = long_short(sc, beta.loc[d]).to_numpy()
    return out


def residual_funding(panel, feat, elig, H: int, mech: str, with_funding: bool, beta_override=None) -> dict:
    """Family C (daily): score = z(sign * residual_H) [+ z(-funding over 3 days)];
    reversal: sign -1 (long recent residual losers); momentum: sign +1."""
    res, beta, _ = residual(feat["ret"], elig, H)
    sign = -1.0 if mech == "reversal" else 1.0
    sc = zscore((sign * res).where(elig))
    if with_funding:
        f = funding_per_day(panel, feat["days"], 3)
        sc = sc.add(zscore((-f).where(elig)), fill_value=np.nan)
    out = {}
    for d in feat["days"]:
        if d in elig.index and elig.loc[d].sum() >= 10:
            out[d] = long_short(sc.loc[d].where(elig.loc[d]), beta.loc[d]).to_numpy()
    return out


def carry(panel, feat, assets: list[str], conditional: bool, cols: list[str]) -> dict:
    """Family A: long spot, short perp in matched units, capital = spot notional + a perp
    collateral reserve of 100% of notional (weight 0.5 of equity per unit of spot per asset,
    split equally across assets). Conditional: flat when trailing 7-day mean funding <= 0."""
    f = funding_per_day(panel, feat["days"], 7) if conditional else None
    per = 0.5 / len(assets)
    out = {}
    for d in feat["days"]:
        w = np.zeros(len(cols))
        for a in assets:
            on = True if f is None else bool(f.loc[d, a + "USDT"] > 0) if pd.notna(f.loc[d, a + "USDT"]) else False
            if on:
                w[cols.index("spot_" + a)] = per
                w[cols.index("perp_" + a)] = -per
        out[d] = w
    return out
