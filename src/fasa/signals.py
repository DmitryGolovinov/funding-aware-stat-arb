"""Decision-time features, all from information strictly before the decision.

Daily decision at 00:00 UTC on day d: the latest price is the close of the 23:00 bar of day d-1
(its close time is before 00:00); funding uses settlements stamped before 00:00; trailing volume
uses the 30 days of hourly bars before d.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily(panel: dict) -> dict:
    close = panel["perp"]["close"]
    qv = panel["perp"]["qv"]
    days = pd.date_range(close.index[0].normalize() + pd.Timedelta(days=1), close.index[-1].normalize(), freq="D", tz="UTC")
    last = close.shift(1)  # at hour h, the close of the bar h-1 (known before h)
    px = last.reindex(days)  # price known at each 00:00 decision
    ret = px.pct_change(fill_method=None)  # r[d] = move from d-1 00:00 to d 00:00 (known at d)
    vol30 = qv.rolling(30 * 24, min_periods=24 * 20).sum().shift(1).reindex(days)
    seen = close.notna().cumsum().shift(1).reindex(days)  # hourly bars before d
    recent = qv.notna().rolling(24).sum().shift(1).reindex(days)
    return {"days": days, "px": px, "ret": ret, "vol30": vol30, "hist_h": seen, "recent": recent}


def eligible(feat: dict, n_top: int = 15, min_days: int = 60) -> pd.DataFrame:
    ok = (feat["hist_h"] >= min_days * 24) & (feat["recent"] > 0) & feat["px"].notna()
    v = feat["vol30"].where(ok)
    rank = v.rank(axis=1, ascending=False, method="first")
    return rank <= n_top


def funding_per_day(panel: dict, times: pd.DatetimeIndex, days: int) -> pd.DataFrame:
    """Mean funding paid per day by a long over the ``days`` before each time (settlements
    strictly before the time; the sum is interval-normalized by construction)."""
    f = panel["funding"]
    wide = f.pivot_table(index="time", columns="symbol", values="rate", aggfunc="sum")
    wide = wide.reindex(columns=panel["perp"]["close"].columns).fillna(0.0)
    has = f.pivot_table(index="time", columns="symbol", values="rate", aggfunc="count")
    has = has.reindex(columns=wide.columns).fillna(0.0)
    cum = wide.cumsum()
    cnt = has.cumsum()
    tpos = cum.index.searchsorted(times, side="left") - 1  # last settlement strictly before t
    lpos = cum.index.searchsorted(times - pd.Timedelta(days=days), side="left") - 1

    def at(frame, pos):
        z = np.zeros((len(pos), frame.shape[1]))
        m = pos >= 0
        z[m] = frame.to_numpy()[pos[m]]
        return z

    total = at(cum, tpos) - at(cum, lpos)
    n = at(cnt, tpos) - at(cnt, lpos)
    out = pd.DataFrame(total / days, index=times, columns=wide.columns)
    return out.where(n > 0)  # unknown funding is not zero


def betas(ret: pd.DataFrame, mkt: pd.Series, window: int = 60) -> pd.DataFrame:
    """Rolling OLS beta of each asset on the market using returns strictly before each day."""
    r, m = ret.shift(1), mkt.shift(1)
    cov = r.mul(m, axis=0).rolling(window, min_periods=40).mean() - r.rolling(window, min_periods=40).mean().mul(
        m.rolling(window, min_periods=40).mean(), axis=0
    )
    var = m.rolling(window, min_periods=40).var(ddof=0)
    return cov.div(var, axis=0)


def residual(ret: pd.DataFrame, elig: pd.DataFrame, H: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Sum over the last H daily returns (known at d) of r_i - beta_i r_m, with the market the
    equal-weight return of the assets eligible at d; beta from the 60 days before."""
    mkt = ret.where(elig).mean(axis=1)
    b = betas(ret, mkt)
    res = ret - b.mul(mkt, axis=0)
    return res.rolling(H, min_periods=H).sum(), b, mkt
