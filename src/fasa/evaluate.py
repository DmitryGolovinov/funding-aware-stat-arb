"""Daily returns from the ledger, metrics, block bootstrap and factor attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 365.0


def daily_returns(led: pd.DataFrame) -> pd.DataFrame:
    """Equity at each 00:00 UTC (after that hour's funding and before the 01:00 fills) and the
    day's P&L components as fractions of the day's starting equity."""
    eq = led["equity"]
    day = eq[eq.index.hour == 0]
    comp = led[["price", "funding", "fee", "slip", "traded"]].resample("D").sum()
    comp = comp.shift(-0)  # components of calendar day d (00:00 through 23:00)
    start = day.shift(1)
    out = pd.DataFrame({"ret": day.pct_change()}, index=day.index)
    # components between consecutive 00:00 marks: hours [d-1 00:00+1h .. d 00:00]
    hourly = led[["price", "funding", "fee", "slip", "traded"]].copy()
    hourly.index = hourly.index - pd.Timedelta(hours=1)
    per_day = hourly.resample("D").sum()
    per_day.index = per_day.index + pd.Timedelta(days=1)
    for c in ("price", "funding", "fee", "slip", "traded"):
        out[c] = per_day[c].reindex(out.index) / start
    out["gross"] = led["gross"].reindex(out.index) / day
    out["net"] = led["net"].reindex(out.index) / day
    return out.iloc[1:].drop(columns=[]) if len(out) > 1 else out


def rf_daily(dtb3: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    y = dtb3.copy()
    y.index = pd.to_datetime(y.index).tz_localize("UTC")
    return (y / 100.0 / ANN).reindex(index, method="ffill").fillna(0.0)


def metrics(d: pd.DataFrame, rf: pd.Series) -> dict:
    r = d["ret"].dropna()
    ex = r - rf.reindex(r.index).fillna(0.0)
    eq = (1 + r).cumprod()
    dd = float((eq / np.maximum.accumulate(np.r_[1.0, eq.to_numpy()][1:]) - 1).min()) if len(eq) else np.nan
    wealth = np.r_[1.0, eq.to_numpy()]
    dd = float((wealth / np.maximum.accumulate(wealth) - 1).min())
    sd = float(r.std(ddof=1))
    tail = r.quantile(0.05)
    return {
        "days": len(r),
        "ann_return": float(r.mean() * ANN),
        "ann_excess": float(ex.mean() * ANN),
        "ann_vol": float(sd * np.sqrt(ANN)),
        "sharpe": float(ex.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan,
        "max_drawdown": dd,
        "cvar5_daily": float(r[r <= tail].mean()),
        "price_ann": float(d["price"].mean() * ANN),
        "funding_ann": float(d["funding"].mean() * ANN),
        "fee_ann": float(d["fee"].mean() * ANN),
        "slip_ann": float(d["slip"].mean() * ANN),
        "turnover_ann": float(d["traded"].mean() * ANN),
        "gross_mean": float(d["gross"].mean()),
        "net_mean": float(d["net"].mean()),
    }


def block_bootstrap(x: np.ndarray, block: int = 20, draws: int = 2000, seed: int = 0) -> np.ndarray:
    """Means of moving-block resamples of the rows of x (columns kept together: paired)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(draws, -1)[:, :n]
    return x[idx].mean(axis=1)


def ci_mean(r: pd.Series, **kw) -> tuple[float, float, float]:
    m = block_bootstrap(r.dropna().to_numpy(), **kw) * ANN
    return float(r.mean() * ANN), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def ols_hac(y: pd.Series, X: pd.DataFrame, lags: int = 5) -> dict:
    """OLS with Newey-West (Bartlett) standard errors; intercept annualized."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    Y = df["y"].to_numpy()
    Z = np.column_stack([np.ones(len(df)), df[X.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    e = Y - Z @ beta
    S = (Z * e[:, None]).T @ (Z * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (Z[L:] * e[L:, None]).T @ (Z[:-L] * e[:-L, None])
        S += w * (G + G.T)
    ZZi = np.linalg.inv(Z.T @ Z)
    se = np.sqrt(np.diag(ZZi @ S @ ZZi))
    names = ["alpha"] + list(X.columns)
    out = {f"{k}_coef": float(b) for k, b in zip(names, beta, strict=True)}
    out.update({f"{k}_t": float(b / s) for k, b, s in zip(names, beta, se, strict=True)})
    out["alpha_ann"] = float(beta[0] * ANN)
    out["r2"] = float(1 - e.var() / Y.var())
    out["days"] = len(df)
    return out
