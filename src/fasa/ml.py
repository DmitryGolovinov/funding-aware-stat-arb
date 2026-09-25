"""One small LightGBM ranker (registered comparison), with and without funding features.

Target: the next day's residual return. A row (day d, asset i) is used for training only when
its target has matured before the refit day (d + 1 < refit). Refit every 90 days on an expanding
window; predictions for day d use the latest fit made before d.
"""

from __future__ import annotations

import pandas as pd

from .signals import funding_per_day, residual


def features(panel, feat, elig, with_funding: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = {}
    for H in (1, 3, 7, 21):
        cols[f"res{H}"] = residual(feat["ret"], elig, H)[0]
    res1, beta, mkt = residual(feat["ret"], elig, 1)
    cols["vol30"] = feat["ret"].rolling(30, min_periods=20).std()
    cols["beta"] = beta
    if with_funding:
        for L in (1, 3, 7):
            cols[f"fund{L}"] = funding_per_day(panel, feat["days"], L)
    X = pd.concat({k: v.where(elig) for k, v in cols.items()}, axis=1).stack(future_stack=True)
    X = X.dropna()
    target = res1.shift(-1).where(elig)  # next day's residual: known after d + 1
    y = target.stack(future_stack=True).reindex(X.index)
    return X, y


def lgbm_scores(panel, feat, elig, with_funding: bool, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import lightgbm as lgb

    X, y = features(panel, feat, elig, with_funding)
    days = X.index.get_level_values(0)
    refits = pd.date_range(start, end, freq="90D", tz="UTC")
    preds = []
    for i, R in enumerate(refits):
        train = (days < R - pd.Timedelta(days=1)) & y.notna().to_numpy()
        nxt = refits[i + 1] if i + 1 < len(refits) else end + pd.Timedelta(days=1)
        test = (days >= R) & (days < nxt)
        if train.sum() < 2000 or not test.any():
            continue
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                              min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=0, verbose=-1)
        m.fit(X[train], y[train])
        preds.append(pd.Series(m.predict(X[test]), index=X.index[test]))
    return pd.concat(preds).unstack() if preds else pd.DataFrame()
