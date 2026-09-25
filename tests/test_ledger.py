"""Hand-checkable ledger cases: fees on flat prices, funding signs, matched carry legs, basis,
settlement-time positions, delisting, drawdown from the first mark, and timing invariance."""

import numpy as np
import pandas as pd
import pytest

from fasa import engine
from fasa.evaluate import daily_returns


def frame(prices, n_hours=48, start="2024-01-01"):
    idx = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    return pd.DataFrame({k: np.full(n_hours, v, float) if np.isscalar(v) else np.asarray(v, float)
                         for k, v in prices.items()}, index=idx)


def costs(n, fee=0.0005, slip=0.0002):
    return engine.Costs(np.full(n, fee), np.full(n, slip))


NOFUND = pd.DataFrame({"time": pd.to_datetime([], utc=True), "symbol": [], "rate": []})


def test_flat_prices_lose_exactly_the_round_trip_costs():
    O = frame({"A": 100.0})
    t0 = O.index[0]
    tg = {t0: np.array([0.5]), O.index[10]: np.array([0.0])}
    led = engine.run(O, O, NOFUND, tg, costs(1), delay=1)["ledger"]
    cost_in = 0.5e6 * 0.0007
    eq_after_entry = 1e6 - cost_in
    cost_out = abs(0.5e6 / 100 * 100) * 0.0007  # units bought at entry equity x 0.5
    assert led["equity"].iloc[-1] == pytest.approx(1e6 - cost_in - cost_out)
    assert led["price"].sum() == pytest.approx(0.0, abs=1e-6)
    assert eq_after_entry > led["equity"].iloc[-1]


def test_funding_sign_long_pays_short_receives_on_the_pre_trade_position():
    O = frame({"L": 100.0, "S": 100.0})
    fund = pd.DataFrame({"time": [O.index[8] + pd.Timedelta(milliseconds=3)] * 2,
                         "symbol": ["L", "S"], "rate": [0.001, 0.001]})
    tg = {O.index[0]: np.array([0.4, -0.4])}
    led = engine.run(O, O, fund, tg, costs(2, 0, 0), delay=1)["ledger"]
    q = 0.4e6 / 100
    assert led["funding"].sum() == pytest.approx(-q * 100 * 0.001 + q * 100 * 0.001)
    one = engine.run(O[["L"]], O[["L"]], fund, {O.index[0]: np.array([0.4])}, costs(1, 0, 0))["ledger"]
    assert one["funding"].sum() == pytest.approx(-q * 100 * 0.001)  # a long pays positive funding
    # an order filling at the settlement hour does not receive that settlement
    late = engine.run(O[["L"]], O[["L"]], fund, {O.index[7]: np.array([0.4])}, costs(1, 0, 0))["ledger"]
    assert late["funding"].sum() == 0.0


def test_matched_carry_is_neutral_to_a_common_price_move_and_earns_basis_convergence():
    n = 48
    spot = np.linspace(100, 150, n)
    perp = spot + 1.0  # constant absolute basis: matched units are exactly delta-neutral
    O = frame({"spot": spot, "perp": perp}, n)
    tg = {O.index[0]: np.array([0.5, -0.5])}
    led = engine.run(O, O, NOFUND, tg, costs(2, 0, 0), perp_mask=np.array([False, True]), pairs=[(0, 1)])["ledger"]
    assert led["equity"].iloc[-1] == pytest.approx(1e6, rel=1e-12)
    perp2 = spot * np.linspace(1.01, 1.0, n)  # basis narrows from 1% to 0: the short perp gains
    O2 = frame({"spot": spot, "perp": perp2}, n)
    led2 = engine.run(O2, O2, NOFUND, tg, costs(2, 0, 0), perp_mask=np.array([False, True]), pairs=[(0, 1)])["ledger"]
    u = 0.5e6 / spot[1]
    expected = u * (perp2[1] - spot[1]) - u * (perp2[-1] - spot[-1])
    assert led2["equity"].iloc[-1] - 1e6 == pytest.approx(expected, rel=1e-9)


def test_quantity_change_at_a_settlement_and_no_double_counted_funding():
    O = frame({"A": 100.0})
    fund = pd.DataFrame({"time": [O.index[k] for k in (8, 16, 24)], "symbol": ["A"] * 3, "rate": [0.001] * 3})
    tg = {O.index[0]: np.array([0.5]), O.index[15]: np.array([0.25])}  # changes at 16:00 open
    led = engine.run(O, O, fund, tg, costs(1, 0, 0), delay=1)["ledger"]
    q1 = 0.5e6 / 100
    q2 = 0.25 * (1e6 - 2 * q1 * 100 * 0.001) / 100  # sized from equity after two payments
    # 08:00 with q1; 16:00 settles before the 16:00 fill, so q1 again; 24:00 with q2
    assert led["funding"].sum() == pytest.approx(-(q1 + q1 + q2) * 100 * 0.001)
    assert (led["funding"] != 0).sum() == 3


def test_delisted_position_is_closed_at_its_last_close_with_costs():
    p = np.full(48, 100.0)
    p[20:] = np.nan
    O = frame({"A": p})
    C = O.copy()
    C.iloc[19, 0] = 90.0  # last bar closes lower
    led = engine.run(O, C, NOFUND, {O.index[0]: np.array([0.5])}, costs(1, 0, 0))["ledger"]
    q = 0.5e6 / 100
    assert led["equity"].iloc[-1] == pytest.approx(1e6 - q * 10)
    assert led["gross"].iloc[-1] == 0.0


def test_drawdown_counts_from_the_first_mark_and_daily_returns_include_the_first_day():
    O = frame({"A": np.r_[np.full(24, 100.0), np.full(24, 80.0), np.full(24, 80.0)]}, 72)
    led = engine.run(O, O, NOFUND, {O.index[0]: np.array([1.0])}, costs(1, 0, 0))["ledger"]
    d = daily_returns(led)
    assert d["ret"].iloc[0] == pytest.approx(-0.2)


def test_future_prices_do_not_change_earlier_trades():
    rng = np.random.default_rng(0)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (72, 3)), axis=0))
    O = frame({c: base[:, i] for i, c in enumerate("ABC")}, 72)
    tg = {O.index[0]: np.array([0.3, -0.3, 0.2]), O.index[24]: np.array([-0.2, 0.3, 0.0])}
    a = engine.run(O, O, NOFUND, tg, costs(3))["ledger"]
    O2 = O.copy()
    O2.iloc[40:] *= 1.5
    b = engine.run(O2, O2, NOFUND, tg, costs(3))["ledger"]
    pd.testing.assert_series_equal(a["equity"].iloc[:40], b["equity"].iloc[:40])


def test_settled_contract_tail_is_cut_and_the_position_closed_at_the_settlement_price():
    """Bars that continue after the last funding settlement (a settled, delisted perpetual whose
    archive keeps a frozen price) are cut one hour after that settlement; a held position is
    closed at the last close before the cut, and the symbol stops trading."""
    from fasa.panel import settle_tails

    n = 24 * 6
    idx = pd.date_range("2025-12-01", periods=n, freq="h", tz="UTC")
    live = np.linspace(100.0, 90.0, n)
    live[80:] = 90.0  # frozen after settlement
    panel = {"perp": {"open": pd.DataFrame({"X": live}, index=idx), "close": pd.DataFrame({"X": live}, index=idx),
                      "qv": pd.DataFrame({"X": np.r_[np.full(80, 1e6), np.zeros(n - 80)]}, index=idx)},
             "funding": pd.DataFrame({"time": [idx[k] for k in range(0, 80, 8)], "symbol": "X", "rate": 0.0})}
    last_funding = idx[72]
    out = settle_tails(panel)
    cut = last_funding + pd.Timedelta(hours=1)
    assert out["settlements"] == {"X": str(cut)}
    assert out["perp"]["open"]["X"].loc[cut:].isna().all() and out["perp"]["open"]["X"].loc[:last_funding].notna().all()
    O, C = out["perp"]["open"], out["perp"]["close"]
    led = engine.run(O, C, NOFUND, {idx[0]: np.array([0.5])}, costs(1, 0, 0))["ledger"]
    q = 0.5e6 / live[1]
    assert led["equity"].iloc[-1] == pytest.approx(1e6 + q * (C["X"].loc[last_funding] - live[1]))
    assert led["gross"].iloc[-1] == 0.0
