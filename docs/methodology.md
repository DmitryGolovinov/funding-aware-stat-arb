# Methodology

**Ledger** (`src/fasa/engine.py`). One USDT account, hourly. Buying q units at price P moves q*P
from cash into the position; equity = cash + sum q_i P_i at each hour's open, which for a linear
perpetual equals collateral plus unrealized P&L. Each hour: mark the open; charge every funding
settlement stamped in that hour to the position held before any fill (cash -= q * price *
rate: a long pays positive funding); execute scheduled orders at the open with fee plus
slippage on traded notional. A symbol with no later bar is closed at its last close with costs.
Carry legs are held in matched units (perp = -spot) and trade together. Cash earns nothing in the
ledger; excess returns over the 3-month T-bill are computed from FRED DTB3.

**Timing.** A decision at time t uses bars that closed before t and funding settled before t, and
fills at the open 1 hour later (2 hours in the delay stress). The funding a strategy receives is
future outcome accounting, never a feature.

**Portfolios.** Long the 5 highest scores and short the 5 lowest among the 15 eligible coins,
equal weight within each leg, legs rescaled to zero estimated beta (60-day beta against the
equal-weight eligible market) with |net notional| <= 0.25 and |w_i| <= 0.125 of equity; gross 1.0.
A name trades only if its weight moves by more than 0.02.

**Selection and evaluation.** 36 candidates (6 carry, 12 funding-only, 18 residual/funding
including a LightGBM pair) on 2021-2023 folds; score = mean fold Sharpe - 0.5 sd; the primary is
the best residual+funding candidate positive in at least 2 of 3 folds. Every candidate was then
evaluated once on 2024-01..2026-08. Uncertainty: 20-day moving-block bootstrap of daily returns
(paired for differences). Attribution: OLS with Newey-West errors on zero-cost factors built in
the same universe (equal-weight market, 21-day momentum long/short, 3-day funding long/short).
After the final, the slow funding rule was tested once on 60 new coins (`scripts/transfer_test.py`,
`docs/protocol_amendments.md`).

**Settled contracts** (`src/fasa/panel.py`, `settle_tails`). A live perpetual pays funding at
every settlement. When a symbol's bars continue more than 48 hours after its last funding
settlement, every bar from one hour after that settlement is set to missing, so a held position is
closed at the last close and the symbol leaves the universe (SXPUSDT, WAVES, OMG and MKR in the
pool). The cut relies on the absence of later settlements, which is not observable in real time; it
stands in for the exchange's advance notice (SXPUSDT's settlement was announced on 2025-12-01, four
days ahead), and none of the four was eligible within months of its end (`docs/data.md`).

**Family-level test** (`scripts/family_significance.py`). The 12 registered funding long/short
configurations, averaged with equal weight over 2021-01..2026-08 (no choice among them): mean return
with a Newey-West t (10 lags) and HAC alphas against two, seven and eight factors. Factors are built
in the same universe: the equal-weight market (MKT), BTC, the S&P 500 (SPX), and zero-cost terciles
(long 5, short 5 of the 15 eligible, held one day) on 21-day momentum (MOM), 1-day reversal (REV),
30-day low volatility (LOWVOL) and low 30-day quote volume (ILLIQ, not a size factor: market caps are
not in the archives), plus the unlevered BTC spot/perpetual carry trade net of costs (CARRY).

**Generation 2** (`scripts/run_positioning.py`; `configs/positioning_protocol.yaml`, frozen before
any positioning file was downloaded). Seven cross-sectional scores from the daily snapshot, each
z-scored among the eligible coins: OI (minus the 7-day log change of open-interest value net of the
7-day log price change), TOP (log top-trader position ratio), CROWD (minus the log all-account
ratio), GAP (TOP + CROWD), FLOW_cont and FLOW_rev (plus or minus the 3-day mean log taker ratio) and
COMPOSITE (mean of the OI, CROWD and GAP z-scores), each rebalanced every 24 h or 72 h: 14
candidates with the Generation-1 construction, costs and limits. Selection on the 2022 and 2023 folds
(mean fold Sharpe - 0.5 sd; requirement: positive in both folds). No candidate met the requirement,
so the best-scoring one, CROWD_72h, became the flagged primary. All 14 were evaluated once on
2024-01..2026-08; the primary is attributed to the eight factors plus the 3-day funding long/short
(FUND), HAC with 10 lags, and stressed with doubled costs, a 2-hour fill delay and every snapshot one
day older.

**Capacity scenario** (`scripts/capacity.py`). For capital A and each one-way trade of the
Generation-2 primary in the final period: participation = traded notional / the coin's quote volume
over the 24 hours before the fill; extra cost per unit traded = Y x sigma_daily x sqrt(participation),
Y = 1 (the square-root law's prefactor is of order one in Toth et al. 2011 and in Donier and Bonart
2015 for Bitcoin), sigma_daily from the 30 daily returns before the decision. The cost is charged on
entries, exits and resizing, on top of the modeled 5 bps slippage (which double-counts part of the
spread, a conservative choice); returns are per year of A. This is a sensitivity scenario under an
assumed impact model, not measured capacity, and the alpha t statistic refers to the modeled-cost
returns only.

**Tests** (`tests/test_ledger.py`, `tests/test_positioning_snapshot.py`): round-trip costs on flat prices; funding signs and the
pre-trade holder at a settlement hour; matched carry neutral to a common move and earning basis
convergence; quantity changes at settlements without double counting; delisting exits; the first
day's return; future prices cannot change earlier trades. The positioning snapshot ignores rows stamped at or
after the decision time.
