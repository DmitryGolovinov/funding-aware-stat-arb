# Funding-Aware Relative Value on Crypto Perpetuals

Can funding rates or trader-positioning data pick long/short positions in Binance USD-M
perpetuals that make money after fees, slippage and funding, and is the result more than
exposure to known factors? I built an hourly self-financing backtest that charges every funding
payment at its actual settlement time, wrote down each strategy family and its selection rule
before computing that family's returns, chose the best candidate on 2021-2023 data (2022-2023 for
positioning) and ran it once on held-out data from 2024-01 to 2026-08.

## Results

| Strategy, net of modeled costs | Selection period: Sharpe / annualized mean return | 2024-01..2026-08: Sharpe / annualized mean return | Alpha, HAC t |
|---|---:|---:|---|
| Against the crowd: short the 5 perpetuals with the most net-long accounts, long the 5 most net-short, rebalanced every 72 h | 0.76 / +19.5% (2022-2023) | 0.72 / +18.3% | +27.3% a year vs nine factors, t 2.33 |
| Funding long/short, average of 12 configurations | | | +25.2% vs seven factors, t 2.89 (2021-2026); +4.3%, t 0.46 once carry is a factor |
| Residual momentum + funding | 0.69 / +18.4% (2021-2023) | 0.10 / +6.2% | no gain over funding alone |
| Spot/perp carry on BTC (benchmark) | 6.20 / +6.8% | -2.67 / +3.4% (below T-bills) | |

**Positioning.** Binance publishes each perpetual's long/short account ratio every 5 minutes.
Trading against the most one-sided accounts earned +18.3% a year net in the held-out period
(+14.2% over T-bills). Against nine factors (equal-weight crypto market, BTC, S&P 500, momentum,
reversal, low volatility, illiquidity, spot/perp carry and funding long/short) the intercept is
+27.3% a year (t 2.33), with no statistically resolved funding loading (0.01, t 0.18). It
earned +13.5% with doubled costs and +25.0% with the positioning data lagged by a full day. The
held-out period is 32 months, so the 95% interval for the mean return is wide (-11.2% to +44.8%),
and the t statistics are not adjusted for the 51 candidates I tested. The development return
was positive in 2022 and negative in 2023, so I treat this as an exploratory result.

**Funding.** The funding long/short family has a large alpha against seven standard factors,
but the estimate falls and is no longer statistically significant once spot/perp carry is added
as a factor, consistent with exposure to the funding-carry premium.

![Against-the-crowd strategy, growth of 1 net of costs](reports/figures/positioning.png)

## Data

- Public Binance archives ([data.binance.vision](https://data.binance.vision)), each file
  checked against its published SHA-256: hourly perpetual and spot klines, funding history and
  daily positioning metrics (5-minute rows, 63,837 daily files).
- Universe: 40 USDT perpetuals fixed by 2020-12 volume, keeping later delistings; on each day
  the 15 with the largest 30-day volume are eligible.
- S&P 500 (FRED SP500) and the 3-month T-bill rate (FRED DTB3) for factors and excess returns.

Fills and costs are modeled; liquidation and counterparty risk are outside the backtest.
Account ratios count accounts rather than capital. Archive publication times were not verified,
so I also report the one-day signal-delay check. The BTC/ETH calendar overlaps other portfolio
studies; data and timing details are in [docs/data.md](docs/data.md).

## How it works

- **Backtest** (`src/fasa/engine.py`). One USDT account marked hourly. Funding is charged at its
  settlement timestamp on the position held at that moment. Decisions are made at 00:00 UTC and
  filled at the next hourly open with a 5 bps taker fee plus 2 bps (BTC, ETH) or 5 bps slippage
  per side. Delisted and settled contracts are closed at their last price.
- **Portfolio.** Long/short, gross exposure 1.0 of equity, legs scaled to zero estimated beta,
  |net| <= 0.25 and |w_i| <= 0.125.
- **Timing.** The positioning signal uses the last 5-minute snapshot before 00:00 UTC (normally
  23:55) and trades at 01:00; a test checks that the snapshot is always stamped before the
  decision.
- **Search.** 51 candidates in total: carry, funding and residual-momentum signals; then open
  interest, top-trader and all-account ratios and taker flow; and one transfer test of a funding
  rule on 60 perpetuals outside the universe.
- **Attribution.** OLS on daily factor returns with Newey-West (10 lags) standard errors; block
  bootstrap intervals.

## Run

```bash
make test                                    # ledger tests (offline)
make demo                                    # synthetic panel through the same ledger
make data reproduce                          # carry, funding and residual strategies (~100 MB of archives)
make data-positioning reproduce-positioning  # positioning strategies (63,837 daily archives, ~15 min)
make report                                  # tables and figures from saved results
```

## Code

- `src/fasa/engine.py`: ledger, funding timing, carry legs, delisting exits.
- `src/fasa/panel.py`: hourly panel and the cut for settled contracts.
- `scripts/run_positioning.py`: positioning signals, selection and nine-factor attribution.
- `scripts/family_significance.py`, `scripts/capacity.py`: factor models and a capacity scenario.
- `tests/`: hand-checked ledger cases and the snapshot-timing rule.

Detailed tables: [reports/results.md](reports/results.md). Method: [docs/methodology.md](docs/methodology.md).

## References

- M. Schmeling, A. Schrimpf, K. Todorov. Crypto carry. BIS Working Papers 1087, 2023.
- Y. Liu, A. Tsyvinski, X. Wu. Common Risk Factors in Cryptocurrency. *Journal of Finance* 77(2), 2022.
- W. K. Newey, K. D. West. A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation
  Consistent Covariance Matrix. *Econometrica* 55(3), 1987.
