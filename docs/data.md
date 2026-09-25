# Data card

- **Source:** https://data.binance.vision (public, no account). USD-M perpetual 1h klines,
  monthly fundingRate archives (`calc_time` settlement time in ms, `funding_interval_hours`,
  `last_funding_rate`) and spot 1h klines for BTCUSDT and ETHUSDT. Every archive is checked
  against its published SHA-256; URLs, digests and retrieval times are in
  `data/download_ledger*.json`. About 100 MB for the study and 70 MB for the transfer test.
- **Pool:** the 40 USDT perpetuals with the largest 2020-12 quote volume among those with a
  2020-12 archive (`data/pool.json`); later listings are excluded, delisted symbols kept.
  **Transfer universe:** 60 non-pool symbols by 2023-12 volume (`data/transfer_universe.json`).
- **Funding intervals** of 1, 2, 4 and 8 hours all occur; features sum settled rates over the
  lookback, so they are interval-normalized. The archive has no mark price; the settlement price
  is the perpetual's hourly open (the funding cash error is rate x (mark - last)).
- **Cash rate:** FRED DTB3 (3-month T-bill), downloaded by `make data`.
- **Positioning metrics (Generation 2):** daily archives under
  https://data.binance.vision/data/futures/um/daily/metrics/, 5-minute rows with a `create_time`
  stamp. Fields used: `sum_open_interest_value`, `sum_toptrader_long_short_ratio` (top traders, by
  position), `count_long_short_ratio` (all accounts, by number of accounts) and
  `sum_taker_long_short_vol_ratio` (taker buy/sell volume). 63,837 archives (0.7 GB), 2021-12-01 to
  2026-08-30, for the 38 pool symbols eligible at some decision in 2022-01..2026-08, each checked
  against its SHA-256. `scripts/download_metrics.py` keeps, per file, the last row stamped strictly
  before 00:00 UTC of the next day (the decision time) and the mean log taker ratio of the rows
  before it, in `data/metrics_daily.parquet`, then deletes the archives. The full download ledger
  (16 MB) stays local; `data/metrics_ledger_summary.json` has per-symbol counts, dates and the
  ledger's SHA-256.
- **Positioning timestamps.** Most files run 00:00:00 to 23:55:00; 656 files (every SXPUSDT file
  from 2024-03 and a few in 2024-04) run 00:05:00 to 00:00:01 of the next day, and their midnight
  row is excluded (`docs/protocol_amendments.md`). After that cut, 63,359 of the 63,837 decision
  snapshots are the 23:55 row; 475 are older because the file ends early (at most 22 h 10 min).
  Binance's REST documentation for the long/short ratio endpoints
  (https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Long-Short-Ratio)
  describes period-end timestamps; when an archive row is published could not be checked from here
  (the REST API refuses this location), so the study also reports the result with every snapshot
  one day older. Account ratios count accounts, not capital, and are not a verified retail population.
- **S&P 500:** FRED SP500 daily close, downloaded by `scripts/family_significance.py` on first use;
  a close on day t enters the factor on day t+1 (UTC), when it is known.
- **Contract ends.** Some settled perpetuals keep frozen-price klines in the archive after their
  last funding settlement: WAVES and OMG (2025-06-19), MKR (2025-09-08) and SXPUSDT (automatic
  settlement 2025-12-05 09:00 UTC, new positions restricted from 08:30,
  https://www.binance.com/en/support/announcement/detail/746056ce732d43dda12c1e5ae1b7a059). Their
  bars are cut one hour after the last settlement (`fasa.panel.settle_tails`). EOSUSDT is the
  reverse case: its klines end with the automatic settlement of 2025-05-21 09:00 UTC (announced
  2025-05-14,
  https://www.binance.com/en/support/announcement/binance-will-support-the-eos-eos-token-swap-and-rebranding-to-vaulta-a-1e89a9ca957c4b0ca7502e60b993e201),
  while its funding archive continues at a constant 0.01% per 8 hours that never reaches a
  position. YFIIUSDT's klines end on 2022-04-12. A position still held when a contract's bars end
  is closed at the last price with fees and slippage (a settlement charges no taker fee, so this is
  conservative). Only EOSUSDT was still eligible near its end (last eligible decision 2025-05-21
  00:00); the others were last eligible months earlier, so no strategy held them. The
  Generation-2 primary held EOSUSDT from its 2025-05-19 decision to the settlement; excluding
  EOSUSDT from decisions after the announcement would have raised that strategy's final return
  from +18.3% to +18.6% a year.
- Raw archives and panels are not redistributed; `make data` rebuilds them.
