# Web-scraped public statistics

Real, publicly available figures collected by `scraping/scrape_public_stats.py` and used by
`scraping/calibrate.py` to calibrate the synthetic transaction generator. Retrieval time,
page revision ids and permanent links are recorded in `sources.json`.

| File | Scraped from | Primary source cited on the page | Used for |
|---|---|---|---|
| `upi_yearly.csv` | Wikipedia, "Unified Payments Interface" (yearly UPI table) | NPCI UPI statistics | Real average UPI ticket size, computed as INR value ÷ volume; sets `amount_scale` |
| `state_crime.csv` | Wikipedia, "Crime in India" (reported crimes by State/UT, 2016–2019) | NCRB "Crime in India" | 2019 crime rate of each generator city's state; sets `city_risk_weight` for where fraud victims live |
| `calibration.json` | derived | — | Parameters read by `data_gen/generate.py` when `DATA_PROFILE=calibrated` |

**Method:** the scraper uses the official MediaWiki API with an identifying User-Agent, at one
request per second. It reads only rendered tables. Content is licensed CC BY-SA 4.0.

**Cleaning:** numbers are parsed from formatted strings (commas, footnote markers removed).
The 2025 USD value is mis-formatted on the source page ("3,436,96"), so it is dropped
whenever the implied INR/USD rate falls outside 50–120 (see `usd_value_flag`).

**Not scraped:** rbi.org.in and npci.org.in refuse automated requests (HTTP 418 and 403).
Their reports are listed in `sources.json` as manual references; we do not get around a
site's block.

**Finding:** the default synthetic generator's everyday personal payments average ₹1,673.
Real UPI transactions averaged ₹1,582 over 2023–2025, so the synthetic amounts are within
6% of reality. The calibrated profile scales them by 0.946.
