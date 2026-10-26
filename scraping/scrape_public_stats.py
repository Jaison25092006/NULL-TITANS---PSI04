"""Web-scrape public statistics used to calibrate the synthetic transaction generator.

Sources (scraped through the official MediaWiki API, which permits automated access;
content licensed CC BY-SA 4.0, each table cites its own primary source on the page):
  1. "Unified Payments Interface" — yearly UPI table: payment partners, transaction
     volume (mn) and value (INR mn / USD bn). Primary source: NPCI statistics.
  2. "Crime in India" — reported crimes by State/UT 2016–2019 with the 2019 crime rate.
     Primary source: NCRB "Crime in India" reports.

Not scraped: rbi.org.in and npci.org.in refuse automated requests (HTTP 418/403), so
their reports are listed in sources.json as manual-download references only. We do not
work around a site that blocks crawlers.

Writes data/scraped/upi_yearly.csv, data/scraped/state_crime.csv and
data/scraped/sources.json. If the network is unavailable, existing files are kept.

Run: python scraping/scrape_public_stats.py
"""

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "scraped"
API = "https://en.wikipedia.org/w/api.php"
UA = "FraudIntelligenceStudentProject/1.0 (HackNex 2026 HNX26PSI04; educational use)"


def fetch_page(title):
    """Rendered HTML + revision id of a Wikipedia page via the MediaWiki parse API."""
    q = urllib.parse.urlencode({"action": "parse", "page": title, "prop": "text|revid",
                                "format": "json", "redirects": 1})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        d = json.load(r)
    if "error" in d:
        raise RuntimeError(d["error"].get("info", "page not found"))
    return d["parse"]["text"]["*"], d["parse"]["revid"], d["parse"]["title"]


def tables(html):
    """All wikitables as lists of rows of cell strings."""
    out = []
    for t in re.findall(r'<table class="wikitable.*?</table>', html, re.S):
        rows = []
        for tr in re.findall(r"<tr.*?</tr>", t, re.S):
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)
            clean = [re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>|\[\d+\]", "", c))).strip()
                     for c in cells]
            if clean:
                rows.append(clean)
        out.append(rows)
    return out


def num(s):
    s = s.replace(",", "").replace("−", "-").strip()
    try:
        return float(s)
    except ValueError:
        return None


def scrape_upi():
    html, revid, title = fetch_page("Unified_Payments_Interface")
    for t in tables(html):
        if t and "Transaction volume" in " ".join(t[0]):
            rows = []
            for r in t[1:]:
                if len(r) >= 4 and re.fullmatch(r"\d{4}", r[0]) and num(r[2]) and num(r[3]):
                    rows.append({"year": int(r[0]), "payment_partners": num(r[1]),
                                 "volume_mn": num(r[2]), "value_inr_mn": num(r[3]),
                                 "value_usd_bn": num(r[4]) if len(r) > 4 else None})
            df = pd.DataFrame(rows).sort_values("year")
            df["avg_ticket_inr"] = (df["value_inr_mn"] / df["volume_mn"]).round(2)
            # Data cleaning: the page occasionally mis-formats a USD figure (e.g. "3,436,96").
            # Keep USD only where the implied INR/USD rate is plausible (50-120).
            rate = (df["value_inr_mn"] / 1000) / df["value_usd_bn"]
            bad = ~rate.between(50, 120)
            df["usd_value_flag"] = bad.map({True: "dropped: implausible INR/USD rate", False: ""})
            df.loc[bad, "value_usd_bn"] = None
            return df, {"title": title, "revid": revid}
    raise RuntimeError("UPI yearly table not found")


def scrape_state_crime():
    html, revid, title = fetch_page("Crime_in_India")
    for t in tables(html):
        head = " ".join(t[0]) if t else ""
        if "State/UT" in head and "2019" in head:
            hdr = t[0]
            rows = []
            for r in t[1:]:
                if len(r) == len(hdr) and num(r[1]) is not None:
                    rows.append({"state": r[0], "crimes_2016": num(r[1]), "crimes_2017": num(r[2]),
                                 "crimes_2018": num(r[3]), "crimes_2019": num(r[4]),
                                 "share_2019_pct": num(r[5]), "crime_rate_2019": num(r[6])})
            return pd.DataFrame(rows), {"title": title, "revid": revid}
    raise RuntimeError("state crime table not found")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sources = {"retrieved_utc": now, "method": "MediaWiki parse API (robots-compliant)",
               "license": "CC BY-SA 4.0 (Wikipedia); primary sources cited on each page",
               "tables": {}, "manual_references": [
                   {"name": "RBI Annual Report — frauds in the banking sector",
                    "url": "https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx",
                    "note": "blocks automated access (HTTP 418); download manually"},
                   {"name": "NPCI UPI product statistics",
                    "url": "https://www.npci.org.in/what-we-do/upi/product-statistics",
                    "note": "blocks automated access (HTTP 403); download manually"}]}
    for name, fn, path in (("upi_yearly", scrape_upi, OUT / "upi_yearly.csv"),
                           ("state_crime", scrape_state_crime, OUT / "state_crime.csv")):
        try:
            df, meta = fn()
            df.to_csv(path, index=False)
            sources["tables"][name] = {
                **meta, "rows": len(df), "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "url": f"https://en.wikipedia.org/w/index.php?title="
                       f"{urllib.parse.quote(meta['title'])}&oldid={meta['revid']}"}
            print(f"✓ {name}: {len(df)} rows → {path.relative_to(ROOT)}")
            print(df.head(12).to_string(index=False))
        except Exception as exc:  # keep cached data on failure
            print(f"✗ {name}: {exc} (keeping existing file)" if path.exists()
                  else f"✗ {name}: {exc}")
        time.sleep(1)  # be polite
    if sources["tables"]:
        (OUT / "sources.json").write_text(json.dumps(sources, indent=2), "utf-8")


if __name__ == "__main__":
    main()
