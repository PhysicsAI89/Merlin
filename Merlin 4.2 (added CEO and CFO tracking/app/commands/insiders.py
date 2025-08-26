import io
import sys
import time
import json
import math
import click
import typing as T
from datetime import datetime, timedelta


import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup


from flask import Blueprint


from models.stockmarket import db_session, Stock, StockDaily
from commands.ml import load_prices, make_features, FEATURE_COLUMNS, train_for_symbol, _load_model


# optional: yfinance ingest
from commands.ingest import ingest_yf


insidersbp = Blueprint("insiders", __name__)


BASE_URL = (
"http://openinsider.com/screener"
)


HEADERS = {
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
"(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}




def _fetch_page(params: dict) -> pd.DataFrame:
"""Fetch a single OpenInsider results page as a DataFrame."""
r = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=20)
r.raise_for_status()
# Parse table safely with BeautifulSoup first, then to DataFrame
soup = BeautifulSoup(r.text, "html.parser")
tbl = soup.find("table", {"class": "tinytable"})
if tbl is None:
return pd.DataFrame()
# Convert to DataFrame
th = [th.get_text(strip=True) for th in tbl.find("thead").find_all("th")]
rows = []
for tr in tbl.find("tbody").find_all("tr"):
tds = [td.get_text(strip=True) for td in tr.find_all("td")]
if len(tds) == len(th):
rows.append(dict(zip(th, tds)))
return pd.DataFrame(rows)




def scrape_openinsider_ceo_cfo(days_back: int = 730, pages: int = 3) -> pd.DataFrame:
"""Scrape CEO/CFO purchases for the last `days_back` across `pages` of results.
Returns a normalized DataFrame with at least: ticker, company, insider, title, trade_type, trade_date, price, qty, value.
"""
# These params mirror your shared URL: purchases only, CEO/CFO, all industries, allow multiple pages
base_params = {
"s": "",
"o": "", # owner filter
"pl": "",
"ph": "",
"ll": "",
"lh": "",
"fd": str(days_back), # from days back
"td": "0", # to days back
"xp": "1", "xs": "1", # include planned? (kept as in your link)
"vl": "", "vh": "",
"ocl": "", "och": "",
"sic1": "-1", "sicl": "100", "sich": "9999", # industry all
"isceo": "1", "iscfo": "1",
"grp": "0",
"sortcol": "0",
"cnt": "100",
}
frames = []
return out