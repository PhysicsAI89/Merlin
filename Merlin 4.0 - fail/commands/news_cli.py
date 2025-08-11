import click
from flask import Blueprint
from models.stockmarket import db_session, Stock
from commands.news import fetch_av_news

newsbp = Blueprint("news", __name__)

@newsbp.cli.command("fetch")
@click.argument("symbol")
def cli_fetch(symbol: str):
    wrote = fetch_av_news(symbol)
    print(f"Inserted {wrote} news rows for {symbol}")
