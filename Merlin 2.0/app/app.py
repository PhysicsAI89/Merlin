
import click
from flask import Flask
from commands.stocks import stocksbp
from commands.ingest import ingestbp
from commands.ml import mlbp

app = Flask(__name__)

@app.route('/')
def root():
    return 'Merlin API'

@app.cli.command("test")
@click.argument("name")
def test(name):
    print("Hi", name)

app.register_blueprint(stocksbp)
app.register_blueprint(ingestbp)
app.register_blueprint(mlbp)
