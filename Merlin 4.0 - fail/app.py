import click
from flask import Flask

from commands.stocks import stocksbp
from commands.ml import mlbp
from commands.news_cli import newsbp

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Merlin 4.0 — Flask CLI host'

app.register_blueprint(stocksbp)
app.register_blueprint(mlbp)
app.register_blueprint(newsbp)
