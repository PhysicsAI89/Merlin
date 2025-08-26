import click
from flask import Flask
#from app.commands.stocks import stocksbp
from app.commands.ml import to_weekly
from commands.stocks import stocksbp

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Merlin 4.0 API'

@app.cli.command("test")
@click.argument("name")
def test(name):
    print('Name:', name)

app.register_blueprint(stocksbp)
