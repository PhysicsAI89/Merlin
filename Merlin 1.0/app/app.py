import click
from flask import Flask

from commands.stocks import stocksbp
from commands.analyse import analysebp




app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.cli.command("test")
@click.argument("name")
def test(name):
    print('Name: ' , name)


app.register_blueprint(stocksbp)
app.register_blueprint(analysebp)


from commands.ml import mlbp
app.register_blueprint(mlbp)

