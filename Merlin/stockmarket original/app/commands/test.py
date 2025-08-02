import click
from flask import Flask

app = Flask(__name__)

@app.cli.command("test")
@click.argument("name")
def test(name):
	print('Name: ' . name)
