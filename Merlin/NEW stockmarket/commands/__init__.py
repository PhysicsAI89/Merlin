from flask import Flask
from .models import init_db, db_session
from .commands.stocks import stocksbp
from .commands.analyse import analysebp

def create_app():
    app = Flask(__name__)

    # Load configuration
    app.config.from_mapping(
        SECRET_KEY='your_secret_key_here',
        # Add other configuration variables as needed
    )

    # Initialize database
    init_db()

    # Register blueprints
    app.register_blueprint(stocksbp)
    app.register_blueprint(analysebp)

    @app.route('/')
    def hello_world():
        return 'Hello World!'

    return app
