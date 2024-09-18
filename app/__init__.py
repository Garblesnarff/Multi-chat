from flask import Flask
from config import Config
import os

def create_app():
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates'),
                static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static'),
                static_url_path='/static')
    app.config.from_object(Config)

    from app import routes
    app.register_blueprint(routes.bp)

    return app
