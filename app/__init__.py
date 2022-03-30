# Librerías
import sys
import os
from flask import Flask

# Paquetes
from config import config
from app.database import database
from app.login_manager import login_manager
from app.csrf_users import csrf
from app.twitter.engine import Twitter

# Instanciar clase Twitter
twitter_api = Twitter()

# Funciones
def create_app(config_name):
    """
    Permite crear aplicación Flask.
    """
    # Instanciar aplicación Flask
    webapp = Flask(__name__, template_folder='templates')

    # Secret Key de la aplicación
    webapp.secret_key = config[config_name].SECRET_KEY

    # Configuración de la aplicación Flask
    webapp.config.from_object(config[config_name])
    config[config_name].init_app(webapp)
    webapp.config.update(config)

    # Iniciar database en la aplicación
    database.init_app(app=webapp)
    
    # Inicializar csrf.
    csrf.init_app(webapp)

    # Iniciar gestor del login de usuarios de la aplicación
    login_manager.init_app(webapp)

    # Registrar planos de la aplicación
    with webapp.app_context():
        # Plano home
        from app.home import home as home_blueprint
        webapp.register_blueprint(home_blueprint)

    return webapp
