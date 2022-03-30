# Librerías
from flask import Blueprint

# Plano principal del plano home
home = Blueprint('home', __name__)

# Cargar vistas del blueprint
from app.home import views