# Librerías
import sys
import os
from flask import redirect, url_for

# Paquetes
import app as flask_appp



# Variables globales
webapp = flask_appp.create_app(config_name=os.getenv('CONFIG_ENVIRONMENT') or 'development')


# Funciones globales
def status_401(error):
    print(error)
    return redirect(url_for('home.login'))

def status_404(error):
    return '<h1>Web not found</h1>', 404

# Función principal
if __name__=='__main__':
    print('Aplicación Flask - Análisis de sentimientos en Twitter - FaysPy')
    webapp.register_error_handler(401, status_401)
    webapp.register_error_handler(404, status_404)
    webapp.run(host='0.0.0.0')