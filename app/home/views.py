# Librerías
from flask import render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
import numpy as np

# Paquetes
from app.home import home
from app.models.ModelUser import ModelUser
from app.models.entities.User import User
from .. import database
from .. import login_manager
from .. import twitter_api
from app.models.Model import Model

    
@login_manager.user_loader
def load_user(user_id):
    return ModelUser.get_by_id(database, user_id)

@home.route('/')
def index():
    return redirect(url_for('home.login'))

@home.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('auth/signup.html')
    else:
        if request.form['name'] == '' or request.form['lastname'] == '' or request.form['username'] == '' or request.form['password'] == '':
            flash('All fields are required!')
            return render_template('auth/signup.html')
        elif len(request.form['password']) < 8:
            flash('Password must have a least 8 characters long!')
            return render_template('auth/signup.html')

        else:

            new_user = User(user_id=0, fullname=request.form['name'] +' '+ request.form['lastname'],
                            username=request.form['username'], password=request.form['password'])

            if not ModelUser.username_exists(database, new_user.username):
                ModelUser.signup(database, new_user)
                return render_template('auth/login.html')
            else:
                flash('Username already exists!')
                return render_template('auth/signup.html')

@home.route('/login', methods=['GET', 'POST'])
def login():
    # Comprobar si se ha enviado el formulario
    if request.method == 'POST':
        if request.form['username'] == '' and request.form['password'] == '':
            flash('Los campos username y password son obligatorios')
        elif request.form['username'] == '':
            flash('El campo username es obligatorio')
        elif request.form['password'] == '':
            flash('El campo password es obligatorio')

        else:
            user = User(user_id=0, username=request.form['username'], password=request.form['password'])
            logged_user = ModelUser.login(database=database, user=user)
            if logged_user != None:
                # Comprobamos si coincide el password
                if logged_user.password:
                    login_user(logged_user)
                    return redirect(url_for('home.main'))
                else:
                    flash('Contraseña invalida!')
            else:
                flash('Usuario no encontrado!')

        return render_template('auth/login.html')
    else:
        return render_template('auth/login.html')
    #return 'Hello world! 201'

@home.route('/home_search', methods=['POST'])
def search_topic():
    tweets = None
    card_list = list()
    if request.method=='POST':

        try:
            tweets = twitter_api.get_tweets_by_topic(q=request.form['topic'])
            models = Model.get_models(database)
            tweets = Model.predict(tweets, models)

        except Exception as e:
            flash('Error. Failed getting topic from twitter')

        else:
            uploaded = ModelUser.upload_tweets(database, current_user, tweets)
            if not uploaded:
                flash('Error. Failed to complete the evaluation')


        card_list = ModelUser.get_topics(database, current_user)

        if not card_list:
            flash('Error. Failed getting top 5 topics')

        return render_template('home.html', card_list=card_list)

    else:
        if tweets != None:
            card_list = ModelUser.get_topics(database, current_user)

            if not card_list:
                flash('Error. Failed getting top 5 topics')

        return render_template('home.html', card_list=card_list)

@home.route('/home')
def main():
    
    card_list = ModelUser.get_topics(database, current_user)
    if not card_list:
        card_list=[]
        
    return render_template('home.html', card_list=card_list)



@home.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home.login'))

