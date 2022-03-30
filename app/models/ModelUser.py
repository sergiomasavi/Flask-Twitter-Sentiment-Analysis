# Librerías
from flask_login import LoginManager, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash
import re
import pandas as pd
import numpy as np

# Paquetes
from app.models.entities.User import User
from app.models.ModelDatabase import ModelDatabase


            
class ModelUser():

    @classmethod
    def get_topics(self, database, user):
        query = f'''
        SELECT 
            topic,
            COUNT(*) AS total_tweets, 
            AVG(prediction) AS sentiment 
        FROM tweets
        WHERE user_id={int(user.get_id())}
        GROUP BY topic
        ORDER BY total_tweets
        '''

        try:
            cursor = database.connection.cursor()
            cursor.execute(query)

            row = cursor.fetchall()
            if row != None:
                result = []
                for r in row:
                    result.append({'topic':r[0], 'total_tweets':r[1], 'sentiment': np.round(100*float(r[2]))})
                return result
            else:
                return list()
        except Exception as e:
            return list()

    @classmethod
    def upload_tweets(self, database, user, tweets):
        """

        :param database:
        :param user:
        :param tweets
        :return:
        """

        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)

        tweets['user_id'] = int(user.get_id())
        cursor = database.connection.cursor()

        try:
            for i, t in tweets.iterrows():
                query = f'''SELECT * FROM tweets where id={t.id}'''
                cursor.execute(query)
                row = cursor.fetchone()

                if row != None:
                    query = f'''DELETE FROM tweets where id={t.id}'''
                    cursor.execute(query)
                    database.connection.commit()

                query ='''
                INSERT INTO tweets (id, topic, datetime, user, text, model_id, prediction, user_id) VALUES 
                ("{}","{}","{}","{}","{}",{},{}, {})'''.format(t.id, t.topic, t.datetime.strftime('%Y-%m-%d %H:%M:%S'), t.user, emoji_pattern.sub(r'', t.text.replace('"','')), t.model_id, t.prediction, t.user_id)

                try:
                    cursor.execute(query)
                except Exception as e:
                    continue

                database.connection.commit()

        except Exception as e:
            print(e)
            return False
        else:
            return True

    @classmethod
    def login(self, database, user):
        """
        Método para realizar el login del usuario comprobando la contraseña.
        :param database:
        :param user:
        :return:
        """

        try:
            cursor = database.connection.cursor()

            query = '''
            SELECT user_id, fullname, username, password FROM usuarios
            WHERE username = "{}"'''.format(user.username)
            cursor.execute(query)
            row = cursor.fetchone()

            if row != None:
                return User(user_id=row[0], fullname=row[1], username=row[2], password=User.check_password(row[3], user.password))
            else:
                return None

        except Exception as e:
            raise e

    @classmethod
    def get_by_id(self, database, user_id):
        """
        Obtener identificador único de usuario logeado.
        :param database:
        :param user_id:
        :return:
        """

        try:
            cursor = database.connection.cursor()

            query = '''
            SELECT user_id, fullname, username FROM usuarios
            WHERE user_id = "{}"'''.format(user_id)
            cursor.execute(query)
            row = cursor.fetchone()

            if row != None:
                logged_user = User(user_id=row[0], fullname=row[1], username=row[2], password=None)
                return logged_user
            else:
                return None

        except Exception as e:
            raise e

    @classmethod
    def username_exists(self, database, username):
        """
        Comprobar si el username existe y se encuentra registrado
        :param database:
        :param user_id:
        :return:
        """

        try:
            cursor = database.connection.cursor()

            query = '''
            SELECT user_id, fullname, username FROM usuarios
            WHERE username = "{}"'''.format(username)
            cursor.execute(query)
            row = cursor.fetchone()
            if row is None:
                return False
            else:
                return True

        except Exception as e:
            raise e

    @classmethod
    def signup(self, database, user):
        """
        Registrar usuario en la base de datos
        :param database:
        :param user:
        :return:
        """
        try:
            cursor = database.connection.cursor()

            query = '''
            INSERT INTO usuarios (user_id, fullname, username, password) 
            VALUES (0,"{}","{}", "{}")'''.format(user.fullname, user.username, generate_password_hash(user.password))
            cursor.execute(query)
            database.connection.commit()
        except Exception as e:
            raise e