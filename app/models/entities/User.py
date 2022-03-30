# Librerías
from werkzeug.security import check_password_hash
from flask_login import UserMixin

class User(UserMixin):

    def __init__(self, user_id, username, password, fullname=''):
        self.id = user_id
        self.username = username
        self.password = password
        self.fullname = fullname

    @classmethod # Puede utilizarse sin necesidad de instanciar la clase
    def check_password(self, hashed_password, password):
        return check_password_hash(hashed_password, password)

    def __repr__(self):
        return '<User {}>'.format(self.username)



