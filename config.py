import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    """
    Clase config.
    """
    SECRET_KEY = 'lLlHsxCSGrf05X8_3wOtpg'

    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    DEBUG = True
    MYSQL_HOST = os.getenv('MYSQL_HOST')
    MYSQL_USER = os.getenv('MYSQL_USER')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
    MYSQL_DB = os.getenv('MYSQL_DB')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT'))
    PORT = int(os.getenv('PORT'))
    CONSUMER_KEY = os.getenv('TWITTER_CONSUMER_KEY')
    CONSUMER_SECRET = os.getenv('TWITTER_CONSUMER_SECRET')
    ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
    ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    
config = {
    'development': DevelopmentConfig
}