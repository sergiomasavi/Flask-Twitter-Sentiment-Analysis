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
    MYSQL_HOST = 'localhost'#os.getenv('MYSQL_HOST')
    MYSQL_USER = 'admin'#os.getenv('MYSQL_USER')
    MYSQL_PASSWORD = 'admin'#os.getenv('MYSQL_PASSWORD')
    MYSQL_DB = 'FaysPy'#os.getenv('MYSQL_DB')
    MYSQL_PORT = 3306#int(os.getenv('MYSQL_PORT'))
    MYSQL_UNIX_SOCKET='/Applications/XAMPP/xamppfiles/var/mysql/mysql.sock'
    PORT = 5001#int(os.getenv('PORT'))
    CONSUMER_KEY ='I4zDy3Y0BkJjPDLgfMEy0ytNS' #os.getenv('TWITTER_CONSUMER_KEY')
    CONSUMER_SECRET = 'MQ9fCfeyUJnYVKAxRHABOQrqECjKOeHgrQfPB1pfCajd7pYd6u'#os.getenv('TWITTER_CONSUMER_SECRET')
    ACCESS_TOKEN = 'I4zDy3Y0BkJjPDLgfMEy0ytNS'#os.getenv('TWITTER_ACCESS_TOKEN')
    ACCESS_TOKEN_SECRET = 'MQ9fCfeyUJnYVKAxRHABOQrqECjKOeHgrQfPB1pfCajd7pYd6u'#os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    
config = {
    'development': DevelopmentConfig
}