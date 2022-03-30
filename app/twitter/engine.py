# Libraries
import os
import tweepy as tw
import pandas as pd


class Twitter():

    def __init__(self, consumer_key=None, consumer_secret=None, access_token=None, access_token_secret=None):
        """
        Constructor
        """
        self.__auth__api()

    def __auth__api(self):
        """
         # Autentificación de la app de twitter (http://tweepy.readthedocs.org/en/v3.1.0/getting_started.html#api)

        Returns:
            None.
        """
        # Claves de acceso
        consumer_key = 'I4zDy3Y0BkJjPDLgfMEy0ytNS'
        consumer_secret = 'MQ9fCfeyUJnYVKAxRHABOQrqECjKOeHgrQfPB1pfCajd7pYd6u'
        access_token = '1495840711366918147-nTrnWgwIybwp7hfOH0I6X7C1j5o4BU'
        access_token_secret = 'wmjfujIYTpJKQfHcg9XKsRauvpB8044wavENXzJJNvW5t'

        try:
            auth = tw.OAuthHandler(consumer_key=consumer_key, consumer_secret=consumer_secret)
            auth.set_access_token(key=access_token, secret=access_token_secret)
            self.api = tw.API(auth, wait_on_rate_limit=True)
            redirect_url = auth.get_authorization_url()

        except Exception as e:
            print('Error! Failed to get request token.')
        else:
            print('twitter api connected!')


    def get_tweets(self, username, tweets_count):
        """
        Método de obtención de tweets a partir del username de un usuario

        Referencias:
            - Cursor tutorial => https://docs.tweepy.org/en/v3.5.0/cursor_tutorial.html
            - User time line in tweepy => https://www.geeksforgeeks.org/python-api-user_timeline-in-tweepy/
        Args:
            usrename [str] -- Username de la cuenta de twitter
            tweets_count [int] -- Número de tweets.
        """

        # Generar cursor de tweets
        for t in tw.Cursor(self.api.user_timeline, screen_name=username).items(tweets_count):
            print('Author info:')
            print('\tName:', t.author.name)
            print('\tDescription:', t.author.description)
            print('\tID:', t.author.id_str)
            print('\tVerified:', t.author.verified)
            print('\tLocation:', t.author.location)
            print('\tFriends count:', t.author.friends_count)
            print('\tFollowers:', t.author.followers_count)
            print('\tGeo enabled:', t.author.geo_enabled)
            print('\tFollowing:', t.author.following)

            print('Tweet:')
            print('\tCreated_at:', t.created_at)
            print('\tCoordinates:', t.coordinates)
            print('\tID:', t.id_str)
            print('\tFavorites_Count:', t.favorite_count)
            print('\tRetweetCount:', t.retweet_count)

            print('\tHasthtags:', t.entities['hashtags'])
            print('\tSymbols:', t.entities['symbols'])
            print('\tUser_mentions:', t.entities['user_mentions'])
            print('\tUrls:', t.entities['urls'])
            print('\tSource:', t.source)
            print('\tSource url:', t.source_url)
            print('\tText:', t.text)
            print(10 * '-', end=2 * '\n')

    def get_tweets_by_topic(self, q):
        MAX_TWEETS = 500

        # Generar cursor de tweets
        tweets = list()
        for t in tw.Cursor(self.api.search_tweets, q=q, result_type='mixed', lang='en', count=MAX_TWEETS).items(MAX_TWEETS):
            tweets.append(pd.Series(
                {'id': int(t.id_str), 'topic': q, 'datetime': t.created_at, 'user': t.author.screen_name,
                 'text': t.text}).to_frame().T)

        return pd.concat(tweets).reset_index(drop=True)