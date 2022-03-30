import pandas as pd
import numpy as np
import os
from app.nlp.nlp import Fayspy_Twitter

class Model():

    @classmethod
    def get_models(self, database):
        query = f'''
              SELECT id AS model_id, name AS model_name
              FROM models
              '''

        try:
            cursor = database.connection.cursor()
            cursor.execute(query)

            row = cursor.fetchall()
            if row != None:
                result = []
                for r in row:
                    result.append(pd.Series({'model_id': r[0], 'model_name': r[1]}).to_frame().T)  # 100*float(r[2])})

                models_df = pd.concat(result).reset_index(drop=True)

                return models_df
            else:
                return pd.DataFrame()

        except Exception as e:
            return pd.DataFrame()

    @classmethod
    def predict(self, tweets, models):

        tweets_df = list()
        for model_id in models.model_id.unique():
            tweets['model_id'] = model_id
            tweets_df.append(tweets.copy())

        tweets_df = pd.concat(tweets_df, axis=0).reset_index(drop=True)
        tweets_predictions = list()
        for i, r in models.iterrows():
            df = tweets_df[tweets_df['model_id']==i+1]
            model_name = models.iloc[i]['model_name']
            filename = os.path.join(os.getenv('MODEL_FILEPATH'),model_name +'.h5')
            if df.shape[0] != 0:
                m = Fayspy_Twitter().load_model_from_file(filename)
                predictions = Fayspy_Twitter().predict_loaded_model(model=m, data=df)
                if type(predictions).__name__=='DataFrame':
                    join_df = df.join(predictions['prediction']).dropna(axis=0)
                    tweets_predictions.append(join_df)

        return pd.concat(tweets_predictions).reset_index(drop=True)