# Procesamiento de datos
from this import d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
import string

# TensorFlow
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras as ks
from tensorflow.keras.callbacks import EarlyStopping

#Library for Text Processing
import texthero as hero

#Sk Learn Library
from sklearn.model_selection import train_test_split

# General
import os
import time
import datetime
import warnings
warnings.filterwarnings("ignore")

class Fayspy_Twitter():
    """
    Objeto diseñado por Fayspy para clasificar el sentimiento de los tweets en dos
    positivo (1) o negativo (0) mediante Transfer Learning
    """
    # Atributos de clase
    models = dict()  # Diccionario de modelos utilizados por el objeto.

    def __init__(self):
        """
        Constructor.
        """
        print('FaysPy Twitter Natural Language Processing created!')

    @classmethod
    def validate_df(self, df):
        """
        Validar que el dataset cumple con las restricciones necesarias.
          - Las columnas tener por nombre ids, text y sentiment.
          - La variable objetivo debe ser binaria (0,1)

        Args:
        -----
          df [pandas.DataFrame] -- Dataset a validar

        Returns:
          validated [bool] -- True si el dataset cumple las restricciones y 0 si no
                              las cumple.
        """
        vars = sorted(['ids', 'text', 'target'])

        # 1. Comprobar que las columnas son 'text' y 'sentiment'.
        if not sorted([x for x in vars if x in df.columns]) == vars:
            raise ('Error! El dataset debe contener las columnas text y target')
            return Fasle

        #  2. Comprobar que la variable target tiene dos posibles valores
        if not df.target.unique().shape[0] == 2:
            raise ('Error! La variable objetivo debe ser binaria (sentimiento positivo = 0; sentimiento negativo = 1)')
            return False

        return True

    @classmethod
    def prepare_dataset(self, data_df):
        """
        Método que permite pre-procesar el dataset original de tweets utilizado
        en el entrenamiento de los modelos de aprendizaje.

        Args:
        ------
          data_df [pandas.DataFrame] -- Conjunto de entrenamiento. Debe contener las
                                        columnas 'text' y 'sentiment' y debe estar
                                        indexado con el id de los tweets.

        Returns:
        ------
          nlp_df [pandas.DataFrame] -- Dataset preparado para ser utilizado por el método
                                       train_val_split().
        """
        nlp_df = data_df.copy()#.sort_values('date', ascending=True)
        print('Dimensionalidad:', data_df.shape)

        #  Seleccionar columnas de interés
        print('Selecting vars')
        nlp_df = nlp_df[['text']]
        #nlp_df.set_index('ids', inplace=True)

        # Elimnar duplicados
        print('Drop duplicates...')
        nlp_df.drop_duplicates(keep='first', inplace=True)
        print('Dimensionalidad:', nlp_df.shape)

        #  Pre-procesamiento del texto
        def clean_ascii(text):
            # function to remove non-ASCII chars from data
            return ''.join(i for i in text if ord(i) < 128)

        print('Cleaning ascii...')
        nlp_df['text'] = nlp_df['text'].apply(clean_ascii)

        # Convertir palabras a minúsculas
        print('Lower text...')
        nlp_df['text'] = nlp_df['text'].str.lower()

        # Limpieza y eliminar stop words en inglés
        print('Cleaning stopwords...')
        stopwords_list = list(stopwords.words('english'))

        def cleaning_stopwords(text):
            return ' '.join([word for word in str(text).split() if word not in stopwords_list])

        nlp_df['text'] = nlp_df['text'].apply(lambda x: cleaning_stopwords(x))

        # Limpieza y eliminar signos de puntuación
        print('Cleaning english punctuations...')
        english_punctuations = string.punctuation

        def cleaning_punctuations(text):
            translator = str.maketrans('', '', english_punctuations)
            return text.translate(translator)

        nlp_df['text'] = nlp_df['text'].apply(lambda x: cleaning_punctuations(x))

        # Limpieza y eliminar caracteres repetidos
        print('Cleaning repeating chars...')

        def cleaning_repeating_char(text):
            return re.sub(r'(.)\1+', r'\1', text)

        nlp_df['text'] = nlp_df['text'].apply(lambda x: cleaning_repeating_char(x))

        # Limpieza y eliminar emails
        print('Cleaning emails...')

        def cleaning_email(data):
            return re.sub('@[^\s]+', ' ', data)

        nlp_df['text'] = nlp_df['text'].apply(lambda x: cleaning_email(x))

        # Limpieza y eliminar URL's
        print('Cleaning urls...')

        def cleaning_URL(data):
            return re.sub('((www\.[^\s]+)(https:?://[^\s]+))', ' ', data)

        nlp_df['text'] = nlp_df['text'].apply(lambda x: cleaning_URL(x))

        # Limpieza y eliminar numeros
        print('Cleaning numbers...')

        def cleaning_numbers(data):
            return re.sub('[0-9]+', ' ', data)

        nlp_df['text'] = nlp_df['text'].apply(lambda x: cleaning_numbers(x))

        # Tokenización del texto de los tweets
        print('Tokenizing text...')
        tokenizer = RegexpTokenizer(r'\w+')
        nlp_df['text'] = nlp_df['text'].apply(tokenizer.tokenize)

        #  Aplicar Stemming
        #print('Applying stemming...')
        #st = nltk.PorterStemmer()

        def stemming_on_text(data):
            text = [st.stem(word) for word in data]
            return data

        #nlp_df['text'] = nlp_df['text'].apply(lambda x: stemming_on_text(x))

        # Aplicar Lemmatizer
        #print('Applying lemmatizer...')
        #nltk.download('wordnet')
        #nltk.download('omw-1.4')
        #lm = nltk.WordNetLemmatizer()

        def lemmatizer_on_text(data):
            text = [lm.lemmatize(word) for word in data]
            return text

        #nlp_df['text'] = nlp_df['text'].apply(lambda x: lemmatizer_on_text(x))

        # Conversión de vector a lista para la red neuronal
        nlp_df['text'] = nlp_df['text'].apply(lambda x: ' '.join(x))

        return nlp_df

    def train_test_split(self, nlp_df, test_size=.2, val_size=.1, feature='text'):
        """
        Método que divide el dataset en conjunto de entrenamiento y validación.

        Args:
        -----
            nlp_df [pandas.DataFrame] -- Conjunto de datos resultado del método
                                          prepare_dataset()

            test_size [float] -- Tamaño del conjunto de test. Debe ser un valor
                                 en el intervalo abierto (0, 1).

            val_size [float] -- Tamaño del conjunto de validación. Debe ser un valor
                                en el intervalo abierto (0, 1).

            text_column [str] -- Columna a utilizar como predictora (text/text_clean)


        Returns:
        -----
            train_df [pandas.DataFrame] -- Conjunto de datos de entrenamiento.
            val_df [pandas.DataFrame] -- Conjunto de datos de validación.
            cleaned_train_df [pandas.DataFrame] -- Conjunto de datos de entrenamiento pre-procesado.
            cleaned_val_df [pandas.DataFrame] -- Conjunto de datos de validación pre-procesado.
        """
        if feature not in ['text', 'text_clean']:
            raise ('Error! La variable utilizada para entrenar debe ser "text" o "text_clean"')

        # Almacenar variables descriptivas y objetivo (X, y)
        X = nlp_df[feature]
        y = nlp_df['target']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train)

        # Formar datasets de entrenamiento y validación
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)

        # Estandarizar nombre de las variables de los dataframes
        renamed_columns = {feature: 'text', 'target': 'sentiment'}
        train_df.rename(renamed_columns, axis=1, inplace=True)
        test_df.rename(renamed_columns, axis=1, inplace=True)
        val_df.rename(renamed_columns, axis=1, inplace=True)

        # Mostrar dimensionalidad de los dataset de entrenamiento y validación.
        print('Tweets in Train Dataset = {} ({:.2f}%)'.format(train_df.shape[0],
                                                              100 * train_df.shape[0] / nlp_df.shape[0]))
        print(
            'Tweets in Test Dataset = {} ({:.2f}%)'.format(test_df.shape[0], 100 * test_df.shape[0] / nlp_df.shape[0]))
        print('Tweets in Validation Dataset = {} ({:.2f}%)'.format(val_df.shape[0],
                                                                   100 * val_df.shape[0] / nlp_df.shape[0]))
        print('\n')

        print('Target distribution on train set')
        print(train_df.sentiment.value_counts())
        print('\n')

        print('Target distribution on test set')
        print(test_df.sentiment.value_counts())
        print('\n')

        print('Target distribution on validation set')
        print(val_df.sentiment.value_counts())
        print('\n')

        return train_df, test_df, val_df

    def download_model(self, model_url):
        """
        Descargar modelo pre-entrenado desde un repositorio público.

        Args:
        -----
          model_url [str] -- Url para utilizar el modelo pre-entrenado.


        Returns:
          tf_model -- Modelo pre-entrenado que se utilizará para realizar el transfer learning.
        """
        print(f'Descargando modelo desde "{model_url}"')
        try:
            tf_model = hub.KerasLayer(model_url, input_shape=[], dtype=tf.string, trainable=True)
            print(tf_model)
        except Exception as e:
            raise ('Error! No se pudo descargar el modelo pre-entrenado!')
            return None
        else:
            print('Modelo pre-entrenado descargado con exito!')
            return tf_model

    def define_model(self, model_url, model_name):
        """
        Permite crear una red neuronal por medio de transfer learning.

        Args:
        -----
          model_url [str] -- Url para utilizar el modelo pre-entrenado.
          model_name [str] -- Nombre del modelo pre-entrenado utilizado.
        """

        # Descargar modelo pre-entrenado
        # Construir modelo de análisis de sentimiento.
        print('Building model...')
        model = tf.keras.Sequential(name=model_name)
        hub_model = self.download_model(model_url)
        model.add(hub_model)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='predictions'))

        print(f'--{model_name} Model Summary --')
        print(model.summary())

        # Compilar modelo
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

        # Guardar modelo en atributo de la clase.
        self.models[model_name] = model
        print(f'Total defined models = {len(self.models.keys())}')
        print(2 * '\n')

    def train(self, model_name, train_df, val_df, test_df, epochs=5, batch_size=216, patience=20, drive_path=None):
        """
        Permite entrenar una red neuronal por medio de transfer learning definida
        previametne.

        Nota: Tanto el conjunto de entrenamiento como el conjunto de validación debe
        ser un dataframe de pandas con las siguientes columnas:
          - text. Texto de los tweets.
          - sentiment. Sentimiento del tweet codificado como positivo (1) o negativo (0).


        Args:
        -----
          model_name [str] -- Nombre del modelo previamente compilado.
          train_df [pandas.DataFrame] -- Conjunto de entrenamiento.
          val_df [pandas.DataFrame] -- Conjunto de validación.
          test_df [pandas.DataFrame] -- Conjunto de test.
          epochs [int] -- Epócas de entrenamiento (Default: 20).
          batch_size [int] -- (Default: 216).
          patience=20 [float] -- Paciencia (Default = 20)
        """
        if model_name not in self.models.keys():
            raise ('Error! El modelo indicado no ha sido compilado. Revise el nombre del modelo!')

        print('Training...')
        print('Patience =', patience)
        print('Epochs =', epochs)
        print('Batch Size =', batch_size)
        m = self.models[model_name]
        print(m.summary())

        # Callbacks
        logdir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        val_loss_callback = EarlyStopping(monitor='val_loss', patience=patience)
        val_accuracy_callback = EarlyStopping(monitor='val_binary_accuracy', patience=patience)

        # Entrenamiento
        t = time.perf_counter()
        history = m.fit(train_df['text'], train_df['sentiment'],
                        epochs=epochs,
                        use_multiprocessing=True,
                        batch_size=batch_size,
                        validation_data=(val_df['text'], val_df['sentiment']),
                        callbacks=[
                            tensorboard_callback,
                            val_loss_callback,
                            val_accuracy_callback
                        ])
        elapsed_time = datetime.timedelta(seconds=(time.perf_counter() - t))
        print('Train elapsed time =', elapsed_time)

        # Guarda el modelo.
        self.models[model_name] = m
        if drive_path is not None:
            ks.models.save_model(m,
                                 filepath=drive_path + f'{model_name}.h5',
                                 overwrite=True,
                                 include_optimizer=True,
                                 save_format='h5'
                                 )

        # Evaluar modelo
        filename_figure = drive_path + f'{model_name}.png'
        self.evaluate_model(model_name, history, test_df, filename_figure)

    def evaluate_model(self, model_name, history, test_df, filename_figure):
        """
        Evaluación del modelo.

        Args:
          model_name [str] -- Nombre del modelo previamente compilado.
          history [] -- Historia de entrenamiento del modelo.
          test_df [pandas.DataFrame] -- Conjunto de test
        """
        print(f'Evaluating {model_name}...')
        if model_name not in self.models.keys():
            raise ('Error! El modelo indicado no ha sido compilado. Revise el nombre del modelo!')

        # Seleccionar modelo
        m = self.models[model_name]

        # Evaluación del modelo
        _, acc = m.evaluate(test_df['text'], test_df['sentiment'])

        print('Accuracy = {:.3f}%'.format(acc * 100))

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(30, 25))

        #  Loss
        _ = sns.lineplot(data=history.history['loss'], color='blue', label='train', marker='o', ax=ax1)
        _ = sns.lineplot(data=history.history['val_loss'], color='orange', label='train', marker='o', ax=ax1)
        _ = ax1.set_xlabel('Epochs', fontsize=20)
        _ = ax1.set_ylabel('Loss', fontsize=20)
        _ = ax1.set_title('Cross Entropy Loss', fontsize=28)
        _ = ax1.tick_params(axis='x', labelsize=14)
        _ = ax1.tick_params(axis='y', labelsize=14)

        # Accuracy
        _ = sns.lineplot(data=history.history['binary_accuracy'], color='blue', label='val', marker='o', ax=ax2)
        _ = sns.lineplot(data=history.history['val_binary_accuracy'], color='orange', label='train', marker='o', ax=ax2)
        _ = ax2.set_xlabel('Epochs', fontsize=20)
        _ = ax2.set_ylabel('Accuracy', fontsize=20)
        _ = ax2.set_title('Binary Classification Accuracy', fontsize=28)
        _ = ax2.tick_params(axis='x', labelsize=14)
        _ = ax2.tick_params(axis='y', labelsize=14)

        _ = plt.show()
        fig.savefig(filename_figure)

    @classmethod
    def load_model_from_file(self, filename):
        """
        Lectura de modelo almacenado localmente en formato 'tf'

        Args:
          model_name [str] -- Nombre del modelo
        """
        try:
            m = ks.models.load_model(filename, custom_objects={'KerasLayer': hub.KerasLayer})
        except Exception as e:
            return None
        else:
            print(m.summary())
            print(2 * '\n')
            return m
        
    @classmethod 
    def predict_loaded_model(self, model, data):
        data = data.copy()
        try:
            data = self.prepare_dataset(data_df=data)
            data['prediction'] = model.predict(data)
        except Exception as e:
            return None
        else:
            return data