# projet-pokemon

import pandas as pd

# Création d'un DataFrame
df = pd.read_csv('pokemon.csv')

# Suppression des données manquantes

df = df.dropna()

df

# Import de bibliothèques
import flask
from flask import request, jsonify

# Création de l'application
app = flask.Flask(__name__)

@app.route('/')
def hello():
    return 'hello pokemon'

app.run()

# Import de bibliothèques
import flask
from flask import request, jsonify

# Création de l'objet Flask
app = flask.Flask(__name__)

# Lancement du Débogueur
app.config["DEBUG"] = True


df

# Route permettant de récupérer toutes les données du dataframe
@app.route('/', methods=['GET'])
def api_df():
    return jsonify(df)

app.run()



# Tensorflow pour la classification d'image


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tkinter.filedialog import askopenfilename



chemin=askopenfilename(file='Desktop/PERSO/POKEMON/archive(3)/images', filetypes=[('JPEG files','.jpeg')('all files','.*')])

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
