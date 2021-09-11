import pickle
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import numpy as np
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import spacy
from tensorflow import keras
from tensorflow.math import reduce_mean
import re


app = Flask(__name__)
Bootstrap(app)
    
def load_model():
    model = keras.models.load_model('model')
    return (model)

def Formatting(text):
    text = str(text)
    text=text.lower()
    text = re.sub(r'[^a-zA-Z0-9_\s]+','',text)
    return text

@app.route('/')
def man():
    model = load_model()
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    maxlen = 50
    Heading = Formatting(request.form['b'])
    tk = Tokenizer()
    with open('tokenizer.pkl', 'rb') as handle:
        tk = pickle.load(handle)
    X = tk.texts_to_sequences([Heading])
    X = pad_sequences(X,maxlen=maxlen,padding='post',value=0)
    model = load_model()
    pred = model.predict(X)[0]
    if np.argmax(pred) == 1:
        data = "The headline is Sarcastic"
    else:
        data ='The headline is not Sarcastic'
    return render_template('after.html', data= data)


if __name__ == "__main__":
    app.run(debug=True)
    def load_model():
        model = pickle.load(open('finalized_model.pickle', 'rb'))
        return(model)