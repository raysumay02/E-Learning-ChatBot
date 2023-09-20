import random
import json
import pickle
import time

import nltk
# nltk.download('popular')
from nltk.stem import WordNetLemmatizer

import numpy as np

import tensorflow as tf

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    # tokenize the pattern - split into arrays
    sentence_words = nltk.word_tokenize(sentence)
    # create short form of words (ignore tense and synonyms)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# return bag of words '0' or '1' for each word in the bag that exists in the sentence
def bag_of_words(sentence):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # matrix of N words(vocabulary matrix)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    # filter out predictions below a threshold
    b_words = bag_of_words(sentence)
    res = model.predict(np.array([b_words]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by greatest probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_Response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            ans = random.choice(i['responses'])
            break
        else:
            ans = "Sorry I couldn't understand what you meant"
    return ans


def chatbot_response(msg):
    ints = predict_class(msg)
    res = get_Response(ints, intents)
    time.sleep(1.5)
    return res


# Uncomment for use in terminal
# while True:
#    message = input("Question\n")
#    ints = predict_class(message)
#    res = get_Response(ints, intents)
#    print(res)

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()
