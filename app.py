# -*- coding: utf-8 -*-
####Dependencies
#1) Python 3
#2) flask, numpy, keras, tensorflow, pickle, logging
#these libraries can be installed one after another by typing library name after the command 'pip install'
#for example: pip install flask

#####Save model file and tokenizer pickle file in a single folder
#model.json
#model.h5
#tokenizer.pickle

#navigate to the folder which has app.py file in your computer and type below command and press enter to initialize the API
#python app.py

#call api by passing text after text = in below command to get the desired label for text
#http://localhost:5000/predict?text=

import os
from flask import Flask, request, jsonify
import numpy as np

from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

import pickle

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

np.random.seed(1337)
# Максимальное количество слов 
num_words = 10000
# Максимальная длина новости
max_news_len = 50
# Количество классов новостей
nb_classes = 64

# graph = tf.get_default_graph()
classes = ['Автомобильное право','Авторские и смежные права','Административное право','Алименты','Арбитраж','Банкротство','Бухгалтерский учет','Взыскание задолженности','Военное право','Гарантии, льготы, компенсации','Гражданское право','Гражданство','ДТП, ГИБДД, ПДД','Доверенности нотариуса','Договорное право','Долевое участие в строительстве','ЖКХ','Жилищное право','Заключение и расторжение брака','Защита прав потребителей','Защита прав призывников','Защита прав работников','Защита прав работодателя','Земельное право','Интеллектуальная собственность','Интернет и право','Ипотека','Исполнительное производство','Конституционное право','Корпоративное право','Кредитование','Лицензирование','Лишение водительских прав','Материнский капитал','Медицинское право','Международное право','Миграционное право','Налоговое право','Наркотики','Наследство','Недвижимость','Нотариат','ОСАГО, Каско','Общие вопросы','Пенсии и пособия','Побои','Получение образования','Права детей','Право собственности','Предпринимательское право','Приватизация','Программы ЭВМ и базы данных','Произвол чиновников','Раздел имущества','Регистрация юридических лиц','Семейное право','Социальное обеспечение','Страхование','Таможенное право','Тендеры, контрактная система в сфере закупок','Товарные знаки, патенты','Трудовое право','Уголовное право','Усыновление, опека и попечительство','Хищения']
#star Flask application
app = Flask(__name__)

#Load model
path = './models'
json_file = open(path+'/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
keras_model_loaded = model_from_json(loaded_model_json)
keras_model_loaded.load_weights(path+'/model.h5')
print('Model loaded...')

#load tokenizer pickle file
with open(path+'/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    print('Tokenizer loaded...')


def preprocess_text(texts,max_news_len = 50):
    #tok = Tokenizer(num_words=num_max)
    #tok.fit_on_texts(texts)
    cnn_texts_seq = tokenizer.texts_to_sequences(texts)
    cnn_texts_mat = pad_sequences(cnn_texts_seq,maxlen=max_news_len)
    return cnn_texts_mat

# URL that we'll use to make predictions using get and post
@app.route('/predict',methods=['GET','POST'])
# @app.route('/predict',methods=['POST'])

def predict():
    text = request.args.get('text')
    x = preprocess_text([text])
    classifier = keras_model_loaded.predict(x)
    classifier = int(np.argmax(classifier)+1)
    return jsonify({'prediction': classes[classifier],"text": [text]})

