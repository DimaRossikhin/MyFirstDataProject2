#!/usr/bin/env python
# coding: utf-8

# ##### Flask-приложение, предсказывающее класс "Genetic Disorder Label"

# In[ ]:


import numpy as np
import click
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('C:/Users/rossi/Desktop/mfdp/models/stacking.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction #round(prediction[0], 2) 
    return render_template('index.html', prediction_text='Genetic Disorder Label :{}'.format(output))

app.run()


# ##### Проверка корректности предсказания

# In[13]:


import pickle
model = pickle.load(open('C:/Users/rossi/Desktop/mfdp/models/stacking.pkl', 'rb'))


# In[14]:


test = np.array([range(1,37)])
test = test.reshape(1, -1)
test_pred = model.predict(test)
test_pred

