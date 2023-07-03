import click
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('models/stacking.pkl', 'rb'))

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


