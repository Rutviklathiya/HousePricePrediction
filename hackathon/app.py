import numpy as np
from flask import Flask, request, jsonify, render_template
import random
import pickle
import pandas as pd
test = pd.read_csv("df_test_final.csv")
rand_idx = random.randrange(1000) 
final_fea = test.iloc[:,1:]
app = Flask(__name__)
model = pickle.load(open('final_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    prediction = model.predict(final_fea)
    # output = round(prediction[0], 2)
    # rand_idx = random.randrange(1000) 
    return render_template('index.html', prediction_text='Price of the house should be $ {}'.format(prediction[-1]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)