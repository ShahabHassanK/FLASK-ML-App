from flask import Flask, render_template, request
from joblib import load
import numpy as np
import pandas as pd
import nbformat
import matplotlib.pyplot as plt

app = Flask(__name__)
@app.route('/',methods=['GET', 'POST'])
def task():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template("index.html", href = "static/smile.png")
    else:
        text = request.form['text']
        path = 'new_prediction.png'
        model = load('model.joblib')
        new_input = float_string_to_npArray(text)
        make_picture('AgesAndHeights.pkl', model, new_input, path)
        return render_template("index.html", href = path)

def make_picture(training_data_filename, model, new_inputs, output_file):
    data = pd.read_pickle(training_data_filename)
    ages = data['Age']
    data = data[ages > 0]
    ages = data['Age']
    heights = data['Height']
    x_new = np.array(list(range(19)))
    x_new_reshaped = x_new.reshape(19,1)
    preds = model.predict(x_new_reshaped)
    new_inputs_array = np.array(new_inputs)
    new_inputs_reshaped = new_inputs_array.reshape(-1,1)
    new_preds = model.predict(new_inputs_reshaped)
    
    plt.figure(figsize=(8,6))
    plt.scatter(ages,heights,color='blue',marker='o', label = 'Data Points')
    plt.scatter(new_inputs_reshaped, new_preds, color='green', marker='x', s=200, label='New Predictions')
    plt.plot(x_new_reshaped, preds, color='red', label = 'Regression Line')
    plt.title('Age vs Height Viz')
    plt.xlabel('Age (years)')
    plt.ylabel('Height (inches)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()

def float_string_to_npArray(floats_str):
    def is_float(x):
        try:
            float(x)
            return True
        except:
            return False
    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)