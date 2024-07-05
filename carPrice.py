from flask import Flask,request,render_template
from source.main_project.pipeline.predict_pipeline import PredicPipeline,UserData
import numpy as np
import pandas as pd

app = Flask(__name__)
data = pd.read_csv('C:/Users/odhia/OneDrive/Desktop/mlopsProjects/notebook/data/cleaned_car.csv')
model_names = data.model.value_counts().index

@app.route('/')
def homepage():
    return render_template('carpricepred.html' ,model_names = model_names)

@app.route('/predict',methods=['POST'])
def do_prediction():
    user_input = UserData(
        model=request.form.get('mod'),
        mileage=request.form.get('mil'),
        vol_engine=request.form.get('vol'),
        fuel=request.form.get('fuel')
        )
    
    user_df = user_input.get_data_as_df()
    
    predictpipe = PredicPipeline()
    results = np.round(predictpipe.predict(user_df),2)
    
    msg = f"Estimated price is ${results}"
    
    return render_template('carpricepred.html',model_names = model_names, text=msg)

if __name__ == "__main__":
    app.run(debug=True)