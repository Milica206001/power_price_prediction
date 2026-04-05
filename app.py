from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.logger import logging

application=Flask(__name__)

app=application

#create route for the page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
        store_nbr = int(request.form.get('store_nbr')),
        family = request.form.get('family'),
        onpromotion = int(request.form.get('onpromotion')),
        dcoilwtico = float(request.form.get('dcoilwtico')),
        transactions = float(request.form.get('transactions')),
        city = request.form.get('city'),
        state = request.form.get('state'),
        type = request.form.get('type'),
        cluster = int(request.form.get('cluster')),
        all_holiday = str(request.form.get('all_holiday'))
        )

        pred_df=data.get_data_as_data_frame()
        # Insert this right before your preprocessor.transform(df) call
        logging.info("--- DEBUGGING COLUMN TYPES ---")
        logging.info(pred_df.dtypes) 

        # This will show exactly which columns have 'NaN' or null values
        logging.info("--- MISSING VALUES PER COLUMN ---")
        logging.info(pred_df.isnull().sum())

        # This will catch strings hiding in numeric columns
        for col in pred_df.columns:
            if pred_df[col].dtype == 'object':
                print(f"Column '{col}' is an OBJECT type. First value: {pred_df[col].iloc[0]}")
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)