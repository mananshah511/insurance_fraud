from flask import render_template,Flask,request
import os,sys,json
from flask_cors import CORS,cross_origin
from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.pipeline.pipeline import Pipeline

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homepage():
    return render_template('index.html')

@app.route('/train',methods=['POST'])
@cross_origin()
def train():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()
        return render_template('index.html',prediction_text = "Model training completed")
    except Exception as e:
        raise InsuranceException(sys,e) from e

if __name__ == "__main__":
    app.run()