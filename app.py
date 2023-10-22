from flask import render_template,Flask,request
import os,sys,json
from flask_cors import CORS,cross_origin
from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.pipeline.pipeline import Pipeline
from insurance.entity.artifact_entity import FinalArtifact
import pandas as pd
import numpy as np
from insurance.util.util import load_object,preprocessing

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homepage():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    try:
        data = [str(x) for x in request.form.values()]
        if not os.path.exists('data.json'):
            return render_template('index.html',output_text = "No model is trained, please start training")

        with open('data.json', 'r') as json_file:
            dict_data = json.loads(json_file.read())

        final_artifact = FinalArtifact(**dict_data)
        logging.info(f"final artifact : {final_artifact}")

        train_df = pd.read_csv(final_artifact.ingested_train_data)
        train_df = train_df.iloc[:,:-1]
        columns = train_df.columns
        
        df = pd.DataFrame(data).T
        df.columns = columns
        df = pd.concat([df,train_df])
        df = preprocessing(df=df)

        df = (np.array(df.iloc[0])).reshape(1,-1)

        cluster_object = load_object(file_path = final_artifact.cluster_model_path)
        cluster_number = int(cluster_object.predict(df))

        model_object = load_object(file_path = final_artifact.export_dir_path[cluster_number])
        output = int(model_object.predict(df))
        
        if output == 0:
            return render_template('index.html',output_text = "Given person's has not commited insurance fraud")
        else:
            return render_template('index.html',output_text = "Given person's has commited insurance fraud")
    except Exception as e:
        raise InsuranceException(sys,e) from e

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