import os,sys
from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance.entity.config_entity import ModelEvulationConfig
from insurance.entity.artifact_entity import ModelTrainerArtifact,ModelEvulationArtifact,DataTransformArtifact
from insurance.util.util import read_yaml,write_yaml_file,load_object
from insurance.entity.model_factory import get_evulated_classification_model
from insurance.constant import BEST_MODEL_KEY,HISTORY_KEY,MODEL_PATH_KEY
import pandas as pd
import numpy as np

class ModelEvulation:

    def __init__(self,model_evulation_config:ModelEvulationConfig,
                 model_trainer_artifact:ModelTrainerArtifact,
                 data_transform_artifact:DataTransformArtifact) -> None:
        try:
            logging.info(f"{'>>'*20}Model evulation log started.{'<<'*20} \n\n")
            self.model_evulation_config = model_evulation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transform_artifact = data_transform_artifact
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def get_best_model(self,cluster_number):
        try:
            logging.info(f"get best model function started")

            model_evulation_path = self.model_evulation_config.evulation_file_path
            model = None

            CLUSTER_NUMBER = 'cluster'+str(cluster_number)

            if not os.path.exists(model_evulation_path):
                write_yaml_file(file_path=model_evulation_path)
                return model
            
            model_evulation_file_content = read_yaml(file_path=model_evulation_path)

            model_evulation_file_content = dict() if model_evulation_file_content is None else model_evulation_file_content

            if CLUSTER_NUMBER not in model_evulation_file_content:
                return model
            
            model = load_object(model_evulation_file_content[CLUSTER_NUMBER][BEST_MODEL_KEY][MODEL_PATH_KEY])

            return model

        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def update_evulation_report(self,cluster_number,model_evulation_artifact:ModelEvulationArtifact):
        try:
            logging.info(f"update evulation report function started")
            model_evulation_path = self.model_evulation_config.evulation_file_path
            
            model_evulation_file_content = read_yaml(file_path=model_evulation_path)

            model_evulation_file_content = dict() if model_evulation_file_content is None else model_evulation_file_content

            CLUSTER_NUMBER = 'cluster'+str(cluster_number)

            previous_best_model = None
            CLUSTER_HISTORY = CLUSTER_NUMBER+'_history'

            if CLUSTER_NUMBER in model_evulation_artifact:
                previous_best_model = model_evulation_file_content[CLUSTER_NUMBER][BEST_MODEL_KEY]

            logging.info(f"previous evulation report: {model_evulation_file_content}")
            
            eval_result = {CLUSTER_NUMBER:{BEST_MODEL_KEY:{MODEL_PATH_KEY:model_evulation_artifact.evulation_model_file_path}}}

            if previous_best_model is not None:
                model_history = {self.model_evulation_config.time_stamp:previous_best_model}

                if CLUSTER_HISTORY not in model_evulation_file_content:
                    cluster_history = {CLUSTER_HISTORY:model_history}
                    eval_result.update(cluster_history)

                else:
                    model_evulation_file_content[CLUSTER_HISTORY].update(model_history)
        
            model_evulation_file_content.update(eval_result)
            logging.info(f"updated evulation result : {model_evulation_file_content}")

            write_yaml_file(file_path=model_evulation_path,data=model_evulation_file_content)
            logging.info(f"updated evulation file")

        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def intiate_model_evulation(self)->ModelEvulationArtifact:
        try:
            logging.info(f"intiate model evulation function started")

            transform_train_files = self.data_transform_artifact.transform_train_dir
            tranform_test_files = self.data_transform_artifact.transform_test_dir

            train_files = os.listdir(transform_train_files)
            test_files = os.listdir(tranform_test_files)

            trained_model_name = os.path.basename(self.model_trainer_artifact.trained_model_path)
            trained_model_dir = os.path.dirname(self.model_trainer_artifact.trained_model_path)

            model_evulation_artifact = None
            cluster_model_path_list = []

            for cluster_number in range(len(train_files)):
                logging.info(f"{'>>'*20}cluster : {cluster_number}{'<<'*20}")

                cluster_model_path = os.path.join(trained_model_dir,'cluster'+str(cluster_number),trained_model_name)
                cluster_model_path_list.append(cluster_model_path)

                train_file_name = train_files[cluster_number]
                test_file_name = test_files[cluster_number]

                train_file_path = os.path.join(transform_train_files,train_file_name)
                test_file_path = os.path.join(tranform_test_files,test_file_name)

                logging.info(f"reading train data from the file : {train_file_path}")    
                train_df = pd.read_csv(train_file_path)
                logging.info(f"train data reading successfull")
                logging.info(f"reading test data from the file : {test_file_path}")    
                test_df = pd.read_csv(test_file_path)
                logging.info(f"test data reading successfull")

                logging.info("splitting data into input and output feature")
                X_train,y_train,X_test,y_test = train_df.iloc[:,:-1],train_df.iloc[:,-1],test_df.iloc[:,:-1],test_df.iloc[:,-1]

                model = self.get_best_model(cluster_number=cluster_number)

                trained_model_object = load_object(file_path=cluster_model_path)

                if model is None:
                    logging.info(f"no model found hence accepting model")

                    model_evulation_artifact = ModelEvulationArtifact(evulation_model_file_path=cluster_model_path,
                                                                      is_model_accepted=True)
                    self.update_evulation_report(cluster_number=cluster_number,model_evulation_artifact=model_evulation_artifact)

                    continue

                model_list = [model , trained_model_object]
                base_accuracy = self.model_trainer_artifact.model_accuracy
                base_accuracy = base_accuracy[cluster_number]

                metric_info_artifact = get_evulated_classification_model(model_list=model_list,X_train=np.array(X_train),
                                                                         X_test=np.array(X_test),y_train=np.array(y_train),
                                                                         y_test=np.array(y_test),base_accuracy=base_accuracy)

                logging.info(f"metric info artifact : {metric_info_artifact}")

                if metric_info_artifact is None:
                    continue

                if metric_info_artifact.index_number == 1:
                    model_evulation_artifact = ModelEvulationArtifact(evulation_model_file_path=cluster_model_path,
                                                                      is_model_accepted=True)
                    self.update_evulation_report(cluster_number=cluster_number,model_evulation_artifact=model_evulation_artifact)
                    logging.info(f"model accepted : {metric_info_artifact}")

                else:

                    logging.info(f"trained model is not better then existing model hence not accepting it")

            model_evulation_artifact = ModelEvulationArtifact(evulation_model_file_path=cluster_model_path_list,
                                                                      is_model_accepted=True)
            
            return model_evulation_artifact

        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Model evulation log completed.{'<<'*20} \n\n")