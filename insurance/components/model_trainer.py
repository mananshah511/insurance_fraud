import os,sys,dill
from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance.entity.config_entity import ModelTrainerConfig
from insurance.entity.artifact_entity import DataTransformArtifact,ModelTrainerArtifact
from insurance.entity.model_factory import ModelFactory,get_evulated_classification_model,GridSearchedBestModel,MetricInfoArtifact
import pandas as pd
import numpy as np
from typing import List

class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transform_artifact:DataTransformArtifact) -> None:
        try:
            logging.info(f"{'>>'*20}Model trainer log completed.{'<<'*20} \n\n")
            self.model_trainer_config = model_trainer_config
            self.data_transform_artifact = data_transform_artifact
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def intiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"intiate model trainer function started")

            transform_train_files = self.data_transform_artifact.transform_train_dir
            transform_test_files = self.data_transform_artifact.transform_test_dir

            logging.info(f"transform train files : {transform_train_files}")
            logging.info(f"transform test files : {transform_test_files}")

            train_files = list(os.listdir(transform_train_files))
            test_files = list(os.listdir(transform_test_files))

            logging.info(f"train files are : {train_files}")
            logging.info(f"test files are : {test_files}")

            model_trainer_artifact = None

            train_accuracy = []
            test_accuracy = []
            model_accuracy = []

            for cluster_number in range(len(train_files)):
                logging.info(f"{'>>'*20}cluster : {cluster_number}{'<<'*20}")

                train_file_name = train_files[cluster_number]
                test_file_name = test_files[cluster_number]

                train_file_path = os.path.join(transform_train_files,train_file_name)
                test_file_path = os.path.join(transform_test_files,test_file_name)

                logging.info(f"reading train data from the file : {train_file_path}")    
                train_df = pd.read_csv(train_file_path)
                logging.info(f"train data reading successfull")
                logging.info(f"reading test data from the file : {test_file_path}")    
                test_df = pd.read_csv(test_file_path)
                logging.info(f"test data reading successfull")

                logging.info("splitting data into input and output feature")
                X_train,y_train,X_test,y_test = train_df.iloc[:,:-1],train_df.iloc[:,-1],test_df.iloc[:,:-1],test_df.iloc[:,-1]

                logging.info(f"extracting model config file path")
                model_config_file_path = self.model_trainer_config.model_config_file_path
                logging.info(f"intlized of model factory class")
                model_factory = ModelFactory(config_path=model_config_file_path)

                base_accuracy = self.model_trainer_config.base_accuracy
                logging.info(f"base accuracy is :{base_accuracy}")

                logging.info(f"finding best model for cluster : {cluster_number}")
                best_model = model_factory.get_best_model(X=np.array(X_train),y=np.array(y_train),base_accuracy=base_accuracy)
                logging.info(f"best model on trained data is : {best_model}")

                grid_searched_best_model_list : List[GridSearchedBestModel] = model_factory.grid_searched_best_model_list
                model_list = [model.best_model for model in grid_searched_best_model_list]
                logging.info(f"individual best model list : {model_list}")

                logging.info(f"finding best model after evulation on train and test data")
                metric_info:MetricInfoArtifact=get_evulated_classification_model(X_train=np.array(X_train),y_train=np.array(y_train),
                                                                        X_test=np.array(X_test),
                                                                            y_test=np.array(y_test),base_accuracy=base_accuracy,model_list=model_list)
                model_object = metric_info.model_object

                logging.info(f"----------best model after train and test evulation : {model_object} accuracy : {metric_info.model_accuracy}-----------")


                model_path = self.model_trainer_config.trained_model_file_path
                model_name = os.path.basename(model_path)
                model_dir = os.path.dirname(model_path)
                cluster_dir = os.path.join(model_dir,'cluster'+str(cluster_number))

                os.makedirs(cluster_dir,exist_ok=True)
                cluster_path = os.path.join(cluster_dir,model_name)

                with open(cluster_path,'wb') as object_file:
                    dill.dump(cluster_path,object_file)
                logging.info(f"model saved successfully")

                train_accuracy.append(train_accuracy)
                test_accuracy.append(test_accuracy)
                model_accuracy.append(model_accuracy)

            model_trainer_artifact = ModelTrainerArtifact(is_trained=True,
                                                          message="successfully",
                                                          trained_model_path=model_path,
                                                          train_accuracy=train_accuracy,
                                                          test_accuracy=test_accuracy,
                                                          model_accuracy=model_accuracy)

            return model_trainer_artifact
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Model trainer log completed.{'<<'*20} \n\n")