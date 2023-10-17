import os,sys
from insurance.exception import InsuranceException
from insurance.logger import logging
from insurance.entity.artifact_entity import DataIngestionArtifact
from insurance.entity.config_entity import DataIngestionConfig
from insurance.constant import DATABASE_NAME
import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig) -> None:
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def download_insurance_data(self):
        try:
            logging.info(f"download insurance data function started")
            dataset_url = self.data_ingestion_config.dataset_download_url
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            logging.info(f"downloading data from {dataset_url} in the {raw_data_dir} folder")

            os.makedirs(raw_data_dir,exist_ok=True)
            insurance_df = pd.read_csv(dataset_url)
            raw_file_path = os.path.join(raw_data_dir,DATABASE_NAME)
            insurance_df.to_csv(raw_file_path,index=False)

            logging.info(f"data saved successfully")
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def get_train_test_split_data(self)->DataIngestionArtifact:
        try:
            logging.info(f"get train test split data function started")
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            logging.info(f"raw data dir is : {raw_data_dir}")

            raw_file_path = os.path.join(raw_data_dir,DATABASE_NAME)
            logging.info(f"-----------data reading started----------")
            insurance_df = pd.read_csv(raw_file_path)
            logging.info(f"-----------data reading completed----------")


            ingested_train_dir = self.data_ingestion_config.ingested_train_dir
            ingested_test_dir = self.data_ingestion_config.ingested_test_dir
            os.makedirs(ingested_train_dir,exist_ok=True)
            os.makedirs(ingested_test_dir,exist_ok=True)


            logging.info(f"ingested train dir is : {ingested_train_dir}")
            logging.info(f"ingested test dir is : {ingested_test_dir}")

            ingested_train_file_path = os.path.join(ingested_train_dir,DATABASE_NAME)
            ingested_test_file_path = os.path.join(ingested_test_dir,DATABASE_NAME)

            X_train, X_test, y_train, y_test = train_test_split(insurance_df.iloc[:,:-1],insurance_df.iloc[:,-1], test_size=0.20, random_state=42)

            train_df = pd.concat([X_train,y_train],axis=1)
            test_df = pd.concat([X_test,y_test],axis=1)

            logging.info(f"saving train file as csv")
            train_df.to_csv(ingested_train_file_path,index=False)
            logging.info(f"train file saved successfully")

            logging.info(f"saving test file as csv")
            test_df.to_csv(ingested_test_file_path,index=False)
            logging.info(f"test file saved successfully")

            data_ingestion_artifact = DataIngestionArtifact(is_ingested=True,
                                                            message="successfully",
                                                            train_file_path=ingested_train_file_path,
                                                            test_file_path=ingested_test_file_path)
            
            return data_ingestion_artifact
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            logging.info(f"initiate data ingestion function started")
            self.download_insurance_data()
            return self.get_train_test_split_data()
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")