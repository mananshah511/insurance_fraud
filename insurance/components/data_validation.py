import os,sys
from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from insurance.entity.config_entity import DataValidationConfig
import pandas as pd
import numpy as np
from insurance.util.util import read_yaml
from insurance.constant import NUMERIC_COULMN_KEY,CATEGORICAL_COLUMN_KEY,COLUMN_KEY
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab


class DataValidation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig) -> None:
        try:
            logging.info(f"{'>>'*20}Data Validation log started.{'<<'*20} \n\n")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def get_train_test_dataframe(self):
        try:
            logging.info(f"get train test dataframe function started")

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info(f"----------reading train data started----------")
            train_df = pd.read_csv(train_file_path)
            logging.info(f"----------reading train data completed----------")

            logging.info(f"-----------reading test data started----------")
            test_df = pd.read_csv(test_file_path)
            logging.info(f"-----------reading test data completed-----------")

            return train_df,test_df
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def check_train_test_dir_exist(self)->bool:
        try:
            logging.info(f"check train test dir exist function started")
            train_dir = self.data_ingestion_artifact.train_file_path
            test_dir = self.data_ingestion_artifact.test_file_path
            

            train_flag = True
            test_flag = True

            if not os.path.exists(train_dir):
                logging.info(f"train file and dir is not available")
                train_flag = False

            if not os.path.exists(test_dir):
                logging.info(f"test file and dir is not available")
                test_flag = False

            return train_flag and test_flag
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def check_column_count_validation(self)->bool:
        try:
            logging.info(f"check column count validation function started")
            schema_file_path =  self.data_validation_config.schema_file_dir
            schema_file_data = read_yaml(file_path=schema_file_path)

            train_df, test_df = self.get_train_test_dataframe()
            train_count = len(train_df.columns)
            test_count = len(test_df.columns)

            schema_count = len(schema_file_data[COLUMN_KEY])

            logging.info(f"column count in train data is : {train_count}")
            logging.info(f"column count in test data is : {test_count}")

            logging.info(f"column count in schema file is : {schema_count}")

            train_flag = False
            test_flag = False

            if train_count == schema_count:
                logging.info(f"column count in train data is correct")
                train_flag = True
            
            if test_count == schema_count:
                logging.info(f"column count in test data is correct")
                test_flag = True

            if train_flag == False:
                logging.info(f"column count in train data is not correct please check")

            if test_flag == False:
                logging.info(f"column count in test data is not correct please check")

            return train_flag and test_flag

        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def check_column_name_validation(self)->bool:
        try:
            logging.info(f"check column name validation function started")
            schema_file_path = self.data_validation_config.schema_file_dir
            schema_file_data = read_yaml(file_path=schema_file_path)

            train_df, test_df = self.get_train_test_dataframe()
            train_columns = list(train_df.columns)
            test_columns = list(test_df.columns)

            schema_columns = list(schema_file_data[NUMERIC_COULMN_KEY])+list(schema_file_data[CATEGORICAL_COLUMN_KEY])

            logging.info(f"column name in train file is : {train_columns}")
            logging.info(f"column name in test file is : {test_columns}")
            logging.info(f"column name in schema file is : {schema_file_data}")

            train_columns.sort()
            test_columns.sort()
            schema_columns.sort()

            train_flag = False
            test_flag = False

            if train_columns == schema_columns:
                logging.info(f"column names in train file is correct")
                train_flag = True

            if test_columns == schema_columns:
                logging.info(f"column names in test file is correct")
                test_flag = True

            if train_flag == False:
                logging.info(f"column names in train file is not correct, please check")

            if test_flag == False:
                logging.info(f"column names in test file is not correct, please check")

            return train_flag and test_flag
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def check_column_data_type_validation(self):
        try:
            logging.info(f"check column data type validation function started")
            schema_file_path = self.data_validation_config.schema_file_dir
            schema_file_data = read_yaml(file_path=schema_file_path)

            train_df, test_df = self.get_train_test_dataframe()
            train_data = dict(train_df.dtypes)
            test_data = dict(test_df.dtypes)

            schema_data = schema_file_data[COLUMN_KEY]

            for column_name in schema_data.keys():
                if train_data[column_name] != schema_data[column_name]:
                    logging.info(f"data type for {column_name} in train data is not correct")
                    return False
                if test_data[column_name] != schema_data[column_name]:
                    logging.info(f"data type for {column_name} in test data is not correct")
                    return False
                
            logging.info(f"data type in train data is correct")
            logging.info(f"data type in test data is correct")

            return True
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def get_and_save_datadrift_report(self):
        try:
            logging.info(f"get and save datadrift report function started")
            report_file_path = self.data_validation_config.report_page_file_dir
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir,exist_ok=True)

            train_df,test_df = self.get_train_test_dataframe()
            dashboard = Dashboard(tabs=[DataDriftTab()])

            dashboard.calculate(train_df,test_df)
            dashboard.save(report_file_path)

            logging.info(f"report saved successfully")
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def intiate_data_validation(self)->DataValidationArtifact:
        try:
            logging.info(f"intiate data validation function started")
            validation4 = False

            validation1 = self.check_train_test_dir_exist()
            if validation1:
                validation2 = self.check_column_count_validation()
            if validation2:
                validation3 = self.check_column_name_validation()
            if validation3:
                validation4 = self.check_column_data_type_validation()

            self.get_and_save_datadrift_report()

            data_validation_config = DataValidationArtifact(is_validated=validation4,
                                                                message="successfully",
                                                                schema_file_path=self.data_validation_config.schema_file_dir,
                                                                reprot_file_path=self.data_validation_config.report_page_file_dir)
            return data_validation_config
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Validation log completed.{'<<'*20} \n\n")