import os,sys
import pandas as pd
import numpy as np
from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance.entity.config_entity import DataTransformConfig
from insurance.entity.artifact_entity import DataIngestionArtifact,DataTransformArtifact,DataValidationArtifact
from insurance.util.util import read_yaml
from insurance.constant import TARGET_COLUMN_KEY,NO_CLUSTER,DROP_COLUMN_LIST
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer

class DataTransform:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_transform_config:DataTransformConfig,
                 data_validation_artifact:DataValidationArtifact) -> None:
        try:
            logging.info(f"{'>>'*20}Data Transformation log completed.{'<<'*20} \n\n")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transform_config = data_transform_config
            self.data_validation_artifact = data_validation_artifact
            self.target_column = read_yaml(file_path=self.data_validation_artifact.schema_file_path)[TARGET_COLUMN_KEY]
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def perform_drop_column(self):
        try:
            logging.info("perform drop column function started")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"dropping not needed column from train file")
            train_df.drop(DROP_COLUMN_LIST,inplace=True,axis=1)
            logging.info(f"dropping not needed column from test file")
            test_df.drop(DROP_COLUMN_LIST,inplace=True,axis=1)

            logging.info(f"column name after dropping columns in train file: {train_df.columns}")
            logging.info(f"column name after dropping columns in test file: {test_df.columns}")
            return train_df,test_df
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def perform_column_mapping(self,df:pd.DataFrame)->pd.DataFrame:
        try:
            logging.info(f"perform column mapping function started")
            df['policy_csl'] = df['policy_csl'].map({'100/300':1,'250/500':2.5,'500/1000':5})
            df['insured_sex'] = df['insured_sex'].map({'MALE':1,'FEMALE':2})
            df['insured_education_level'] = df['insured_education_level'].map({'JD':1,'High School':2, 'College':3,'Masters':4,'Associate':5,'MD':6,'PhD':7})
            df['incident_severity'] = df['incident_severity'].map({'Trivial Damage':1,'Minor Damage':2,'Major Damage':3,'Total Loss':4})
            df['property_damage'] = df['property_damage'].map({'YES':1,'NO':0})
            df['police_report_available'] = df['police_report_available'].map({'YES':1,'NO':0})
            df['fraud_reported'] = df['fraud_reported'].map({'Y':1,'N':0})
            logging.info(f"column mapping completed")
            return df
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def perform_onehot_encoding(self,df:pd.DataFrame)->pd.DataFrame:
        try:
            logging.info(f"perform onehot encoding function started")
            return pd.get_dummies(df,columns=['insured_occupation','insured_relationship','incident_type','authorities_contacted',
            'collision_type'],drop_first=True,dtype='int64')
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def get_preprocessing_object(self)->Pipeline:
        try:
            logging.info(f"get preprocessing object function started")
            
            logging.info(f"making pipeline started")
            pipeline = Pipeline(steps=[('scaler',StandardScaler())])
            logging.info(f"pipline completed")
            return pipeline

        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def perform_preprocessing(self,preprocessing_object:Pipeline,df:pd.DataFrame,is_test_data:bool=False):
        try:
            logging.info(f"perform preprocessing function started")
            
            df = self.perform_column_mapping(df=df)
            columns = df.columns

            imp = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
            df = imp.fit_transform(df)
            df = pd.DataFrame(df,columns=columns)
            
            target_df = df.iloc[:,-1]
            df.drop(self.target_column,axis=1,inplace=True)
            df = self.perform_onehot_encoding(df=df)
            columns = df.columns
            '''
            if is_test_data == False:
                df = preprocessing_object.fit_transform(df)
            else:
                df = preprocessing_object.transform(df)'''

            df = pd.DataFrame(df,columns=columns)
            df = pd.concat([df,target_df],axis=1)
            return df,preprocessing_object
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def get_and_save_graph_cluster(self,df:pd.DataFrame):
        try:
            logging.info(f"get and asve graph cluster function started")
            logging.info(f"making k-means object")

            kmeans = KMeans(init='k-means++',random_state=42)

            logging.info(f"making visulizer object and fitting train data")
            visulizer = KElbowVisualizer(kmeans,k=(2,11))
            visulizer.fit((df.drop(self.target_column,axis=1)))

            graph_dir = self.data_transform_config.graph_save_dir
            os.makedirs(graph_dir,exist_ok=True)
            graph_file_path = os.path.join(graph_dir,'graph_cluster.png')
            visulizer.show(graph_dir)

            logging.info(f"graph saved successfully")
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def get_and_save_silhouette_score_graph(self):
        try:
            pass
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def save_data_based_on_cluster(self):
        try:
            pass
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def intiate_data_transform(self)->DataTransformArtifact:
        try:
            logging.info(f"intiate data transform function started")
            train_df,test_df = self.perform_drop_column()
            preprocessing_obj = self.get_preprocessing_object()

            train_df,preprocessing_obj = self.perform_preprocessing(df=train_df,preprocessing_object=preprocessing_obj)
            train_df.to_csv('train.csv')

            test_df,preprocessing_obj = self.perform_preprocessing(df=test_df,preprocessing_object=preprocessing_obj,is_test_data=True)
            self.get_and_save_graph_cluster(df=train_df)
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Transformation log completed.{'<<'*20} \n\n")
        

        
    