import os,sys,csv,dill
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
import matplotlib.pyplot as plt
from pathlib import Path

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
        
        
    def perform_preprocessing(self,df:pd.DataFrame,is_test_data:bool=False):
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

            df = pd.DataFrame(df,columns=columns)
            df = pd.concat([df,target_df],axis=1)
            return df
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
            visulizer.show(graph_file_path)

            logging.info(f"graph saved successfully")
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def get_and_save_silhouette_score_graph(self,df:pd.DataFrame):
        try:
            logging.info(f"get and save silhouette score graph function started")
            fig, ax = plt.subplots(2, 2, figsize=(15,8))

            for no_clusters in [2,3,4,5]:
                logging.info(f"finding and saving graph of silhouette score for {no_clusters} clusters")
                kmeans = KMeans(n_clusters=no_clusters,init='k-means++',random_state=42)
                q,mod = divmod(no_clusters,2)

                visulizer = SilhouetteVisualizer(kmeans,colors='yellowbrick',ax=ax[q-1][mod])
                visulizer.fit((df.drop(self.target_column,axis=1)))

                graph_dir = self.data_transform_config.graph_save_dir
                os.makedirs(graph_dir,exist_ok=True)
                graph_file_path = os.path.join(graph_dir,'cluster_'+str(no_clusters)+'silhouetter_score.png')
                visulizer.show(graph_file_path)
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def save_data_based_on_cluster(self,train_df:pd.DataFrame,test_df:pd.DataFrame,n_clusters):
        try:
            logging.info(f"save data based on cluster function started")

            logging.info(f"making k-means object and fitting data")
            kmeans = KMeans(n_clusters=n_clusters,init='k-means++',random_state=42)
            kmeans.fit((train_df.drop(self.target_column,axis=1)))

            logging.info(f"prediction of train data's cluster")
            train_predict = kmeans.predict((train_df.drop(self.target_column,axis=1)))

            transform_train_folder = self.data_transform_config.transform_train_dir
            os.makedirs(transform_train_folder,exist_ok=True)

            column_name = train_df.columns
            logging.info(f"column name in train file is : {column_name}")

            cluster_numbers = list(np.unique(np.array(train_predict)))
            logging.info(f"cluster numbers are : {cluster_numbers}")

            logging.info(f"making csv file for train data")

            for cluster_number in cluster_numbers:
                train_file_path = os.path.join(transform_train_folder,'train_cluster'+str(cluster_number)+'.csv')
                with Path(train_file_path).open('w',newline='') as csvfiles:

                    csvwriter = csv.writer(csvfiles)
                    csvwriter.writerow(column_name)

                    for index in range(len(train_df)):
                        if train_predict[index] == cluster_number:
                            csvwriter.writerow(train_df.iloc[index])

            logging.info(f"csv files write for train data is completed")

            logging.info(f"prediction of test data's cluster")
            test_predict = kmeans.predict((test_df.drop(self.target_column,axis=1)))

            transform_test_folder = self.data_transform_config.transform_test_dir
            os.makedirs(transform_test_folder,exist_ok=True)

            column_name = test_df.columns
            logging.info(f"column name in test file is : {column_name}")

            cluster_numbers = list(np.unique(np.array(test_predict)))
            logging.info(f"cluster numbers are : {cluster_numbers}")

            logging.info(f"making csv file for test data")

            for cluster_number in cluster_numbers:
                test_file_path = os.path.join(transform_test_folder,'test_cluster'+str(cluster_number)+'.csv')
                with Path(test_file_path).open('w',newline='') as csvfiles:

                    csvwriter = csv.writer(csvfiles)
                    csvwriter.writerow(column_name)

                    for index in range(len(test_df)):
                        if test_predict[index] == cluster_number:
                            csvwriter.writerow(test_df.iloc[index])

            logging.info(f"csv files write for test data is completed")

            return kmeans

        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def intiate_data_transform(self)->DataTransformArtifact:
        try:
            logging.info(f"intiate data transform function started")
            train_df,test_df = self.perform_drop_column()

            train_df = self.perform_preprocessing(df=train_df)
            test_df = self.perform_preprocessing(df=test_df,is_test_data=True)

            self.get_and_save_graph_cluster(df=train_df)
            self.get_and_save_silhouette_score_graph(df=train_df)
            
            kmeans = self.save_data_based_on_cluster(train_df=train_df,test_df=test_df,n_clusters=NO_CLUSTER)

            logging.info(f"saving cluster model object")

            cluster_dir = os.path.dirname(self.data_transform_config.cluster_model_file_path)
            os.makedirs(cluster_dir,exist_ok=True)
            with open(self.data_transform_config.cluster_model_file_path,'wb') as objfile:
                dill.dump(kmeans,objfile)
            
            logging.info(f"cluster object saved")


            data_transform_artifact = DataTransformArtifact(is_transform=True,
                                                            message="successfully",
                                                            transform_train_dir=self.data_transform_config.transform_train_dir,
                                                            transform_test_dir=self.data_transform_config.transform_test_dir,
                                                            cluster_model_dir=self.data_transform_config.cluster_model_file_path)
            return data_transform_artifact
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Transformation log completed.{'<<'*20} \n\n")
        

        
    