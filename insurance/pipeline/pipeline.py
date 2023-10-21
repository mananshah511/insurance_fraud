import os,sys
from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance.config.configuration import Configuration
from insurance.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformArtifact,ModelTrainerArtifact
from insurance.components.data_ingestion import DataIngestion
from insurance.components.data_validation import DataValidation
from insurance.components.data_transform import DataTransform
from insurance.components.model_trainer import ModelTrainer

class Pipeline:

    def __init__(self,config:Configuration=Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.intiate_data_validation()
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def start_data_transform(self,data_ingestion_artifact:DataIngestionArtifact,
                            data_validation_artifact:DataValidationArtifact)->DataTransformArtifact:
        try:
            data_transform = DataTransform(data_transform_config=self.config.get_data_transform_config(),
                                           data_ingestion_artifact=data_ingestion_artifact,
                                           data_validation_artifact=data_validation_artifact)
            return data_transform.intiate_data_transform()
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def start_model_trainer(self,data_transform_artifact:DataTransformArtifact)->ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(data_transform_artifact=data_transform_artifact,
                                         model_trainer_config=self.config.get_model_trainer_config())
            return model_trainer.intiate_model_trainer()
        except Exception as e:
            raise InsuranceException(sys,e) from e
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transform_artifact = self.start_data_transform(data_ingestion_artifact=data_ingestion_artifact,
                                                                data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transform_artifact=data_transform_artifact)
        except Exception as e:
            raise InsuranceException(sys,e) from e