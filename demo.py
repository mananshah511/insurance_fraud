from insurance.config.configuration import Configuration
from insurance.pipeline.pipeline import Pipeline

#config = Configuration()
#config.get_data_ingestion_config()

pipeline = Pipeline()
pipeline.start_data_ingestion()