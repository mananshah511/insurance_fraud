import os,sys,yaml
from insurance.exception import InsuranceException

def read_yaml(file_path:str):
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise InsuranceException(sys,e) from e