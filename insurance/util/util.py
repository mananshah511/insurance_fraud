import os,sys,yaml,dill
from insurance.exception import InsuranceException
import pandas as pd
from insurance.constant import DROP_COLUMN_LIST

def read_yaml(file_path:str):
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise InsuranceException(sys,e) from e

def write_yaml_file(file_path:str, data:dict=None):
    try:
        file_dir=os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        
        with open(file_path,"w") as yaml_file:
            yaml.dump(data, yaml_file)
    except Exception as e:
        raise InsuranceException(sys,e) from e  
    
def load_object(file_path:str):
    try:
        with open(file_path,"rb") as object_file:
            return dill.load(object_file)
    except Exception as e:
        raise InsuranceException(sys,e) from e
    

def preprocessing(df:pd.DataFrame):
    try:
        df.drop(DROP_COLUMN_LIST,inplace=True,axis=1)
        df['policy_csl'] = df['policy_csl'].map({'100/300':1,'250/500':2.5,'500/1000':5})
        df['insured_sex'] = df['insured_sex'].map({'MALE':1,'FEMALE':2})
        df['insured_education_level'] = df['insured_education_level'].map({'JD':1,'High School':2, 'College':3,'Masters':4,'Associate':5,'MD':6,'PhD':7})
        df['incident_severity'] = df['incident_severity'].map({'Trivial Damage':1,'Minor Damage':2,'Major Damage':3,'Total Loss':4})
        df['property_damage'] = df['property_damage'].map({'YES':1,'NO':0})
        df['police_report_available'] = df['police_report_available'].map({'YES':1,'NO':0})
        return pd.get_dummies(df,columns=['insured_occupation','insured_relationship','incident_type','authorities_contacted',
            'collision_type'],drop_first=True,dtype='int64')
    except Exception as e:
        raise InsuranceException(sys,e) from e