import os
import sys 
from src.exception import customException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformerConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str = os.path.join('artifacts',"test.csv")
    raw_data_path:str = os.path.join('artifacts',"data.csv")
    
    
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiateDataIngestion(self):
        logging.info("Entered the data Ingestion method or components")
        try:
            df = pd.read_csv('/Users/faux/Models/Notebook/data/stud.csv')
            logging.info('Exported or Read the dataset as dataframe')
                
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            
            logging.info(' Train Test Split initiated')
            train_set, test_set = train_test_split(df, random_state=42, test_size = 0.2)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            
            logging.info('Ingestion of Data Completed')
            
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            
            raise customException(e,sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiateDataIngestion()
    # print(train_data)
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
        