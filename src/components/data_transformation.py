import sys
import pandas as pd
import numpy as np
# import logger
import os

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from src.logger import logging
from src.exception import customException
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object 


@dataclass
class DataTransformerConfig:
    data_transformer_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformerConfig()
        
    def data_preprocessing(self):
        '''
        this function is responsible for data transformation based on a differnet type of datas
        '''
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            
            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean = False))
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('oneHotEncoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean = False))
                ]
            )
            logging.info(f'Num_columns_transformation:{numerical_columns}')
            logging.info(f'Cat_columns_transformation:{categorical_columns}')
            
            preprocessor = ColumnTransformer(
                [
                     ('num_features',num_pipeline,numerical_columns),
                    ('cat_features',cat_pipeline,categorical_columns)
                ]
               
            )
            return preprocessor
        except Exception as e:
            return customException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            print(train_df)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')
            preprocessing_object = self.data_preprocessing()
            target_column_name = ['math_score']
            numerical_columns= ['writing_score', 'reading_score']
            input_feature_train_df = train_df.drop(columns = target_column_name, axis = 1)
            target_feature_train = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns = target_column_name, axis = 1)
            target_feature_test = test_df[target_column_name]
            
            logging.info(f'applying the preprocessing object on training dataframe and testing dataframe')
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df) 
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test)
            ]
            logging.info(f'saved preprocessing object')
            save_object(
                file_path = self.data_transformation_config.data_transformer_path,
                obj=  preprocessing_object
            )
            
            return (
                train_arr, 
                test_arr,
                self.data_transformation_config.data_transformer_path
            )
                
                
        except Exception as e:
            raise customException(e,sys)
    
        