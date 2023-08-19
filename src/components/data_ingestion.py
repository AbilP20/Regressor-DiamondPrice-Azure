# in this component, the input is the raw data and ouput train-test data 
# NOTE: eda should be done before - which includes duplicates, analyzing, graphs etc. - feature engineering is done is data transformation
 
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

# initialize dataclass which stores the input and ouput of data ingestion (i.e path to input and output)

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','raw.csv')

# create to class to give the above path to data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_Config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Starts')
        try:
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            logging.info('Dataset read')

            os.makedirs(os.path.dirname(self.ingestion_Config.raw_data_path), exist_ok = True)
            df.to_csv(self.ingestion_Config.raw_data_path, index = False)
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=32)
            logging.info('Train-Test split done.')

            train_set.to_csv(self.ingestion_Config.train_data_path)
            test_set.to_csv(self.ingestion_Config.test_data_path)
            
            logging.info('Data Ingestion completed.')

            return (self.ingestion_Config.train_data_path, self.ingestion_Config.test_data_path)

        except Exception as e:
            logging.info('Exception occured at Data Ingestion Stage.')
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    obj  = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    