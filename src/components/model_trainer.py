import numpy as np
import pandas as pd
import os
import sys
from src.utils import save_object, evaluate_model
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting target and features i.e Dependant and Independant variables')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'LinearRegression' : LinearRegression(),
                'Ridge':Ridge(),
                'Lasoo':Lasso(),
                'ElasticNet':ElasticNet()
            }

            model_report : dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print('\n========================================================================================')
            logging.info(f'Model Report : {model_report}')

            best_model_score = max(model_report.values()) 
            best_model_name = list(model_report.keys())[(list(model_report.values()).index(best_model_score))] 
            best_model = models[best_model_name]
            
            print(f'Best Model : {best_model_name} with accuracy = {round(best_model_score*100, 5)}%')
            print('\n========================================================================================')
            logging.info(f'Best Model : {best_model_name},{best_model_score}')


            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj = best_model
            )
            logging.info('Model Pickle saved')

        except Exception as e:
            logging.info('Exception occured while Initiating Model trainer')
            raise CustomException(e, sys)