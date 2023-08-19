import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        logging.info('Exception occured while saving the object.')
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for name,model in models.items():
            current_model = model
            
            # train the model
            current_model.fit(X_train,y_train)

            # predict
            y_test_pred = current_model.predict(X_test)

            # get r2 score
            r2 = r2_score(y_test, y_test_pred)
            
            report[name] = r2
        logging.info('Successfully evaluated all models.')
        return report

    except Exception as e:
        logging.info('Exception occured while Evaulutaing the models')
        raise CustomException(e, sys)