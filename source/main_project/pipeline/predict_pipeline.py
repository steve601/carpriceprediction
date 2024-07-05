import sys
import pandas as pd
from source.commons import load_object
from source.exception import UserException
from source.logger import logging

class PredicPipeline:
    def __init__(self):
        pass
    
    logging.info('Preprocessing user input and making predictions')
    def predict(self,features):
        model_path = 'elements\model.pkl'
        scaler_path = 'elements\scaler.pkl'
        enc_path = 'elements\enc.pkl'
        # loaeding objects
        model = load_object(model_path)
        scaler = load_object(scaler_path)
        encoder = load_object(enc_path)
        
        features.iloc[:, 0] = encoder.transform(features.iloc[:, 0])
        data_processed = scaler.transform(features)
        prediction = model.predict(data_processed)
        
        return prediction
logging.info('This class is responsible for mapping all the inputs from html to flask')
class UserData:
    def __init__(self,model,mileage,vol_engine,fuel):
        self.mod = model
        self.mil = mileage
        self.vol = vol_engine
        self.fuel = fuel
        
    # let's write a function that returns the user input as a pandas dataframe
    def get_data_as_df(self):
        try:
            user_data = {
                "model":[self.mod],
                "mileage":[self.mil],
                "vol_engine":[self.vol],
                "fuel":[self.fuel]
            }
            return pd.DataFrame(user_data)
        except Exception as e:
            raise UserException(e,sys)
        