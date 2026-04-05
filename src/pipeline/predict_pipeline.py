import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from sklearn.metrics import r2_score

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifact\model_tuned.pkl'
            preprocessor_path='artifact\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 store_nbr: int,
                 family: str,
                 onpromotion: int,
                 dcoilwtico: float,
                 transactions: float,
                 city: str,
                 state: str,
                 type: str,
                 cluster: int,
                 all_holiday: str):
        
        self.store_nbr = store_nbr
        self.family = family
        self.onpromotion = onpromotion
        self.dcoilwtico = dcoilwtico
        self.transactions = transactions
        self.city = city
        self.state = state
        self.type = type
        self.cluster = cluster
        self.all_holiday = all_holiday

    def get_data_as_data_frame(self):
        try:
            data_dict = {
                "store_nbr": [self.store_nbr],
                "family": [self.family],
                "onpromotion": [self.onpromotion],
                "dcoilwtico": [self.dcoilwtico],
                "transactions": [self.transactions],
                "city": [self.city],
                "state": [self.state],
                "type": [self.type],
                "cluster": [self.cluster],
                "all_holidays": [str(self.all_holiday)]
            }
            
            df = pd.DataFrame(data_dict)
            
            current_date = pd.Timestamp('17.8.2017')
            
            df['month'] = current_date.month
            df['day_of_week'] = current_date.dayofweek
            df['year'] = current_date.year
            df['is_weekend'] = 1 if current_date.dayofweek >= 5 else 0
            
            df['is_payday'] = (current_date.day == 15 or current_date.is_month_end)
            df['is_payday'] = df['is_payday'].astype(int) 

            earthquake_date = pd.Timestamp('2016-04-16')
            df['is_impacted_by_earthquake'] = current_date >= (earthquake_date + pd.Timedelta(weeks=3))
            df['is_impacted_by_earthquake'] = df['is_impacted_by_earthquake'].astype(int)

            #providing 0 for the testing enivironment
            df['lag_250'] = 0.0 
            df['sales_lag_1'] = 0.0
            df['sales_lag_7'] = 0.0
            df['rolling_mean_7'] = 0.0

            return df

        except Exception as e:
            raise CustomException(e, sys)
