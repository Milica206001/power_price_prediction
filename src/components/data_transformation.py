import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from scipy.signal import find_peaks
from category_encoders import CatBoostEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifact', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
        self.categories = {
            'Foundation_Provincialization_Independence': [
                'Fundacion de Manta', 'Provincializacion de Cotopaxi', 'Fundacion de Cuenca',
                'Cantonizacion de Libertad', 'Cantonizacion de Riobamba', 'Cantonizacion del Puyo',
                'Cantonizacion de Guaranda', 'Provincializacion de Imbabura', 'Cantonizacion de Latacunga',
                'Fundacion de Machala', 'Fundacion de Santo Domingo', 'Cantonizacion de El Carmen',
                'Cantonizacion de Cayambe', 'Fundacion de Esmeraldas', 'Primer Grito de Independencia',
                'Fundacion de Riobamba', 'Fundacion de Ambato', 'Fundacion de Ibarra',
                'Cantonizacion de Quevedo', 'Independencia de Guayaquil', 'Traslado Independencia de Guayaquil',
                'Cantonizacion de Salinas', 'Independencia de Cuenca', 'Provincializacion de Santo Domingo',
                'Provincializacion Santa Elena', 'Independencia de Guaranda', 'Independencia de Latacunga',
                'Independencia de Ambato', 'Fundacion de Quito-1', 'Fundacion de Quito', 'Fundacion de Loja',
                'Traslado Fundacion de Guayaquil', 'Traslado Primer Grito de Independencia',
                'Traslado Fundacion de Quito', 'Fundacion de Guayaquil-1', 'Fundacion de Guayaquil'
            ],
            'Holidays': [
                'Navidad-4','Dia de Difuntos', 'Navidad-3', 'Navidad-2', 'Puente Navidad',
                'Navidad-1', 'Navidad', 'Navidad+1', 'Puente Primer dia del ano', 'Primer dia del ano-1',
                'Primer dia del ano', 'Recupero puente Navidad', 'Recupero puente primer dia del ano',
                'Carnaval', 'Viernes Santo', 'Dia del Trabajo', 'Dia de la Madre-1', 'Dia de la Madre',
                'Batalla de Pichincha','Traslado Batalla de Pichincha', 'Puente Dia de Difuntos'
            ],
            'Entertainment': ['Black Friday','Cyber Monday', 'Mundial'],
            'Earthquake': ['Terremoto Manabi']
        }

    def _categorize_event(self, event):
        """Helper to map specific holiday names to broader categories."""
        for category, events_list in self.categories.items():
            if any(substring in str(event) for substring in events_list):
                return category
        return 'Other'

    def _is_payday(self, date):
        """Logic for 15th and last day of month."""
        return date.day == 15 or date.is_month_end
    
    def get_the_highest_peak(self,store_no,max_lag_value, transactions):
        max_lag = max_lag_value
        autocorrelations = np.array([transactions['transactions'].autocorr(lag) for lag in range(max_lag+1)]) 
       
        peaks, properties = find_peaks(autocorrelations,height=0) 
        peak_lags = peaks 
        peak_values = autocorrelations[peaks]
        max_peak_ind = -1
        max_peak_val = 0
        for lag, value in zip(peak_lags,peak_values):
            if value > max_peak_val:
                max_peak_val = value
                max_peak_ind = lag
    
        return max_peak_ind,max_peak_val
    
    def _fill_store_nans(self, df):
        """Custom seasonal imputation for store transactions."""
        logging.info("Filling NaNs using seasonal store-specific lags")
        
        stores = df['store_nbr'].unique()
        
        final_df_list = []
        for store_no in stores:
            subset = df[df['store_nbr'] == store_no].copy()
            store_daily_signal = subset.groupby('date')['transactions'].mean().reset_index()
            subset = subset.set_index('date').sort_index()
            lag_param, _ = self.get_the_highest_peak(store_no, 400, store_daily_signal)

            # Handle January 1st logic
            years = subset.index.year.unique()
            for year in years:
                jan_1st = f"{year}-01-01"
                if jan_1st in subset.index:
                    subset.at[jan_1st, 'transactions'] = 0

            # Seasonal Imputation via shift
            # If transactions is NaN, fill with value from 'lag_param' days ago
            shifted_values = subset['transactions'].shift(periods=lag_param)
            subset['transactions'] = subset['transactions'].fillna(shifted_values)

            # Final safety catch for any remaining NaNs
            subset['transactions'] = subset['transactions'].ffill().bfill()
            
            final_df_list.append(subset.reset_index())

        return pd.concat(final_df_list).sort_values(by=['date', 'store_nbr'])
    
    def _extract_time_features(self, df,target):
        """
        Extracts numeric features from the date index and 
        adds seasonal lags/rolling means.
        """
        df.index = pd.to_datetime(df.index)
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['year'] = df.index.year
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        #250 was the highest peak
        df[f'lag_{250}'] = df['transactions'].shift(250)
        df['rolling_mean_7'] = df['transactions'].shift(1).rolling(window=7).mean()
        df['sales_lag_1'] = df[target].shift(1)
        df['sales_lag_7'] = df[target].shift(7)

        df = df.ffill().bfill()
        
        return df

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation logic
        """
        try:
            cols_one_hot_other = ['city', 'state', 'type', 'family', 'all_holidays']
            cols_one_hot_binary = ['is_payday', 'is_impacted_by_earthquake']
            cols_cat_boost = ['store_nbr']
            
            logging.info("Pipeline creation started")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("onehot_min", OneHotEncoder(min_frequency=2000, handle_unknown='ignore'), cols_one_hot_other),
                    ("binary", OneHotEncoder(drop='if_binary', handle_unknown='ignore'), cols_one_hot_binary),
                    ("catboost", CatBoostEncoder(), cols_cat_boost)
                ],
                remainder='passthrough'
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path, parse_dates=['date'])
            test_df = pd.read_csv(test_path, parse_dates=['date'])

            logging.info("Applying feature engineering logic")

            train_df = self._fill_store_nans(train_df)
            test_df = self._fill_store_nans(test_df)

            for df in [train_df, test_df]:
                df['dcoilwtico'] = df['dcoilwtico'].interpolate(method='linear').bfill()
                
                df['all_holidays'] = df['all_holidays'].apply(self._categorize_event)
                
                df['is_payday'] = df['date'].apply(self._is_payday)
                earthquake_date = pd.Timestamp('2016-04-16')
                df['is_impacted_by_earthquake'] = df['date'] >= (earthquake_date + pd.Timedelta(weeks=3))
                
            train_df = self._extract_time_features(train_df,target='sales')
            test_df = self._extract_time_features(test_df,target='sales')

            target_column_name = "sales"
            
            drop_columns = [target_column_name, "date", "id"] 
            
            input_feature_train_df = train_df.drop(columns=drop_columns)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            preprocessing_obj = self.get_data_transformer_object()

            train_arr = preprocessing_obj.fit_transform(input_feature_train_df, target_feature_train_df)
            test_arr = preprocessing_obj.transform(input_feature_test_df)

            target_train = np.array(target_feature_train_df).reshape(-1, 1)
            target_test = np.array(target_feature_test_df).reshape(-1, 1)

            if hasattr(train_arr, 'toarray'):
                train_arr = train_arr.toarray()
            if hasattr(test_arr, 'toarray'):
                test_arr = test_arr.toarray()

            train_arr = np.c_[train_arr, target_train]
            test_arr = np.c_[test_arr, target_test]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)