import os
import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass(frozen=True)
class DataIngestionConfig:
    
    root_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = root_dir / "notebook" / "data"
    
    holidays_path: Path = data_dir / 'holidays_events.csv'
    oil_path: Path = data_dir / 'oil.csv'
    stores_path: Path = data_dir / 'stores.csv'
    trans_path: Path = data_dir / 'transactions.csv'
    train_path: Path = data_dir / 'train.csv'
    test_path: Path = data_dir / 'test.csv'
    sample_sub_path: Path = data_dir / 'sample_submission.csv'

    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def _determine_holiday(self, row):
        l_h, n_h = row['description_x'], row['description_y']
        if l_h == 'No local holiday' and n_h == 'No national holiday': return 'Work Day'
        if l_h != 'No local holiday' and n_h == 'No national holiday': return l_h
        if l_h == 'No local holiday' and n_h != 'No national holiday': return n_h
        return f"{l_h}, {n_h}"

    def _clean_holidays(self, h_e):
        logging.info("Cleaning holiday and event data")

        ev_df = h_e.copy()
        ev_df['description'] = ev_df['description'].apply(lambda x: re.sub(r'[+\d-]+$', '', str(x)))
        ev_df['description'] = ev_df['description'].apply(lambda s: 'Mundial' if 'Mundial' in s else s)
        
        h_cleaned = ev_df.groupby(['date', 'locale_name', 'locale']).agg({
            'description': lambda x: ','.join(sorted(set(x)))
        }).reset_index()
        return h_cleaned

    def load_data(self):
        try:
            h_e = pd.read_csv(self.ingestion_config.holidays_path, parse_dates=['date'])
            oil = pd.read_csv(self.ingestion_config.oil_path, parse_dates=['date'])
            stores = pd.read_csv(self.ingestion_config.stores_path)
            trans = pd.read_csv(self.ingestion_config.trans_path, parse_dates=['date'])
            train = pd.read_csv(self.ingestion_config.train_path, parse_dates=['date'])
            test = pd.read_csv(self.ingestion_config.test_path, parse_dates=['date'])
            return h_e, oil, stores, trans, train, test
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info("Started Ingestion and Merging process")
        try:
            h_e, oil, stores, trans, train, test = self.load_data()
            
            test['sales'] = np.nan
            full = pd.concat([train, test], axis=0).sort_values('date')
            
            df = pd.merge(full, oil, on='date', how='left')
            df = pd.merge(df, trans, on=['date', 'store_nbr'], how='left')
            df = pd.merge(df, stores, on='store_nbr', how='left')
            
            h_cleaned = self._clean_holidays(h_e)

            df = pd.merge(df, h_cleaned, left_on=['date', 'city'], right_on=['date', 'locale_name'], how='left')
            df.rename(columns={'description': 'description_x'}, inplace=True)
            df['description_x'] = df['description_x'].fillna('No local holiday')

            nat_h = h_cleaned[h_cleaned['locale'] == 'National'][['date', 'description']].drop_duplicates('date')
            df = pd.merge(df, nat_h, on='date', how='left')
            df.rename(columns={'description': 'description_y'}, inplace=True)
            df['description_y'] = df['description_y'].fillna('No national holiday')

            df['all_holidays'] = df.apply(self._determine_holiday, axis=1)
            
            final_df = df.drop(columns=['description_x', 'description_y', 'locale', 'locale_name'], errors='ignore')
            logging.info("Dropping rows where sales are NaN (from the original train set)")

            final_df.dropna(subset=['sales'], inplace=True)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            final_df.to_csv(self.ingestion_config.raw_data_path, index=False)

            train_set, test_set = train_test_split(final_df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion complete. Artifacts saved.")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        train_data_path = os.path.join('artifact', 'train.csv')
        test_data_path = os.path.join('artifact', 'test.csv')

        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_path=train_data_path,
            test_path=test_data_path
        )

        print(f"Ingestion and Transformation complete.")
        print(f"Training array shape: {train_arr.shape}")
        print(f"Test array shape: {test_arr.shape}")

        model_trainer= ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr,test_arr)

    except Exception as e:
        raise CustomException(e, sys)