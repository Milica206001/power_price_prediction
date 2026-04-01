import pandas as pd
import numpy as np
import os
import re
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
print(CURRENT_DIR)
# 2. This goes UP one level to 'C:\Projects\power_price_prediction'
# This is the "Parent" house that contains both 'src' and 'notebook'
ROOT_DIR = CURRENT_DIR.parent
print(ROOT_DIR)
# 3. We tell Python: "Look in the Parent house for modules"
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from src.logger import logging
from src.exception import CustomException

print(CURRENT_DIR)
# 2. Point to the 'data' folder inside 'notebook'
DATA_DIR = ROOT_DIR / "notebook" / "data"
print(DATA_DIR)

def load_data():
    try:
        paths = {
            'holidays': os.path.join(DATA_DIR, 'holidays_events.csv'),
            'oil': os.path.join(DATA_DIR, 'oil.csv'),
            'stores': os.path.join(DATA_DIR, 'stores.csv'),
            'trans': os.path.join(DATA_DIR, 'transactions.csv'),
            'train': os.path.join(DATA_DIR, 'train.csv'),
            'test': os.path.join(DATA_DIR, 'test.csv')
        }
        
        holidays_events = pd.read_csv(paths['holidays'], parse_dates=['date'])
        oil = pd.read_csv(paths['oil'], parse_dates=['date'])
        stores = pd.read_csv(paths['stores'])
        transactions = pd.read_csv(paths['trans'], parse_dates=['date'])
        sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
        train = pd.read_csv(paths['train'], parse_dates=['date'])
        test = pd.read_csv(paths['test'], parse_dates=['date'])
        
        return holidays_events, oil, stores, transactions, sample_sub, train, test
    except Exception as e:
        raise CustomException(e, sys)

def find_transfered_date(holidays_events, holiday, year):
    mask = (holidays_events['description'].str.contains(str(holiday), na=False)) & \
           (holidays_events['date'].dt.year == year) & \
           (holidays_events['type'] == 'Transfer')
    
    list_of_dates = holidays_events.loc[mask, 'date'].to_list()
    return pd.DataFrame({'date': list_of_dates, 'description': [holiday]}) if list_of_dates else None

def find_transferred_holidays(holidays_events):
    trans_mask = holidays_events['transferred'] == True
    names = holidays_events.loc[trans_mask, 'description'].unique().tolist()
    years = holidays_events.loc[trans_mask, 'date'].dt.year.unique().tolist()
    
    df_list = []
    for holiday in names:
        for year in years:
            res = find_transfered_date(holidays_events, holiday, year)
            if res is not None:
                df_list.append(res)
    return pd.concat(df_list, axis=0) if df_list else pd.DataFrame()

def find_transferred_holidays_date_locale_name(holidays_events):
    trans_final = find_transferred_holidays(holidays_events)
    merged = pd.merge(trans_final, holidays_events, on=['date'])
    res = merged.groupby(['date', 'locale_name']).agg({
        'description_x': lambda x: ','.join(sorted(set(x))),
        'locale': lambda x: ','.join(sorted(set(x)))
    }).reset_index()
    return res.rename(columns={'description_x': 'description'})

def find_non_transferred_holidays(holidays_events):
    return holidays_events.loc[holidays_events['type'] == 'Holiday'].groupby(['date', 'locale_name']).agg({
        'description': ''.join, 'locale': ''.join
    }).reset_index()

def find_additional_days(holidays_events):
    add_df = holidays_events.loc[holidays_events['type'] == 'Additional'].copy()
    add_df['description'] = add_df['description'].apply(lambda x: re.sub(r'[+-123456789]+$', '', x))
    return add_df.groupby(['date', 'locale_name']).agg({
        'description': ''.join, 'locale': ''.join
    }).reset_index()

def find_all_events(holidays_events):
    ev_df = holidays_events.loc[holidays_events['type'] == 'Event'].copy()
    ev_df['description'] = ev_df['description'].apply(lambda x: re.sub(r'[+\d-]+$', '', x))
    ev_df['description'] = ev_df['description'].apply(lambda s: 'Mundial' if 'Mundial' in s else s)
    return ev_df.groupby(['date', 'locale_name']).agg({
        'description': lambda x: ','.join(sorted(set(x))),
        'locale': lambda x: ','.join(sorted(set(x)))
    }).reset_index()

def find_bridge_dates(holidays_events):
    br_df = holidays_events.loc[holidays_events['type'] == 'Bridge'].copy()
    br_df['description'] = br_df['description'].str.replace(r'^Puente\s*', '', regex=True)
    return br_df.groupby(['date', 'locale_name']).agg({
        'description': ''.join, 'locale': ''.join
    }).reset_index()

def find_unusuall_work_days(holidays_events):
    res = holidays_events.loc[holidays_events['type'] == 'Work Day'].groupby(['date', 'locale_name']).agg({
        'type': ''.join, 'locale': ''.join
    }).reset_index()
    return res.rename(columns={'type': 'description'})

def clean_data_holidays_events(holidays_events):
    parts = [
        find_transferred_holidays_date_locale_name(holidays_events),
        find_non_transferred_holidays(holidays_events),
        find_additional_days(holidays_events),
        find_all_events(holidays_events),
        find_bridge_dates(holidays_events),
        find_unusuall_work_days(holidays_events)
    ]
    combined = pd.concat(parts, axis=0)
    return combined.groupby(['date', 'locale_name']).agg({
        'description': lambda x: ','.join(sorted(set(x))) if not ''.join(x).endswith('-1') else ''.join(x).rstrip('-1'),
        'locale': lambda x: ','.join(sorted(set(x)))
    }).reset_index()

def determine_holiday(row):
    l_h, n_h = row['description_x'], row['description_y']
    if l_h == 'No local holiday' and n_h == 'No national holiday': return 'Work Day'
    if l_h != 'No local holiday' and n_h == 'No national holiday': return l_h
    if l_h == 'No local holiday' and n_h != 'No national holiday': return n_h
    return f"{l_h}, {n_h}"

def create_dataset():
    try:
        h_e, oil, stores, trans, s_s, train, test = load_data()
        test['sales'] = np.nan
        full = pd.concat([train, test], axis=0).sort_values('date')
        
        df = pd.merge(full, oil, on='date', how='left')
        df = pd.merge(df, trans, on=['date', 'store_nbr'], how='left')
        df = pd.merge(df, stores, on='store_nbr', how='left')
        
        h_cleaned = clean_data_holidays_events(h_e)
        df = pd.merge(df, h_cleaned, left_on=['date', 'city'], right_on=['date', 'locale_name'], how='left')
        
        nat_h = h_cleaned[h_cleaned['locale'] == 'National'][['date', 'description']]
        df = pd.merge(df, nat_h, on='date', how='left')
        
        df['description_x'] = df['description_x'].fillna('No local holiday')
        df['description_y'] = df['description_y'].fillna('No national holiday')
        df['all_holidays'] = df.apply(determine_holiday, axis=1)
        
        return df.drop(columns=['description_x', 'description_y', 'locale', 'locale_name'])
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    dataset = create_dataset()
    print(dataset.head())