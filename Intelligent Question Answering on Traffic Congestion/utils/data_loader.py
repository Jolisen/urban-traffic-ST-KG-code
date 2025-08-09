from pyproj import CRS
import pickle
import pandas as pd

def CRS_load(crs_path):
    with open(crs_path, "r") as file:
        crs_text = file.read().strip()
    roadcrs = CRS.from_string(crs_text)
    return roadcrs

def CSV_load(file_path):
    batch_data_df = pd.read_csv(file_path)
    batch_data_df['start_datetime'] = pd.to_datetime(batch_data_df['start_datetime'], format="%Y-%m-%d %H:%M:%S")
    batch_data_df['end_datetime'] = pd.to_datetime(batch_data_df['end_datetime'], format="%Y-%m-%d %H:%M:%S")
    # 添加纯时间列用于筛选
    batch_data_df['start_time_only'] = batch_data_df['start_datetime'].dt.time
    batch_data_df['end_time_only'] = batch_data_df['end_datetime'].dt.time
    return batch_data_df

def pkl_load(pkl_path):
    with open(pkl_path, 'rb') as file:
        pkl = pickle.load(file)
    return pkl