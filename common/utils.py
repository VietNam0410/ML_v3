import pandas as pd
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def save_data(df, filepath):
    # Tạo thư mục cha nếu chưa tồn tại
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)