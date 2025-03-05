import pandas as pd
import os
from pathlib import Path

def load_data(filepath):
    """
    Tối ưu hóa việc đọc file CSV
    """
    # Sử dụng engine='c' (nhanh hơn python engine) và chỉ load các cột cần thiết nếu có thể
    try:
        # Nếu biết trước các cột cần dùng, có thể thêm parameter usecols=['col1', 'col2']
        df = pd.read_csv(
            filepath,
            engine='c',          # Engine C nhanh hơn Python
            dtype_backend='pyarrow',  # Sử dụng pyarrow nếu có để tăng tốc
            encoding='utf-8'     # Chỉ định encoding để tránh đoán
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading file {filepath}: {str(e)}")
    return df

def save_data(df, filepath):
    """
    Tối ưu hóa việc lưu file CSV
    """
    try:
        # Sử dụng Path để xử lý đường dẫn hiệu quả hơn
        path = Path(filepath)
        # Tạo thư mục cha với hiệu suất tốt hơn
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Tối ưu hóa việc lưu với compression và engine nhanh
        df.to_csv(
            filepath,
            index=False,
            engine='c',          # Engine C nhanh hơn
            compression='infer', # Tự động nén nếu file lớn
            encoding='utf-8'     # Chuẩn hóa encoding
        )
    except Exception as e:
        raise Exception(f"Error saving file {filepath}: {str(e)}")

# Bonus: Hàm load nhanh hơn cho file lớn
def load_data_chunked(filepath, chunksize=10000):
    """
    Load dữ liệu theo chunk cho file lớn
    """
    return pd.read_csv(
        filepath,
        engine='c',
        dtype_backend='pyarrow',
        encoding='utf-8',
        chunksize=chunksize
    )