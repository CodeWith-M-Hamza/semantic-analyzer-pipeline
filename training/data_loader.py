import pandas as pd
from pathlib import Path
RAW_DATA_PATH=Path("data/raw/training.1600000.processed.noemoticon.csv")
# print(RAW_DATA_PATH)
COLUMNS=[
    'Sentiments',
    'Tweet_id',
    'Timestamp',
    'Query',
    'User',
    'Text'
]
def load_raw_data()->pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"DataSet not found{RAW_DATA_PATH}")
    df=pd.read_csv(
            RAW_DATA_PATH,
            names=COLUMNS,
            encoding='latin-1'
        )
    df=df[['Sentiments','Text']]
    return df    
