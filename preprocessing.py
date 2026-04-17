import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def importData(path = "C:\\Users\\maxma\\OneDrive\\ETH\\Bachelors_Thesis\\Data Collection\\CSV Files\\"):
    # Load all CSV files
    csv_files = glob.glob(f'{path}*.csv')
    datasets = []
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        df['dataset_id'] = i  # Track which dataset each row came from
        df['file_name'] = Path(file).stem  # Store filename
        datasets.append(df)
        print(f"Loaded {file}: {len(df)} rows")

    # Combine all datasets
    df_combined = pd.concat(datasets, ignore_index=True)
    print(f"\nTotal combined rows: {len(df_combined)}")
