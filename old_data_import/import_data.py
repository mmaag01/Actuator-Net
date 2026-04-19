import pandas as pd
import numpy as np
from pathlib import Path
import glob


class ImportData:
    def __init__(self,
        path=r"C:\Users\maxma\OneDrive\ETH\Bachelors_Thesis\Data Collection\Actuator Net\Data",
        units={'t':'s', 'tor':'Nm', 'vel':'rad/s', 
               'pos':'rad', 'accel':'rad/s^2', 'i':'A'},
        zero_offset=False,
    ):
        self.units = units
        self.zero_offset = zero_offset
        self.cols_in = ['t', 'torDes', 'posDes', 'velDes', 'torAct', 'posAct', 'velAct','accelAct', 'i', 'velErr', 'posErr', 'torKdEst', 'torEst', 'i2t']
        self.base_path = Path(path)
        self.file_list = []

        self.path_main = self.base_path / "Main"
        self.path_fails = self.base_path / "Crashes"
        self.path_other = self.base_path / "Other"

        self.file_list_main = sorted(glob.glob(str(self.path_main / "*.csv")))
        self.file_list_fails = sorted(glob.glob(str(self.path_fails / "*.csv")))
        self.file_list_other = sorted(glob.glob(str(self.path_other / "*.csv")))

        self.main_dataset = []
        self.fails_dataset = []
        self.other_dataset = []
        self.combined = []

    def importMain(self):
        assert self.path_main.exists(), f"Main input folder does not exist: {self.path_main}"
        file_list = sorted(glob.glob(str(self.path_main / "*.csv")))
        assert file_list, f"No .csv files found in {self.path_main}"
        for i, file in enumerate(file_list):
            df = pd.read_csv(file, usecols=self.cols)
            assert len(df.columns.tolist()) == len(self.cols), f"{Path(file).stem}: expected columns {self.cols}, got {df.columns.tolist()}."
            df['dataset_id'] = i  # Track which dataset each row came from
            df['file_name'] = Path(file).stem  # Store filename
            self.main_dataset.append(df)
            print(f"Loaded {file}: {len(df)} rows")
            self.main_combined = pd.concat(self.main_dataset, ignore_index=True)
        return self.main_dataset

    def importFails(self):
        assert self.path_fails.exists(), f"Crashes input folder does not exist: {self.path_fails}"
        file_list = sorted(glob.glob(str(self.path_fails / "*.csv")))
        assert self.file_list_fails, f"No .csv files found in {self.path_fails}"
        self.fails_dataset = []
        for i, file in enumerate(file_list):
            df = pd.read_csv(file, usecols=self.cols)
            assert len(df.columns.tolist()) == len(self.cols), f"{Path(file).stem}: expected columns {self.cols}, got {df.columns.tolist()}."
            df['dataset_id'] = i  # Track which dataset each row came from
            df['file_name'] = Path(file).stem  # Store filename
            self.fails_dataset.append(df)
            print(f"Loaded {file}: {len(df)} rows")
        return self.fails_dataset

    def importOther(self):
        assert self.path_other.exists(), f"Other input folder does not exist: {self.path_other}"
        file_list = sorted(glob.glob(str(self.path_other / "*.csv")))
        assert self.file_list_other, f"No .csv files found in {self.path_other}"
        for i, file in enumerate(file_list):
            df = pd.read_csv(file)
            assert df.columns.tolist() == self.cols, f"{Path(file).stem}: expected columns {self.cols}, got {df.columns.tolist()}."
            df['dataset_id'] = i  # Track which dataset each row came from
            df['file_name'] = Path(file).stem  # Store filename
            self.other_dataset.append(df)
            print(f"Loaded {file}: {len(df)} rows")
        return self.other_dataset

    def combineDatasets(self):
        datasets = []
        
        # Select datasets based on categories
        if self.main_dataset != []:
            for df in self.main_dataset:
                df['category'] = 'main'
            datasets.extend(self.main_dataset)
            print(f"Including Main: {len(self.main_dataset)} files")
        
        if self.fails_dataset != []:
            for df in self.fails_dataset:
                df['category'] = 'fails'
            datasets.extend(self.fails_dataset)
            print(f"Including Fails: {len(self.fails_dataset)} files")
        
        if self.other_dataset != []:
            for df in self.other_dataset:
                df['category'] = 'other'
            datasets.extend(self.other_dataset)
            print(f"Including Other: {len(self.other_dataset)} files")
        
        # Combine all datasets
        if datasets:
            df_combined = pd.concat(datasets, ignore_index=True)
            print(f"\nDatasets combined. Total combined rows: {len(df_combined)}")
            self.combined = df_combined
            return self.combined
        else:
            print("No datasets to combine!")
            return None

    # Process all CSV files in input folder and export edited versions to output folder
    def convertUnits(self):
        units = self.units
        zero_offset = self.zero_offset
        to_milli = 1e3
        rad_s_to_rpm = 60 / (2 * np.pi)
        rad_to_inc = 2 * np.pi / 4096.0
        for df in self.combined:
            if units['t'] == 'ms':
                df['t'] *= to_milli
            if units['tor'] == 'mNm':
                df['torDes'] *= to_milli
                df['torAct'] *= to_milli
                df['torKdEst'] *= to_milli
                df['torEst'] *= to_milli
            if units['vel'] == 'rpm':
                df['velDes'] *= rad_s_to_rpm
                df['velAct'] *= rad_s_to_rpm
            if units['vel'] == 'mrpm':
                df['velDes'] *= rad_s_to_rpm * to_milli
                df['velAct'] *= rad_s_to_rpm * to_milli
            if units['vel'] == 'inc/s':
                df['velDes'] *= rad_to_inc
                df['velAct'] *= rad_to_inc
            if units['pos'] == 'inc':
                df['posDes'] *= rad_to_inc
                df['posAct'] *= rad_to_inc
            if units['accel'] == 'rpm/s':
                df['accelAct'] *= rad_s_to_rpm
            if units['accel'] == 'mrpm/s':
                df['accelAct'] *= rad_s_to_rpm * to_milli
            if units['accel'] == 'inc/s^2':
                df['accelAct'] *= rad_to_inc
            if units['i'] == 'mA':
                df['i'] *= to_milli
            if zero_offset:
                df['posAct'] -= df['posAct'].iloc[0]
                df['posDes'] -= df['posDes'].iloc[0]
        return self.combined
            