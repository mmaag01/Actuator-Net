import pandas as pd
import numpy as np
from pathlib import Path
import glob

class ProcessRawData:
    def __init__(
        self,
        locIn="Actuator Net/Raw Data",
        locOut="Actuator Net/Data",
        path=r"C:\Users\maxma\OneDrive\ETH\Bachelors_Thesis\Data Collection",
        MainDir = ["tStep", "TMS", "PMS", "PLC"],
        CrashesDir = ["tStep", "TMS"],
        OtherDir = ["tStep", "tStep CmdErrs", "TMS", "PMS", "Misc"],
    ):
        self.MainDir = MainDir
        self.CrashesDir = CrashesDir
        self.OtherDir = OtherDir
        self.locIn = locIn
        self.locOut = locOut

        self.base_path = Path(path)
        self.input_dir = self.base_path / locIn
        self.output_dir = self.base_path / locOut
        self.input_dir_main = self.input_dir / "Main"
        self.output_dir_main = self.output_dir / "Main"
        self.input_dir_fails = self.input_dir / "Crashes"
        self.output_dir_fails = self.output_dir / "Crashes"
        self.input_dir_other = self.input_dir / "Other"
        self.output_dir_other = self.output_dir / "Other"

        self.file_list = []
        self.file_list_main, self.file_list_fails, self.file_list_other = [], [], []

        if MainDir == [] and CrashesDir == [] and OtherDir == []:
            raise ValueError("No Input folder selected.")

        for loc in MainDir:
            file_import = self.input_dir_main / loc
            file_export = self.output_dir_main / loc
            
            if not file_import.exists():
                print("Input folder does not exist:", file_import)
                break
            if not file_import.exists():
                print("Output folder does not exist:", file_export)
                break

            file_list = sorted(glob.glob(str(file_import / "*.csv")))
            self.file_list_main.extend(file_list)

        for loc in CrashesDir:
            file_import = self.input_dir_fails / loc
            file_export = self.output_dir_fails / loc
            
            if not file_import.exists():
                print("Input folder does not exist:", file_import)
                break
            if not file_import.exists():
                print("Output folder does not exist:", file_export)
                break

            file_list = sorted(glob.glob(str(file_import / "*.csv")))
            self.file_list_fails.extend(file_list)

        for loc in OtherDir:
            file_import = self.input_dir_other / loc
            file_export = self.output_dir_other / loc
            
            if not file_import.exists():
                print("Input folder does not exist:", file_import)
                break
            if not file_import.exists():
                print("Output folder does not exist:", file_export)
                break
            file_list = sorted(glob.glob(str(file_import / "*.csv")))
            self.file_list_other.extend(file_list)

        self.file_list.extend(self.file_list_main)
        self.file_list.extend(self.file_list_fails)
        self.file_list.extend(self.file_list_other)

        self.raw = None
        self.df = None

    # Read one .csv file with the format output by the motor controller,
    # apply unit conversions, and store the data in a pandas dataframe
    def importRawData(self, file_path):
        file_path = Path(file_path)
        assert file_path.exists(), f"Input file does not exist: {file_path}"

        # unit conversions
        rad_s_to_rpm = 60.0 / (2 * np.pi)
        from_milli = 0.001
        rad_to_inc = 4096.0 / (2 * np.pi)

        # expected columns from controller CSV
        raw_cols = {
            'User Parameter [0x2201/31]': 't',           # ms
            'User Parameter [0x2201/99]': 'stable',      # 1/0
            'User Parameter [0x2201/20]': 'torDes',      # mNm
            'User Parameter [0x2201/93]': 'posDes',      # inc
            'User Parameter [0x2201/97]': 'velDes',      # mrpm
            'User Parameter [0x2201/1]': 'torAct',       # mNm
            '1: PO_BUSMOD_VALUE [0x4C01/2]': 'posAct',   # inc
            '1: PO_BUSMOD_VALUE [0x4C01/5]': 'velAct',   # mrpm
            '1: PO_BUSMOD_VALUE [0x4C01/6]': 'accelAct', # mrpm/s
            '1: PO_BUSMOD_VALUE [0x4C01/3]': 'i',        # mA
            'User Parameter [0x2201/69]': 'kd',          # mNm/(rad/s)
            '1: PO_BUSMOD_VALUE [0x4C01/7]': 'torEst',   # mNm
            '1: PO_BUSMOD_VALUE [0x4C01/8]': 'i2t'
        }

        cols = list(raw_cols.values())
        new_cols = ['t', 'torDes', 'posDes', 'velDes', 'torAct', 'posAct', 'velAct', 'accelAct', 'i', 'velErr', 'posErr', 'torKdEst', 'torEst', 'i2t']
        df = pd.read_csv(file_path, skiprows=1)

        # keep every second column, because the CSV alternates X/Y columns
        self.raw = df.copy()

        if len(df[df.columns[1::2]].columns) != len(cols) and len(df.columns) != len(new_cols):
            raise ValueError(
                f"{file_path.name}: expected {len(cols)} usable columns, got {len(df.columns)}."
            )
        
        if len(cols) == len(df[df.columns[1::2]].columns):
            df = df[df.columns[1::2]]
            df = df.set_axis(cols, axis=1)
            df = df[df["stable"] == 1].copy()
            df.drop("stable", axis=1, inplace=True)

            velErr = df["velDes"] - df["velAct"]
            posErr = df["posDes"] - df["posAct"]
            torKdEst = df["kd"] * velErr / rad_s_to_rpm

            idx = df.columns.get_loc("torEst")
            
            df.insert(idx, "velErr", velErr)
            df.insert(idx + 1, "posErr", posErr)
            df.insert(idx + 2, "torKdEst", torKdEst)

            df.drop("kd", axis=1, inplace=True)

            df["t"] *= from_milli
            df["torDes"] *= from_milli
            df["torAct"] *= from_milli
            df["torEst"] *= from_milli
            df["torKdEst"] *= from_milli
            df["i"] *= from_milli
            df["velDes"] *= from_milli
            df["velAct"] *= from_milli
            df["velErr"] *= from_milli
            df["accelAct"] *= from_milli

            df["velDes"] /= rad_s_to_rpm
            df["velAct"] /= rad_s_to_rpm
            df["velErr"] /= rad_s_to_rpm
            df["accelAct"] /= rad_s_to_rpm
            df["posDes"] /= rad_to_inc
            df["posAct"] /= rad_to_inc
            df["posErr"] /= rad_to_inc

        else: df = df.set_axis(new_cols, axis = 1)
        if all(df.columns) != all(new_cols):
            raise ValueError(f"Weird Columns: {df.columns}, expected {new_cols}")

        self.df = df
        return df

    # Process all CSV files in input folder and export edited versions to output folder
    def process_all(self, suffix="_fin"):
        if self.MainDir != []:
            for file_name in self.file_list_main:
                file_path = Path(file_name)
                output_path = Path(file_name.replace("Raw Data", "Data")[:-4]+suffix+".csv")
                assert file_path != output_path
                try:
                    print(f"Processing: {file_path.name}")
                    file = self.importRawData(file_path)
                    file.to_csv(output_path, index=False)
                    print(f"Saved to:   {output_path.name}")
                except Exception as e:
                    print(f"Failed for {file_path.name}: {e}")
        if self.CrashesDir != []:
            for file_name in self.file_list_fails:
                file_path = Path(file_name)
                output_path = Path(file_name.replace("Raw Data", "Data")[:-4]+suffix+".csv")
                try:
                    print(f"Processing: {file_path.name}")
                    file = self.importRawData(file_path)
                    file.to_csv(output_path, index=False)
                    print(f"Saved to:   {output_path.name}")
                except Exception as e:
                    print(f"Failed for {file_path.name}: {e}")
        if self.OtherDir != []:
            for file_name in self.file_list_other:
                file_path = Path(file_name)
                output_path = Path(file_name.replace("Raw Data", "Data")[:-4]+suffix+".csv")
                try:
                    print(f"Processing: {file_path.name}")
                    file = self.importRawData(file_path)
                    file.to_csv(output_path, index=False)
                    print(f"Saved to:   {output_path.name}")
                except Exception as e:
                    print(f"Failed for {file_path.name}: {e}")

class ImportData:
    def __init__(self,
        path=r"C:\Users\maxma\OneDrive\ETH\Bachelors_Thesis\Data Collection\Actuator Net\Data",
        units={'t':'s', 'tor':'Nm', 'vel':'rad/s', 
               'pos':'rad', 'accel':'rad/s^2', 'i':'A'},
        zero_offset=False,
    ):
        self.units = units
        self.zero_offset = zero_offset
        self.cols = ['t', 'torDes', 'posDes', 'velDes', 'torAct', 'posAct', 'velAct','accelAct', 'i', 'velErr', 'posErr', 'torKdEst', 'torEst', 'i2t']
        self.base_path = Path(path)
        self.file_list = []

        self.path_main = self.base_path / "Main"
        self.path_fails = self.base_path / "Crashes"
        self.path_other = self.base_path / "Other"
        self.main_types = ["tStep", "TMS", "PMS", "PLC"]
        self.fails_types = ["tStep", "TMS"]
        self.other_types = ["tStep", "tStep CmdErrs", "TMS", "PMS", "Misc"]

        self.file_list_main = sorted(glob.glob(str(self.path_main / "*.csv")))
        self.file_list_fails = sorted(glob.glob(str(self.path_fails / "*.csv")))
        self.file_list_other = sorted(glob.glob(str(self.path_other / "*.csv")))

        self.main_dataset = []
        self.fails_dataset = []
        self.other_dataset = []
        self.combined = []

    def importMain(self, types = None):
        if types is None: types = self.main_types
        for type in types:
            path = self.path_main / type
            if not path.exists():
                print(f"Failed to import Main: Folder does not exist ({path})")
                break
            file_list = sorted(glob.glob(str(path / "*.csv")))
            print(f"{len(file_list)} found files at {path}.")
            for i, file in enumerate(file_list):
                print("Reading file:", file)
                df = pd.read_csv(file, usecols=self.cols)
                assert len(df.columns.tolist()) == len(self.cols), f"{Path(file).stem}: expected columns {self.cols}, got {df.columns.tolist()}."
                df['category'] = 'main'
                df['dataset_id'] = i  # Track which dataset each row came from
                df['file_name'] = Path(file).stem  # Store filename
                df['type'] = type # Store input type
                self.main_dataset.append(df)
                print(f"Loaded File Main/{type}/{Path(file).name}: {len(df)} rows")
        self.main_combined = pd.concat(self.main_dataset, ignore_index=True)
        return self.main_dataset

    def importFails(self, types = None):
        if types is None: types = self.fails_types
        for type in types:
            path = self.path_fails / type
            if not path.exists():
                print(f"Failed to import Crashes: Folder does not exist ({path})")
                break
            file_list = sorted(glob.glob(str(path / "*.csv")))
            print(f"{len(file_list)} found files at {path}.")
            for i, file in enumerate(file_list):
                df = pd.read_csv(file, usecols=self.cols)
                assert len(df.columns.tolist()) == len(self.cols), f"{Path(file).stem}: expected columns {self.cols}, got {df.columns.tolist()}."
                df['category'] = 'crashes'
                df['dataset_id'] = i  # Track which dataset each row came from
                df['file_name'] = Path(file).stem  # Store filename
                df['type'] = type # Store input type
                self.fails_dataset.append(df)
                print(f"Loaded File Crashes/{type}/{Path(file).name}: {len(df)} rows")
        return self.fails_dataset

    def importOther(self, types = None):
        if types is None: types = self.other_types
        for type in types:
            path = self.path_other / type
            if not path.exists():
                print(f"Failed to import Other: Folder does not exist ({path})")
                break
            file_list = sorted(glob.glob(str(path / "*.csv")))
            print(f"{len(file_list)} found files at {path}.")
            for i, file in enumerate(file_list):
                df = pd.read_csv(file, usecols=self.cols)
                assert len(df.columns.tolist()) == len(self.cols), f"{Path(file).stem}: expected columns {self.cols}, got {df.columns.tolist()}."
                df['category'] = 'other'
                df['dataset_id'] = i  # Track which dataset each row came from
                df['file_name'] = Path(file).stem  # Store filename
                df['type'] = type # Store input type
                self.other_dataset.append(df)
                print(f"Loaded File Other/{type}/{Path(file).name}: {len(df)} rows")
        return self.other_dataset
    
    def importAll(self):
        df = []
        df_main = self.importMain()
        df_fails = self.importFails()
        df_other = self.importOther()

    def combineDatasets(self, datasets = [], all = False):
        if all:
            datasets.extend(self.main_dataset)
            print(f"Including Main: {len(self.main_dataset)} files")
            datasets.extend(self.fails_dataset)
            print(f"Including Fails: {len(self.fails_dataset)} files")
            datasets.extend(self.other_dataset)
            print(f"Including Other: {len(self.other_dataset)} files")
        
        # Combine all datasets
        if datasets:
            df_combined = pd.concat(datasets, ignore_index=True)
            print(f"\nDatasets combined. Total combined rows: {len(df_combined)}")
            df_combined = pd.get_dummies(df_combined, columns=['input'])
            df_combined = pd.get_dummies(df_combined, columns=['category'])
            print(f"\nFirst Row Combined: {df_combined.iloc[-1]}")
            self.combined = df_combined
            return self.combined
        else:
            print("No datasets to combine!")
            return None

    # Process all CSV files in input folder and export edited versions to output folder
    def convertUnits(self, df):
        units = self.units
        to_milli = 1e3
        rad_s_to_rpm = 60 / (2 * np.pi)
        rad_to_inc = 2 * np.pi / 4096.0
        #for df in self.combined:
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
        if self.zero_offset:
            df["posDes"] -= df["posDes"].iloc[0]
            df["posAct"] -= df["posAct"].iloc[0]
        return df