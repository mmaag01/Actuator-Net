import pandas as pd
import numpy as np
from pathlib import Path
import glob


class Data:
    def __init__(
        self,
        locIn="Actuator Net/Raw Data",
        locOut="Actuator Net/Data",
        path=r"C:\Users\maxma\OneDrive\ETH\Bachelors_Thesis\Data Collection",
        main_experiments=True,
        crashed_experiments=False,
        other_experiments=False,
        use_milli=False,
        rad=True,
        zero_offset=False,
        rename_cols=False
    ):
        self.general = main_experiments
        self.fails = crashed_experiments
        self.other = other_experiments
        self.use_milli = use_milli
        self.rad = rad
        self.zero_offset = zero_offset
        self.rename_cols = rename_cols

        self.base_path = Path(path)
        self.input_dir = self.base_path / locIn
        self.output_dir = self.base_path / locOut
        self.file_list = []

        assert self.general or self.fails or self.other, "No Input folder selected."

        if self.general:
            self.input_dir_main = self.input_dir / "Main"
            self.output_dir_main = self.output_dir / "Main"
            assert self.input_dir_main.exists(), f"Main input folder does not exist: {self.input_dir_main}"
            assert self.output_dir_main.exists(), f"Main output folder does not exist: {self.output_dir_main}"
            self.file_list_main = sorted(glob.glob(str(self.input_dir_main / "*.csv")))
            assert self.file_list_main, f"No .csv files found in {self.input_dir_main}"
            self.file_list.extend(self.file_list_main)
        else:
            self.input_dir_main = None
            self.output_dir_main = None
            self.file_list_main = None

        if self.fails:
            self.input_dir_fails = self.input_dir / "Crashes"
            self.output_dir_fails = self.output_dir / "Crashes"
            assert self.input_dir_fails.exists(), f"Crashes input folder does not exist: {self.input_dir_fails}"
            assert self.output_dir_fails.exists(), f"Crashes output folder does not exist: {self.output_dir_fails}"
            self.file_list_fails = sorted(glob.glob(str(self.input_dir_fails / "*.csv")))
            assert self.file_list_fails, f"No .csv files found in {self.input_dir_fails}"
            self.file_list.extend(self.file_list_fails)
        else:
            self.input_dir_fails = None
            self.output_dir_fails = None
            self.file_list_fails = None

        if self.other:
            self.input_dir_other = self.input_dir / "Other"
            self.output_dir_other = self.output_dir / "Other"
            assert self.input_dir_other.exists(), f"Other input folder does not exist: {self.input_dir_other}"
            assert self.output_dir_other.exists(), f"Other output folder does not exist: {self.output_dir_other}"
            self.file_list_other = sorted(glob.glob(str(self.input_dir_other / "*.csv")))
            assert self.file_list_other, f"No .csv files found in {self.input_dir_other}"
            self.file_list.extend(self.file_list_other)
        else:
            self.input_dir_other = None
            self.output_dir_other = None
            self.file_list_other = None

        self.raw = None
        self.df = None
        self.dataOut = None

    # Read one .csv file with the format output by the motor controller,
    # apply unit conversions, and store the data in a pandas dataframe
    def importRawData(self, file_path):
        file_path = Path(file_path)
        assert file_path.exists(), f"Input file does not exist: {file_path}"

        # unit conversions
        rad_s_to_rpm = 60.0 / (2 * np.pi)
        from_milli = 0.001
        rad_to_inc = 4096.0 / (2 * np.pi)
        use_milli = self.use_milli
        rad = self.rad

        # expected columns from controller CSV
        new_cols = {
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
        cols = list(new_cols.values())

        df = pd.read_csv(file_path, skiprows=1)

        # keep every second column, because the CSV alternates X/Y columns
        df = df[df.columns[1::2]]
        self.raw = df.copy()

        if len(df.columns) != len(cols):
            raise ValueError(
                f"{file_path.name}: expected {len(cols)} usable columns, got {len(df.columns)}."
            )

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

        if self.zero_offset:
            df["posDes"] -= df["posDes"].iloc[0]
            df["posAct"] -= df["posAct"].iloc[0]

        if not use_milli:
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

        if rad:
            df["velDes"] /= rad_s_to_rpm
            df["velAct"] /= rad_s_to_rpm
            df["velErr"] /= rad_s_to_rpm
            df["accelAct"] /= rad_s_to_rpm
            df["posDes"] /= rad_to_inc
            df["posAct"] /= rad_to_inc
            df["posErr"] /= rad_to_inc

        self.df = df
        return df

    # Write one dataframe to one output CSV
    def exportData(self, output_path):
        output_path = Path(output_path)

        colsOut = {
            't': 'Time [s]',
            'torDes': 'Demanded Joint Torque [Nm]',
            'posDes': 'Demanded Joint Position [rad]',
            'velDes': 'Demanded Joint Velocity [rad/s]',
            'torAct': 'Joint Torque [Nm]',
            'posAct': 'Joint Position [rad]',
            'velAct': 'Sensor Fusion Filtered Fused Velocity [rad/s]',
            'accelAct': 'Sensor Fusion Filtered Fused Acceleration [rad/s]',
            'i': 'Current Average [A]',
            'torEst': 'Filtered Estimated Joint Torque [Nm]',
            'posErr': 'Position Error [rad]',
            'velErr': 'Velocity Error [rad/s]',
            'torKdEst': 'Kd Torque Estimate [Nm]',
            'i2t': 'i2t'
        }

        if self.rename_cols:
            dataOut = self.df.rename(colsOut, axis=1)
        else:
            dataOut = self.df.copy()
        self.dataOut = dataOut
        dataOut.to_csv(output_path, index=False)

    # Process all CSV files in input folder and export edited versions to output folder
    def process_all(self, suffix="_exp"):
        if self.general:
            for file_name in self.file_list_main:
                file_path = Path(file_name)
                output_path = self.output_dir_main / f"{file_path.stem}{suffix}.csv"
                try:
                    print(f"Processing: {file_path.name}")
                    self.importRawData(file_path)
                    self.exportData(output_path)
                    print(f"Saved to:   {output_path.name}")
                except Exception as e:
                    print(f"Failed for {file_path.name}: {e}")
        if self.fails:
            for file_name in self.file_list_fails:
                file_path = Path(file_name)
                output_path = self.output_dir_fails / f"{file_path.stem}{suffix}.csv"
                try:
                    print(f"Processing: {file_path.name}")
                    self.importRawData(file_path)
                    self.exportData(output_path)
                    print(f"Saved to:   {output_path.name}")
                except Exception as e:
                    print(f"Failed for {file_path.name}: {e}")
        if self.other:
            for file_name in self.file_list_other:
                file_path = Path(file_name)
                output_path = self.output_dir_other / f"{file_path.stem}{suffix}.csv"
                try:
                    print(f"Processing: {file_path.name}")
                    self.importRawData(file_path)
                    self.exportData(output_path)
                    print(f"Saved to:   {output_path.name}")
                except Exception as e:
                    print(f"Failed for {file_path.name}: {e}")