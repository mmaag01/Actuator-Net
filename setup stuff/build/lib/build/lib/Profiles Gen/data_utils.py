import pandas as pd
import numpy as np
from pathlib import Path
import glob


class Data:
    def __init__(
        self,
        locIn="Aposs Code/.mh/Raw CSV Files",
        locOut="CSV Files",
        path=r"C:\Users\maxma\OneDrive\ETH\Bachelors_Thesis\Data Collection",
    ):
        self.base_path = Path(path)
        self.input_dir = self.base_path / locIn
        self.output_dir = self.base_path / locOut

        assert self.input_dir.exists(), f"Input folder does not exist: {self.input_dir}"
        assert self.output_dir.exists(), f"Output folder does not exist: {self.output_dir}"

        self.raw = None
        self.df = None
        self.dataOut = None

    # Read one .csv file with the format output by the motor controller,
    # apply unit conversions, and store the data in a pandas dataframe
    def importData(self, file_path, convert_time=False, convert_torque=False, convert_angle=False):
        file_path = Path(file_path)
        assert file_path.exists(), f"Input file does not exist: {file_path}"

        # unit conversions
        rad_s_to_mrpm = 60000.0 / (2 * np.pi)
        from_milli = 0.001
        rad_to_inc = 4096.0 / (2 * np.pi)

        # expected columns from controller CSV
        new_cols = {
            'User Parameter [0x2201/31]': 't',          # ms
            'User Parameter [0x2201/99]': 'stable',     # 1/0
            'User Parameter [0x2201/20]': 'torDes',     # mNm
            'User Parameter [0x2201/93]': 'posDes',     # inc
            'User Parameter [0x2201/97]': 'velDes',     # mrpm
            'User Parameter [0x2201/1]': 'torAct',      # mNm
            '1: PO_BUSMOD_VALUE [0x4C01/2]': 'posAct',  # inc
            '1: PO_BUSMOD_VALUE [0x4C01/5]': 'velAct',  # mrpm
            '1: PO_BUSMOD_VALUE [0x4C01/6]': 'accelAct',# mrpm/s
            '1: PO_BUSMOD_VALUE [0x4C01/3]': 'i',       # mA
            'User Parameter [0x2201/69]': 'kd',         # mNm/(rad/s)
            '1: PO_BUSMOD_VALUE [0x4C01/7]': 'torEst',  # mNm
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

        idx = df.columns.get_loc("torEst")
        df.insert(idx, "velErr", df["velDes"] - df["velAct"])
        df.insert(idx + 1, "posErr", df["posDes"] - df["posAct"])
        df.insert(idx + 2, "torKdEst", df["kd"] * df["velErr"] / rad_s_to_mrpm)

        df.drop("kd", axis=1, inplace=True)

        # optional: zero position offsets
        # df["posDes"] -= df["posDes"].iloc[0]
        # df["posAct"] -= df["posAct"].iloc[0]

        if convert_time:
            df["t"] *= from_milli

        if convert_torque:
            df["torDes"] *= from_milli
            df["torAct"] *= from_milli
            df["torEst"] *= from_milli
            df["torKdEst"] *= from_milli

        if convert_angle:
            df["velDes"] /= rad_s_to_mrpm
            df["velAct"] /= rad_s_to_mrpm
            df["velErr"] /= rad_s_to_mrpm
            df["accelAct"] /= rad_s_to_mrpm
            df["posDes"] /= rad_to_inc
            df["posAct"] /= rad_to_inc
            df["posErr"] /= rad_to_inc

        self.df = df
        return df

    # Write one dataframe to one output CSV
    def exportData(self, output_path):
        output_path = Path(output_path)

        colsOut = {
            't': 'Time [ms]',
            'torDes': 'Demanded Joint Torque [mNm]',
            'posDes': 'Demanded Joint Position [inc]',
            'velDes': 'Demanded Joint Velocity [mrpm]',
            'torAct': 'Joint Torque [mNm]',
            'posAct': 'Joint Position [inc]',
            'velAct': 'Sensor Fusion Filtered Fused Velocity [mrpm]',
            'accelAct': 'Sensor Fusion Filtered Fused Acceleration [mrpm/s]',
            'i': 'Current Average [mA]',
            'torEst': 'Filtered Estimated Joint Torque [mNm]',
            'posErr': 'Position Error [inc]',
            'velErr': 'Velocity Error [mrpm]',
            'torKdEst': 'Kd Torque Estimate [mNm]',
            'i2t': 'i2t'
        }

        dataOut = self.df.rename(colsOut, axis=1)
        self.dataOut = dataOut
        dataOut.to_csv(output_path, index=False)

    # Process all CSV files in input folder and export edited versions to output folder
    def process_all(
        self,
        pattern="*.csv",
        suffix="_exp",
        convert_time=False,
        convert_torque=False,
        convert_angle=False,
    ):
        file_list = sorted(glob.glob(str(self.input_dir / pattern)))

        if not file_list:
            print(f"No files found in {self.input_dir} matching '{pattern}'")
            return

        for file_name in file_list:
            file_path = Path(file_name)
            output_path = self.output_dir / f"{file_path.stem}{suffix}.csv"

            try:
                print(f"Processing: {file_path.name}")
                self.importData(
                    file_path,
                    convert_time=convert_time,
                    convert_torque=convert_torque,
                    convert_angle=convert_angle,
                )
                self.exportData(output_path)
                print(f"Saved to:   {output_path.name}")
            except Exception as e:
                print(f"Failed for {file_path.name}: {e}")