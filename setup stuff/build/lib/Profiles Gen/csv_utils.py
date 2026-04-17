import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
class Data:
    def __init__(self, fName, locIn = "Aposs Code\\.mh\\Raw CSV Files\\", locOut = "CSV Files\\", path = "C:\\Users\\maxma\\OneDrive\\ETH\\Bachelors_Thesis\\Data Collection\\"):
        self.fNameIn = path + locIn + fName + ".csv"
        print(self.fNameIn)
        self.fNameOut = path + locOut + fName + "_exp.csv"
        self.base_path = Path(path)
        self.input_dir = self.base_path / locIn
        self.output_dir = self.base_path / locOut
        
        assert os.path.exists(self.fNameIn), "Input File does not exist."
        assert os.path.exists(path+locOut), "Output Location does not exist."
        assert self.input_dir.exists(), f"Input folder does not exist: {self.input_dir}"
        assert self.output_dir.exists(), f"Output folder does not exist: {self.output_dir}"

        self.raw = None
        self.df = None
        self.dataOut = None

    # Read a .csv file with the format output by the motor controller, apply unit conversions and store the data in a pandas dataframe
    def importData(self):
        #define unit conversions
        rad_s_to_mrpm, from_milli, rad_to_inc = 60000.0/(2*np.pi), 0.001, 4096.0/(2*np.pi)
        
        #define column names for the dataframe, based on the format of the .csv file output by the motor controller, and apply unit conversions to the relevant columns
        new_cols = {'User Parameter [0x2201/31]' : 't', # ms    
                'User Parameter [0x2201/99]' : 'stable', # 1/0
                'User Parameter [0x2201/20]' : 'torDes', # mNm
                'User Parameter [0x2201/93]' : 'posDes', # inc
                'User Parameter [0x2201/97]' : 'velDes', # mrpm
                'User Parameter [0x2201/1]' : 'torAct', # mNm
                '1: PO_BUSMOD_VALUE [0x4C01/2]' : 'posAct', # inc
                '1: PO_BUSMOD_VALUE [0x4C01/5]' : 'velAct', # mrpm
                '1: PO_BUSMOD_VALUE [0x4C01/6]' : 'accelAct', # mrpm/s
                '1: PO_BUSMOD_VALUE [0x4C01/3]' : 'i', # mA
                'User Parameter [0x2201/69]' : 'kd', # mNm/(rad/s)
                '1: PO_BUSMOD_VALUE [0x4C01/7]' : 'torEst', # mNm
                '1: PO_BUSMOD_VALUE [0x4C01/8]' : 'i2t'}
        cols = list(new_cols.values())
        df = pd.read_csv(self.fNameIn, skiprows = 1) #= range(15)
        df = df[df.columns[1::2]] #ignore duplicate time columns
        self.raw = df.copy()
        df = df.set_axis(cols, axis=1) #rename columns
        df = df[df.stable == 1] #only keep rows where the motor controller was in stable state
        df.drop('stable', axis=1, inplace=True) 
        
        idx = df.columns.get_loc('torEst') #get index of torque demand column
        df.insert(idx, 'velErr', df.velDes - df.velAct) #insert velocity error column after velocity demand column
        df.insert(idx+1, 'posErr', df.posDes - df.posAct) #insert position error column after position demand column
        df.insert(idx+2, 'torKdEst', df.kd * df.velErr * 1/rad_s_to_mrpm) #insert Kd torque estimate column after position error
        df.drop('kd', axis=1, inplace=True)
        #df['posDes'] = df['posDes'] - df['posDes'].iloc[0]
        #df['posAct'] = df['posAct'] - df['posAct'].iloc[0]
        
        rad, Nm, s = False, False, False
        if s is True: 
            #convert time from ms to s
            df.t *= from_milli
        if Nm is True:
            #convert derivative gain from mNm/(rad/s) to Nm/(rad/s)
            df.kd *= from_milli
            #convert torque from mNm to Nm
            df.torDes *= from_milli
            df.torAct *= from_milli
            df.torEst *= from_milli
            df.torKdEst *= from_milli
        if rad is True:
            #convert velocity from mrpm to rad/s
            df.velDes /= (rad_s_to_mrpm)
            df.velAct /= (rad_s_to_mrpm)
            df.velErr /= (rad_s_to_mrpm)
            #convert acceleration from mrpm/s to rad/s^2
            df.accelAct /= (rad_s_to_mrpm)
            #convert position from inc to rad
            df.posDes /= rad_to_inc
            df.posAct /= rad_to_inc
            df.posErr /= rad_to_inc
        
        self.df = df #store the data in a pandas dataframe for further processing and testing

    # Write the data stored in the dataframe to a .csv file with appropriate column names and unit conversions applied
    def exportData(self):
        colsOut = {'t' : 'Time [ms]', 
                'torDes' : 'Demanded Joint Torque [mNm]', 
                'posDes' : 'Demanded Joint Position [inc]', 
                'velDes' : 'Demanded Joint Velocity [mrpm]',
                'torAct' : 'Joint Torque [mNm]',
                'posAct' : 'Joint Position [inc]',
                'velAct' : 'Sensor Fusion Filtered Fused Velocity [mrpm]',
                'accelAct' : 'Sensor Fusion Filtered Fused Acceleration [mrpm/s]',
                'i' : 'Current Average [mA]',
                'torEst' : 'Filtered Estimated Joint Torque [mNm]',
                'posErr' : 'Position Error [inc]',
                'velErr' : 'Velocity Error [mrpm]',
                'torKdEst' : 'Kd Torque Estimate [mNm]',
                'i2t' : 'i2t'}
        
        dataOut = self.df.rename(colsOut, axis=1) #rename columns
        self.dataOut = dataOut
        dataOut.to_csv(self.fNameOut) #write the data to a .csv file with the new column names and unit conversions applied
