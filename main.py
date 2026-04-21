import pandas as pd
import numpy as np
from pathlib import Path
import torch # type: ignore
import matplotlib.pyplot as plt
from torch import nn # type: ignore
import xgboost as xgb
from data_utils import ProcessRawData, ImportData
#from import_data import ImportData
#git add .
#git commit -m "short, specific message"
#git push

units = {'t':'s', 'tor':'Nm', 'vel':'rad/s', 
         'pos':'rad', 'accel':'rad/s^2', 'i':'A'}

def convertData():
    DataProcessor = ProcessRawData()
    DataProcessor.process_all(suffix="_exp")

if __name__ == "__main__":
    convertData()
    df = pd.read_csv("C:\\Users\\maxma\\OneDrive\\ETH\\Bachelors_Thesis\\Data Collection\\Actuator Net\\Data\\Main\\tStep\\Nm105_0.15_1.5_0.09_exp.csv")
    print(len(df.columns))
    print(df.columns)
    importer = ImportData(units=units, zero_offset=True)
    main_data = importer.importMain()
    fails_data = importer.importFails()
    other_data = importer.importOther()
    combined_data = importer.combineDatasets()
    file_names = combined_data['file_name']
    combined_data = combined_data.drop(columns=['file_name'])