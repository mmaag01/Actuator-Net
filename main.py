import pandas as pd
import numpy as np
from pathlib import Path
import glob
import torch
import matplotlib.pyplot as plt
from torch import nn
import xgboost as xgb
from data_utils import Data
from import_data import ImportData

units = {'t':'s', 'tor':'Nm', 'vel':'rad/s', 
         'pos':'rad', 'accel':'rad/s^2', 'i':'A'}

def processData(main_data=True, fails_data=False, other_data=False):
    DataProcessor = Data(main_experiments=main_data, crashed_experiments=fails_data, other_experiments=other_data)
    DataProcessor.process_all(suffix="_exp")

if __name__ == "__main__":
    #processData(main_data=True, fails_data=True, other_data=False)
    importer = ImportData(units=units, zero_offset=True)
    main_data = importer.importMain()
    fails_data = importer.importFails()
    other_data = importer.importOther()
    combined_data = importer.combineDatasets()
    combined_data = importer.convertUnits()

   