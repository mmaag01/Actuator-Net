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

def convertData():
    DataProcessor = ProcessRawData(locIn="Raw Data", locOut="Data")
    DataProcessor.process_all(suffix_remove="_raw", suffix_add="")

if __name__ == "__main__":
    convertData()