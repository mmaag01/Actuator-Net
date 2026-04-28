## Evaluation Results

## v1 = [torDes, posDes, velDes, posAct, posErr, velErr, velAct, accelAct, i, torEst]

## Evaluation Results

| Date       | Model      | RMSE    | MAE    | Max Err. | Ep. | Val MSE  | Main/Crashes/Other | tStep/TMS/PMS/PLC   | Smooth | Scaler | Excluded Features    | Hyperparameters |
|------------|------------|---------|--------|----------|-----|----------|--------------------|---------------------|--------|--------|----------------------|-----------------|
| 4-27 16:54 | WH         | 13.7197 | 4.3576 | 221.5468 | 24  | 0.009226 | True/False/False   | True/True/True/True | True   | power  | i2t, torKdEst, kd, t | 8,32,True 
| 4-28 10:44 | MLP        | 13.9573 | 4.5603 | 229.0313 | 14  | 0.008893 | True/False/False   | True/True/True/True | True   | power  | i2t, torKdEst, kd, t | 64,4 
| 4-28 11:41 | MLP        | 13.8131 | 4.5519 | 223.1466 | 28  | 0.008542 | True/False/False   | True/True/True/True | True   | power  | i2t, torKdEst, kd, t | 32,3
| 4-28 13:13 | MLP        | 13.8416 | 4.5922 | 223.4447 | 28  | 0.009205 | True/False/False   | True/True/True/True | True   | power | i2t, torKdEst, kd, t | 32,3 |
| 4-28 13:26 | MLP        | 13.7795 | 4.2720 | 220.6429 | 73  | 0.007720 | True/False/False   | True/True/True/True | True   | power | i2t, torKdEst, kd, t | 32,3 |
