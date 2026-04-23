# Ensemble Metrics

# Position Log Chirp (Main)
metric,             MLP,        GRU,        Ensemble
RMSE [Nm],          51.310169,  67.246284,  58.701542
MAE [Nm],           43.261726,  57.881138,  50.519234
Max Abs Error [Nm], 177.194534, 137.150879, 156.472717

# Position Multisine (Main)
metric,             MLP,        GRU,        Ensemble
RMSE [Nm],          8.177079,   8.079439,   8.042926
MAE [Nm],           4.470948,   4.199528,   4.161286
Max Abs Error [Nm], 51.309437,  52.255428,  51.528286

# PLC, PMS, TMS (Main)
metric,MLP,GRU,Ensemble
RMSE [Nm],5.393996,5.261635,5.228040
MAE [Nm],2.742030,2.546872,2.533227
Max Abs Error [Nm],55.279171,52.068981,53.311260

# Torque Multisine (Main)
metric,MLP,GRU,Ensemble
RMSE [Nm],4.731154,4.555592,4.458689
MAE [Nm],3.582587,3.339812,3.300291
Max Abs Error [Nm],26.872868,23.553421,25.213146

# Torque Step (Main)
metric,             MLP,        GRU,        Ensemble
RMSE [Nm],          1.734929,   0.365783,   0.952068
MAE [Nm],           1.257564,   0.294351,   0.713138
Max Abs Error [Nm], 8.390741,   2.478211,   4.036317

# Torque Step + CmdErrs (Main, Crashes, Other)
metric,MLP,GRU,Ensemble
RMSE [Nm],1.012267,0.254295,0.487346
MAE [Nm],0.500963,0.186458,0.261780
Max Abs Error [Nm],10.264744,2.112515,4.817434

# Torque Step CmdErrs (Other)
metric,MLP,GRU,Ensemble
RMSE [Nm],1.549247,0.215878,0.807437
MAE [Nm],0.808816,0.162230,0.445069
Max Abs Error [Nm],6.392404,1.405609,3.322324