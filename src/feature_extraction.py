import pandas as pd

import config

# Read data
df_folds = pd.read_csv(config.THREE_CLASS_FOLDS_v2)
df_test = pd.read_csv(config.THREE_CLASS_TEST_v2)

# Extract ACC features
# chest
df_folds['chest_ACC_0_mean'] = df_folds['chest_ACC_0'].rolling(window=20).mean()
df_folds['chest_ACC_0_std'] = df_folds['chest_ACC_0'].rolling(window=20).std()
df_folds['chest_ACC_0_min'] = df_folds['chest_ACC_0'].rolling(window=20).min()
df_folds['chest_ACC_0_max'] = df_folds['chest_ACC_0'].rolling(window=20).max()
df_folds['chest_ACC_1_mean'] = df_folds['chest_ACC_1'].rolling(window=20).mean()
df_folds['chest_ACC_1_std'] = df_folds['chest_ACC_1'].rolling(window=20).std()
df_folds['chest_ACC_1_min'] = df_folds['chest_ACC_1'].rolling(window=20).min()
df_folds['chest_ACC_1_max'] = df_folds['chest_ACC_1'].rolling(window=20).max()
df_folds['chest_ACC_2_mean'] = df_folds['chest_ACC_2'].rolling(window=20).mean()
df_folds['chest_ACC_2_std'] = df_folds['chest_ACC_2'].rolling(window=20).std()
df_folds['chest_ACC_2_min'] = df_folds['chest_ACC_2'].rolling(window=20).min()
df_folds['chest_ACC_2_max'] = df_folds['chest_ACC_2'].rolling(window=20).max()
df_folds['chest_ACC_sum_mean'] = (df_folds['chest_ACC_0_mean'] + df_folds['chest_ACC_1_mean'] + df_folds[
    'chest_ACC_2_mean']) / 3
df_folds['chest_ACC_sum_std'] = (df_folds['chest_ACC_0_std'] + df_folds['chest_ACC_1_std'] + df_folds[
    'chest_ACC_2_std']) / 3
# wrist
df_folds['wrist_ACC_0_mean'] = df_folds['wrist_ACC_0'].rolling(window=20).mean()
df_folds['wrist_ACC_0_std'] = df_folds['wrist_ACC_0'].rolling(window=20).std()
df_folds['wrist_ACC_0_min'] = df_folds['wrist_ACC_0'].rolling(window=20).min()
df_folds['wrist_ACC_0_max'] = df_folds['wrist_ACC_0'].rolling(window=20).max()
df_folds['wrist_ACC_1_mean'] = df_folds['wrist_ACC_1'].rolling(window=20).mean()
df_folds['wrist_ACC_1_std'] = df_folds['wrist_ACC_1'].rolling(window=20).std()
df_folds['wrist_ACC_1_min'] = df_folds['wrist_ACC_1'].rolling(window=20).min()
df_folds['wrist_ACC_1_max'] = df_folds['wrist_ACC_1'].rolling(window=20).max()
df_folds['wrist_ACC_2_mean'] = df_folds['wrist_ACC_2'].rolling(window=20).mean()
df_folds['wrist_ACC_2_std'] = df_folds['wrist_ACC_2'].rolling(window=20).std()
df_folds['wrist_ACC_2_min'] = df_folds['wrist_ACC_2'].rolling(window=20).min()
df_folds['wrist_ACC_2_max'] = df_folds['wrist_ACC_2'].rolling(window=20).max()
df_folds['wrist_ACC_sum_mean'] = (df_folds['wrist_ACC_0_mean'] + df_folds['wrist_ACC_1_mean'] + df_folds[
    'wrist_ACC_2_mean']) / 3
df_folds['wrist_ACC_sum_std'] = (df_folds['wrist_ACC_0_std'] + df_folds['wrist_ACC_1_std'] + df_folds[
    'wrist_ACC_2_std']) / 3

# Extract ECG
# chest
df_folds['chest_ECG_mean'] = df_folds['chest_ECG'].rolling(window=240).mean()
df_folds['chest_ECG_std'] = df_folds['chest_ECG'].rolling(window=240).std()
df_folds['chest_ECG_min'] = df_folds['chest_ECG'].rolling(window=240).min()
df_folds['chest_ECG_max'] = df_folds['chest_ECG'].rolling(window=240).max()

# Extract BVP
df_folds['wrist_BVP_mean'] = df_folds['wrist_BVP'].rolling(window=240).mean()
df_folds['wrist_BVP_std'] = df_folds['wrist_BVP'].rolling(window=240).std()
df_folds['wrist_BVP_min'] = df_folds['wrist_BVP'].rolling(window=240).min()
df_folds['wrist_BVP_max'] = df_folds['wrist_BVP'].rolling(window=240).max()

# Extract EDA
# chest
df_folds['chest_EDA_mean'] = df_folds['chest_EDA'].rolling(window=240).mean()
df_folds['chest_EDA_std'] = df_folds['chest_EDA'].rolling(window=240).std()
df_folds['chest_EDA_min'] = df_folds['chest_EDA'].rolling(window=240).min()
df_folds['chest_EDA_max'] = df_folds['chest_EDA'].rolling(window=240).max()

# wrist
df_folds['wrist_EDA_mean'] = df_folds['wrist_EDA'].rolling(window=240).mean()
df_folds['wrist_EDA_std'] = df_folds['wrist_EDA'].rolling(window=240).std()
df_folds['wrist_EDA_min'] = df_folds['wrist_EDA'].rolling(window=240).min()
df_folds['wrist_EDA_max'] = df_folds['wrist_EDA'].rolling(window=240).max()

# Extract EMG
# chest
df_folds['chest_EMG_mean'] = df_folds['chest_EMG'].rolling(window=20).mean()
df_folds['chest_EMG_std'] = df_folds['chest_EMG'].rolling(window=20).std()
df_folds['chest_EMG_min'] = df_folds['chest_EMG'].rolling(window=20).min()
df_folds['chest_EMG_max'] = df_folds['chest_EMG'].rolling(window=20).max()

# Extract RESP
# chest
df_folds['chest_RESP_mean'] = df_folds['chest_Resp'].rolling(window=240).mean()
df_folds['chest_RESP_std'] = df_folds['chest_Resp'].rolling(window=240).std()
df_folds['chest_RESP_min'] = df_folds['chest_Resp'].rolling(window=240).min()
df_folds['chest_RESP_max'] = df_folds['chest_Resp'].rolling(window=240).max()

# TEMP
# chest
df_folds['chest_TEMP_mean'] = df_folds['chest_Temp'].rolling(window=240).mean()
df_folds['chest_TEMP_std'] = df_folds['chest_Temp'].rolling(window=240).std()
df_folds['chest_TEMP_min'] = df_folds['chest_Temp'].rolling(window=240).min()
df_folds['chest_TEMP_max'] = df_folds['chest_Temp'].rolling(window=240).max()

# wrist
df_folds['wrist_TEMP_mean'] = df_folds['wrist_TEMP'].rolling(window=240).mean()
df_folds['wrist_TEMP_std'] = df_folds['wrist_TEMP'].rolling(window=240).std()
df_folds['wrist_TEMP_min'] = df_folds['wrist_TEMP'].rolling(window=240).min()
df_folds['wrist_TEMP_max'] = df_folds['wrist_TEMP'].rolling(window=240).max()

# On the test set
df_test['chest_ACC_0_mean'] = df_test['chest_ACC_0'].rolling(window=20).mean()
df_test['chest_ACC_0_std'] = df_test['chest_ACC_0'].rolling(window=20).std()
df_test['chest_ACC_0_min'] = df_test['chest_ACC_0'].rolling(window=20).min()
df_test['chest_ACC_0_max'] = df_test['chest_ACC_0'].rolling(window=20).max()
df_test['chest_ACC_1_mean'] = df_test['chest_ACC_1'].rolling(window=20).mean()
df_test['chest_ACC_1_std'] = df_test['chest_ACC_1'].rolling(window=20).std()
df_test['chest_ACC_1_min'] = df_test['chest_ACC_1'].rolling(window=20).min()
df_test['chest_ACC_1_max'] = df_test['chest_ACC_1'].rolling(window=20).max()
df_test['chest_ACC_2_mean'] = df_test['chest_ACC_2'].rolling(window=20).mean()
df_test['chest_ACC_2_std'] = df_test['chest_ACC_2'].rolling(window=20).std()
df_test['chest_ACC_2_min'] = df_test['chest_ACC_2'].rolling(window=20).min()
df_test['chest_ACC_2_max'] = df_test['chest_ACC_2'].rolling(window=20).max()
df_test['chest_ACC_sum_mean'] = (df_test['chest_ACC_0_mean'] + df_test['chest_ACC_1_mean'] + df_test[
    'chest_ACC_2_mean']) / 3
df_test['chest_ACC_sum_std'] = (df_test['chest_ACC_0_std'] + df_test['chest_ACC_1_std'] + df_test[
    'chest_ACC_2_std']) / 3
# wrist
df_test['wrist_ACC_0_mean'] = df_test['wrist_ACC_0'].rolling(window=20).mean()
df_test['wrist_ACC_0_std'] = df_test['wrist_ACC_0'].rolling(window=20).std()
df_test['wrist_ACC_0_min'] = df_test['wrist_ACC_0'].rolling(window=20).min()
df_test['wrist_ACC_0_max'] = df_test['wrist_ACC_0'].rolling(window=20).max()
df_test['wrist_ACC_1_mean'] = df_test['wrist_ACC_1'].rolling(window=20).mean()
df_test['wrist_ACC_1_std'] = df_test['wrist_ACC_1'].rolling(window=20).std()
df_test['wrist_ACC_1_min'] = df_test['wrist_ACC_1'].rolling(window=20).min()
df_test['wrist_ACC_1_max'] = df_test['wrist_ACC_1'].rolling(window=20).max()
df_test['wrist_ACC_2_mean'] = df_test['wrist_ACC_2'].rolling(window=20).mean()
df_test['wrist_ACC_2_std'] = df_test['wrist_ACC_2'].rolling(window=20).std()
df_test['wrist_ACC_2_min'] = df_test['wrist_ACC_2'].rolling(window=20).min()
df_test['wrist_ACC_2_max'] = df_test['wrist_ACC_2'].rolling(window=20).max()
df_test['wrist_ACC_sum_mean'] = (df_test['wrist_ACC_0_mean'] + df_test['wrist_ACC_1_mean'] + df_test[
    'wrist_ACC_2_mean']) / 3
df_test['wrist_ACC_sum_std'] = (df_test['wrist_ACC_0_std'] + df_test['wrist_ACC_1_std'] + df_test[
    'wrist_ACC_2_std']) / 3

# Extract ECG
# chest
df_test['chest_ECG_mean'] = df_test['chest_ECG'].rolling(window=240).mean()
df_test['chest_ECG_std'] = df_test['chest_ECG'].rolling(window=240).std()
df_test['chest_ECG_min'] = df_test['chest_ECG'].rolling(window=240).min()
df_test['chest_ECG_max'] = df_test['chest_ECG'].rolling(window=240).max()

# Extract BVP
df_test['wrist_BVP_mean'] = df_test['wrist_BVP'].rolling(window=240).mean()
df_test['wrist_BVP_std'] = df_test['wrist_BVP'].rolling(window=240).std()
df_test['wrist_BVP_min'] = df_test['wrist_BVP'].rolling(window=240).min()
df_test['wrist_BVP_max'] = df_test['wrist_BVP'].rolling(window=240).max()

# Extract EDA
# chest
df_test['chest_EDA_mean'] = df_test['chest_EDA'].rolling(window=240).mean()
df_test['chest_EDA_std'] = df_test['chest_EDA'].rolling(window=240).std()
df_test['chest_EDA_min'] = df_test['chest_EDA'].rolling(window=240).min()
df_test['chest_EDA_max'] = df_test['chest_EDA'].rolling(window=240).max()

# wrist
df_test['wrist_EDA_mean'] = df_test['wrist_EDA'].rolling(window=240).mean()
df_test['wrist_EDA_std'] = df_test['wrist_EDA'].rolling(window=240).std()
df_test['wrist_EDA_min'] = df_test['wrist_EDA'].rolling(window=240).min()
df_test['wrist_EDA_max'] = df_test['wrist_EDA'].rolling(window=240).max()

# Extract EMG
# chest
df_test['chest_EMG_mean'] = df_test['chest_EMG'].rolling(window=20).mean()
df_test['chest_EMG_std'] = df_test['chest_EMG'].rolling(window=20).std()
df_test['chest_EMG_min'] = df_test['chest_EMG'].rolling(window=20).min()
df_test['chest_EMG_max'] = df_test['chest_EMG'].rolling(window=20).max()

# Extract RESP
# chest
df_test['chest_RESP_mean'] = df_test['chest_Resp'].rolling(window=240).mean()
df_test['chest_RESP_std'] = df_test['chest_Resp'].rolling(window=240).std()
df_test['chest_RESP_min'] = df_test['chest_Resp'].rolling(window=240).min()
df_test['chest_RESP_max'] = df_test['chest_Resp'].rolling(window=240).max()

# TEMP
# chest
df_test['chest_TEMP_mean'] = df_test['chest_Temp'].rolling(window=240).mean()
df_test['chest_TEMP_std'] = df_test['chest_Temp'].rolling(window=240).std()
df_test['chest_TEMP_min'] = df_test['chest_Temp'].rolling(window=240).min()
df_test['chest_TEMP_max'] = df_test['chest_Temp'].rolling(window=240).max()

# wrist
df_test['wrist_TEMP_mean'] = df_test['wrist_TEMP'].rolling(window=240).mean()
df_test['wrist_TEMP_std'] = df_test['wrist_TEMP'].rolling(window=240).std()
df_test['wrist_TEMP_min'] = df_test['wrist_TEMP'].rolling(window=240).min()
df_test['wrist_TEMP_max'] = df_test['wrist_TEMP'].rolling(window=240).max()

# Drop nan's
df_folds.dropna(inplace=True)
df_test.dropna(inplace=True)

# save the new dataframes
df_folds.to_csv('../data/3class/folds_extracted_v2.csv', index=False)
df_test.to_csv('../data/3class/test_extracted_v2.csv', index=False)
