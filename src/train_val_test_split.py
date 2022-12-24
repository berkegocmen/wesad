import pandas as pd
import feature_engineering
import config

df_train = pd.DataFrame()
for S in range(1, 12):
    print(S)
    df = feature_engineering.generate_features(config.training_files[f'tf_{S}'], original=True)
    df_train = pd.concat([df_train.reset_index(drop=True), df.reset_index(drop=True)])

# df_1 = feature_engineering.generate_features(config.TRAINING_FILE, original=False)
# df_2 = feature_engineering.generate_features(config.TRAINING_FILE_2, original=False)
# df_train = pd.concat([df_1.reset_index(drop=True), df_2.reset_index(drop=True)])
# print(df_train.label.value_counts())

# Get test file and generate features
df_valid = feature_engineering.generate_features(config.validation_files['vf_1'], original=True)
df_valid_2 = feature_engineering.generate_features(config.validation_files['vf_2'], original=True)
df_valid = pd.concat([df_valid.reset_index(drop=True), df_valid_2.reset_index(drop=True)])

df_test_ = feature_engineering.generate_features(config.test_files['f_1'], original=True)
df_test_2 = feature_engineering.generate_features(config.test_files['f_2'], original=True)
df_test = pd.concat([df_test_.reset_index(drop=True), df_test_2.reset_index(drop=True)])

# save train file
df_train.to_csv('../data/s_train.csv', index=False)
# save test file
df_test.to_csv('../data/s_test.csv', index=False)
# save validation file
df_valid.to_csv('../data/s_valid.csv', index=False)
