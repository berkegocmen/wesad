import pandas as pd

import config
import feature_engineering

df_train = pd.DataFrame()
fold = 0
for S in range(1, 14):
    print(S, int(fold // 1))
    df = feature_engineering.generate_features(config.training_files[f'tf_{S}'], original=True,
                                               sample_rate='250000000ns')
    print(df['chest_EDA'].mean(), df['chest_EDA'].std())

    df['kfold'] = int(fold // 1)
    df_train = pd.concat([df_train.reset_index(drop=True), df.reset_index(drop=True)])

    fold = fold + 0.5

# df_1 = feature_engineering.generate_features(config.TRAINING_FILE, original=False)
# df_2 = feature_engineering.generate_features(config.TRAINING_FILE_2, original=False)
# df_train = pd.concat([df_1.reset_index(drop=True), df_2.reset_index(drop=True)])
# print(df_train.label.value_counts())

# Get test file and generate features
df_valid = feature_engineering.generate_features(config.test_files['f_1'], original=True, sample_rate='250000000ns')
df_valid_2 = feature_engineering.generate_features(config.test_files['f_2'], original=True, sample_rate='250000000ns')
df_valid = pd.concat([df_valid.reset_index(drop=True), df_valid_2.reset_index(drop=True)])

# save validation file
df_valid.to_csv('../data/3class/s_folds_test_v2.csv', index=False)

# df = pd.concat([df_valid.reset_index(drop=True), df_train.reset_index(drop=True)])
# df = df_train
#
# df['kfold'] = -1
#
# # Randomize the rows of the dataframe
# df = df.sample(frac=1).reset_index(drop=True)
#
# # fetch labels
# y = df.label.values
#
# # Initiate the kfold class from model_selection
# kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#
# # fill the new kfold column
# for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
#     df.loc[v_, 'kfold'] = f

# save the new csv with kfold column
df_train.to_csv('../data/3class/test_v2.csv', index=False)
