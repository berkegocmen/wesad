import pandas as pd
import feature_engineering
from sklearn.model_selection import StratifiedKFold
import config

df_train = pd.DataFrame()
for S in range(1, 14):
    print(S)
    df = feature_engineering.generate_features(config.training_files[f'tf_{S}'], original=True)
    df_train = pd.concat([df_train.reset_index(drop=True), df.reset_index(drop=True)])

# df_1 = feature_engineering.generate_features(config.TRAINING_FILE, original=False)
# df_2 = feature_engineering.generate_features(config.TRAINING_FILE_2, original=False)
# df_train = pd.concat([df_1.reset_index(drop=True), df_2.reset_index(drop=True)])
# print(df_train.label.value_counts())

# Get test file and generate features
df_valid = feature_engineering.generate_features(config.validation_files['vf_1'], original=True)
df_valid_2 = feature_engineering.generate_features(config.validation_files['vf_1'], original=True)
df_valid = pd.concat([df_valid.reset_index(drop=True), df_valid_2.reset_index(drop=True)])

df = pd.concat([df_valid.reset_index(drop=True), df_train.reset_index(drop=True)])

df['kfold'] = -1

# Randomize the rows of the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# fetch labels
y = df.label.values

# Initiate the kfold class from model_selection
kf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

# fill the new kfold column
for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

# save the new csv with kfold column
df.to_csv('../data/s_folds.csv', index=False)
