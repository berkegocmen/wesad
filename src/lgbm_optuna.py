import lightgbm as lgb
import optuna
import feature_engineering
import pandas as pd
import config
from sklearn.metrics import accuracy_score

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
df_valid = feature_engineering.generate_features(config.VALIDATION_FILE, original=True)
df_valid_2 = feature_engineering.generate_features(config.VALIDATION_FILE, original=True)
df_valid = pd.concat([df_valid.reset_index(drop=True), df_valid_2.reset_index(drop=True)])

features = [f for f in df_train.columns if f not in ['label']]

print(df_train['label'].value_counts())
print(df_valid['label'].value_counts())


# 1. Define an objective function to be maximized.

def objective(trial):


    # 2. Suggest values of the hyperparameters using a trial object.
    param = {
        'boosting_type': 'dart',
        'objective': 'multiclas',
        'verbosity': 1,
        'max_bin': trial.suggest_int('max_bin', 2, 256),
        'max_depth': trial.suggest_int('max_depth', 2, 256),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 30.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 30.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate ': trial.suggest_float('learning_rate', 0.001, 0.1),

    }

    gbm = lgb.LGBMClassifier(**param)
    gbm.fit(df_train[features].values, df_train['label'].values)

    # validate
    preds = gbm.predict(df_valid[features].values)
    accuracy = accuracy_score(df_valid['label'].values, preds)

    ...
    return accuracy


# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)
