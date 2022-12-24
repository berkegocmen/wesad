import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import f1_score

df = pd.read_csv('../data/3class/train_folds.csv')

df_test = pd.read_csv('../data/s_test.csv')

# Split data into train and validation
df_train = df[df.kfold != 5]
df_valid = df[df.kfold == 5]

features = [f for f in df_train.columns if f not in ['label', 'kfold']]

print(df_train['label'].value_counts())
print(df_valid['label'].value_counts())


# 1. Define an objective function to be maximized.

def objective(trial):
    # 2. Suggest values of the hyperparameters using a trial object.
    param = {
        'boosting_type': 'dart',
        'objective': 'multiclas',
        'verbosity': 1,
        'max_bin': trial.suggest_int('max_bin', 2, 32),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 32, 128),
        'max_depth': trial.suggest_int('max_depth', 2, 32),
        'lambda_l1': trial.suggest_float('lambda_l1', 20.0, 100.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 20.0, 100.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 32),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.8),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.8),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 50, 150),
        'learning_rate ': trial.suggest_float('learning_rate', 0.001, 0.1),

    }

    gbm = lgb.LGBMClassifier(**param)
    gbm.fit(df_train[features].values, df_train['label'].values)

    # train auc
    train_pred = gbm.predict(df_train[features].values)
    train_auc = f1_score(df_train.label.values, train_pred, average='weighted')

    # validate
    validation_acc_prob = gbm.predict(df_valid[features].values)
    validation_auc = f1_score(df_valid.label.values, validation_acc_prob, average='weighted')

    # test
    test_acc_prob = gbm.predict(df_test[features].values)
    test_auc = f1_score(df_test.label.values, test_acc_prob, average='weighted')

    # print the results
    print(f'Train AUC: {train_auc}, Validation AUC: {validation_auc}, Test AUC: {test_auc}')

    # preds = gbm.predict(df_valid[features].values)
    # accuracy = accuracy_score(df_valid['label'].values, preds)

    ...
    return validation_auc


# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40)
