import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

df_train = pd.read_csv('../data/s_train.csv')
df_valid = pd.read_csv('../data/s_valid.csv')
df_test = pd.read_csv('../data/s_valid.csv')

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
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 50.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 50.0, log=True),
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
    validation_acc_prob = gbm.predict_proba(df_valid[features].values)
    validation_auc = roc_auc_score(df_valid.label.values, validation_acc_prob, multi_class='ovr')

    # test
    test_acc_prob = gbm.predict_proba(df_test[features].values)
    test_auc = roc_auc_score(df_test.label.values, test_acc_prob, multi_class='ovr')

    # print the results
    print(f'Validation AUC: {validation_auc}, Test AUC: {test_auc}')

    # preds = gbm.predict(df_valid[features].values)
    # accuracy = accuracy_score(df_valid['label'].values, preds)

    ...
    return validation_auc


# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)
