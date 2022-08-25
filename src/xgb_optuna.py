import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

import config

df = pd.read_csv(config.THREE_CLASS_FOLDS)

df_test = pd.read_csv(config.THREE_CLASS_TEST)

# Split data into train and validation
df_train = df[df.kfold != 5]
df_valid = df[df.kfold == 5]

features = [f for f in df_train.columns if f not in ['label', 'kfold']]

eval_set = [(df_valid[features].values, df_valid.label.values)]


# 1. Define an objective function to be maximized.

def objective(trial):
    # 2. Suggest values of the hyperparameters using a trial object.
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eval_set": eval_set,
        "early_stopping_rounds": 10,
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": 'gbtree',
        # L2 regularization weight.
        "reg_lambda": trial.suggest_float("reg_lambda", 20.0, 100.0, log=True),
        # L1 regularization weight.
        "reg_alpha": trial.suggest_float("reg_alpha", 20.0, 100.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 0.8),
        # sampling according to each tree.
        "min_child_weight": trial.suggest_float("min_child_weight", 0.4, 1.0),
        "max_depth": trial.suggest_int("max_depth", 1, 6),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 0.7),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),

    }

    gbm = xgb.XGBClassifier(**param)
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
    print(f'Train F1: {train_auc}, Validation F1: {validation_auc}, Test F1: {test_auc}')

    # preds = gbm.predict(df_valid[features].values)
    # accuracy = accuracy_score(df_valid['label'].values, preds)

    ...
    return validation_auc


# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
