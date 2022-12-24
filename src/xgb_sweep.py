import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

import wandb

wandb.login()

# Sweep Configuration
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'validation_f1',
               'goal': 'maximize'},
    'parameters': {'max_bin': {'values': [2, 4, 8, 16, 32]},
                   'verbosity': {'values': [0]},
                   'col_sample_bytree': {'distribution': 'uniform', 'min': 0.01, 'max': 0.7},
                   'reg_lambda': {'distribution': 'log_uniform_values', 'min': 20, 'max': 100},
                   'reg_alpha': {'distribution': 'log_uniform_values', 'min': 20, 'max': 100},
                   'min_child_weight': {'distribution': 'log_uniform_values', 'min': 0.1, 'max': 1.0},
                   'learning_rate': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 0.1},
                   'max_depth': {'values': [2, 3, 4, 5, 6]},
                   'subsample': {'distribution': 'log_uniform_values', 'min': 0.1, 'max': 0.8},
                   'gamma': {'distribution': 'log_uniform_values', 'min': 0.1, 'max': 100.0},
                   }}

sweep_id = wandb.sweep(sweep_config, project="wesad", entity='berkegocmen')


def train(config=None):
    with wandb.init(config=config):
        # set sweep config
        config = wandb.config

        df = pd.read_csv('../data/3class/train_folds.csv')
        df_test = pd.read_csv('../data/s_test.csv')

        # Split data into train and validation
        df_train = df[df.kfold != 1]
        df_valid = df[df.kfold == 1]
        features = [f for f in df_train.columns if f not in ['label', 'kfold']]

        # fit the model on train
        model = xgb.XGBClassifier(**config)

        model.fit(df_train[features].values, df_train['label'].values)

        # make predictions on train set
        preds_train = model.predict(df_train[features].values)
        train_f1 = f1_score(df_train['label'].values, preds_train, average='weighted')

        # make predictions on validation set
        preds_valid = model.predict(df_valid[features].values)
        # calculate f1 score
        score = f1_score(df_valid['label'].values, preds_valid, average='weighted')

        # make predictions on test set
        preds_test = model.predict(df_test[features].values)
        # calculate f1 score
        score_test = f1_score(df_test['label'].values, preds_test, average='weighted')

        # print results
        print(f'F1 Score on train: {train_f1:.4f}', f'F1 Score on valid: {score:.4f}',
              f'F1 Score on test: {score_test:.4f}')

        # log results
        wandb.log({'train_f1': train_f1, 'validation_f1': score, 'test_f1': score_test})


wandb.agent(sweep_id, train, count=200)
