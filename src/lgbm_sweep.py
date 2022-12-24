import lightgbm as lgb
import pandas as pd
from sklearn.metrics import f1_score

import wandb
from config import *

wandb.login()

# Sweep Configuration
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'validation_f1',
               'goal': 'maximize'},
    'parameters': {'max_bin': {'values': [2, 4, 8, 16, 32]},
                   'min_data_in_leaf': {'values': [32, 64, 128, 256, 512]},
                   'max_depth': {'values': [2, 3, 4, 5, 6, 8, 16]},
                   'lamda_l1': {'distribution': 'log_uniform_values', 'min': 20, 'max': 100},
                   'lamda_l2': {'distribution': 'log_uniform_values', 'min': 20, 'max': 100},
                   'num_leaves': {'values': [2, 4, 6, 8, 16, 32]},
                   'feature_fraction': {'distribution': 'log_uniform_values', 'min': 0.4, 'max': 0.8},
                   'bagging_fraction': {'distribution': 'log_uniform_values', 'min': 0.4, 'max': 0.8},
                   'bagging_freq': {'values': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
                   'min_child_samples': {'distribution': 'log_uniform_values', 'min': 50, 'max': 150},
                   'learning_rate': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 0.1}}}

sweep_id = wandb.sweep(sweep_config, project="wesad", entity='berkegocmen')


def train(config=None):
    with wandb.init(config=config):
        # set sweep config
        config = wandb.config

        df = pd.read_csv(THREE_CLASS_EXTRACTED_FOLDS_v2)
        df_test = pd.read_csv(THREE_CLASS_EXTRACTED_TEST_v2)

        # Split data into train and validation
        df_train = df[df.kfold != 1]
        df_valid = df[df.kfold == 1]
        features = [f for f in df_train.columns if f not in ['label', 'kfold']]

        # fit the model on train
        model = lgb.LGBMClassifier(**config)

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


wandb.agent(sweep_id, train, count=100)
