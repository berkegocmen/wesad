import pandas as pd
import feature_engineering
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import src.config as config
import time
from sklearn.metrics import roc_auc_score

PARAMS = {'max_bin': 141, 'max_depth': 180, 'lambda_l1': 19.23179979173314, 'lambda_l2': 28.82949791945566,
          'num_leaves': 244, 'feature_fraction': 0.5822945319860839, 'bagging_fraction': 0.778955613699345,
          'bagging_freq': 2, 'min_child_samples': 75, 'learning_rate': 0.06814947496610763}


def run(fold):
    # Get training file and generate features
    df = pd.read_csv('../data/s_folds.csv')

    df_train = df[df.kfold != fold]  # get the training data
    df_valid = df[df.kfold == fold]  # get the validation data

    # fill nan's in the df_
    # df_train.fillna(method='bfill', inplace=True)
    # df_valid.fillna(method='bfill', inplace=True)

    # get the feature names
    features = [f for f in df_train.columns if f not in ['label', 'kfold']]

    # initiate a Logistic Regression
    clf = LGBMClassifier(**PARAMS)

    # fit the model
    print('Training started')
    start = time.time()
    clf.fit(df_train[features].values, df_train.label.values)
    finish = time.time() - start

    # get the training results
    print('Making predictions')
    training_pred = clf.predict(df_train[features].values)

    # training accuracy
    training_acc = accuracy_score(df_train.label.values, training_pred)

    # get the validation results
    validation_pred = clf.predict(df_valid[features].values)

    # validation accuracy
    validation_acc = accuracy_score(df_valid.label.values, validation_pred)

    # ROC AUC Score
    training_acc_prob = clf.predict_proba(df_train[features].values)
    validation_acc_prob = clf.predict_proba(df_valid[features].values)

    training_auc = roc_auc_score(df_train.label.values, training_acc_prob, multi_class='ovr')
    validation_auc = roc_auc_score(df_valid.label.values, validation_acc_prob, multi_class='ovr')

    print(
        f'Fold:{fold}, Training Accuracy:{training_acc}, Validation Accuracy:{validation_acc}'
        f'\nTraining AUC:{training_auc}, Valdidation_auc:{validation_auc}'
        f'\nTraining time:{finish} seconds')

    # # save the model
    # joblib.dump(
    #     clf,
    #     os.path.join(config.MODEL_OUTPUT, 'lgbm_simple_full_data.bin')
    # )


if __name__ == '__main__':
    for fold_ in range(8):
        print(fold_)
        run(fold_)
