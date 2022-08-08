import argparse
import time
import joblib
import os

import config
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

PARAMS = {'max_bin': 141, 'max_depth': 180, 'lambda_l1': 19.23179979173314, 'lambda_l2': 28.82949791945566,
          'num_leaves': 244, 'feature_fraction': 0.5822945319860839, 'bagging_fraction': 0.778955613699345,
          'bagging_freq': 2, 'min_child_samples': 75, 'learning_rate': 0.06814947496610763}


def run(fold):
    # Get training file and generate features
    df = pd.read_csv('../data/s_folds.csv')

    df_train = df[df.kfold != fold]  # get the training data
    df_valid = df[df.kfold == fold]  # get the validation data

    # Get unique labels
    labels = df_train['label'].unique()
    labels_map = {1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation'}

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

    cm = confusion_matrix(df_valid['label'], validation_pred)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    plt.set_cmap('Blues')
    ax.set_xticklabels([''] + labels.map(labels_map).tolist())
    ax.set_yticklabels([''] + labels.map(labels_map).tolist())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.savefig(f'../figures/lgbm_optuna_cf_{fold}_confusion_matrix.png')

    # save the model
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f'lgbm_optuna_cf_{fold}.bin')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--fold',
        type=int
    )

    args = parser.parse_args()
    print(args.fold)
    run(args.fold)
