import argparse
import time
import joblib
import os
import wandb

import config
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

PARAMS = {'max_bin': 141, 'max_depth': 180, 'lambda_l1': 29.23179979173314, 'lambda_l2': 38.82949791945566,
          'num_leaves': 244, 'feature_fraction': 0.5822945319860839, 'bagging_fraction': 0.778955613699345,
          'bagging_freq': 2, 'min_child_samples': 75, 'learning_rate': 0.06814947496610763}

PARAMS_2 = {'lambda_l1': 19.23179979173314, 'lambda_l2': 28.82949791945566, 'num_leaves': 244,
            'feature_fraction': 0.4697931264052483, 'bagging_fraction': 0.5653221736593377, 'bagging_freq': 6,
            'min_child_samples': 54, 'learning_rate': 0.006610920247541726}


def run(fold):
    # WANDB
    wandb.init(project="wesad", entity='berkegocmen',
               name=f"lgbm_default_cf_{fold}",
               tags=["3class", "default_params", 'lgbm', '4HZ'],
               group="3class_4hz")

    # Get training file and generate features
    df_folds = pd.read_csv(config.THREE_CLASS_FOLDS)
    df_test = pd.read_csv(config.THREE_CLASS_TEST)

    df_train = df_folds[df_folds['kfold'] != fold]
    df_valid = df_folds[df_folds['kfold'] == fold]

    # Get unique labels
    labels = df_train['label'].unique()
    labels_map = {1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation'}

    # get the feature names
    features = [f for f in df_train.columns if f not in ['label', 'kfold']]

    # initiate a Logistic Regression
    clf = LGBMClassifier()

    # fit the model
    print('Training started')
    start = time.time()
    clf.fit(df_train[features].values, df_train.label.values)
    finish = time.time() - start

    # get the training results
    print('Making predictions')

    training_pred = clf.predict(df_train[features].values)
    training_acc = accuracy_score(df_train.label.values, training_pred)
    training_f1 = f1_score(df_train.label.values, training_pred, average='weighted')

    # get the validation results
    validation_pred = clf.predict(df_valid[features].values)
    validation_acc = accuracy_score(df_valid.label.values, validation_pred)
    validation_f1 = f1_score(df_valid.label.values, validation_pred, average='weighted')

    # get the test predictions
    test_pred = clf.predict(df_test[features].values)
    test_acc = accuracy_score(df_test.label.values, test_pred)
    test_f1 = f1_score(df_test.label.values, test_pred, average='weighted')

    # # ROC AUC Score
    # training_acc_prob = clf.predict_proba(df_train[features].values)
    # validation_acc_prob = clf.predict_proba(df_valid[features].values)
    # test_acc_prob = clf.predict_proba(df_test[features].values)
    #
    # # training_auc = roc_auc_score(df_train.label.values, training_acc_prob, multi_class='ovr')
    # validation_auc = roc_auc_score(df_valid.label.values, validation_acc_prob, multi_class='ovr')
    # test_auc = roc_auc_score(df_test.label.values, test_acc_prob, multi_class='ovr')

    print(
        f'Fold:{fold}, Training Accuracy:{training_acc}, Validation Accuracy:{validation_acc}, Test Accuracy:{test_acc}'
        # f'\nTraining AUC:{training_auc}, Validation_auc:{validation_auc}'
        f'\nTraining F1:{training_f1}, Validation F1:{validation_f1}, Test F1:{test_f1}'
        f'\nTraining time:{finish} seconds')

    # log the results to wandb
    wandb.log({'Training Accuracy': training_acc, 'Validation Accuracy': validation_acc, 'Test Accuracy': test_acc,
               'Training F1': training_f1, 'Validation F1': validation_f1, 'Test F1': test_f1,
               'Confusion Matrix': wandb.plot.confusion_matrix(y_true=df_valid.label.values, preds=validation_pred,
                                                               class_names=[None, 'baseline', 'stress', 'amusement'])})

    # cm = confusion_matrix(df_test['label'], test_pred)
    # print(cm)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(cm)
    # plt.title('Confusion matrix of the classifier')
    # fig.colorbar(cax)
    # plt.set_cmap('Blues')
    # ax.set_xticklabels([''] + list(map(str, labels)))
    # ax.set_yticklabels([''] + list(map(str, labels)))
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()
    # plt.savefig(f'../figures/lgbm_optuna_cf_{fold}_confusion_matrix.png')

    # save the model
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f'3class_lgbm__cf_{fold}.bin')
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
