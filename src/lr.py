import argparse
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import config
import wandb


def run(fold):
    # WANDB
    wandb.init(project="wesad", entity='berkegocmen',
               name=f"lr_default_extracted_cf_{fold}",
               tags=["3class", "default_params", 'lr', '4HZ', 'extracted_feat'],
               group="3class_4hz")

    # Get training file and generate features
    df_folds = pd.read_csv(config.THREE_CLASS_EXTRACTED_FOLDS_v2)
    df_test = pd.read_csv(config.THREE_CLASS_EXTRACTED_TEST_v2)

    df_train = df_folds[df_folds['kfold'] != fold]
    df_valid = df_folds[df_folds['kfold'] == fold]

    # # Get unique labels
    # labels = df_train['label'].unique()
    # labels_map = {1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation'}

    # get the feature names
    features = [f for f in df_train.columns if f not in ['label', 'kfold']]

    # Scale the features

    scalers = [StandardScaler() for i in range(len(features))]
    for i, scaler in enumerate(scalers):
        scaler.fit(df_train[features[i]].values.reshape(-1, 1))
        df_train.loc[:, features[i]] = scaler.transform(df_train[features[i]].values.reshape(-1, 1))
        df_valid.loc[:, features[i]] = scaler.transform(df_valid[features[i]].values.reshape(-1, 1))
        df_test.loc[:, features[i]] = scaler.transform(df_test[features[i]].values.reshape(-1, 1))
    # initiate a Logistic Regression
    clf = LogisticRegression()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--fold',
        type=int
    )

    args = parser.parse_args()
    print(args.fold)
    run(args.fold)
