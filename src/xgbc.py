import feature_engineering
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib
import os
import src.config as config
import time
import pandas as pd


def run():
    # Get training file and generate features
    # Get training file and generate features
    df_1 = feature_engineering.generate_features(config.TRAINING_FILE, original=True)
    df_2 = feature_engineering.generate_features(config.TRAINING_FILE_2, original=True)
    df_train = pd.concat([df_1.reset_index(drop=True), df_2.reset_index(drop=True)])
    print(df_train.label.value_counts())

    # Get test file and generate features
    df_valid = feature_engineering.generate_features(config.VALIDATION_FILE, original=True)

    # fill nan's in the df_
    df_train.fillna(method='bfill', inplace=True)
    df_valid.fillna(method='bfill', inplace=True)

    # get the feature names
    features = [f for f in df_train.columns if f not in ['label']]

    # initiate a Logistic Regression
    clf = xgb.XGBClassifier()

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

    print(f'Training Accuracy:{training_acc}, Validation Accuracy:{validation_acc}, Training time:{finish} seconds')

    # save the model
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, 'xgb_initial.bin')
    )


if __name__ == '__main__':
    run()
