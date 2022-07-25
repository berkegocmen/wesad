from src.feature_engineering import generate_features
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import src.config as config


def run():
    # Get training file and generate features
    df_train = generate_features(config.TRAINING_FILE)
    print(df_train.label.value_counts())

    # Get test file and generate features
    df_valid = generate_features(config.VALIDATION_FILE)

    # fill nan's in the df_
    df_train.fillna(method='bfill', inplace=True)
    df_valid.fillna(method='bfill', inplace=True)

    # get the feature names
    features = [f for f in df_train.columns if f not in ['label']]

    # initiate a Logistic Regression
    lr = LogisticRegression(solver='saga', max_iter=10, verbose=1, n_jobs=-1)

    # fit the model
    lr.fit(df_train[features].values, df_train.label.values)

    # get the training results
    training_pred = lr.predict(df_train[features].values)

    # training accuracy
    training_acc = accuracy_score(df_train.label.values, training_pred)

    # get the validation results
    validation_pred = lr.predict(df_valid[features].values)

    # validation accuracy
    validation_acc = accuracy_score(df_valid.label.values, validation_pred)

    # save the model
    joblib.dump(
        lr,
        os.path.join(config.MODEL_OUTPUT, 'lr_initial.bin')
    )


if __name__ == '__main__':
    run()
