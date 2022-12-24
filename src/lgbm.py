import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import os
import src.config as config
import time
import wandb
import seaborn as sns
import matplotlib.pyplot as plt


def run():
    # Get training, validation and test_file
    df_train = pd.read_csv(config.TRAINING_FILE)
    df_valid = pd.read_csv(config.VALIDATION_FILE)
    df_test = pd.read_csv(config.TEST_FILE)

    # CHEST FEATURES
    # features = ['chest_ACC_0', 'chest_ACC_1', 'chest_ACC_2', 'chest_ECG', 'chest_EMG', 'chest_EDA', 'chest_Temp',
    #                  'chest_Resp']
    # WRIST FEATURES
    # features = ['wrist_ACC_0', 'wrist_ACC_1', 'wrist_ACC_2', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP']
    # get the feature names
    features = [f for f in df_train.columns if f not in ['label']]

    # initiate wandb
    wandb.init(project="wesad", entity="berkegocmen")
    # name the run
    wandb.run.name = 'lgbm'
    wandb.run.save()

    # Classifier only with chest features
    clf = LGBMClassifier(**config.PARAMS)

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

    # validation f1 score
    validation_f1 = f1_score(df_valid.label.values, validation_pred, average='weighted')
    # training f1 score
    training_f1 = f1_score(df_train.label.values, training_pred, average='weighted')

    # confusion matrix
    cm = confusion_matrix(df_valid.label.values, validation_pred)

    # plot the confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('confusion_matrix_lgbm.png')
    # log the plot
    # wandb.log({"Confusion Matrix": wandb.Image(plt)})
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                               y_true=df_valid.label.values, preds=validation_pred)})

    # log the results
    wandb.log({'Training Accuracy': training_acc, 'Validation Accuracy': validation_acc, 'Validation F1': validation_f1,
               'Training F1': training_f1})
    # finish the run
    wandb.run.finish()

    print(f'Training Accuracy:{training_acc}, Validation Accuracy:{validation_acc}, Training time:{finish} seconds')

    # save the model
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, 'lgbm.bin')
    )


if __name__ == '__main__':
    run()
