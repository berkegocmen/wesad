import argparse
import time

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

import config


def run(fold):
    # WANDB
    # wandb.init(project="wesad", entity='berkegocmen',
    #            name=f"lgbm_default_cf_{fold}",
    #            tags=["3class", "default_params", 'lgbm', '4HZ'],
    #            group="3class_4hz")

    # Get training file and generate features
    df_folds = pd.read_csv(config.THREE_CLASS_FOLDS)
    df_test = pd.read_csv(config.THREE_CLASS_TEST)

    df_train = df_folds[df_folds['kfold'] != fold]
    df_valid = df_folds[df_folds['kfold'] == fold]

    features = [f for f in df_train.columns if f not in ['label', 'kfold']]

    PARAMS = {'reg_lambda': 16.73529729880662, 'reg_alpha': 20.635150267902112, 'subsample': 0.2745205398226232,
              'min_child_weight': 0.3259365410678807, 'max_depth': 4, 'colsample_bytree': 0.23839555397557413,
              'learning_rate': 0.0029683651691049537, "verbosity": 0,
              "objective": "binary:logistic",
              "eval_metric": "auc",
              "eval_set": [(df_valid[features].values, df_valid.label.values)],
              "early_stopping_rounds": 10,
              # use exact for small dataset.
              "tree_method": "exact", }

    # Get unique labels
    labels = df_train['label'].unique()
    labels_map = {1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation'}

    # get the feature names

    # initiate a Logistic Regression
    clf = xgb.XGBClassifier(max_depth=4, learning_rate=0.001, n_estimators=50, colsample_bytree=0.23839555397557413,
                            min_child_weight=0.3259365410678807)

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
    # wandb.log({'Training Accuracy': training_acc, 'Validation Accuracy': validation_acc, 'Test Accuracy': test_acc,
    #            'Training F1': training_f1, 'Validation F1': validation_f1, 'Test F1': test_f1,
    #            'Confusion Matrix': wandb.plot.confusion_matrix(y_true=df_valid.label.values, preds=validation_pred,
    #                                                            class_names=[None, 'baseline', 'stress', 'amusement'])})

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
    # #plt.savefig(f'../figures/lgbm_optuna_cf_{fold}_confusion_matrix.png')

    # # save the model
    # joblib.dump(
    #     clf,
    #     os.path.join(config.MODEL_OUTPUT, f'3class_lgbm__cf_{fold}_P3.bin')
    # )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--fold',
        type=int
    )

    args = parser.parse_args()
    print(args.fold)
    run(args.fold)
