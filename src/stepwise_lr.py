import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd


def run():
    df_train = pd.read_csv('../data/s_train.csv')
    df_valid = pd.read_csv('../data/s_valid.csv')
    # df_test = pd.read_csv('../data/s_valid.csv')

    features = [f for f in df_train.columns if f not in ['label']]

    for feature in features:
        wandb.init(project="wesad", entity="berkegocmen")

        # name the run
        wandb.run.name = f'lr_stepwise_{feature}'
        wandb.run.save()

        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(df_train[feature].values.reshape(-1, 1), df_train.label.values)

        # get the training results
        training_pred = lr.predict(df_train[feature].values.reshape(-1, 1))

        # training accuracy
        training_acc = accuracy_score(df_train.label.values, training_pred)

        # get the validation results
        validation_pred = lr.predict(df_valid[feature].values.reshape(-1, 1))

        # validation accuracy
        validation_acc = accuracy_score(df_valid.label.values, validation_pred)

        # training f1 score
        training_f1 = f1_score(df_train.label.values, training_pred, average='weighted')

        # validation f1 score
        validation_f1 = f1_score(df_valid.label.values, validation_pred, average='weighted')

        # log the results
        wandb.log({'Training Accuracy': training_acc, 'Validation Accuracy': validation_acc, 'Training F1': training_f1,
                   'Validation F1': validation_f1})

        # finish the run
        wandb.run.finish()


if __name__ == '__main__':
    run()
