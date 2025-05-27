import os
import pickle
import click
import mlflow

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment('nyc-taxi-random-forest')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    mlflow.sklearn.autolog()

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():
        print('Inside mlflow...')
        rf = RandomForestRegressor(max_depth=10, random_state=1)
        print('Now fittif....')
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        print(rmse)
        mlflow.log_metric('rmse', rmse)
 


if __name__ == '__main__':
    run_train()
