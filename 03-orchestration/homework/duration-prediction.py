
import pickle
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import root_mean_squared_error

import mlflow 
import mlflow.xgboost

import xgboost as xgb 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment('nyc-taxi-experiment')

models_folder = Path("models")
models_folder.mkdir(exist_ok=True)


def read_dataframe(year, month):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"

    df_original = pd.read_parquet(url)

    print(f"Record Count: {len(df_original)}")

    df_original['duration'] = (df_original.tpep_dropoff_datetime - df_original.tpep_pickup_datetime)
    df_original['duration'] = df_original.duration.dt.total_seconds()/60

    df = df_original[(df_original.duration >= 1) & (df_original.duration <= 60) ]

    print(f"Filtered Record Count: {len(df)}")



    return df


def create_X(df, dv=None):
    print("1. Creating X matrice.....")

    categorical = ["PULocationID", "DOLocationID"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    print("2. Training model.....")
    with mlflow.start_run() as run:

        lr = LinearRegression()

        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)

        print(f"Intercept: {lr.intercept_}")



        rmse = root_mean_squared_error(y_val, y_pred)
        
        mlflow.log_metric("rmse", rmse)

        with open("./models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("./models/preprocessor.b", artifact_path="preprocessor")
        


    
    run_id = run.info.run_id#["run_id"]
    return run_id


def run(year, month):
    print("0. Into run.....")

    train_df = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year +1 
    next_month = month + 1 if month < 12 else 1
    val_df = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(train_df)
    X_val, _ = create_X(val_df, dv)


    target = 'duration'
    y_train = train_df[target].values
    y_val = val_df[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MlFlow Run ID: {run_id}")
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model for taxi trip duration prediction")
    parser.add_argument("--year", type=int, required=True, help="Year of data to train on")
    parser.add_argument("--month", type=int, required=True, help="Month of data to train on")

    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    # save run id to a file 
    with open("run_id.txt", "w") as f:
        f.write(run_id)

