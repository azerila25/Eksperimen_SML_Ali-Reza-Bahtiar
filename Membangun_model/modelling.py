import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "dataset_preprocessing.csv")
    df = pd.read_csv(data_path)

    y = df["target"]
    if y.dtype != int:
        y = np.where(y > y.mean(), 1, 0)

    X = df.drop(columns=["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        
        # Tanpa conda_env sama sekali — biarkan MLflow pakai default
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Accuracy: {acc}")
        print(f"Run ID: {run.info.run_id}")