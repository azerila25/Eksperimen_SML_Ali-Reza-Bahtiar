import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    # Load data (pastikan file csv ada di folder yang sama)
    df = pd.read_csv('dataset_preprocessing.csv')
    
    # Menangani target jika masih hasil scaling
    import numpy as np
    y = df['target']
    if y.dtype != 'int':
        y = np.where(y > y.mean(), 1, 0)
    
    X = df.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logging lokal ke folder mlruns
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        mlflow.log_metric("accuracy", acc)
        # Tambahkan parameter conda_env agar Docker tahu harus pakai Python 3.12.7
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model", 
            conda_env="conda.yaml"
        )
        print(f"Model trained with accuracy: {acc}")

if __name__ == "__main__":
    train_model()