import dagshub
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Inisialisasi Dagshub untuk Tracking Online
dagshub.init(repo_owner='azerila25', repo_name='Eksperimen_SML_Ali-Reza-Bahtiar', mlflow=True)

# 2. Load Data
df = pd.read_csv('dataset_preprocessing.csv')

# Asumsi target column adalah 'target' (sesuaikan jika nama kolom di datasetmu berbeda)
X = df.drop('target', axis=1) 
y = df['target']

import numpy as np
y = np.where(y > y.mean(), 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Mulai MLflow Run
with mlflow.start_run(run_name="RandomForest_Tuning_AliReza"):
    # Hyperparameter Tuning dengan GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # --- MANUAL LOGGING (Kriteria Advance) ---
    
    # 1. Logging Parameters & Metrics
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)

    # 2. Logging Model
    mlflow.sklearn.log_model(best_model, "random_forest_model")

    # 3. Artifact Tambahan 1: Confusion Matrix Image
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - Acc: {accuracy:.4f}")
    
    cm_path = "training_confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # 4. Artifact Tambahan 2: Metric Info JSON
    metric_info = {
        "model_name": "Random Forest Classifier",
        "final_accuracy": accuracy,
        "best_params": grid_search.best_params_,
        "cv_splits": 3
    }
    
    json_path = "metric_info.json"
    with open(json_path, "w") as f:
        json.dump(metric_info, f, indent=4)
    mlflow.log_artifact(json_path)

    print("-" * 30)
    print("Training Selesai!")
    print(f"Best Accuracy: {accuracy:.4f}")
    print("Semua metrik dan artifak berhasil di-log ke DagsHub.")
    print("-" * 30)