"""
Model Training script for the "Visit with Us" Wellness Tourism Package
prediction project.

- Loads the train / test data prepared by data_prep.py
- Builds a preprocessing + classification pipeline
- Trains and evaluates several candidate models
- Tracks every run (params, metrics, artifacts) with MLflow
- Selects the best model (by test ROC-AUC) and saves it to disk
  and registers it in the local MLflow Model Registry
"""
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

DATA_DIR = "tourism_project/data"
MODEL_DIR = "tourism_project/model_building"
TARGET = "ProdTaken"
EXPERIMENT_NAME = "Tourism_Wellness_Package_Prediction"


def load_data():
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]
    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath('mlflow.db')}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()
    preprocessor = build_preprocessor(X_train)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        ),
    }

    results = []
    best_model_name, best_pipeline, best_score = None, None, -1.0

    for name, clf in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)

            mlflow.log_param("model_type", name)
            for p_name, p_val in clf.get_params().items():
                if isinstance(p_val, (int, float, str, bool)) or p_val is None:
                    mlflow.log_param(p_name, p_val)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipeline, artifact_path="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )

            results.append({"model": name, **metrics})

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model_name = name
                best_pipeline = pipeline

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\n=== Model comparison (test set) ===")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model_name} (ROC-AUC = {best_score:.4f})")

    # Save the best model locally for deployment
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = f"{MODEL_DIR}/best_tourism_model_v1.joblib"
    joblib.dump(best_pipeline, best_model_path)
    print(f"Best model saved to: {best_model_path}")

    results_df.to_csv(f"{MODEL_DIR}/model_comparison_results.csv", index=False)

    # Register the best run in the local MLflow Model Registry
    with mlflow.start_run(run_name=f"{best_model_name}_final") as run:
        mlflow.log_metrics({f"best_{k}": v for k, v in
                             results_df.iloc[0].drop("model").to_dict().items()})
        mlflow.sklearn.log_model(
            best_pipeline, artifact_path="model",
            registered_model_name="tourism_wellness_package_model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )

    return results_df, best_model_name, best_model_path


if __name__ == "__main__":
    main()
