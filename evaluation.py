import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")

class ModelEvaluator:
    """Handles pulling the pipeline back out from MLflow tracking and validating it."""
    
    def run(self, run_id: str, x_test: pd.DataFrame, y_test: pd.Series):
        print("--- Step 4: Evaluation ---")
        
        # Fetch the trained model/pipeline bundle from MLflow using run_id
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro")
        rec = recall_score(y_test, preds, average="macro")
        f1_macro = f1_score(y_test, preds, average='macro')

        # Log metric results to the active MLflow run context
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_macro", f1_macro)

        print(f"Evaluation completed | Accuracy = {acc:.3f} | F1_Macro = {f1_macro:.3f}")
        return acc, f1_macro