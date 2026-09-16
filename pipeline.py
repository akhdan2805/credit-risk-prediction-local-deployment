from pathlib import Path
from data_ingestion import DataIngestion
from train import CreditModelTrainer
from evaluation import ModelEvaluator

class PredictionPipeline:
    """The master pipeline orchestrating ingestion, training, evaluation, and approval."""
    
    def __init__(self, raw_data_path: str | Path):
        self.base_dir = Path(__file__).parent
        self.raw_data_path = Path(raw_data_path)
        self.ingested_dir = self.base_dir / "ingested"
        
        # Core components instantiation
        self.ingestor = DataIngestion(self.raw_data_path, self.ingested_dir)
        self.trainer = CreditModelTrainer()
        self.evaluator = ModelEvaluator()

    def execute(self):
        print("Executing Credit Prediction Pipeline...")
        
        # 1. Data Ingestion
        ingested_file_path = self.ingestor.run()
        
        # 2. Preprocessing & Training
        run_id, x_test, y_test = self.trainer.run(ingested_file_path)
        
        # 3. Evaluation
        accuracy, f1_macro = self.evaluator.run(run_id, x_test, y_test)
        
        # 4. Final Validation
        print("\n--- Pipeline Running Successfully ---")
        print(f"Model accuracy: ({accuracy:.3f})")
        print(f"Model f1_macro: ({f1_macro:.3f})")

if __name__ == "__main__":
    # Point directly to your primary relative data source file
    DATA_INPUT = Path(__file__).parent / "data_C.csv"
    
    # Initialize and execute
    credit_pipeline = PredictionPipeline(raw_data_path=DATA_INPUT)
    credit_pipeline.execute()