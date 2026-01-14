from src.pipeline.train import Trainer
from src.pipeline.evaluation import Evaluation

def run():
    trainer = Trainer()
    model, X_test, y_test = trainer.initiate_training()
    
    evaluator = Evaluation()
    metrics = evaluator.evaluate_model(model, X_test, y_test)
    
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print(f"{"Evaluation Completed Successfully":^50}")
    print("-"*50)
    
if __name__ == "__main__":
    run()