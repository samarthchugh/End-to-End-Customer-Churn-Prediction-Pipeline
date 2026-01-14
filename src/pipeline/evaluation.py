from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluation:
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        Evaluate the model using various metrics.
        
        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            
        Returns:
            dict: A dictionary containing accuracy, precision, recall, and F1-score.
        """
        try:
            print("-"*50)
            print(f"{"Evaluating the Model":^50}")
            y_pred = model.predict(X_test)
            y_true = y_test
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            raise e