from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.utils.helper import calculate_scale_pos_weight

class TrainModel:
    def __init__(self, preprocessor, X_train, y_train):
        self.preprocessor = preprocessor

        self.X_train = X_train
        self.y_train = y_train
        
        
    def build_pipeline(self):
        try:
            model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', XGBClassifier(eval_metric='auc', random_state=42, scale_pos_weight=calculate_scale_pos_weight(self.y_train)))
            ])
            model.fit(self.X_train, self.y_train)
            return model
        except Exception as e:
            print(f"Error during building pipeline: {e}")
            raise e