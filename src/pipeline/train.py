from sklearn.model_selection import train_test_split
from src.pipeline.preprocessing import Preprocessing
from src.pipeline.modeling import TrainModel
import joblib
import os
from src.constant import RAW_DATA_PATH, CLEAN_DATA_PATH, SAVED_MODEL_PATH

class Trainer:
    def __init__(self):
        self.raw_data_path = RAW_DATA_PATH
        self.clean_data_path = CLEAN_DATA_PATH
    
    def initiate_preprocessing(self):
        try:
            print("-"*50)
            print(f"{"Starting Data Preprocessing...":^50}")
            preprocessor = Preprocessing()
            data = preprocessor.load_data(self.raw_data_path)
            clean_data = preprocessor.clean_data(data)
            preprocessor.save_clean_data(clean_data, self.clean_data_path)
            return clean_data, preprocessor
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            raise e
    
        
    def initiate_training(self):
        try:
            clean_data, preprocessor = self.initiate_preprocessing()
            print("-"*50)
            print(f"{"Starting the Training Pipeline":^50}")
            
            # Splitting features and target
            X = clean_data.drop('Churn', axis=1)
            y = clean_data['Churn']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Build preprocessor
            preproc = preprocessor.build_processor(X_train)
            
            # Modeling
            model_trainer = TrainModel(preproc, X_train, y_train)
            model = model_trainer.build_pipeline()
            
            # save the trained model
            if not os.path.exists(os.path.dirname(SAVED_MODEL_PATH)):
                os.makedirs(os.path.dirname(SAVED_MODEL_PATH))
            joblib.dump(model, SAVED_MODEL_PATH)
            print(f"{"Training Completed Successfully":^50}\n")
            
            return model, X_test, y_test
        except Exception as e:
            print(f"Error during training: {e}")
            raise e
