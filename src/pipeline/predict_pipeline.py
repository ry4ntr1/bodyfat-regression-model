import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, save_object, evaluate_models


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist):
        self.age = age
        self.weight = weight
        self.height = height
        self.neck = neck
        self.chest = chest
        self.abdomen = abdomen
        self.hip = hip
        self.thigh = thigh
        self.knee = knee
        self.ankle = ankle
        self.biceps = biceps
        self.forearm = forearm
        self.wrist = wrist

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Age": [self.age],
                "Weight": [self.weight],
                "Height": [self.height],
                "Neck": [self.neck],
                "Chest": [self.chest],
                "Abdomen": [self.abdomen],
                "Hip": [self.hip],
                "Thigh": [self.thigh],
                "Knee": [self.knee],
                "Ankle": [self.ankle],
                "Biceps": [self.biceps],
                "Forearm": [self.forearm],
                "Wrist": [self.wrist]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)