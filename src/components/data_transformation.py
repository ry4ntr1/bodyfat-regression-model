import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    logging.info(
        "data_transformation.py: DataTransformationConfig Instance Initialized")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        logging.info(
            "data_transformation.py: DataTransformation Instance Initialized")

    def get_data_transformer_object(self):
        """
        returns a ColumnTransformer object that can be used to preprocess the input data before fitting a machine learning model. The function defines two separate Pipeline objects for numerical and categorical columns, respectively. Each pipeline contains two steps: SimpleImputer and StandardScaler.
        The SimpleImputer step fills missing values in the data using either the median (for numerical columns) or the most frequent value (for categorical columns). The StandardScaler step scales the data by subtracting the mean and dividing by the standard deviation (for numerical columns) or by dividing by the L2 norm (for categorical columns). This is done to ensure that the features have similar scales, which can improve the performance of machine learning models by reducing the impact of differences in scale among features.
        The categorical pipeline additionally includes the OneHotEncoder step, which performs one-hot encoding on the categorical features to represent them as binary vectors. The resulting sparse matrix can be used as input to a machine learning model after any necessary scaling or preprocessing steps.
        The ColumnTransformer object is defined using the preprocessor variable, which consists of two pipelines: num_pipeline for numerical columns and cat_pipeline for categorical columns. The preprocessor object can then be used to preprocess the input data by applying the appropriate pipeline to each column.
        """
        try:
            logging.info(
                "data_transformation.py: get_data_transformer_object() called")

            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            """
            SimpleImputer: step fills missing values with either the median (for numerical columns) or the most frequent value (for categorical columns)

            StandardScaler: step standardizes the data by removing the mean and scaling to unit variance
            - feature scaling: is the process of transforming numerical features to have a similar scale, typically by subtracting the mean and dividing by the standard deviation or by scaling to a fixed range
                - done to improve the performance of machine learning models by reducing the impact of differences in scale among features
            """

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                ]
            )

            logging.info(
                "data_transformation.py: get_data_transformer_object() finished")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("data_transformation.py: initiate_data_transformation() called")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformer_object()

            # MODIFY THIS TO CHANGE THE TARGET COLUMN
            target_column_name = "math_score"

            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(
                columns=[target_column_name], axis=1)

            input_feature_test_df = test_df.drop(
                columns=[target_column_name], axis=1)

            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            logging.info(
                "data_transformation.py: initiate_data_transformation() finished")

            return (
                train_arr,
                test_arr,
            )

        except Exception as e:
            raise CustomException(e, sys)
