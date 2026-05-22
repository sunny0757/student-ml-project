# ============================================================
# FILE: src/components/data_transformation.py
# PURPOSE: Prepares the data for the ML model by:
#          - Scaling number columns (math, science) to a standard range
#          - Encoding the gender column into numbers (0 or 1)
#
# HOW IT CONNECTS TO OTHER FILES:
#   - READS FROM  → artifacts/train.csv and artifacts/test.csv
#                   (created by data_ingestion.py)
#                   (paths come from src/config.py)
#   - CALLED BY   → pipelines/training_pipeline.py (STEP 3)
#   - FEEDS INTO  → model_trainer.py
#                   (returns X_train, X_test, y_train, y_test, preprocessor)
#   - LOGS TO     → logs/ folder via src/logger.py
#   - ERRORS VIA  → src/exception.py → CustomException
#
#   - Think of this as STEP 3 of the training process —
#     "Clean and prepare the data so the model can understand it."
# ============================================================

import sys
import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Importing logger.py triggers the log file setup for the whole project
import src.logger

# CustomException gives us detailed error messages (file + line number)
from src.exception import CustomException

# Import file paths from the central config file
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH


class DataTransformation:

    def data_transformation(self):
        try:
            logging.info("Data Transformation started")

            # Load the training and testing data saved by data_ingestion.py
            train_df = pd.read_csv(TRAIN_DATA_PATH)
            test_df  = pd.read_csv(TEST_DATA_PATH)
            logging.info(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")

            # Separate features (inputs) from the target label (result column)
            X_train = train_df.drop(columns=["result"])
            y_train = train_df["result"]

            X_test  = test_df.drop(columns=["result"])
            y_test  = test_df["result"]
            logging.info("Features and target column separated successfully")

            # Define which columns are numbers and which are categories
            numerical_columns   = ["math", "science"]  # Will be scaled to a standard range
            categorical_columns = ["gender"]            # Will be converted to 0s and 1s

            # Pipeline for number columns — StandardScaler brings all numbers
            # to a similar range so one column doesn't dominate the model
            num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

            # Pipeline for category columns — OneHotEncoder converts
            # "male"/"female" into separate 0/1 columns
            cat_pipeline = Pipeline(steps=[("encoder", OneHotEncoder())])

            # Combine both pipelines into one preprocessor
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )
            logging.info("Preprocessor (scaler + encoder) built successfully")

            # Fit the preprocessor on training data and transform it
            X_train = preprocessor.fit_transform(X_train)

            # Transform test data using the SAME rules learned from training data
            X_test  = preprocessor.transform(X_test)
            logging.info("Data transformation applied — train and test data are ready")

            logging.info("Data Transformation completed successfully")

            # Return everything — model_trainer.py uses X_train/X_test/y_train/y_test
            # The preprocessor is saved by model_trainer.py and later loaded
            # by prediction_pipeline.py to transform new student data
            return (X_train, X_test, y_train, y_test, preprocessor)

        except Exception as e:
            # Log the error, then raise CustomException with file + line info
            logging.error(f"Error during Data Transformation: {e}")
            raise CustomException(e, sys)
