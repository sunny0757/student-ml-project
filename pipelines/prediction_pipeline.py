# ============================================================
# FILE: pipelines/prediction_pipeline.py
# PURPOSE: Used AFTER training is done. Takes new student data
#          (gender, math score, science score) and predicts
#          whether the student will PASS or FAIL.
#
# HOW IT CONNECTS TO OTHER FILES:
#   - LOADS FROM  → artifacts/model.pkl (saved by model_trainer.py)
#                 → artifacts/preprocessor.pkl (saved by model_trainer.py)
#                   Both paths come from src/config.py
#   - USES        → src/utils.py → load_object()
#   - CALLED BY   → app.py (FastAPI /predict route)
#   - LOGS TO     → logs/ folder via src/logger.py
#   - ERRORS VIA  → src/exception.py → CustomException
#
#   - This file has TWO classes:
#       1. CustomData        → packages raw user input as a DataFrame
#       2. PredictionPipeline → loads model + preprocessor, runs prediction
# ============================================================

import sys
import logging

import pandas as pd

# Importing logger.py triggers the log file setup for the whole project
import src.logger

# CustomException gives us detailed error messages (file + line number)
from src.exception import CustomException

# load_object from utils.py loads saved .pkl files from disk
from src.utils import load_object

# File paths for the saved model and preprocessor
from src.config import MODEL_PATH, PREPROCESSOR_PATH


# ── Class 1: PredictionPipeline ──────────────────────────────
# Loads the saved model and preprocessor, then makes a prediction
# Called by app.py → predict() route
class PredictionPipeline:

    def __init__(self):
        try:
            logging.info("Loading model and preprocessor from disk")

            # Load the trained ML model saved by model_trainer.py
            self.model = load_object(MODEL_PATH)

            # Load the preprocessor (scaler + encoder) saved by model_trainer.py
            self.preprocessor = load_object(PREPROCESSOR_PATH)

            logging.info("Model and preprocessor loaded successfully")

        except Exception as e:
            logging.error(f"Failed to load model or preprocessor: {e}")
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            logging.info("Running prediction on new student data")

            # Step 1: Transform the raw input using the same preprocessor from training
            data_scaled = self.preprocessor.transform(features)

            # Step 2: Use the loaded model to predict pass/fail
            prediction = self.model.predict(data_scaled)

            logging.info(f"Prediction result: {prediction}")
            return prediction

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise CustomException(e, sys)


# ── Class 2: CustomData ──────────────────────────────────────
# Takes the user's raw input and turns it into a format the model accepts
# Called by app.py → predict() route BEFORE PredictionPipeline.predict()
class CustomData:

    def __init__(self, gender, math, science):
        try:
            # Validate that gender is one of the two accepted values
            if gender not in ["male", "female"]:
                raise ValueError(f"gender must be 'male' or 'female', got '{gender}'")

            # Validate that math score is between 0 and 100
            if not (0 <= math <= 100):
                raise ValueError(f"math score must be between 0 and 100, got '{math}'")

            # Validate that science score is between 0 and 100
            if not (0 <= science <= 100):
                raise ValueError(f"science score must be between 0 and 100, got '{science}'")

            # Store the validated input values
            self.gender  = gender
            self.math    = math
            self.science = science

            logging.info(f"CustomData created — gender: {gender}, math: {math}, science: {science}")

        except Exception as e:
            logging.error(f"Invalid input data: {e}")
            raise CustomException(e, sys)

    def get_data_as_dataframe(self):
        try:
            # Convert the student's input into a DataFrame (table with one row)
            # Column names must match exactly what was used during training
            custom_data_input_dict = {
                "gender":  [self.gender],
                "math":    [self.math],
                "science": [self.science],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f"Input data converted to DataFrame successfully")
            return df

        except Exception as e:
            logging.error(f"Error converting data to DataFrame: {e}")
            raise CustomException(e, sys)
