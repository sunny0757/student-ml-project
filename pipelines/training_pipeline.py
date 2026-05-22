# ============================================================
# FILE: pipelines/training_pipeline.py
# PURPOSE: The main script you run to TRAIN the model.
#          It connects all 4 components in the correct order
#          and runs them one after another.
#
# HOW IT CONNECTS TO OTHER FILES:
#   - CALLS (in order):
#       1. data_ingestion.py       → reads CSV, saves train.csv + test.csv
#       2. data_validation.py      → checks train.csv for quality issues
#       3. data_transformation.py  → preprocesses data, returns arrays
#       4. model_trainer.py        → trains + saves model.pkl + preprocessor.pkl
#   - LOGS TO   → logs/ folder via src/logger.py
#   - ERRORS VIA → src/exception.py → CustomException
#
#   HOW TO RUN:
#       python pipelines/training_pipeline.py
# ============================================================

import sys
import logging

# Importing logger.py triggers the log file setup — MUST be first
import src.logger

# CustomException gives us detailed error messages (file + line number)
from src.exception import CustomException

# Import all 4 component classes
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

try:
    logging.info("=" * 50)
    logging.info("Training Pipeline started")

    # STEP 1: Data Ingestion
    # Reads data/students.csv → saves train.csv and test.csv to artifacts/
    logging.info("STEP 1: Data Ingestion")
    obj = DataIngestion()
    obj.data_ingestion()

    # STEP 2: Data Validation
    # Reads train.csv → checks for missing values, correct columns, data types
    logging.info("STEP 2: Data Validation")
    validation_obj = DataValidation()
    validation_obj.data_validation()

    # STEP 3: Data Transformation
    # Reads train.csv + test.csv → scales numbers, encodes categories
    # Returns processed arrays and the preprocessor object
    logging.info("STEP 3: Data Transformation")
    transformation_obj = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor = (
        transformation_obj.data_transformation()
    )

    # STEP 4: Model Training
    # Trains the model → evaluates accuracy → saves model.pkl + preprocessor.pkl
    logging.info("STEP 4: Model Training")
    trainer_obj = ModelTrainer()
    trainer_obj.model_training(X_train, X_test, y_train, y_test, preprocessor)

    logging.info("Training Pipeline completed successfully")
    logging.info("=" * 50)

except Exception as e:
    # If anything fails in any step, log it and raise the detailed error
    logging.error(f"Training Pipeline failed: {e}")
    raise CustomException(e, sys)
