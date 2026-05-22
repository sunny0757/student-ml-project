# ============================================================
# FILE: src/components/data_validation.py
# PURPOSE: Checks the training data to make sure it is clean
#          and has the right columns before training begins.
#
# HOW IT CONNECTS TO OTHER FILES:
#   - READS FROM  → artifacts/train.csv (created by data_ingestion.py)
#                   (path comes from src/config.py → TRAIN_DATA_PATH)
#   - CALLED BY   → pipelines/training_pipeline.py (STEP 2)
#   - LOGS TO     → logs/ folder via src/logger.py
#   - ERRORS VIA  → src/exception.py → CustomException
#
#   - Think of this as STEP 2 of the training process —
#     "Check the data is clean and correct before using it."
# ============================================================

import sys
import logging

import pandas as pd

# Importing logger.py triggers the log file setup for the whole project
import src.logger

# CustomException gives us detailed error messages (file + line number)
from src.exception import CustomException

# Import the training data file path from the central config file
from src.config import TRAIN_DATA_PATH


class DataValidation:

    def data_validation(self):
        try:
            logging.info("Data Validation started")

            # Load the training CSV that was saved by data_ingestion.py
            df = pd.read_csv(TRAIN_DATA_PATH)
            logging.info(f"Training data loaded for validation — shape: {df.shape}")

            # Check how many values are missing in each column
            print("MISSING VALUES")
            missing = df.isnull().sum()
            print(missing)
            logging.info(f"Missing values per column:\n{missing}")

            # Show the data type of each column (e.g. int, float, object)
            print("\nDATA TYPES")
            print(df.dtypes)
            logging.info(f"Column data types:\n{df.dtypes}")

            # Check that the columns are exactly what we expect
            expected_columns = ["gender", "math", "science", "result"]

            if list(df.columns) == expected_columns:
                print("\nAll columns are correct")
                logging.info("Column validation passed — all expected columns are present")
            else:
                # If columns don't match, log a warning and print a message
                print("\nColumn mismatch found")
                logging.warning(
                    f"Column mismatch! Expected: {expected_columns}, "
                    f"Got: {list(df.columns)}"
                )

            logging.info("Data Validation completed successfully")

        except Exception as e:
            # Log the error, then raise CustomException with file + line info
            logging.error(f"Error during Data Validation: {e}")
            raise CustomException(e, sys)
