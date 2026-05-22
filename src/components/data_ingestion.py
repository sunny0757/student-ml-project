# ============================================================
# FILE: src/components/data_ingestion.py
# PURPOSE: Reads the raw student data CSV and splits it into
#          a training set and a testing set, then saves both.
#
# HOW IT CONNECTS TO OTHER FILES:
#   - READS FROM  → data/students.csv
#                   (path comes from src/config.py → RAW_DATA_PATH)
#   - SAVES TO    → artifacts/train.csv and artifacts/test.csv
#                   (paths come from src/config.py)
#   - CALLED BY   → pipelines/training_pipeline.py (STEP 1)
#   - FEEDS INTO  → data_validation.py and data_transformation.py
#   - LOGS TO     → logs/ folder via src/logger.py
#   - ERRORS VIA  → src/exception.py → CustomException
#
#   - Think of this as STEP 1 of the training process —
#     "Get the data and split it into train and test."
# ============================================================

import os
import sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

# Importing logger.py triggers the log file setup for the whole project
import src.logger

# CustomException gives us detailed error messages (file + line number)
from src.exception import CustomException

# Import file paths from the central config file
from src.config import RAW_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, ARTIFACTS_DIR


class DataIngestion:

    def data_ingestion(self):
        try:
            logging.info("Data Ingestion started")

            # Create the artifacts folder if it doesn't already exist
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            logging.info(f"Artifacts directory ready: {ARTIFACTS_DIR}")

            # Read the raw student data from the data/ folder
            df = pd.read_csv(RAW_DATA_PATH)
            logging.info(f"Raw data loaded — shape: {df.shape}")

            # Split the data: 80% for training, 20% for testing
            # random_state=42 makes the split the same every time you run it
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f"Data split — Train: {train_set.shape}, Test: {test_set.shape}")

            # Save the training data — used by data_validation.py and data_transformation.py
            train_set.to_csv(TRAIN_DATA_PATH, index=False)

            # Save the testing data — used by data_transformation.py and model_trainer.py
            test_set.to_csv(TEST_DATA_PATH, index=False)

            logging.info("Data Ingestion completed successfully")
            print("Data Ingestion completed")

        except Exception as e:
            # Log the error, then raise CustomException with file + line info
            logging.error(f"Error during Data Ingestion: {e}")
            raise CustomException(e, sys)
