# ============================================================
# FILE: src/config.py
# PURPOSE: Stores all the folder and file paths used across
#          the entire project in one central place.
#
# HOW IT CONNECTS TO OTHER FILES:
#   - This is the "address book" of the project.
#     Every file that needs to read or save something
#     comes here to get the correct path.
#
#   - data_ingestion.py  → uses RAW_DATA_PATH (to read raw CSV),
#                          TRAIN_DATA_PATH, TEST_DATA_PATH (to save splits),
#                          ARTIFACTS_DIR (to create the folder)
#   - data_validation.py → uses TRAIN_DATA_PATH (to check the training data)
#   - data_transformation.py → uses TRAIN_DATA_PATH, TEST_DATA_PATH
#                               (to load data for preprocessing)
#   - model_trainer.py   → uses MODEL_PATH, PREPROCESSOR_PATH
#                          (to save the trained model and preprocessor)
#   - prediction_pipeline.py → uses MODEL_PATH, PREPROCESSOR_PATH
#                              (to load the saved model and preprocessor)
#
#   - If you ever change a folder name, just change it here —
#     all other files will automatically pick up the new path.
# ============================================================

import os

# ROOT_DIR = the main project folder (two levels up from this file)
# Example: d:/Data Science/Python/ML
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ARTIFACTS_DIR = where trained model files and split data are saved
# Example: d:/Data Science/Python/ML/artifacts
ARTIFACTS_DIR     = os.path.join(ROOT_DIR, "artifacts")

# MODEL_PATH = where the trained ML model is saved after training
MODEL_PATH        = os.path.join(ARTIFACTS_DIR, "model.pkl")

# PREPROCESSOR_PATH = where the data preprocessor (scaler + encoder) is saved
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")

# TRAIN_DATA_PATH = where the training portion of the data is saved
TRAIN_DATA_PATH   = os.path.join(ARTIFACTS_DIR, "train.csv")

# TEST_DATA_PATH = where the testing portion of the data is saved
TEST_DATA_PATH    = os.path.join(ARTIFACTS_DIR, "test.csv")

# RAW_DATA_PATH = the original raw CSV file with all student records
# This is the starting point — data_ingestion.py reads from here first
RAW_DATA_PATH     = os.path.join(ROOT_DIR, "data", "students.csv")
