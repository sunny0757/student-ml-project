# ============================================================
# FILE: src/utils.py
# PURPOSE: Provides helper functions for saving and loading
#          files (like trained models) to/from disk.
#
# HOW IT CONNECTS TO OTHER FILES:
#   - model_trainer.py       → calls save_object() to save the
#                              trained model and preprocessor to disk
#   - prediction_pipeline.py → calls load_object() to load
#                              the saved model and preprocessor from disk
#   - LOGS TO                → logs/ folder via src/logger.py
#   - ERRORS VIA             → src/exception.py → CustomException
#
#   - Think of this file as the "storage helper" —
#     it handles reading and writing .pkl files so
#     other files don't have to repeat that code.
# ============================================================

import sys
import logging
import pickle

# Importing logger.py triggers the log file setup for the whole project
import src.logger

# CustomException gives us detailed error messages (file + line number)
from src.exception import CustomException


# Saves any Python object (like a trained model) to a file on disk
# file_path → where to save it (comes from config.py paths)
# obj       → the object to save (e.g., a trained model or preprocessor)
def save_object(file_path, obj):
    try:
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully to: {file_path}")

    except Exception as e:
        logging.error(f"Failed to save object to {file_path}: {e}")
        raise CustomException(e, sys)


# Loads a previously saved Python object back from disk
# file_path → path to the saved file (comes from config.py paths)
# Returns   → the object that was saved (e.g., the trained model)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object loaded successfully from: {file_path}")
        return obj

    except Exception as e:
        logging.error(f"Failed to load object from {file_path}: {e}")
        raise CustomException(e, sys)
