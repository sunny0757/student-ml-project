# ============================================================
# FILE: src/logger.py
# PURPOSE: Sets up logging so every file can write messages
#          to a log file automatically.
#
# HOW IT CONNECTS TO OTHER FILES:
#   - Any file can import "logging" and use it AFTER this
#     file has been imported once at the start.
#   - Used by: data_ingestion.py, data_validation.py,
#              data_transformation.py, model_trainer.py,
#              training_pipeline.py, prediction_pipeline.py
#   - Think of this as the "diary" of the project — every
#     important step gets written here for you to check later.
# ============================================================

import logging
import os
from datetime import datetime

# Create a folder called "logs" if it doesn't already exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Name the log file using the current date and time
# so each run creates a new log file (e.g. 2024_01_15_10_30_00.log)
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Set up the logging format: [time] LEVEL - message
# All log messages from all files will be saved here
logging.basicConfig(
    filename=LOG_PATH,
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    level=logging.INFO
)
