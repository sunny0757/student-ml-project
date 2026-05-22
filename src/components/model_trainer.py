# ============================================================
# FILE: src/components/model_trainer.py
# PURPOSE: Trains a Logistic Regression model on the prepared
#          data, evaluates how well it performs, and saves
#          both the model and the preprocessor to disk.
#
# HOW IT CONNECTS TO OTHER FILES:
#   - RECEIVES DATA FROM → data_transformation.py
#                          (X_train, X_test, y_train, y_test, preprocessor)
#   - USES               → src/utils.py → save_object()
#   - SAVES TO           → artifacts/model.pkl
#                        → artifacts/preprocessor.pkl
#                          (paths from src/config.py)
#   - CALLED BY          → pipelines/training_pipeline.py (STEP 4 — final step)
#   - FEEDS INTO         → prediction_pipeline.py loads the saved model + preprocessor
#   - LOGS TO            → logs/ folder via src/logger.py
#   - ERRORS VIA         → src/exception.py → CustomException
#
#   - Think of this as STEP 4 (the final step) of the training process —
#     "Train the model, check its accuracy, and save it for later use."
# ============================================================

import sys
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Importing logger.py triggers the log file setup for the whole project
import src.logger

# CustomException gives us detailed error messages (file + line number)
from src.exception import CustomException

# save_object from utils.py handles saving Python objects as .pkl files
from src.utils import save_object

# File paths where the trained model and preprocessor will be saved
from src.config import MODEL_PATH, PREPROCESSOR_PATH


class ModelTrainer:

    # Receives the prepared data from data_transformation.py
    def model_training(self, X_train, X_test, y_train, y_test, preprocessor):
        try:
            logging.info("Model Training started")

            # Create a Logistic Regression model
            model = LogisticRegression()
            logging.info("Logistic Regression model created")

            # Train the model using the training data
            model.fit(X_train, y_train)
            logging.info("Model training completed")

            # Use the trained model to predict results on the test data
            predictions = model.predict(X_test)

            # Check how accurate the model is
            accuracy = accuracy_score(y_test, predictions)
            print("\nModel Accuracy:", accuracy)
            logging.info(f"Model Accuracy: {accuracy}")

            # Confusion matrix shows correct vs wrong predictions per class
            conf_matrix = confusion_matrix(y_test, predictions)
            print("\nConfusion_Matrix:", conf_matrix)
            logging.info(f"Confusion Matrix:\n{conf_matrix}")

            # Full classification report: precision, recall, f1-score
            report = classification_report(y_test, predictions)
            print(report)
            logging.info(f"Classification Report:\n{report}")

            # Save the trained model to artifacts/model.pkl
            # prediction_pipeline.py will load this later to make predictions
            save_object(MODEL_PATH, model)
            logging.info(f"Model saved to: {MODEL_PATH}")

            # Save the preprocessor to artifacts/preprocessor.pkl
            # prediction_pipeline.py will load this to transform new student data
            save_object(PREPROCESSOR_PATH, preprocessor)
            logging.info(f"Preprocessor saved to: {PREPROCESSOR_PATH}")

            print("\nModel Saved")
            logging.info("Model Training completed successfully")

        except Exception as e:
            # Log the error, then raise CustomException with file + line info
            logging.error(f"Error during Model Training: {e}")
            raise CustomException(e, sys)
