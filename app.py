# ============================================================
# FILE: app.py
# PURPOSE: The main API file. This is what users or other
#          systems talk to when they want a prediction.
#          Built with FastAPI — a fast and simple web framework.
#
# HOW IT CONNECTS TO OTHER FILES:
#   - IMPORTS FROM → pipelines/prediction_pipeline.py
#                    → PredictionPipeline (loads model + makes prediction)
#                    → CustomData (packages user input as a DataFrame)
#   - INDIRECTLY USES:
#       → artifacts/model.pkl        (loaded by PredictionPipeline)
#       → artifacts/preprocessor.pkl (loaded by PredictionPipeline)
#       → src/utils.py               (load_object called by prediction_pipeline.py)
#       → src/config.py              (file paths used by prediction_pipeline.py)
#   - LOGS TO   → logs/ folder via src/logger.py
#   - ERRORS VIA → FastAPI HTTPException (returns clean error to the API caller)
#
#   HOW TO RUN:
#       uvicorn app:app --reload
#   Then open: http://127.0.0.1:8000/docs  to test the API visually
# ============================================================

import logging

# FastAPI is our web framework — it handles incoming HTTP requests
from fastapi import FastAPI, HTTPException

# BaseModel defines the shape of data the API expects from the user
from pydantic import BaseModel

# Importing logger.py triggers the log file setup for the whole project
# This is an intentional side-effect import — it must come before any logging.info() call
import src.logger  # noqa: F401

# PredictionPipeline → loads model and preprocessor, runs prediction
# CustomData         → packages raw user input into a DataFrame
from pipelines.prediction_pipeline import PredictionPipeline, CustomData

# CustomException gives us detailed error messages (file + line number)
from src.exception import CustomException

# Database connection and cursor for saving predictions
from src.database import conn, cursor

# Create the FastAPI app (this is our main application)
app = FastAPI()


# ── Input Schema ─────────────────────────────────────────────
# This defines the shape of data the user must send when making
# a prediction request. Think of it as a form with three fields.
class StudentData(BaseModel):
    gender: str  # Student's gender — must be "male" or "female"
    math: float  # Student's math score — a number between 0 and 100
    science: float  # Student's science score — a number between 0 and 100


# ── Route 1: Home Page ───────────────────────────────────────
# When someone visits "/" they get a simple welcome message
# Used to check that the API is running correctly
@app.get("/")
def home():
    logging.info("Home page accessed")
    return {"message": "Home Page"}


# ── Route 2: Prediction ──────────────────────────────────────
# When someone sends a POST request to "/predict" with student data,
# this function runs the model and returns PASS or FAIL
@app.post("/predict")
def predict(data: StudentData):
    try:
        logging.info(
            f"Prediction request received — gender: {data.gender}, math: {data.math}, science: {data.science}"
        )

        # Step 1: Pack the incoming student data into a CustomData object
        # CustomData also validates the input (e.g. math must be 0-100)
        custom_data = CustomData(
            gender=data.gender, math=data.math, science=data.science
        )

        # Step 2: Convert the student data into a DataFrame the model understands
        pred_df = custom_data.get_data_as_dataframe()

        # Step 3: Load the prediction pipeline (loads model.pkl + preprocessor.pkl)
        predict_pipeline = PredictionPipeline()

        # Step 4: Run the model on the student's data and get a raw prediction
        result = predict_pipeline.predict(pred_df)

        # Step 5: Convert the model's raw output into a human-readable label
        if result[0] == "pass":
            prediction = "PASS"
        else:
            prediction = "FAIL"

        logging.info(f"Prediction response: {prediction}")

        # Step 6: Save the prediction to the database
        cursor.execute(
            """
            INSERT INTO predictions (gender, math, science, prediction)
            VALUES (?, ?, ?, ?)
            """,
            (data.gender, data.math, data.science, prediction),
        )
        conn.commit()

        # Step 7: Send the prediction back to whoever called the API
        return {"prediction": prediction}

    except CustomException as ce:
        # If our own custom error was raised, log it and return a 400 error to the caller
        logging.error(f"Prediction failed — CustomException: {ce}")
        raise HTTPException(status_code=400, detail=str(ce))

    except Exception as e:
        # If any unexpected error happened, log it and return a 500 server error
        logging.error(f"Prediction failed — Unexpected error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error. Please try again."
        )
