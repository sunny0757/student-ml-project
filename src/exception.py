# ============================================================
# FILE: src/exception.py
# PURPOSE: Creates a helpful error message whenever something
#          goes wrong in any part of the project.
#
# HOW IT CONNECTS TO OTHER FILES:
#   - CustomException is used by: data_ingestion.py,
#     data_validation.py, data_transformation.py,
#     model_trainer.py, prediction_pipeline.py, app.py
#   - Works together with logger.py — each file logs the
#     error message before raising CustomException.
#   - Think of this as the "error reporter" of the project.
# ============================================================

import sys


# This function takes an error and returns a detailed message
# telling you exactly where the error happened (file name + line number)
def error_message_detail(error, error_detail):

    # Get the traceback object — it holds info about where the error occurred
    _, _, exc_tb = error_detail.exc_info()

    # Extract the name of the file where the error happened
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Build and return a clear error message with file name and line number
    return (
        f"Error occurred in script: [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] "
        f"error message [{str(error)}]"
    )


# CustomException is a custom error class that automatically
# includes the file name and line number in its message.
# Any file can raise CustomException(e, sys) inside an except block.
class CustomException(Exception):

    def __init__(self, error, error_detail):
        super().__init__(str(error))
        # Build the full detailed error message using the function above
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        # When you print this exception, it shows the full detailed message
        return self.error_message
