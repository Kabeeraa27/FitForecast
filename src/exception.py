import sys
import traceback
from src.logger import logging

def error_message_detail(error, error_detail):
    exc_type, exc_obj, exc_tb = error_detail
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_num = exc_tb.tb_lineno
    error_message = f"ERROR OCCURRED IN PYTHON SCRIPT: {file_name}, LINE: {line_num}, ERROR MESSAGE: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
