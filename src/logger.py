import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_XY_%H_%M_%S')}.log"
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s -%(levelname)s %(message)s",
    level=logging.INFO  # Change to DEBUG
)
