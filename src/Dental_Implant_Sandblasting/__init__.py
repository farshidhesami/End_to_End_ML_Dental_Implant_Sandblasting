import os
import sys
import logging

# Log formatting string
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Configuration for logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Default to 'INFO' if not set
log_dir = os.getenv('LOG_DIR', 'logs')  # Default log directory is 'logs'
logger_name = os.getenv('LOGGER_NAME', 'Dental_Implant_Sandblasting')  # Default logger name

# Path to log file
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create log directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, log_level),
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger
logger = logging.getLogger(logger_name)

# Ensure logger is available for import
__all__ = ['logger']


