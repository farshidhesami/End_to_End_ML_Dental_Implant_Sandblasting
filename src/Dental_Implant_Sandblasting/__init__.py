import os
import sys
import logging

"""
Dental_Implant_Sandblasting package initialization module.

This module sets up logging for the package. The logger can be configured using environment variables:
- LOG_LEVEL: The logging level (default: INFO).
- LOG_DIR: The directory where logs will be stored (default: logs).
- LOGGER_NAME: The name of the logger (default: Dental_Implant_Sandblasting).

Usage:
    from . import logger
    logger.info("This is an info message")
"""

# Log formatting string
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Configuration for logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()                       # # Default to 'INFO' if not set
if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
    log_level = 'INFO'

log_dir = os.getenv('LOG_DIR', 'logs')                                    # Default to 'logs' if not set
logger_name = os.getenv('LOGGER_NAME', 'Dental_Implant_Sandblasting')     # Default to 'Dental_Implant_Sandblasting' if not set

# Path to log file
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create log directory if it doesn't exist
try:
    os.makedirs(log_dir, exist_ok=True)
except Exception as e:
    print(f"Error creating log directory {log_dir}: {e}", file=sys.stderr)
    log_dir = '.'
    log_filepath = os.path.join(log_dir, "running_logs.log")

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