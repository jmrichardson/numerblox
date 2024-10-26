import os
import logging
from datetime import datetime

# Create a directory for logs if it doesn't exist
log_dir = 'tmp/logs'
os.makedirs(log_dir, exist_ok=True)

# Logger setup
logger = logging.getLogger('my_package_logger')
logger.setLevel(logging.DEBUG)  # Set the minimum logging level

# Create handlers
console_handler = logging.StreamHandler()  # Log to console
log_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
file_handler = logging.FileHandler(os.path.join(log_dir, f'walk_forward_{log_time}.log'))

# Set log level for each handler
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)

# Create formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger if not already added
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
