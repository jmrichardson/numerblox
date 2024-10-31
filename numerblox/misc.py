import json
import pickle
import hashlib
import os
import logging
from datetime import datetime


class AttrDict(dict):
    """ Access dictionary elements as attributes. """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Key:
    """Numerai credentials."""
    def __init__(self, pub_id: str, secret_key: str):
        self.pub_id = pub_id
        self.secret_key = secret_key

    def __repr__(self):
        return f"Numerai Auth Key. pub_id = '{self.pub_id}'"

    def __str__(self):
        return self.__repr__()


def load_key_from_json(file_path: str, *args, **kwargs):
    """
    Initialize Key object from JSON file. \n
    Credentials file must have the following format: \n
    `{"pub_id": "PUBLIC_ID", "secret_key": "SECRET_KEY"}`
    """
    with open(file_path) as json_file:
        json_data = json.load(json_file, *args, **kwargs)
    pub_id = json_data["pub_id"]
    secret_key = json_data["secret_key"]
    return Key(pub_id=pub_id, secret_key=secret_key)


def get_cache_hash(*args, **kwds):
    """
    Generate a unique cache key based on positional and keyword arguments.

    This function serializes the provided arguments, creates a SHA-256 hash,
    and returns the first 12 characters of the hash, which can be used as
    a cache key for functions that need to store results based on input.

    Parameters:
    -----------
    *args : tuple
        Positional arguments to include in the cache key.
    **kwds : dict
        Keyword arguments to include in the cache key.

    Returns:
    --------
    str
        A 12-character hexadecimal string representing the cache key.
    """

    # Sort keyword arguments to ensure consistent ordering and convert them to a string
    sorted_kwargs = str(sorted(kwds.items()))

    # Serialize both positional and keyword arguments using pickle to generate a byte stream
    serialized = pickle.dumps((args, sorted_kwargs))

    # Generate a SHA-256 hash from the serialized arguments
    hash_bytes = hashlib.sha256(serialized).digest()

    # Generate another SHA-256 hash from the first hash and convert to hexadecimal format
    hash_hex = hashlib.sha256(hash_bytes).hexdigest()

    # Return the first 12 characters of the hash as the cache key
    return hash_hex[:12]


class Logger:
    def __init__(self, log_dir='logs', log_file=None):
        os.makedirs(log_dir, exist_ok=True)

        if log_file is None:
            log_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_file = f'numberblox_{log_time}.log'

        log_path = os.path.join(log_dir, log_file)

        self.logger = logging.getLogger('numerblox_logger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Remove existing handlers to prevent duplication
        self.logger.handlers.clear()

        # Set up console and file handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Apply a consistent formatter to both handlers
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Attach handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger
