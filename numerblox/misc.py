import json
import pickle
import hashlib
import pandas as pd
import numpy as np


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

