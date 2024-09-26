import json
import pickle
import hashlib
import pandas as pd
import numpy as np
from scipy import stats


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


def get_sample_weights(data, wfactor=.2, eras=None):
    """
    Calculate sample weights for the given data, optionally handling weights by era.

    The weights are calculated using an exponential decay function, and if eras are provided
    (or exist as a column in the data), the weights are averaged across each era.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset for which sample weights will be calculated.
    wfactor : float, optional (default=0.2)
        A weighting factor that controls the decay rate of the weights. The lower the value, the more
        weight is concentrated on later entries in the data.
    eras : pandas.Series, optional (default=None)
        The era data to assign weights. If not provided, the function will attempt to use the 'era' column
        from the data.

    Returns:
    --------
    pandas.Series
        A series of sample weights for each row in the input data.
    """

    data_copy = data.copy()  # Create a copy to avoid modifying the original data

    num_weights = len(data_copy)

    # First, calculate the weights as if we are not handling eras
    weights = np.exp(-np.arange(num_weights) / (wfactor * num_weights))[::-1]
    normalized_weights = weights * (num_weights / weights.sum())

    if eras is None and 'era' in data_copy.columns:
        # If eras is not supplied, try to get it from the data's "era" column
        eras = data_copy['era']

    if eras is not None:
        # Ensure eras are treated as string or categorical values
        data_copy['era'] = eras.values
        unique_eras = eras.unique()

        weights = np.zeros(num_weights)

        # Apply the same average weight within each era
        for era in unique_eras:
            era_indices = data_copy[data_copy['era'] == era].index

            # Convert era_indices to positional indices
            pos_indices = data_copy.index.get_indexer(era_indices)

            # Take the average of the calculated weights within the era
            avg_weight = normalized_weights[pos_indices].mean()

            # Assign the average weight to all the positions in the era
            weights[pos_indices] = avg_weight

        data_copy['sample_weight'] = weights
        data_copy = data_copy.drop(columns=['era'])  # Drop the era column if added
    else:
        data_copy['sample_weight'] = normalized_weights

    return data_copy['sample_weight']


def numerai_corr_weighted(true_targets: pd.Series, predictions: pd.Series, sample_weight: pd.Series = None) -> float:
    """
    Computes Numerai correlation (Corrv2).
    Returns negative correlation as loss (lower is better).

    Parameters
    ----------
    true_targets : pd.Series
        True target values, expected to be in the range [0, 1].
    predictions : pd.Series
        Predicted values.
    sample_weight : pd.Series or None, default=None
        Sample weights to apply in the correlation calculation.

    Returns
    -------
    loss : float
        Negative Numerai correlation (since lower is better in loss functions).
    """
    # Rank and gaussianize predictions
    ranked_preds = predictions.fillna(0.5).rank(pct=True, method='average')
    gauss_ranked_preds = stats.norm.ppf(ranked_preds)

    # Center target from [0...1] to [-0.5...0.5] range
    centered_target = true_targets - true_targets.mean()

    # Accentuate tails of predictions and targets
    preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
    target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5

    # If sample_weight is provided, apply it
    if sample_weight is not None:
        weighted_preds = preds_p15 * sample_weight
        weighted_target = target_p15 * sample_weight
    else:
        weighted_preds = preds_p15
        weighted_target = target_p15

    # Remove inf and NaN values from weighted_preds and weighted_target
    valid_mask = np.isfinite(weighted_preds) & np.isfinite(weighted_target)
    weighted_preds = weighted_preds[valid_mask]
    weighted_target = weighted_target[valid_mask]

    # Check if we still have enough valid data points to compute correlation
    if len(weighted_preds) < 2:
        return np.nan  # Not enough data to compute correlation

    # Pearson correlation
    corr, _ = stats.pearsonr(weighted_preds, weighted_target)

    # Return negative correlation as loss (since lower is better)
    return corr

