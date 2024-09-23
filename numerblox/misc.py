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
    sorted_kwargs = str(sorted(kwds.items()))
    serialized = pickle.dumps((args, sorted_kwargs))
    hash_bytes = hashlib.sha256(serialized).digest()
    hash_hex = hashlib.sha256(hash_bytes).hexdigest()
    return hash_hex[:12]


def get_sample_weights(data, wfactor=.2, eras=None):
    data_copy = data.copy()  # Create a copy to avoid modifying the original data

    num_weights = len(data_copy)

    # First, calculate the weights as if we are not handling eras
    weights = np.exp(-np.arange(num_weights) / (wfactor * num_weights))[::-1]
    normalized_weights = weights * (num_weights / weights.sum())

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
    return -corr

