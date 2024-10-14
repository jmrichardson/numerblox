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


def numerai_corr_weighted(targets: pd.Series, predictions: pd.Series, eras: pd.Series = None, sample_weight: pd.Series = None) -> float:
    # Align eras with the predictions' index if eras are provided
    if eras is not None:
        eras = eras.reindex(predictions.index)

    # Rank and gaussianize predictions
    ranked_preds = predictions.fillna(0.5).rank(pct=True, method='average')
    gauss_ranked_preds = stats.norm.ppf(ranked_preds)

    # Center target from [0...1] to [-0.5...0.5] range
    centered_target = targets - targets.mean()

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
    valid_eras = eras[valid_mask] if eras is not None else None

    # Initialize correlation result as NaN
    corr_result = np.nan

    # Group by eras and calculate correlation for each era
    if valid_eras is not None:
        correlations = []
        unique_eras = valid_eras.unique()
        for era in unique_eras:
            era_mask = valid_eras == era
            if era_mask.sum() >= 2:  # Ensure there are enough data points in the era
                era_corr, _ = stats.pearsonr(weighted_preds[era_mask], weighted_target[era_mask])
                correlations.append(era_corr)

        # If valid correlations are available, calculate the mean correlation
        if correlations:
            corr_result = np.mean(correlations)

    # If no eras provided, calculate overall correlation
    elif len(weighted_preds) >= 2:
        corr_result, _ = stats.pearsonr(weighted_preds, weighted_target)

    # Return the final correlation result
    return corr_result


def mmc_weighted(
    targets: pd.Series,
    predictions: pd.Series,
    meta_model: pd.Series,
    eras: pd.Series = None,
    sample_weight: pd.Series = None,
) -> float:

    DEFAULT_MAX_FILTERED_INDEX_RATIO = 0.2

    # Align eras with the predictions' index if eras are provided
    if eras is not None:
        eras = eras.reindex(predictions.index)

    # Step 1: Filter and sort indices to match
    ids = meta_model.dropna().index.intersection(predictions.dropna().index)
    assert len(ids) / len(meta_model) >= (1 - DEFAULT_MAX_FILTERED_INDEX_RATIO), (
        "meta_model does not have enough overlapping ids with predictions,"
        f" must have >= {round(1 - DEFAULT_MAX_FILTERED_INDEX_RATIO, 2) * 100}% overlapping ids"
    )
    assert len(ids) / len(predictions) >= (1 - DEFAULT_MAX_FILTERED_INDEX_RATIO), (
        "predictions do not have enough overlapping ids with meta_model,"
        f" must have >= {round(1 - DEFAULT_MAX_FILTERED_INDEX_RATIO, 2) * 100}% overlapping ids"
    )
    meta_model = meta_model.loc[ids].sort_index()
    predictions = predictions.loc[ids].sort_index()

    ids = targets.dropna().index.intersection(predictions.index)
    assert len(ids) / len(targets) >= (1 - DEFAULT_MAX_FILTERED_INDEX_RATIO), (
        "targets do not have enough overlapping ids with predictions,"
        f" must have >= {round(1 - DEFAULT_MAX_FILTERED_INDEX_RATIO, 2) * 100}% overlapping ids"
    )
    targets = targets.loc[ids].sort_index()
    predictions = predictions.loc[ids].sort_index()
    meta_model = meta_model.loc[ids].sort_index()

    if sample_weight is not None:
        ids = sample_weight.dropna().index.intersection(predictions.index)
        sample_weight = sample_weight.loc[ids].sort_index()
        predictions = predictions.loc[ids].sort_index()
        targets = targets.loc[ids].sort_index()
        meta_model = meta_model.loc[ids].sort_index()

    # Step 2: Rank and gaussianize predictions and meta_model
    predictions_ranked = (predictions.rank(method="average") - 0.5) / predictions.count()
    meta_model_ranked = (meta_model.rank(method="average") - 0.5) / meta_model.count()

    predictions_ranked = predictions_ranked.clip(lower=1e-6, upper=1 - 1e-6)
    meta_model_ranked = meta_model_ranked.clip(lower=1e-6, upper=1 - 1e-6)

    predictions_gaussianized = stats.norm.ppf(predictions_ranked)
    meta_model_gaussianized = stats.norm.ppf(meta_model_ranked)

    # Step 3: Orthogonalize predictions with respect to meta_model
    p = predictions_gaussianized.values
    m = meta_model_gaussianized.values

    m_dot_m = np.dot(m.T, m)
    projection = np.dot(p.T, m) / m_dot_m
    neutral_preds = p - m * projection

    # Step 4: Adjust targets
    if (targets >= 0).all() and (targets <= 1).all():
        targets = targets * 4
    targets = targets - targets.mean()
    targets_arr = targets.values

    # Step 5: Compute MMC
    mmc_score = np.nan
    if eras is not None:
        mmc_scores = []
        unique_eras = eras.unique()
        for era in unique_eras:
            era_mask = eras == era

            if sample_weight is not None:
                sample_weight_arr = sample_weight.values[era_mask]
                numerator = np.sum(sample_weight_arr * targets_arr[era_mask] * neutral_preds[era_mask])
                denominator = np.sum(sample_weight_arr)
                mmc_scores.append(numerator / denominator)
            else:
                mmc_scores.append(np.dot(targets_arr[era_mask], neutral_preds[era_mask]) / len(targets_arr[era_mask]))

        if mmc_scores:
            mmc_score = np.mean(mmc_scores)

    else:
        if sample_weight is not None:
            sample_weight_arr = sample_weight.values
            numerator = np.sum(sample_weight_arr * targets_arr * neutral_preds)
            denominator = np.sum(sample_weight_arr)
            mmc_score = numerator / denominator
        else:
            mmc_score = np.dot(targets_arr, neutral_preds) / len(targets_arr)

    return mmc_score
