from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import VotingRegressor
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from typing import Callable, Union
from sklearn.utils import check_random_state
from scipy import stats
from .misc import Logger
from typing import Optional, Union

# Setup logger
logger = Logger(log_dir='logs', log_file='meta_model.log').get_logger()


def numerai_corr_score(
    targets: pd.Series,
    predictions: pd.Series,
    eras: pd.Series,
    meta_data: Optional[dict] = None,
    sample_weight: Optional[pd.Series] = None
) -> float:
    """
    Calculate the Numerai correlation score between targets and predictions with optional per-era and weighted correlations.

    Parameters:
    - targets (pd.Series): Series of true target values.
    - predictions (pd.Series): Series of predicted values.
    - eras (pd.Series): Series indicating the era of each observation.
    - meta_data (Optional[dict]): Optional metadata for additional information.
    - sample_weight (Optional[pd.Series]): Optional series of weights for each prediction.

    Returns:
    - float: The mean per-era correlation score or the overall correlation if no eras are provided.
    """
    # Align `eras` and `sample_weight` to the index of `predictions`
    eras = eras.reindex(predictions.index)
    if sample_weight is not None:
        sample_weight = sample_weight.reindex(predictions.index)

    # Prepare predictions and targets for correlation computation
    preds_filled = predictions.fillna(0.5).values
    targets_values = targets.values

    # Rank and Gaussianize predictions
    ranked_preds = stats.rankdata(preds_filled, method='average') / len(preds_filled)
    gauss_ranked_preds = stats.norm.ppf(ranked_preds)

    # Center targets around zero
    centered_target = targets_values - targets_values.mean()

    # Accentuate the tails for more sensitivity in the extreme values
    preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
    target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5

    # Apply sample weights if provided
    if sample_weight is not None:
        weighted_preds = preds_p15 * sample_weight.values
        weighted_target = target_p15 * sample_weight.values
    else:
        weighted_preds = preds_p15
        weighted_target = target_p15

    # Filter out non-finite values (e.g., NaNs, infs)
    valid_mask = np.isfinite(weighted_preds) & np.isfinite(weighted_target)
    weighted_preds = weighted_preds[valid_mask]
    weighted_target = weighted_target[valid_mask]
    valid_eras = eras[valid_mask] if eras is not None else None

    # Calculate per-era correlation if `eras` is provided and has sufficient data
    corr_result = np.nan  # Default result if conditions aren't met
    if valid_eras is not None and not valid_eras.empty:
        # Create DataFrame for efficient grouping
        df = pd.DataFrame({
            'weighted_preds': weighted_preds,
            'weighted_target': weighted_target,
            'eras': valid_eras
        })

        # Filter out eras with less than 2 samples
        era_counts = df['eras'].value_counts()
        valid_eras_list = era_counts[era_counts >= 2].index
        df = df[df['eras'].isin(valid_eras_list)]

        # Compute per-era correlations if we have valid data
        if not df.empty:
            correlations = df.groupby('eras').apply(
                lambda group: np.corrcoef(group['weighted_preds'], group['weighted_target'])[0, 1]
            )
            corr_result = correlations.mean() if not correlations.empty else np.nan
        else:
            logger.warning("No eras with sufficient samples for correlation calculation.")

    # Fallback to overall correlation if no valid per-era data
    elif len(weighted_preds) >= 2:
        corr_result = np.corrcoef(weighted_preds, weighted_target)[0, 1]

    return corr_result


def mmc_score(
        targets: pd.Series,
        predictions: pd.Series,
        eras: pd.Series,
        meta_data: pd.Series,
        sample_weight: Optional[pd.Series] = None
) -> float:
    """
    Calculate the Mean Model Correlation (MMC) score, adjusted for orthogonality with meta data.

    Args:
        targets (pd.Series): Target values, expected to be in range [0, 1].
        predictions (pd.Series): Model predictions.
        eras (pd.Series): Era identifiers.
        meta_data (pd.Series): Meta data for orthogonalization.
        sample_weight (Optional[pd.Series]): Sample weights, if provided; otherwise assumed to be 1.

    Returns:
        float: The mean MMC score, NaN if eras are insufficient for calculation.
    """
    # Align sample weights with targets
    if sample_weight is not None:
        sample_weight = sample_weight.reindex(targets.index)
        sample_weight_values = sample_weight.values
    else:
        sample_weight_values = np.ones_like(targets)

    # Convert series to numpy arrays for optimized computations
    targets_arr, predictions_arr = targets.values, predictions.values
    meta_data_arr, eras_arr = meta_data.values, eras.values

    # Rank and gaussianize predictions and meta_data
    predictions_ranked = (stats.rankdata(predictions_arr, method='average') - 0.5) / len(predictions_arr)
    meta_data_ranked = (stats.rankdata(meta_data_arr, method='average') - 0.5) / len(meta_data_arr)
    predictions_gaussianized = stats.norm.ppf(np.clip(predictions_ranked, 1e-6, 1 - 1e-6))
    meta_data_gaussianized = stats.norm.ppf(np.clip(meta_data_ranked, 1e-6, 1 - 1e-6))

    # Orthogonalize predictions with respect to meta_data
    projection = np.dot(predictions_gaussianized, meta_data_gaussianized) / np.dot(meta_data_gaussianized, meta_data_gaussianized)
    neutral_predictions = predictions_gaussianized - meta_data_gaussianized * projection

    # Adjust targets scaling if in range [0, 1]
    if np.all((targets_arr >= 0) & (targets_arr <= 1)):
        targets_arr *= 4
    targets_arr -= targets_arr.mean()

    # Construct DataFrame to facilitate per-era calculations
    df = pd.DataFrame({
        'targets': targets_arr,
        'neutral_predictions': neutral_predictions,
        'eras': eras_arr,
        'sample_weight': sample_weight_values
    })

    # Filter out eras with fewer than 2 samples
    valid_eras = df['eras'].value_counts()[lambda x: x >= 2].index
    df = df[df['eras'].isin(valid_eras)]
    if df.empty:
        return np.nan

    # Compute MMC per era
    df['numerator'] = df['sample_weight'] * df['targets'] * df['neutral_predictions']
    grouped = df.groupby('eras')
    per_era_mmc = grouped['numerator'].sum() / grouped['sample_weight'].sum()

    return per_era_mmc.mean()


def numerai_payout_score(
        targets: pd.Series,
        predictions: pd.Series,
        eras: pd.Series,
        meta_data: pd.Series,
        sample_weight: pd.Series = None
) -> float:
    """
    Calculate the Numerai payout score based on correlation and MMC (Meta Model Contribution) scores.

    Args:
        targets (pd.Series): True target values for each sample.
        predictions (pd.Series): Model predictions for each sample.
        eras (pd.Series): Era identifiers for each sample to group by eras.
        meta_data (pd.Series): Additional metadata for custom handling.
        sample_weight (pd.Series, optional): Weights for each sample, defaults to None.

    Returns:
        float: The calculated payout score, a weighted combination of correlation and MMC scores.
    """
    # Calculate correlation and MMC scores
    numerai_corr = numerai_corr_score(targets, predictions, eras, meta_data, sample_weight)
    mmc = mmc_score(targets, predictions, eras, meta_data, sample_weight)

    # Weighted combination of scores
    score = numerai_corr * 0.5 + mmc * 2
    return score


def get_sample_weights(
        data: pd.DataFrame,
        wfactor: float = 0.2,
        eras: Optional[Union[pd.Series, np.ndarray]] = None
) -> pd.Series:
    """
    Generate sample weights for data, optionally adjusting weights based on eras.

    Args:
        data (pd.DataFrame): Input data containing samples and optional "era" column.
        wfactor (float): Weight factor controlling the rate of decay in weights.
        eras (Optional[Union[pd.Series, np.ndarray]]): Series or array specifying eras;
            if None, will use 'era' column from `data` if present.

    Returns:
        pd.Series: Sample weights, with decay applied across samples, averaged within eras if provided.
    """
    num_weights = len(data)

    # Calculate decaying weights and normalize
    weights = np.exp(-np.arange(num_weights) / (wfactor * num_weights))[::-1]
    normalized_weights = weights * (num_weights / weights.sum())

    # Use the provided eras or retrieve from data if not supplied
    if eras is None and 'era' in data.columns:
        eras = data['era']

    if eras is not None:
        # Compute weights averaged within each era
        temp_df = pd.DataFrame({
            'era': eras,
            'normalized_weights': normalized_weights
        }, index=data.index)
        sample_weights = temp_df.groupby('era')['normalized_weights'].transform('mean')
    else:
        # Return weights directly if no eras are provided
        sample_weights = pd.Series(normalized_weights, index=data.index)

    return sample_weights


class GreedyEnsemble:
    def __init__(
        self,
        max_ensemble_size: int = 10,
        metric: Union[str, Callable] = "corr",
        use_replacement: bool = True,
        sorted_initialization: bool = True,
        initial_n: int = None,
        num_bags: int = 50,
        random_state: Union[int, None] = None
    ):

        self.max_ensemble_size = max_ensemble_size
        self.use_replacement = use_replacement
        self.sorted_initialization = sorted_initialization
        self.initial_n = initial_n
        self.num_bags = num_bags
        self.random_state = random_state

        self.weights_ = None
        self.selected_model_names_ = None

        if isinstance(metric, str):
            if metric == "corr":
                self.metric = numerai_corr_score
            elif metric == "mmc":
                self.metric = mmc_score
            elif metric == "payout":
                self.metric = numerai_payout_score
            else:
                raise ValueError("Unsupported metric string. Choose 'corr', 'mmc', or 'payout'.")
        elif callable(metric):
            self.metric = metric
        else:
            raise TypeError("Metric must be a string or a callable.")

    def fit(
            self,
            oof: pd.DataFrame,
            sample_weights: pd.Series = None
    ):

        rng = check_random_state(self.random_state)
        oof_targets = oof['target']
        oof_eras = oof['era']
        meta_data = oof.get('meta_data', None)
        oof_predictions = oof.drop(columns=['era', 'target', 'meta_data'], errors='ignore')
        model_names = oof_predictions.columns.tolist()

        if self.max_ensemble_size < 1:
            raise ValueError("max_ensemble_size cannot be less than one!")
        if self.num_bags < 1:
            raise ValueError("num_bags cannot be less than one!")

        current_ensemble_predictions = pd.Series(0.0, index=oof.index)
        ensemble_indices = []
        used_model_counts = Counter()

        ensemble_scores = []
        ensemble_sizes = []

        if self.sorted_initialization:
            model_scores = {}
            for model_name in model_names:
                predictions = oof_predictions[model_name]
                score = self.metric(
                    oof_targets,
                    predictions,
                    oof_eras,
                    meta_data=meta_data,
                    sample_weight=sample_weights
                )
                logger.info(f"Calculating initial sorted metric - Model: {model_name}: {score}")
                model_scores[model_name] = score

            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

            if self.initial_n is None:
                N = self._determine_initial_n([score for name, score in sorted_models])
            else:
                N = self.initial_n

            N = min(N, self.max_ensemble_size)

            for i in range(N):
                model_name = sorted_models[i][0]
                model_score = sorted_models[i][1]
                current_ensemble_predictions += oof_predictions[model_name]
                ensemble_indices.append(model_name)
                used_model_counts[model_name] += 1
                logger.info(f"Selected Model {i + 1}: {model_name} with score {model_score}")

            current_ensemble_predictions /= N

            initial_score = self.metric(
                oof_targets,
                current_ensemble_predictions,
                oof_eras,
                meta_data=meta_data,
                sample_weight=sample_weights
            )
            ensemble_scores.append(initial_score)
            ensemble_sizes.append(len(ensemble_indices))
            logger.info(f"Initial ensemble score with {len(ensemble_indices)} models ({', '.join(ensemble_indices)}): {initial_score}")
        else:
            # Initialize empty ensemble_scores and ensemble_sizes if no sorted initialization
            ensemble_scores = []
            ensemble_sizes = []

        # Greedy addition of models to the ensemble
        while len(ensemble_indices) < self.num_bags:
            if len(used_model_counts) >= self.max_ensemble_size and not self.use_replacement:
                break  # Reached maximum number of unique models

            best_score = None
            best_model_name = None

            # Determine candidate models
            if self.use_replacement:
                candidate_model_names = model_names
            else:
                candidate_model_names = [name for name in model_names if name not in used_model_counts]

            if not candidate_model_names:
                break  # No more candidates to consider

            # Evaluate each candidate model
            for model_name in candidate_model_names:
                model_predictions = oof_predictions[model_name]
                combined_predictions = (current_ensemble_predictions * len(ensemble_indices) + model_predictions) / (len(ensemble_indices) + 1)

                # Calculate the score of the new ensemble
                score = self.metric(
                    oof_targets,
                    combined_predictions,
                    oof_eras,
                    meta_data=meta_data,
                    sample_weight=sample_weights
                )

                if best_score is None or score > best_score:
                    best_score = score
                    best_model_name = model_name

            if best_model_name is None:
                break  # No improvement found

            # Update ensemble with the best model
            ensemble_indices.append(best_model_name)
            used_model_counts[best_model_name] += 1
            current_ensemble_predictions = (current_ensemble_predictions * (len(ensemble_indices) - 1) + oof_predictions[best_model_name]) / len(ensemble_indices)

            ensemble_scores.append(best_score)
            ensemble_sizes.append(len(ensemble_indices))

            logger.info(f"Iteration {len(ensemble_sizes)}: Added model '{best_model_name}' with score {best_score}. Bag ensemble size: {len(ensemble_indices)}")

            if len(used_model_counts) >= self.max_ensemble_size and not self.use_replacement:
                break  # Reached maximum number of unique models

        # Select the best ensemble size based on the highest score
        if ensemble_scores:
            max_score = max(ensemble_scores)
            best_indices = [i for i, score in enumerate(ensemble_scores) if score == max_score]
            best_index = best_indices[-1]  # Pick the last occurrence
            best_ensemble_size = ensemble_sizes[best_index]
            best_ensemble_indices = ensemble_indices[:best_ensemble_size]
            logger.info(f"Best bagged ensemble size: {best_ensemble_size} with score {ensemble_scores[best_index]}")
        else:
            best_ensemble_indices = []

        # Calculate the final weights
        model_counts = Counter(best_ensemble_indices)
        total_counts = sum(model_counts.values()) or 1
        weights = pd.Series({model_name: count / total_counts for model_name, count in model_counts.items()})
        self.weights_ = weights.reindex(model_names).fillna(0.0)
        self.selected_model_names_ = self.weights_[self.weights_ > 0].index.tolist()

    def _determine_initial_n(self, scores):
        if len(scores) <= 1:
            return 1

        threshold = 0.001  # You may adjust this threshold based on your needs
        N = 1
        for i in range(1, len(scores)):
            improvement = scores[i - 1] - scores[i]
            if improvement < threshold:
                break
            N += 1
        return N

    def predict(self, base_models_predictions: pd.DataFrame) -> pd.Series:
        weighted_predictions = base_models_predictions.multiply(self.weights_, axis=1)
        ensemble_predictions = weighted_predictions.sum(axis=1)
        return ensemble_predictions


class MetaModel(BaseEstimator, RegressorMixin):
    def __init__(self, max_ensemble_size: int = 15, num_bags=50, random_state: int = 42, weight_factor: float = None, metric='corr'):
        self.metric = metric
        self.max_ensemble_size = max_ensemble_size
        self.num_bags = num_bags
        self.random_state = random_state
        self.weight_factor = weight_factor  # Weight factor used in sample weighting
        self.selected_model_names = []  # List to store the names of selected models
        self.ensemble_model = None

    def fit(self, oof, models, ensemble_method=GreedyEnsemble):

        if oof.isnull().values.any():
            raise ValueError("Out of fold predictions contains NaN values.")

        if not all(col in oof.columns for col in ['era', 'target', 'meta_data']):
            raise KeyError("The dataframe is missing 'era', 'target' or 'meta_data' columns.")

        if not isinstance(models, dict):
            raise TypeError("The 'models' variable must be a dictionary.")

        # Scale oof predictions to [0, 1]
        oof_predictions = oof.drop(columns=['era', 'target', 'meta_data'], errors='ignore')
        oof_predictions = oof_predictions.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

        # Reinsert scaled predictions into oof DataFrame
        oof_scaled = oof.copy()
        oof_scaled.update(oof_predictions)

        # If weight_factor is provided, calculate sample weights based on eras
        if self.weight_factor is not None:
            logger.info(f"Generating sample weights - Weight factor: {self.weight_factor}")
            sample_weights = get_sample_weights(oof_scaled.drop(columns=['era', 'target', 'meta_data'], errors='ignore'), wfactor=self.weight_factor, eras=oof_scaled['era'])
        else:
            sample_weights = None

        # Instantiate ensemble
        if isinstance(ensemble_method, type):
            ensemble_method = ensemble_method(max_ensemble_size=self.max_ensemble_size, num_bags=self.num_bags, metric=self.metric, random_state=self.random_state)

        # Fit the ensemble method with scaled predictions, true targets, and sample weights
        logger.info(f"Ensembling out of sample predictions - Metric: {self.metric}")
        ensemble_method.fit(oof_scaled, sample_weights=sample_weights)

        # Get the names of the selected models and their weights
        self.selected_model_names = ensemble_method.selected_model_names_
        self.weights_ = ensemble_method.weights_

        # Load the selected models from disk
        logger.info(f"Generating meta model")
        selected_models = []
        for model_name in self.selected_model_names:
            model_path = models[model_name]['model_path']
            with open(model_path, 'rb') as f:
                model = pickle.load(f)  # Load the model from the file
            selected_models.append((model_name, model))  # Add the model to the list

        # Prepare the final VotingRegressor ensemble model
        weights_list = []
        estimators_list = []
        models = []
        for model_name, model in selected_models:
            weight = self.weights_.loc[model_name]  # Get the weight for the model
            weights_list.append(weight)  # Add the weight to the list
            logger.info(f"Adding model - Model: {model_name}, Weight: {weight}")
            estimators_list.append((model_name, model))  # Add the model and its name to the list
            models.append(model)

        # Create the VotingRegressor with the selected models and their corresponding weights
        self.ensemble_model = VotingRegressor(estimators=estimators_list, weights=weights_list)
        self.ensemble_model.estimators_ = models

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # Use the ensemble model to predict and return the results as a pandas Series
        results = pd.Series(self.ensemble_model.predict(X), index=X.index)
        return results


