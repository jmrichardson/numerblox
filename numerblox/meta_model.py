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
from typing import Optional, Union, Dict, Any

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


from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats


def mmc_score(
    targets: pd.Series,
    predictions: pd.Series,
    eras: pd.Series,
    meta_data: pd.Series,
    sample_weight: Optional[pd.Series] = None,
) -> float:
    """
    Calculate the Mean Model Correlation (MMC) score, adjusted for orthogonality with meta-data.

    The MMC score is computed by orthogonalizing the model predictions with respect to the meta-data,
    and then calculating the weighted correlation between the targets and the orthogonalized predictions
    within each era. The final score is the mean of these per-era correlations.

    Args:
        targets (pd.Series): Target values, expected to be in the range [0, 1].
        predictions (pd.Series): Model predictions.
        eras (pd.Series): Era identifiers for each sample.
        meta_data (pd.Series): Meta-data used for orthogonalization.
        sample_weight (Optional[pd.Series], optional): Sample weights for each observation.
            If None, each sample is assigned a weight of 1. Defaults to None.

    Returns:
        float: The mean MMC score across eras. Returns NaN if there are insufficient eras for calculation.

    Raises:
        ValueError: If input series lengths do not match or if meta-data variance is zero.
    """

    # Input validation
    if not all(len(arr) == len(targets) for arr in [predictions, eras, meta_data]):
        raise ValueError("All input Series must be of the same length.")

    if sample_weight is not None and len(sample_weight) != len(targets):
        raise ValueError("Sample weights must be of the same length as targets.")

    # Align sample weights with targets
    if sample_weight is not None:
        sample_weight = sample_weight.reindex(targets.index).fillna(1.0)
    else:
        sample_weight = pd.Series(1.0, index=targets.index)

    # Convert inputs to numpy arrays for optimized computations
    targets_arr = targets.to_numpy(copy=True).astype(np.float64)
    predictions_arr = predictions.to_numpy().astype(np.float64)
    meta_data_arr = meta_data.to_numpy().astype(np.float64)
    eras_arr = eras.to_numpy()
    sample_weight_arr = sample_weight.to_numpy().astype(np.float64)

    # Rank and gaussianize predictions and meta_data
    predictions_ranked = (stats.rankdata(predictions_arr, method='average') - 0.5) / len(predictions_arr)
    meta_data_ranked = (stats.rankdata(meta_data_arr, method='average') - 0.5) / len(meta_data_arr)

    # Clip ranked data to avoid extremes and apply inverse normal transformation
    predictions_clipped = np.clip(predictions_ranked, 1e-6, 1 - 1e-6)
    meta_data_clipped = np.clip(meta_data_ranked, 1e-6, 1 - 1e-6)

    predictions_gaussianized = stats.norm.ppf(predictions_clipped)
    meta_data_gaussianized = stats.norm.ppf(meta_data_clipped)

    # Orthogonalize predictions with respect to meta_data
    m_dot_m = np.dot(meta_data_gaussianized, meta_data_gaussianized)
    if m_dot_m == 0:
        raise ValueError("Meta-data has zero variance; cannot orthogonalize.")

    projection = np.dot(predictions_gaussianized, meta_data_gaussianized) / m_dot_m
    neutral_preds = predictions_gaussianized - meta_data_gaussianized * projection

    # Adjust targets
    if np.all((targets_arr >= 0) & (targets_arr <= 1)):
        targets_arr *= 4
    targets_arr -= targets_arr.mean()

    # Create DataFrame for grouping
    df = pd.DataFrame({
        'targets': targets_arr,
        'neutral_preds': neutral_preds,
        'eras': eras_arr,
        'sample_weight': sample_weight_arr
    })

    # Filter out eras with less than 2 samples
    era_counts = df['eras'].value_counts()
    valid_eras = era_counts[era_counts >= 2].index
    df = df[df['eras'].isin(valid_eras)]

    if df.empty:
        return np.nan

    # Compute numerator and denominator per era
    grouped = df.groupby('eras')
    numerators = grouped.apply(
        lambda x: np.sum(x['sample_weight'] * x['targets'] * x['neutral_preds'])
    )
    denominators = grouped['sample_weight'].sum()

    # Compute per-era MMC scores
    per_era_mmc = numerators / denominators

    # Compute the mean MMC score
    mmc_score = per_era_mmc.mean()

    return mmc_score



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
    """
    A greedy ensemble model that selects base models for optimal weighted prediction.

    Attributes:
        weights_ (pd.Series): Final weights of the models in the ensemble.
        selected_model_names_ (list): Names of selected models in the ensemble.
    """

    def __init__(
            self,
            max_ensemble_size: int = 10,
            metric: Union[str, Callable] = "corr",
            use_replacement: bool = True,
            sorted_initialization: bool = True,
            initial_n: Optional[int] = None,
            num_bags: int = 50,
            random_state: Optional[int] = None
    ):
        self.max_ensemble_size = max_ensemble_size
        self.use_replacement = use_replacement
        self.sorted_initialization = sorted_initialization
        self.initial_n = initial_n
        self.num_bags = num_bags
        self.random_state = random_state

        self.weights_ = None
        self.selected_model_names_ = None

        # Configure metric
        if isinstance(metric, str):
            if metric == "corr":
                self.metric = numerai_corr_score
            elif metric == "mmc":
                self.metric = mmc_score
            elif metric == "payout":
                self.metric = numerai_payout_score
            else:
                raise ValueError("Unsupported metric. Choose 'corr', 'mmc', or 'payout'.")
        elif callable(metric):
            self.metric = metric
        else:
            raise TypeError("Metric must be a string or callable.")

    def fit(self, oof: pd.DataFrame, sample_weights: Optional[pd.Series] = None) -> None:
        """
        Fits the ensemble using Out-of-Fold (OOF) predictions.

        Args:
            oof (pd.DataFrame): DataFrame containing OOF predictions, with columns 'target' and 'era'.
            sample_weights (Optional[pd.Series]): Weights for samples in scoring.
        """
        rng = check_random_state(self.random_state)
        oof_targets = oof['target']
        oof_eras = oof['era']
        meta_data = oof.get('meta_data', None)
        oof_predictions = oof.drop(columns=['era', 'target', 'meta_data'], errors='ignore')
        model_names = oof_predictions.columns.tolist()

        if self.max_ensemble_size < 1 or self.num_bags < 1:
            raise ValueError("max_ensemble_size and num_bags must each be at least one.")

        # Initialize predictions and variables to track the ensemble process
        current_ensemble_predictions = pd.Series(0.0, index=oof.index)
        ensemble_indices, used_model_counts = [], Counter()
        ensemble_scores, ensemble_sizes = [], []

        if self.sorted_initialization:
            # Compute initial model scores for sorted initialization
            model_scores = {name: self.metric(
                oof_targets, oof_predictions[name], oof_eras, meta_data=meta_data, sample_weight=sample_weights
            ) for name in model_names}

            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            initial_size = self._determine_initial_n([score for _, score in sorted_models]) if self.initial_n is None else self.initial_n
            initial_size = min(initial_size, self.max_ensemble_size)

            for i in range(initial_size):
                model_name, model_score = sorted_models[i]
                current_ensemble_predictions += oof_predictions[model_name]
                ensemble_indices.append(model_name)
                used_model_counts[model_name] += 1
                logger.info(f"Selected Model {i + 1}: {model_name} with score {model_score}")

            current_ensemble_predictions /= initial_size
            initial_score = self.metric(oof_targets, current_ensemble_predictions, oof_eras, meta_data=meta_data, sample_weight=sample_weights)
            ensemble_scores.append(initial_score)
            ensemble_sizes.append(len(ensemble_indices))
            logger.info(f"Initial ensemble score with {len(ensemble_indices)} models: {initial_score}")

        # Greedy addition to the ensemble
        while len(ensemble_indices) < self.num_bags:
            if len(used_model_counts) >= self.max_ensemble_size and not self.use_replacement:
                break  # Stop if max unique models reached

            best_score, best_model_name = None, None

            candidate_model_names = model_names if self.use_replacement else [name for name in model_names if name not in used_model_counts]
            if not candidate_model_names:
                break  # Stop if no candidates left

            # Evaluate each candidate model's score if added to the ensemble
            for model_name in candidate_model_names:
                combined_predictions = (current_ensemble_predictions * len(ensemble_indices) + oof_predictions[model_name]) / (len(ensemble_indices) + 1)
                score = self.metric(oof_targets, combined_predictions, oof_eras, meta_data=meta_data, sample_weight=sample_weights)

                if best_score is None or score > best_score:
                    best_score, best_model_name = score, model_name

            if not best_model_name:
                break  # Stop if no improvement found

            # Add the best model to the ensemble
            ensemble_indices.append(best_model_name)
            used_model_counts[best_model_name] += 1
            current_ensemble_predictions = (current_ensemble_predictions * (len(ensemble_indices) - 1) + oof_predictions[best_model_name]) / len(ensemble_indices)

            ensemble_scores.append(best_score)
            ensemble_sizes.append(len(ensemble_indices))
            logger.info(f"Added model '{best_model_name}' with score {best_score}. Ensemble size: {len(ensemble_indices)}")

        # Final selection of the best ensemble size based on scores
        max_score = max(ensemble_scores, default=None)
        if max_score is not None:
            best_index = [i for i, score in enumerate(ensemble_scores) if score == max_score][-1]
            best_ensemble_size = ensemble_sizes[best_index]
            best_ensemble_indices = ensemble_indices[:best_ensemble_size]
            logger.info(f"Best ensemble size: {best_ensemble_size} with score {max_score}")
        else:
            best_ensemble_indices = []

        # Set final model weights
        model_counts = Counter(best_ensemble_indices)
        total_counts = sum(model_counts.values()) or 1
        self.weights_ = pd.Series({model_name: count / total_counts for model_name, count in model_counts.items()}).reindex(model_names).fillna(0.0)
        self.selected_model_names_ = self.weights_[self.weights_ > 0].index.tolist()

    def _determine_initial_n(self, scores: list) -> int:
        """
        Determines initial number of models to include based on sorted scores.

        Args:
            scores (list): List of model scores in descending order.

        Returns:
            int: Number of initial models to include.
        """
        threshold = 0.001  # Threshold for score improvement
        return next((i for i in range(1, len(scores)) if scores[i - 1] - scores[i] < threshold), len(scores))

    def predict(self, base_models_predictions: pd.DataFrame) -> pd.Series:
        """
        Predicts ensemble output based on weighted base models.

        Args:
            base_models_predictions (pd.DataFrame): DataFrame of model predictions.

        Returns:
            pd.Series: Ensemble predictions as weighted sum.
        """
        weighted_predictions = base_models_predictions.multiply(self.weights_, axis=1)
        return weighted_predictions.sum(axis=1)


class MetaModel(BaseEstimator, RegressorMixin):
    """
    MetaModel class for ensemble learning using out-of-fold (OOF) predictions.
    This model selects the best subset of models based on a specified metric and
    combines them into a weighted ensemble.

    Attributes:
        max_ensemble_size (int): Maximum number of models to include in the ensemble.
        num_bags (int): Number of bagging iterations.
        random_state (int): Random state for reproducibility.
        weight_factor (Optional[float]): Factor for sample weighting.
        metric (str): Performance metric for ensemble optimization.
        selected_model_names (list): List of selected models for the ensemble.
        ensemble_model (VotingRegressor): The final ensemble model.
    """

    def __init__(self, max_ensemble_size: int = 15, num_bags: int = 50, random_state: int = 42,
                 weight_factor: Optional[float] = None, metric: str = 'corr'):
        self.max_ensemble_size = max_ensemble_size
        self.num_bags = num_bags
        self.random_state = random_state
        self.weight_factor = weight_factor
        self.metric = metric
        self.selected_model_names = []
        self.ensemble_model = None
        self.weights_ = None

    def fit(self, oof: pd.DataFrame, models: Dict[str, Dict[str, Any]], ensemble_method: Callable[..., Any] = GreedyEnsemble) -> 'MetaModel':
        """
        Fit the meta-model by selecting and ensembling the best models based on OOF predictions.

        Args:
            oof (pd.DataFrame): Out-of-fold predictions with columns ['era', 'target', 'meta_data'].
            models (dict): Dictionary of models with model names as keys and model metadata.
            ensemble_method: Class implementing the ensemble selection algorithm.

        Returns:
            self: Fitted meta-model.
        """
        # Validate input data
        if oof.isnull().values.any():
            raise ValueError("Out-of-fold predictions contain NaN values.")
        required_cols = {'era', 'target', 'meta_data'}
        if not required_cols.issubset(oof.columns):
            raise KeyError(f"Dataframe is missing columns: {required_cols - set(oof.columns)}")
        if not isinstance(models, dict):
            raise TypeError("Expected 'models' to be a dictionary.")

        # Scale predictions to [0, 1]
        oof_predictions = oof.drop(columns=required_cols, errors='ignore')
        oof_scaled = oof.copy()
        oof_scaled.update(oof_predictions.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0))

        # Compute sample weights if weight_factor is set
        sample_weights = None
        if self.weight_factor is not None:
            logger.info(f"Generating sample weights with weight factor: {self.weight_factor}")
            sample_weights = get_sample_weights(
                oof_scaled.drop(columns=required_cols, errors='ignore'),
                wfactor=self.weight_factor,
                eras=oof_scaled['era']
            )

        # Initialize and fit ensemble method
        if isinstance(ensemble_method, type):
            ensemble_method = ensemble_method(
                max_ensemble_size=self.max_ensemble_size,
                num_bags=self.num_bags,
                metric=self.metric,
                random_state=self.random_state
            )

        logger.info(f"Ensembling out-of-sample predictions using metric: {self.metric}")
        ensemble_method.fit(oof_scaled, sample_weights=sample_weights)

        # Store selected model names and weights
        self.selected_model_names = ensemble_method.selected_model_names_
        self.weights_ = ensemble_method.weights_

        # Load models and prepare ensemble
        logger.info("Building ensemble model with selected models.")
        estimators_list = []
        weights_list = []

        for model_name in self.selected_model_names:
            model_info = models.get(model_name)
            if not model_info:
                raise KeyError(f"Model '{model_name}' not found in provided models dictionary.")

            model_path = model_info.get('model_path')
            if not model_path:
                raise ValueError(f"Model '{model_name}' has no associated 'model_path'.")

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            weight = self.weights_.loc[model_name]
            logger.info(f"Adding model '{model_name}' with weight {weight:.4f}")
            estimators_list.append((model_name, model))
            weights_list.append(weight)

        # Create and store the VotingRegressor ensemble model
        self.ensemble_model = VotingRegressor(estimators=estimators_list, weights=weights_list)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict using the ensemble model.

        Args:
            X (pd.DataFrame): Input features for prediction.

        Returns:
            pd.Series: Predictions from the ensemble model.
        """
        if self.ensemble_model is None:
            raise ValueError("The ensemble model has not been fitted yet.")

        predictions = self.ensemble_model.predict(X)
        return pd.Series(predictions, index=X.index)

