import numpy as np
import pandas as pd
from collections import Counter
from typing import Callable, Union
from sklearn.utils import check_random_state
from scipy import stats
from . import logger


def numerai_corr_score(
    targets: pd.Series,
    predictions: pd.Series,
    eras: pd.Series,
    meta_data=None,
    sample_weight: pd.Series = None
) -> float:
    # Align eras and sample_weight with the predictions' index
    eras = eras.reindex(predictions.index)
    if sample_weight is not None:
        sample_weight = sample_weight.reindex(predictions.index)

    # Convert to numpy arrays for faster computations
    preds_filled = predictions.fillna(0.5).values
    targets_values = targets.values

    # Rank and gaussianize predictions using scipy's rankdata
    ranked_preds = stats.rankdata(preds_filled, method='average') / len(preds_filled)
    gauss_ranked_preds = stats.norm.ppf(ranked_preds)

    # Center target to zero mean
    target_mean = targets_values.mean()
    centered_target = targets_values - target_mean

    # Accentuate tails of predictions and targets
    preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
    target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5

    # Apply sample weights if provided
    if sample_weight is not None:
        sample_weight_values = sample_weight.values
        weighted_preds = preds_p15 * sample_weight_values
        weighted_target = target_p15 * sample_weight_values
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

    if valid_eras is not None:
        df = pd.DataFrame({
            'weighted_preds': weighted_preds,
            'weighted_target': weighted_target,
            'eras': valid_eras
        })

        # Filter out eras with less than 2 samples
        era_counts = df['eras'].value_counts()
        valid_eras_list = era_counts[era_counts >= 2].index
        df = df[df['eras'].isin(valid_eras_list)]

        if not df.empty:
            # Compute per-era correlations using groupby and vectorized operations
            correlations = df.groupby('eras').apply(
                lambda group: np.corrcoef(group['weighted_preds'], group['weighted_target'])[0, 1]
            )
            # Compute mean correlation
            if not correlations.empty:
                corr_result = correlations.mean()

    elif len(weighted_preds) >= 2:
        corr_result = np.corrcoef(weighted_preds, weighted_target)[0, 1]

    # Return the final correlation result
    return corr_result


def mmc_score(
    targets: pd.Series,
    predictions: pd.Series,
    eras: pd.Series,
    meta_data: pd.Series,
    sample_weight: pd.Series = None,
) -> float:
    # Align sample_weight with targets
    if sample_weight is not None:
        sample_weight = sample_weight.reindex(targets.index)

    # Convert to numpy arrays for faster computations
    targets_arr = targets.values
    predictions_arr = predictions.values
    meta_data_arr = meta_data.values
    eras_arr = eras.values

    # Step 2: Rank and gaussianize predictions and meta_data using scipy's rankdata
    predictions_ranked = (stats.rankdata(predictions_arr, method='average') - 0.5) / len(predictions_arr)
    meta_data_ranked = (stats.rankdata(meta_data_arr, method='average') - 0.5) / len(meta_data_arr)

    predictions_ranked = np.clip(predictions_ranked, 1e-6, 1 - 1e-6)
    meta_data_ranked = np.clip(meta_data_ranked, 1e-6, 1 - 1e-6)

    predictions_gaussianized = stats.norm.ppf(predictions_ranked)
    meta_data_gaussianized = stats.norm.ppf(meta_data_ranked)

    # Step 3: Orthogonalize predictions with respect to meta_data
    m_dot_m = np.dot(meta_data_gaussianized, meta_data_gaussianized)
    projection = np.dot(predictions_gaussianized, meta_data_gaussianized) / m_dot_m
    neutral_preds = predictions_gaussianized - meta_data_gaussianized * projection

    # Step 4: Adjust targets
    if np.all((targets_arr >= 0) & (targets_arr <= 1)):
        targets_arr = targets_arr * 4
    targets_arr = targets_arr - targets_arr.mean()

    # Step 5: Prepare sample_weight
    if sample_weight is not None:
        sample_weight_arr = sample_weight.values
    else:
        sample_weight_arr = np.ones_like(targets_arr)

    # Step 6: Create DataFrame for grouping
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
    df['numerator'] = df['sample_weight'] * df['targets'] * df['neutral_preds']
    grouped = df.groupby('eras')
    numerators = grouped['numerator'].sum()
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
    numerai_corr = numerai_corr_score(targets, predictions, eras, meta_data, sample_weight)
    mmc = mmc_score(targets, predictions, eras, meta_data, sample_weight)
    score = numerai_corr * 0.5 + mmc * 2
    return score


class GreedyEnsemble:
    def __init__(
        self,
        max_ensemble_size: int = 10,
        metric: Union[str, Callable] = "corr",
        use_replacement: bool = True,
        sorted_initialization: bool = True,
        initial_n: int = None,
        bag_fraction: float = 0.5,
        num_bags: int = 20,
        random_state: Union[int, None] = None
    ):

        self.max_ensemble_size = max_ensemble_size
        self.use_replacement = use_replacement
        self.sorted_initialization = sorted_initialization
        self.initial_n = initial_n
        self.bag_fraction = bag_fraction
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
            raise ValueError("Ensemble size cannot be less than one!")

        # Initialize ensemble predictions and other variables
        current_ensemble_predictions = pd.Series(0.0, index=oof.index)
        ensemble_indices = []
        used_model_counts = Counter()

        # Initialize ensemble_scores list
        ensemble_scores = []

        # Sorted initialization
        if self.sorted_initialization:
            model_scores = {}
            for model_name in model_names:
                predictions = oof_predictions[model_name]
                logger.info(f"Calculating initial sorted metric - Model: {model_name}")
                score = self.metric(
                    oof_targets,
                    predictions,
                    oof_eras,
                    meta_data=meta_data,
                    sample_weight=sample_weights
                )
                model_scores[model_name] = score

            # Sort models by score in descending order
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

            # Determine the number of models to initialize
            if self.initial_n is None:
                N = self._determine_initial_n([score for name, score in sorted_models])
            else:
                N = self.initial_n

            # Add the top N models to the ensemble
            for i in range(N):
                model_name = sorted_models[i][0]
                current_ensemble_predictions += oof_predictions[model_name]
                ensemble_indices.append(model_name)
                used_model_counts[model_name] += 1
            current_ensemble_predictions /= N

            # Compute initial ensemble score
            initial_score = self.metric(
                oof_targets,
                current_ensemble_predictions,
                oof_eras,
                meta_data=meta_data,
                sample_weight=sample_weights
            )
            ensemble_scores.append(initial_score)
            logger.info(f"Initial ensemble score with {N} models: {initial_score}")
        else:
            # Initialize empty ensemble_scores if no sorted initialization
            ensemble_scores = []

        # Greedy addition of models to the ensemble
        for iteration in range(self.max_ensemble_size - len(ensemble_indices)):
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
                combined_predictions = (
                    current_ensemble_predictions * len(ensemble_indices) + model_predictions
                ) / (len(ensemble_indices) + 1)

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
            current_ensemble_predictions = (
                current_ensemble_predictions * (len(ensemble_indices) - 1) +
                oof_predictions[best_model_name]
            ) / len(ensemble_indices)

            ensemble_scores.append(best_score)

            logger.info(f"Iteration {iteration + 1}: Added model '{best_model_name}' with score {best_score}. Ensemble size: {len(ensemble_indices)}")

        # Select the best ensemble size based on the highest score
        if ensemble_scores:
            best_index = np.argmax(ensemble_scores)
            best_ensemble_indices = ensemble_indices[:best_index + 1]
            logger.info(f"Best ensemble size: {best_index + 1} with score {ensemble_scores[best_index]}")
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
