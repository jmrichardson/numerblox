import scipy
import warnings
import numpy as np
import pandas as pd
from typing import Union, List
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter


class NumeraiEnsemble(BaseEstimator, TransformerMixin):
    """
    Ensembler that standardizes predictions by era and averages them.
    :param weights: Sequence of weights (float or int), optional, default: None.
    If None, then uniform weights are used.
    :param n_jobs: The number of jobs to run in parallel for fit.
    Will revert to 1 CPU core if not defined.
    -1 means using all processors.
    :param donate_weighted: Whether to use Donate et al.'s weighted average formula.
    Often used when ensembling predictions from multiple folds over time.
    Paper Link: https://doi.org/10.1016/j.neucom.2012.02.053
    Example donate weighting for 5 folds: [0.0625, 0.0625, 0.125, 0.25, 0.5]
    """
    def __init__(self, weights=None, donate_weighted=False):
        sklearn.set_config(enable_metadata_routing=True)
        self.set_transform_request(era_series=True)
        self.set_predict_request(era_series=True)
        super().__init__()
        self.weights = weights
        if self.weights and sum(self.weights) != 1:
            warnings.warn(f"Warning: Weights do not sum to 1. Got {sum(self.weights)}.")
        self.donate_weighted = donate_weighted

    def fit(self, X: Union[np.array, pd.DataFrame], y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[np.array, pd.DataFrame], era_series: pd.Series) -> np.array:
        """
        Standardize by era and ensemble.
        :param X: Input data where each column contains predictions from an estimator.
        :param era_series: Era labels (strings) for each row in X.
        :return: Ensembled predictions.
        """
        assert not era_series is None, "Era series must be provided for NumeraiEnsemble."
        assert len(X) == len(era_series), f"input X and era_series must have the same length. Got {len(X)} != {len(era_series)}."

        if len(X.shape) == 1:
            raise ValueError("NumeraiEnsemble requires at least 2 prediction columns. Got 1.")

        n_models = X.shape[1]
        if n_models <= 1:
            raise ValueError(f"NumeraiEnsemble requires at least 2 predictions columns. Got {len(n_models)}.")

        # Override weights if donate_weighted is True
        if self.donate_weighted:
            weights = self._get_donate_weights(n=n_models)
        else:
            weights = self.weights

        if isinstance(X, pd.DataFrame):
            X = X.values
        # Standardize predictions by era
        standardized_pred_list = []
        for i in range(n_models):
            # Skip standardization if all predictions are the same
            pred = X[:, i]
            if np.isnan(pred).any():
                warnings.warn(f"Warning: Some predictions in column '{i}' contain NaNs. Consider checking your estimators. Ensembled predictions will also be a NaN.")
            if np.all(pred == pred[0]):
                warnings.warn(f"Warning: Predictions in column '{i}' are all constant. Consider checking your estimators. Skipping these estimator predictions in ensembling.")
            else:
                standardized_pred = self._standardize_by_era(pred, era_series)
                standardized_pred_list.append(standardized_pred)
        standardized_pred_arr = np.asarray(standardized_pred_list).T

        if not standardized_pred_list:
            raise ValueError("Predictions for all columns are constant. No valid predictions to ensemble.")

        # Average out predictions
        ensembled_predictions = np.average(standardized_pred_arr, axis=1, weights=weights)
        return ensembled_predictions.reshape(-1, 1)

    def fit_transform(self, X: Union[np.array, pd.DataFrame], y=None, era_series: pd.Series = None) -> np.array:
        self.fit(X, y)
        return self.transform(X, era_series)

    def predict(self, X: Union[np.array, pd.DataFrame], era_series: pd.Series) -> np.array:
        """
        For if a NumeraiEnsemble happens to be the last step in the pipeline. Has same behavior as transform.
        """
        return self.transform(X, era_series=era_series)

    def _standardize(self, X: np.array) -> np.array:
        """
        Standardize single era.
        :param X: Predictions for a single era.
        :return: Standardized predictions.
        """
        percentile_X = (scipy.stats.rankdata(X, method="ordinal") - 0.5) / len(X)
        return percentile_X

    def _standardize_by_era(self, X: np.array, era_series: Union[np.array, pd.Series, pd.DataFrame]) -> np.array:
        """
        Standardize predictions of a single estimator by era.
        :param X: All predictions of a single estimator.
        :param era_series: Era labels (strings) for each row in X.
        :return: Standardized predictions.
        """
        if isinstance(era_series, (pd.Series, pd.DataFrame)):
            era_series = era_series.to_numpy().flatten()
        df = pd.DataFrame({'prediction': X, 'era': era_series})
        df['standardized_prediction'] = df.groupby('era')['prediction'].transform(self._standardize)
        return df['standardized_prediction'].values.flatten()

    def _get_donate_weights(self, n: int) -> list:
        """
        Exponential weights as per Donate et al.'s formula.
        Example donate weighting for 3 folds: [0.25, 0.25, 0.5]
        Example donate weighting for 5 folds: [0.0625, 0.0625, 0.125, 0.25, 0.5]

        :param n: Number of estimators.
        :return: List of weights.
        """
        weights = []
        for j in range(1, n + 1):
            j = 2 if j == 1 else j
            weights.append(1 / (2 ** (n + 1 - j)))
        return weights

    def get_feature_names_out(self, input_features = None) -> List[str]:
        return ["numerai_ensemble_predictions"] if not input_features else input_features


class PredictionReducer(BaseEstimator, TransformerMixin):
    """
    Reduce multiclassification and proba preds to 1 column per model.
    If predictions were generated with a regressor or regular predict you don't need this step.
    :param n_models: Number of resulting columns.
    This indicates how many models were trained to generate the prediction array.
    :param n_classes: Number of classes for each prediction.
    If predictions were generated with predict_proba and binary classification -> n_classes = 2.
    """
    def __init__(self, n_models: int, n_classes: int):
        super().__init__()
        if n_models < 1:
            raise ValueError(f"n_models must be >= 1. Got '{n_models}'.")
        self.n_models = n_models
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2. If n_classes = 1 you don't need PredictionReducer. Got '{n_classes}'.")
        self.n_classes = n_classes
        self.dot_array = [i for i in range(self.n_classes)]

    def fit(self, X: np.array, y=None):
        return self

    def transform(self, X: np.array):
        """
        :param X: Input predictions.
        :return: Reduced predictions of shape (X.shape[0], self.n_models).
        """
        reduced = []
        expected_n_cols = self.n_models * self.n_classes
        if len(X.shape) != 2:
            raise ValueError(f"Expected X to be a 2D array. Got '{len(X.shape)}' dimension(s).")
        if X.shape[1] != expected_n_cols:
            raise ValueError(f"Input X must have {expected_n_cols} columns. Got {X.shape[1]} columns while n_models={self.n_models} * n_classes={self.n_classes} = {expected_n_cols}. ")
        for i in range(self.n_models):
            # Extracting the predictions of the i-th model
            model_preds = X[:, i*self.n_classes:(i+1)*self.n_classes]
            r = model_preds @ self.dot_array
            reduced.append(r)
        reduced_arr = np.column_stack(reduced)
        return reduced_arr

    def predict(self, X: np.array):
        """
        For if PredictionReducer happens to be the last step in the pipeline. Has same behavior as transform.
        :param X: Input predictions.
        :return: Reduced predictions of shape (X.shape[0], self.n_models).
        """
        return self.transform(X)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return [f"reduced_prediction_{i}" for i in range(self.n_models)] if not input_features else input_features


class GreedyEnsemble:
    def __init__(self, ensemble_size: int, random_state: int = None):
        self.ensemble_size = ensemble_size
        self.random_state = random_state
        self.indices_ = []
        self.trajectory_ = []
        self.weights_ = None

    def fit(self, base_models_predictions: pd.DataFrame, true_targets: pd.Series, sample_weights: pd.Series) -> None:
        """Fit the greedy ensemble by selecting models based on their performance."""
        n_samples, n_models = base_models_predictions.shape

        if self.ensemble_size < 1:
            raise ValueError("Ensemble size cannot be less than one!")

        # Initialize ensemble predictions
        current_ensemble_predictions = pd.Series(0, index=base_models_predictions.index)

        for _ in range(self.ensemble_size):
            best_loss = float("inf")
            best_model_name = None

            # Iterate through all models to find the one that minimizes the loss when added to the ensemble
            for model_name in base_models_predictions.columns:
                model_predictions = base_models_predictions[model_name]
                # Combine current ensemble predictions with this model's predictions
                combined_predictions = (current_ensemble_predictions * len(self.indices_) + model_predictions) / (len(self.indices_) + 1)

                # Calculate loss for this combined model, weighted by sample weights
                loss = self._calculate_loss(combined_predictions, true_targets, sample_weights)

                # If this model improves the performance, select it as the best model
                if loss < best_loss:
                    best_loss = loss
                    best_model_name = model_name

            # Add the best model's predictions to the ensemble
            self.indices_.append(best_model_name)
            current_ensemble_predictions = (current_ensemble_predictions * (len(self.indices_) - 1) + base_models_predictions[best_model_name]) / len(self.indices_)
            self.trajectory_.append(best_loss)

        # Calculate final model weights based on the frequency of their selection
        self._calculate_weights(base_models_predictions)

    def _calculate_loss(self, predictions: pd.Series, true_targets: pd.Series, sample_weights: pd.Series) -> float:
        """Calculate the weighted loss of predictions with respect to the true targets."""
        return (((predictions - true_targets) ** 2) * sample_weights).mean()  # Weighted MSE

    def _calculate_weights(self, base_models_predictions: pd.DataFrame) -> None:
        """Calculate the weights of the models based on their frequency of selection in the ensemble."""
        ensemble_members = Counter(self.indices_).most_common()
        total_counts = sum(count for model_name, count in ensemble_members)
        self.weights_ = pd.Series(0, index=base_models_predictions.columns)
        for model_name, count in ensemble_members:
            weight = float(count) / total_counts
            self.weights_.loc[model_name] = weight
