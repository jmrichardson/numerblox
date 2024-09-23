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


import numpy as np
import pandas as pd
from collections import Counter
from typing import Callable, List, Union
from sklearn.utils import check_random_state
from numerblox.misc import numerai_corr_weighted

class GreedyEnsemble:
    def __init__(self,
                 max_ensemble_size: int = 10,
                 metric: Callable = numerai_corr_weighted,
                 use_replacement: bool = True,
                 sorted_initialization: bool = False,
                 initial_n: int = None,
                 bagging: bool = False,
                 bag_fraction: float = 0.5,
                 num_bags: int = 20,
                 random_state: Union[int, None] = None):
        """
        GreedyEnsemble class implementing ensemble selection from libraries of models.

        This class implements the ensemble selection method described in the paper:
        "Ensemble Selection from Libraries of Models" by Caruana et al.

        Parameters
        ----------
        max_ensemble_size : int, default=10
            The maximum number of models to include in the ensemble.
        metric : Callable, default=None
            The function to evaluate performance. If None, uses `numerai_corr`.
            Should accept (y_true, y_pred, sample_weight) and return a scalar loss.
            Lower values indicate better performance.
        use_replacement : bool, default=True
            Whether to allow selection with replacement (models can be added multiple times).
        sorted_initialization : bool, default=False
            Whether to initialize the ensemble with N best models before starting greedy selection.
        initial_n : int or None, default=None
            Number of models to use in initialization if sorted_initialization is True.
            If None, N will be determined based on validation performance.
        bagging : bool, default=False
            Whether to use bagged ensemble selection to reduce overfitting.
        bag_fraction : float, default=0.5
            Fraction of models to include in each bag when bagging is True.
        num_bags : int, default=20
            Number of bags to use when bagging is True.
        random_state : int or None, default=None
            Random state for reproducibility.

        Attributes
        ----------
        weights_ : pd.Series
            Weights of the models in the ensemble after fitting.
        selected_model_names_ : list of str
            Names of the models selected in the ensemble.
        """

        self.max_ensemble_size = max_ensemble_size
        if metric is None:
            self.metric = numerai_corr
        else:
            self.metric = metric
        self.use_replacement = use_replacement
        self.sorted_initialization = sorted_initialization
        self.initial_n = initial_n
        self.bagging = bagging
        self.bag_fraction = bag_fraction
        self.num_bags = num_bags
        self.random_state = random_state

        self.weights_ = None
        self.selected_model_names_ = None

    def fit(self,
            base_models_predictions: pd.DataFrame,
            true_targets: pd.Series,
            sample_weights: pd.Series = None):
        """
        Fit the greedy ensemble by selecting models based on their performance.

        Parameters
        ----------
        base_models_predictions : pd.DataFrame
            A DataFrame where each column contains predictions from a base model
            on the validation (hillclimbing) set.
        true_targets : pd.Series
            The true target values for the validation set.
        sample_weights : pd.Series or None, default=None
            Sample weights to be used in evaluating performance.

        """
        rng = check_random_state(self.random_state)

        model_names = base_models_predictions.columns.tolist()
        n_models = len(model_names)

        if self.max_ensemble_size < 1:
            raise ValueError("Ensemble size cannot be less than one!")

        if self.bagging:
            # Perform bagged ensemble selection
            ensemble_weights_list = []
            for _ in range(self.num_bags):
                # Sample models to include in this bag
                n_bag_models = max(1, int(n_models * self.bag_fraction))
                bag_model_names = rng.choice(model_names, size=n_bag_models, replace=False)
                bag_predictions = base_models_predictions[bag_model_names]

                # Perform ensemble selection on this bag
                bag_weights = self._fit_ensemble(bag_predictions, true_targets, sample_weights)

                # Store the weights
                ensemble_weights_list.append(bag_weights)

            # Average the weights across all bags
            weights_df = pd.concat(ensemble_weights_list, axis=1).fillna(0)
            self.weights_ = weights_df.mean(axis=1)
            self.weights_ = self.weights_ / self.weights_.sum()

        else:
            # Perform ensemble selection on all models
            self.weights_ = self._fit_ensemble(base_models_predictions, true_targets, sample_weights)

        # Store selected model names
        self.selected_model_names_ = self.weights_[self.weights_ > 0].index.tolist()

    def _fit_ensemble(self,
                      base_models_predictions: pd.DataFrame,
                      true_targets: pd.Series,
                      sample_weights: pd.Series):
        """
        Perform ensemble selection on given models.

        This method implements the core ensemble selection algorithm:
        - Start with an empty ensemble (or initialize with N best models if sorted_initialization is True).
        - At each iteration, add the model that when combined with the current ensemble
          yields the best performance on the validation set.
        - Repeat for max_ensemble_size iterations or until no improvement can be made.
        - Select the ensemble from the nested ensembles that has the best validation performance.
        - Calculate model weights based on their frequency in the selected ensemble.

        Parameters
        ----------
        base_models_predictions : pd.DataFrame
            Predictions from the base models on the validation set.
        true_targets : pd.Series
            True target values for the validation set.
        sample_weights : pd.Series or None
            Sample weights for performance evaluation.

        Returns
        -------
        weights : pd.Series
            Weights of the models in the ensemble.

        """
        model_names = base_models_predictions.columns.tolist()

        # Initialize variables
        current_ensemble_predictions = pd.Series(0.0, index=base_models_predictions.index)
        ensemble_indices = []
        ensemble_scores = []
        used_model_counts = Counter()

        # If sorted_initialization is True, initialize with N best models
        if self.sorted_initialization:
            # Compute individual model performances
            model_losses = {}
            for model_name in model_names:
                predictions = base_models_predictions[model_name]
                loss = self.metric(true_targets, predictions, sample_weight=sample_weights)
                model_losses[model_name] = loss
            # Sort models by performance (lower loss is better)
            sorted_models = sorted(model_losses.items(), key=lambda x: x[1])
            if self.initial_n is None:
                # Determine N based on performance improvements
                N = self._determine_initial_n([loss for name, loss in sorted_models])
            else:
                N = self.initial_n
            # Add N best models to the ensemble
            for i in range(N):
                model_name = sorted_models[i][0]
                current_ensemble_predictions += base_models_predictions[model_name]
                ensemble_indices.append(model_name)
                used_model_counts[model_name] += 1
            # Normalize current ensemble predictions
            current_ensemble_predictions /= N

        # Perform greedy ensemble selection
        for _ in range(self.max_ensemble_size):
            best_score = None
            best_model_name = None

            # Loop over candidate models
            if self.use_replacement:
                candidate_model_names = model_names
            else:
                candidate_model_names = [name for name in model_names if name not in used_model_counts]

            if not candidate_model_names:
                # No more models to select
                break

            for model_name in candidate_model_names:
                model_predictions = base_models_predictions[model_name]
                # Combine current ensemble predictions with this model's predictions
                combined_predictions = (current_ensemble_predictions * len(ensemble_indices) + model_predictions) / (len(ensemble_indices) + 1)

                # Calculate loss for this combined ensemble
                loss = self.metric(true_targets, combined_predictions, sample_weight=sample_weights)

                if best_score is None or loss < best_score:
                    best_score = loss
                    best_model_name = model_name

            if best_model_name is None:
                # No improvement can be made
                break

            # Update the ensemble
            ensemble_indices.append(best_model_name)
            used_model_counts[best_model_name] += 1
            current_ensemble_predictions = (current_ensemble_predictions * (len(ensemble_indices) - 1) + base_models_predictions[best_model_name]) / len(ensemble_indices)
            ensemble_scores.append(best_score)

        # Select the ensemble with the best performance on validation set
        if ensemble_scores:
            best_index = np.argmin(ensemble_scores)
            best_ensemble_indices = ensemble_indices[:best_index+1]
        else:
            best_ensemble_indices = []

        # Calculate weights
        model_counts = Counter(best_ensemble_indices)
        total_counts = sum(model_counts.values())
        weights = pd.Series({model_name: count / total_counts for model_name, count in model_counts.items()})
        # Ensure all models are represented in weights (with zero weight if not used)
        weights = weights.reindex(model_names).fillna(0.0)

        return weights

    def _determine_initial_n(self, losses):
        """
        Determine the number N of models to use in sorted initialization.

        This method selects N based on the performance improvements between
        successive models in the sorted list. When the improvement falls below
        a threshold, it stops adding models.

        Parameters
        ----------
        losses : list of float
            Losses of the models sorted in ascending order.

        Returns
        -------
        N : int
            Number of models to use in initialization.

        """
        threshold = 0.001  # Performance improvement threshold
        N = 1
        for i in range(1, len(losses)):
            improvement = losses[i-1] - losses[i]
            if improvement < threshold:
                break
            N += 1
        return N

    def predict(self, base_models_predictions: pd.DataFrame) -> pd.Series:
        """
        Predict using the ensemble.

        Parameters
        ----------
        base_models_predictions : pd.DataFrame
            A DataFrame where each column contains predictions from a base model.

        Returns
        -------
        predictions : pd.Series
            The ensemble predictions.

        """
        # Weight the predictions
        weighted_predictions = base_models_predictions.multiply(self.weights_, axis=1)
        ensemble_predictions = weighted_predictions.sum(axis=1)
        return ensemble_predictions

    def predict_proba(self, base_models_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Predict class probabilities using the ensemble.

        Parameters
        ----------
        base_models_predictions : pd.DataFrame
            A DataFrame where each column contains predicted probabilities from a base model.

        Returns
        -------
        predictions : pd.DataFrame
            The ensemble predicted probabilities.

        """
        # For classification tasks where predictions are probabilities
        weighted_predictions = base_models_predictions.multiply(self.weights_, axis=1)
        ensemble_predictions = weighted_predictions.sum(axis=1)
        return ensemble_predictions
