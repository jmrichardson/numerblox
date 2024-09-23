import scipy
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple
from typing import Union, List
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
        

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


from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class Meta(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            task_type: int,
            ensemble_size: int = 50,
            random_state: int = None,
    ):
        """Meta Ensemble

        Parameters:
        -----------
        task_type : int
            The task type, typically 1 for classification, 2 for regression.

        ensemble_size : int
            Number of models to include in the ensemble.

        random_state : int
            Seed for reproducibility.
        """
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.random_state = random_state
        self.random_generator = np.random.RandomState(random_state) if random_state else np.random.RandomState()
        self.indices_ = []
        self.weights_ = None

    def fit(
            self,
            base_models_predictions: np.ndarray,
            true_targets: np.ndarray,
    ) -> 'Meta':
        """Fit the ensemble by selecting base models based on their performance.

        Parameters:
        -----------
        base_models_predictions : np.ndarray
            Predictions of the base models, shape: (n_samples, n_models)

        true_targets : np.ndarray
            True labels/targets for the task, shape: (n_samples,)

        Returns:
        --------
        self
        """
        # Check that the ensemble size is valid
        if self.ensemble_size < 1:
            raise ValueError("Ensemble size cannot be less than one!")

        n_samples, n_models = base_models_predictions.shape

        # Initialize ensemble predictions and other attributes
        self.indices_ = []  # Store selected model indices
        self.trajectory_ = []  # Losses after each iteration
        self.weights_ = np.zeros(n_models)  # Model weights in the final ensemble

        current_ensemble_predictions = np.zeros(n_samples)  # Initial empty ensemble predictions

        for _ in range(self.ensemble_size):
            best_loss = float("inf")
            best_model_idx = None

            # Iterate through all models to find the one that minimizes the loss when added to the ensemble
            for idx in range(n_models):
                model_predictions = base_models_predictions[:, idx]
                # Combine current ensemble predictions with this model's predictions
                combined_predictions = (current_ensemble_predictions + model_predictions) / (len(self.indices_) + 1)

                # Calculate loss for this combined model
                loss = self._calculate_loss(combined_predictions, true_targets)

                # If this model improves the performance, select it as the best model
                if loss < best_loss:
                    best_loss = loss
                    best_model_idx = idx

            # Add the best model's predictions to the ensemble
            self.indices_.append(best_model_idx)
            current_ensemble_predictions += base_models_predictions[:, best_model_idx]
            self.trajectory_.append(best_loss)

        # Calculate final model weights based on the frequency of their selection
        self._calculate_weights()
        return self

    def _calculate_loss(self, predictions: np.ndarray, true_targets: np.ndarray) -> float:
        """Calculate the loss of predictions with respect to the true targets."""
        if self.task_type == 1:  # Classification (accuracy)
            correct = np.sum(np.argmax(predictions, axis=1) == true_targets)
            accuracy = correct / len(true_targets)
            return 1 - accuracy  # We want to minimize the loss (1 - accuracy)
        elif self.task_type == 2:  # Regression (mean squared error)
            return np.mean((predictions - true_targets) ** 2)
        else:
            raise ValueError("Unknown task type!")

    def _calculate_weights(self) -> None:
        """Calculate the weights of the models based on their frequency of selection in the ensemble."""
        ensemble_members = Counter(self.indices_).most_common()
        total_counts = sum(count for idx, count in ensemble_members)
        self.weights_ = np.zeros_like(self.weights_)
        for idx, count in ensemble_members:
            weight = float(count) / total_counts
            self.weights_[idx] = weight

    def predict(self, base_models_predictions: np.ndarray) -> np.ndarray:
        """Create ensemble predictions from the base model predictions using the selected model weights.

        Parameters:
        -----------
        base_models_predictions : np.ndarray
            Predictions of the base models, shape: (n_samples, n_models)

        Returns:
        --------
        np.ndarray
            Final ensemble predictions.
        """
        average = np.zeros(base_models_predictions.shape[0], dtype=np.float64)
        for idx, weight in enumerate(self.weights_):
            if weight > 0.0:
                average += base_models_predictions[:, idx] * weight
        return average

    def get_validation_performance(self) -> float:
        """Return the final validation performance (loss) of the ensemble."""
        return self.trajectory_[-1]
