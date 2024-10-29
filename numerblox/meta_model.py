import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import VotingRegressor
import pickle
from .meta_ensemble import GreedyEnsemble
from . import logger

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


class MetaModel(BaseEstimator, RegressorMixin):
    def __init__(self, max_ensemble_size: int = 7, random_state: int = 42, weight_factor: float = None, metric='corr'):
        self.metric = metric
        self.max_ensemble_size = max_ensemble_size
        self.random_state = random_state
        self.weight_factor = weight_factor  # Weight factor used in sample weighting
        self.selected_model_names = []  # List to store the names of selected models
        self.ensemble_model = None

    def fit(
            self,
            oof,
            models,
            ensemble_method=GreedyEnsemble,
    ):

        if oof.isnull().values.any():
            raise ValueError("Out of fold predictions contains NaN values.")

        if not all(col in oof.columns for col in ['era', 'target', 'meta_data']):
            raise KeyError("The dataframe is missing 'era', 'target' or 'meta_data' columns.")

        if not isinstance(models, dict):
            raise TypeError("The 'models' variable must be a dictionary.")

        # If weight_factor is provided, calculate sample weights based on eras
        if self.weight_factor is not None:
            logger.info(f"Generating sample weights - Weight factor: {self.weight_factor}")
            sample_weights = get_sample_weights(oos_model_predictions, wfactor=self.weight_factor, eras=oos_model_predictions.era)
        else:
            sample_weights = None

        # Instantiate ensemble
        if isinstance(ensemble_method, type):
            ensemble_method = ensemble_method(max_ensemble_size=self.max_ensemble_size, metric=self.metric, random_state=self.random_state)

        # Fit the ensemble method with base model predictions, true targets, and sample weights
        logger.info(f"Ensembling out of sample predictions - Metric: {self.metric}")
        ensemble_method.fit(oof, sample_weights=sample_weights)

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


