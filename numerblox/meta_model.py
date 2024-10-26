import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import VotingRegressor
import pickle
from .misc import get_sample_weights
from .meta_ensemble import GreedyEnsemble
from . import logger


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

        # Test no nans, meta_data, era and target exists...

        # If weight_factor is provided, calculate sample weights based on eras
        if self.weight_factor is not None:
            # Fix this
            sample_weights = get_sample_weights(oos_model_predictions, wfactor=self.weight_factor, eras=oos_model_predictions.era)
        else:
            sample_weights = None

        # Instantiate ensemble
        if isinstance(ensemble_method, type):
            ensemble_method = ensemble_method(max_ensemble_size=self.max_ensemble_size, metric=self.metric, random_state=self.random_state)

        # Fit the ensemble method with base model predictions, true targets, and sample weights
        logger.info("Start meta fit")
        ensemble_method.fit(oof, sample_weights=sample_weights)
        print("End meta fit")

        # Get the names of the selected models and their weights
        self.selected_model_names = ensemble_method.selected_model_names_
        self.weights_ = ensemble_method.weights_

        # Load the selected models from disk
        selected_models = []
        for model_name in self.selected_model_names:
            print(f"meta {model_name}")
            model_path = models[model_name]['model_path']
            with open(model_path, 'rb') as f:
                model = pickle.load(f)  # Load the model from the file
            selected_models.append((model_name, model))  # Add the model to the list

        # Prepare the final VotingRegressor ensemble model
        weights_list = []
        estimators_list = []
        models = []
        for model_name, model in selected_models:
            print(f"meta2 {model_name}")
            weight = self.weights_.loc[model_name]  # Get the weight for the model
            weights_list.append(weight)  # Add the weight to the list
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


