import os
import hashlib
import base64
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from joblib import load
from tqdm import tqdm
from numerblox.evaluation import NumeraiClassicEvaluator


# Cache function provided by user, with improvements
def _cache(*args, **kwds):
    try:
        sorted_kwargs = str(sorted(kwds.items()))
        serialized = pickle.dumps((args, sorted_kwargs))
        hash_bytes = hashlib.sha256(serialized).digest()
        hash_hex = hashlib.sha256(hash_bytes).hexdigest()
        return hash_hex[:12]
    except Exception as e:
        raise ValueError(f"Failed to generate cache hash: {e}")


# Helper function to check sklearn compatibility
def _check_sklearn_compatibility(model):
    print(f"Checking compatibility of model {model}")
    if not hasattr(model, "fit") or not callable(getattr(model, "fit")):
        raise ValueError(f"Model {model} does not implement a 'fit' method.")
    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise ValueError(f"Model {model} does not implement a 'predict' method.")


# Walk-forward training class
class WalkForward(BaseEstimator, RegressorMixin):

    def __init__(self, model_paths, n_eras=52, cache_dir=None, era_column="era", ensemble=None, model_save_path=None, metrics_list=None):
        """
        Parameters:
        - model_paths: List of paths to pre-trained sklearn models (.pkl)
        - n_eras: Number of last eras to iterate over (for walk-forward iteration)
        - cache_dir: Directory to save cached results (if None, no caching will be done)
        - era_column: Column name in X that contains the era indicator
        - ensemble: Instance of ensemble class for ensemble selection
        - model_save_path: Path to save the last trained model for each era (default: None)
        - metrics_list: List of metrics to use for evaluation (default: None)
        """
        print("Initializing WalkForward class")

        self.model_paths = model_paths
        self.models = [self._load_model(path) for path in model_paths]  # Load models from disk with validation
        self.model_names = [os.path.basename(path).replace('.pkl', '') for path in model_paths]  # Extract model names
        self.n_eras = n_eras
        self.cache_dir = cache_dir
        self.era_column = era_column
        self.oof_predictions = []
        self.oof_targets = []
        self.eras_trained_on = []
        self.ensemble = ensemble  # Ensemble instance passed from outside
        self.model_save_path = model_save_path  # Path where models will be saved after last era (if provided)
        self.metrics_list = metrics_list or ["mean_std_sharpe", "apy", "max_drawdown"]

        # Create cache directory if caching is enabled and directory doesn't exist
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Cache directory set to: {self.cache_dir}")

        # Create model save directory if it doesn't exist and model_save_path is provided
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
            print(f"Model save directory set to: {self.model_save_path}")

        # To store benchmark and evaluation results
        self.evaluation_results = None
        self.benchmark_predictions = None
        self.per_era_results = None  # To store per-era results

    def _load_model(self, path):
        """Load a model from the given path and check its compatibility with sklearn."""
        print(f"Loading model from: {path}")
        model = load(path)
        _check_sklearn_compatibility(model)
        return model

    def _save_model(self, model, model_name):
        """Save the trained model to the specified path after the last era (if model_save_path is provided)."""
        if self.model_save_path is not None:
            model_path = os.path.join(self.model_save_path, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved trained model: {model_name} to {model_path}")

    def fit(self, X, y):
        """
        Iteratively trains on past data and makes out-of-sample (OOF) predictions.

        Parameters:
        - X: DataFrame containing feature columns and an 'era' column.
        - y: Series or array containing the target values.
        """
        print(f"Starting walk-forward training over the last {self.n_eras} eras")

        # Extract unique eras from the data
        eras = sorted(X[self.era_column].unique())

        benchmark_predictions = pd.DataFrame(index=X.index)

        total_tasks = len(self.model_names) * self.n_eras  # Total tasks (model * era)

        task_count = 0  # To track progress

        # Use tqdm to track progress over eras and models
        for i in tqdm(range(-self.n_eras, 0), desc="Walk-forward training"):  # Iterate over the last n_eras
            test_era = eras[i]  # The current era used for testing
            print(f"Processing era: {test_era}")

            # Training data: all data up to but not including test_era
            train_data = X[X[self.era_column] < test_era].drop(columns=[self.era_column])
            train_targets = y[train_data.index]

            # Test data: data corresponding to test_era
            test_data = X[X[self.era_column] == test_era].drop(columns=[self.era_column])
            test_targets = y[test_data.index]

            # Check if train_data and test_data are not empty
            if train_data.empty or test_data.empty:
                raise ValueError(f"Empty training or testing data for era {test_era}. Please check your data!")

            # Loop through models to train and predict
            for model, model_name in zip(self.models, self.model_names):
                task_count += 1
                print(f"Training model: {model_name} on era: {test_era} ({task_count}/{total_tasks})")

                # Skip training if it's the first test era, but still get predictions
                if i == -self.n_eras:  # If it's the first test era, skip training and only get predictions
                    print(f"Skipping training for the first test era: {test_era} for model: {model_name}")
                    # Get predictions using the pre-trained model
                    test_predictions = pd.Series(model.predict(test_data), index=test_data.index, name=model_name)
                else:
                    # Train the model for subsequent test eras
                    if self.cache_dir is not None:
                        # Generate cache hash based on training data shape and test_era
                        cache_id = [train_data.shape, test_era]
                        cache_hash = _cache(cache_id)
                        cache_file = os.path.join(self.cache_dir, f"{test_era}_{model_name}_{cache_hash}.pkl")

                        if os.path.exists(cache_file):
                            # Load cached predictions if available
                            with open(cache_file, 'rb') as f:
                                test_predictions = pickle.load(f)
                            print(
                                f"Loaded cached predictions for era {test_era} and model {model_name} from {cache_file}")
                        else:
                            # Train the model on training data and predict on the test era (out-of-sample)
                            print(f"Training model: {model_name} and generating predictions for era {test_era}")
                            test_predictions = pd.Series(model.fit(train_data, train_targets).predict(test_data),
                                                         index=test_data.index, name=model_name)
                            # Save the predictions to cache
                            with open(cache_file, 'wb') as f:
                                pickle.dump(test_predictions, f)
                            print(f"Saved predictions for era {test_era} and model {model_name} to cache {cache_file}")
                    else:
                        # No caching, train and predict directly
                        print(f"No caching. Training model {model_name} and generating predictions for era {test_era}")
                        test_predictions = pd.Series(model.fit(train_data, train_targets).predict(test_data),
                                                     index=test_data.index, name=model_name)

                # If we already have predictions from other models, append current model predictions
                if task_count % len(self.model_names) == 1:
                    # If it's the first model for this era, create new DataFrame
                    combined_predictions = pd.DataFrame(test_predictions)
                else:
                    # Otherwise, add the current model's predictions to the existing DataFrame
                    combined_predictions[model_name] = test_predictions

            # Ensure that all OOF predictions are aligned by reindexing
            combined_predictions = combined_predictions.reindex(test_data.index)

            # Store OOF predictions and corresponding targets
            self.oof_predictions.append(combined_predictions)
            self.oof_targets.append(test_targets.values)
            self.eras_trained_on.append(test_era)

            # Collect original model benchmark predictions
            benchmark_predictions.loc[test_data.index, 'benchmark'] = self.models[0].predict(test_data)

        # After iterating over all eras, concatenate OOF predictions and targets
        all_oof_predictions = pd.concat(self.oof_predictions)
        all_oof_targets = np.concatenate(self.oof_targets)

        # Train and save the final model only if model_save_path is provided
        if self.model_save_path is not None:
            print("Training the final model on all data")
            final_train_data = X.drop(columns=[self.era_column])
            final_train_targets = y
            final_model = self.models[0].fit(final_train_data, final_train_targets)

            # Save the final model trained on all eras
            final_model_name = self.model_names[0]  # Assuming you're saving the primary model
            model_save_file = os.path.join(self.model_save_path, f"final_{final_model_name}.pkl")
            with open(model_save_file, 'wb') as f:
                pickle.dump(final_model, f)
            print(f"Saved the final model trained on all eras to {model_save_file}")

        # Perform ensemble selection using the ensemble class (if provided)
        if self.ensemble:
            print("Performing ensemble selection")
            self.ensemble.fit(all_oof_predictions.values, all_oof_targets)
            print("Ensemble selection completed")

        # Store benchmark predictions
        self.benchmark_predictions = benchmark_predictions

        # Evaluate
        self.evaluate(X, y)

        return self

    def evaluate(self, X, y):
        """
        Evaluates the model predictions on the given data for each model individually.

        Parameters:
        - X: DataFrame containing the features and 'era' column
        - y: Series or array containing the target values
        """
        print("Starting evaluation...")

        # Initialize evaluator
        evaluator = NumeraiClassicEvaluator(metrics_list=self.metrics_list, era_col=self.era_column)

        # Filter X to only include indices for which we have OOF predictions
        all_oof_predictions = pd.concat(self.oof_predictions)
        oof_index = all_oof_predictions.index
        eval_data = pd.concat([X.loc[oof_index], y.loc[oof_index]], axis=1)

        # Prepare OOF predictions and benchmark for the filtered eras
        eval_data['target'] = y.loc[oof_index]

        # Add predictions for each model into eval_data
        for model_name in self.model_names:
            eval_data[model_name] = all_oof_predictions[model_name].loc[oof_index]

        # Add benchmark predictions
        eval_data['benchmark'] = self.benchmark_predictions.loc[oof_index]

        # Perform evaluation for each model's predictions (store overall and per-era results)
        pred_cols = self.model_names + ['benchmark']
        self.evaluation_results = evaluator.full_evaluation(
            dataf=eval_data,
            pred_cols=pred_cols,  # Evaluate each model and the benchmark separately
            target_col='target',
        )
        self.evaluation_results = self.evaluation_results.sort_values(by='mean', ascending=False)

        # Compute Numerai correlation per era for each model and the benchmark
        per_era_numerai_corr = {}
        for model_name in self.model_names:
            per_era_numerai_corr[model_name] = evaluator.per_era_numerai_corrs(
                dataf=eval_data,
                pred_col=model_name,
                target_col='target'
            )

        # Compute for benchmark
        per_era_numerai_corr['benchmark'] = evaluator.per_era_numerai_corrs(
            dataf=eval_data,
            pred_col='benchmark',
            target_col='target'
        )

        # Combine them into a single DataFrame for easier comparison
        self.per_era_numerai_corr = pd.DataFrame(per_era_numerai_corr)

        print("Evaluation and per-era Numerai correlation computation completed.")

    def predict(self, X):
        """
        Predict using the ensemble model (if available).

        Parameters:
        - X: DataFrame to predict on (same format as during training).
        """
        print("Generating predictions using the models")

        test_predictions = pd.DataFrame(
            {model_name: model.predict(X.drop(columns=[self.era_column])) for model, model_name in
             zip(self.models, self.model_names)},
            index=X.index
        )

        if self.ensemble:
            print("Using ensemble to generate final predictions")
            return self.ensemble.predict(test_predictions.values)
        else:
            return test_predictions  # Return individual model predictions if no ensemble is provided
