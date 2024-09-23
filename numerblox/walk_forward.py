import os
import hashlib
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from joblib import load
from tqdm import tqdm
from numerblox.evaluation import NumeraiClassicEvaluator
from collections import Counter

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

    def __init__(self, model_paths, cache_dir=None, era_column="era", meta_eras=[1,3,12], model_save_path=None,
                 metrics_list=None):
        """
        Parameters:
        - model_paths: List of paths to pre-trained sklearn models (.pkl)
        - cache_dir: Directory to save cached results (if None, no caching will be done)
        - era_column: Column name in X that contains the era indicator
        - meta_eras: List of integers specifying window sizes for meta models
        - model_save_path: Path to save the last trained model for each era (default: None)
        - metrics_list: List of metrics to use for evaluation (default: None)
        """
        print("Initializing WalkForward class")

        self.model_paths = model_paths
        self.models = [self._load_model(path) for path in model_paths]  # Load models from disk with validation
        self.model_names = [os.path.basename(path).replace('.pkl', '') for path in model_paths]  # Extract model names
        self.cache_dir = cache_dir
        self.era_column = era_column
        self.oof_predictions = []
        self.oof_targets = []
        self.eras_trained_on = []
        self.meta_eras = meta_eras  # List of window sizes for meta models
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
        with open(path, 'rb') as f:
            model = pickle.load(f)
        _check_sklearn_compatibility(model)
        return model

    def _save_model(self, model, model_name):
        """Save the trained model to the specified path after the last era (if model_save_path is provided)."""
        if self.model_save_path is not None:
            model_path = os.path.join(self.model_save_path, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved trained model: {model_name} to {model_path}")

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Fits the models on provided training data and generates out-of-sample (OOF) predictions.

        Parameters:
        - X_train: DataFrame containing training feature columns and an 'era' column.
        - y_train: Series or array containing the training target values.
        - X_test: DataFrame containing testing feature columns and an 'era' column.
        - y_test: Series or array containing the test target values.
        """
        print("Starting training and walk-forward prediction")

        # Extract unique eras from test data
        eras_test = sorted(X_test[self.era_column].unique())

        # Initialize train_data and train_targets
        train_data = X_train.drop(columns=[self.era_column])  # Initial training data
        train_targets = y_train

        # Benchmark DataFrame to collect predictions
        benchmark_predictions = pd.DataFrame(index=X_test.index)

        total_tasks = len(self.model_names) * len(eras_test)  # Models * test eras
        task_count = 0  # To track progress

        # Use tqdm to track progress over test eras and models
        for test_era in tqdm(eras_test, desc="Walk-forward training"):
            print(f"Processing test era: {test_era}")

            # Test data: data corresponding to the test_era
            test_data = X_test[X_test[self.era_column] == test_era].drop(
                columns=[self.era_column])  # Drop the 'era' column
            test_targets = y_test[test_data.index]

            # Check if train_data and test_data are not empty
            if train_data.empty or test_data.empty:
                raise ValueError(f"Empty training or testing data for era {test_era}. Please check your data!")

            # Initialize DataFrame to collect predictions for current era
            combined_predictions = pd.DataFrame(index=test_data.index)

            # Loop through models to train and predict
            for model, model_name in zip(self.models, self.model_names):
                task_count += 1
                print(f"Processing model: {model_name} on test era: {test_era} ({task_count}/{total_tasks})")

                # Cache handling for model predictions
                cache_id = [train_data.shape, test_era]
                cache_hash = _cache(cache_id)
                cache_file = os.path.join(self.cache_dir,
                                          f"{test_era}_{model_name}_{cache_hash}.pkl") if self.cache_dir else None

                if cache_file and os.path.exists(cache_file):
                    # Load cached predictions if available
                    with open(cache_file, 'rb') as f:
                        test_predictions = pickle.load(f)
                    print(f"Loaded cached predictions for era {test_era} and model {model_name} from {cache_file}")
                else:
                    # Train the model on the training data and predict on the test era
                    print(f"Training model: {model_name} and generating predictions for test era {test_era}")
                    model.fit(train_data, train_targets)
                    test_predictions = pd.Series(model.predict(test_data),
                                                 index=test_data.index, name=model_name)
                    # Save predictions to cache if applicable
                    if cache_file:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(test_predictions, f)
                        print(f"Saved predictions for era {test_era} and model {model_name} to cache {cache_file}")

                # Add the current model's predictions to the combined DataFrame
                combined_predictions[model_name] = test_predictions

            # Ensure that all OOF predictions are aligned by reindexing
            combined_predictions = combined_predictions.reindex(test_data.index)

            # Store OOF predictions and corresponding targets
            self.oof_predictions.append(combined_predictions)
            self.oof_targets.append(test_targets.values)
            self.eras_trained_on.append(test_era)

            # Collect original model benchmark predictions
            benchmark_predictions.loc[test_data.index, 'benchmark'] = self.models[0].predict(test_data)

            # After base model predictions, create meta models
            for window_size in self.meta_eras:
                if len(self.oof_predictions) >= window_size:
                    print(f"Creating meta model with window size: {window_size}")
                    # Collect OOF predictions and targets from the last 'window_size' eras
                    recent_oof_preds_list = self.oof_predictions[-window_size:]
                    recent_oof_preds = pd.concat(recent_oof_preds_list)
                    recent_oof_targets = np.concatenate(self.oof_targets[-window_size:])
                    recent_eras = self.eras_trained_on[-window_size:]

                    # Prepare base model predictions as numpy array
                    base_models_predictions = recent_oof_preds.values  # shape: (n_samples, n_models)
                    true_targets = recent_oof_targets  # shape: (n_samples,)

                    # Create Meta model
                    meta_model = Meta(task_type=2, ensemble_size=5)
                    meta_model.fit(base_models_predictions, true_targets)

                    # Save the meta model
                    meta_model_name = f"meta_model_{window_size}"
                    meta_model_path = os.path.join(self.model_save_path or "", f"{meta_model_name}.pkl")
                    with open(meta_model_path, 'wb') as f:
                        pickle.dump(meta_model, f)
                    print(f"Saved meta model: {meta_model_name} to {meta_model_path}")

                    # Use meta model to predict on current test_data
                    base_test_preds = combined_predictions.values  # shape: (n_samples, n_models)
                    meta_predictions = meta_model.predict(base_test_preds)
                    # Add meta predictions to combined_predictions
                    combined_predictions[meta_model_name] = meta_predictions

                    # Add meta model to models list if not already present
                    if meta_model_name not in self.model_names:
                        self.models.append(meta_model)
                        self.model_names.append(meta_model_name)

            # Append the test data and corresponding targets to the training data for future iterations
            # Include the era column back to test_data for concatenation
            test_data_with_era = test_data.copy()
            test_data_with_era[self.era_column] = test_era
            train_data = pd.concat([train_data, test_data])
            train_targets = pd.concat([train_targets, test_targets])

        # After iterating over all eras, concatenate OOF predictions and targets
        all_oof_predictions = pd.concat(self.oof_predictions)
        all_oof_targets = np.concatenate(self.oof_targets)

        # Perform evaluation
        self.evaluate(X_test, y_test)

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
        Predict using the models including meta models.

        Parameters:
        - X: DataFrame to predict on (same format as during training).
        """
        print("Generating predictions using the models")

        # Collect predictions from base models
        base_model_predictions = pd.DataFrame(
            {model_name: model.predict(X.drop(columns=[self.era_column])) for model, model_name in
             zip(self.models[:len(self.model_paths)], self.model_names[:len(self.model_paths)])},
            index=X.index
        )

        # Collect predictions from meta models
        for model, model_name in zip(self.models[len(self.model_paths):], self.model_names[len(self.model_paths):]):
            base_preds_array = base_model_predictions.values  # shape: (n_samples, n_models)
            meta_predictions = model.predict(base_preds_array)
            base_model_predictions[model_name] = meta_predictions

        return base_model_predictions
