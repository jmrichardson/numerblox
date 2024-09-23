import os
import hashlib
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from tqdm import tqdm
from numerblox.evaluation import NumeraiClassicEvaluator
from numerblox.misc import get_cache_hash



def _check_sklearn_compatibility(model):
    print(f"Checking compatibility of model {model}")
    if not hasattr(model, "fit") or not callable(getattr(model, "fit")):
        raise ValueError(f"Model {model} does not implement a 'fit' method.")
    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise ValueError(f"Model {model} does not implement a 'predict' method.")

# Walk-forward training class
class WalkForward(BaseEstimator, RegressorMixin):

    def __init__(self, model_paths, cache_dir=None, era_column="era", meta=None, meta_eras=[1, 3, 12], model_save_path=None,
                 metrics_list=None):
        """
        Parameters:
        - model_paths: List of paths to pre-trained sklearn models (.pkl)
        - cache_dir: Directory to save cached results (if None, no caching will be done)
        - era_column: Column name in X that contains the era indicator
        - meta: Meta ensemble instance (can be None)
        - meta_eras: List of integers specifying window sizes for meta models
        - model_save_path: Path to save the models (default: None)
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
        self.meta = meta  # Ensemble instance passed from outside
        self.meta_eras = meta_eras  # List of window sizes for meta models
        self.model_save_path = model_save_path  # Path where models will be saved
        self.metrics_list = metrics_list or ["mean_std_sharpe", "apy", "max_drawdown"]

        # Dictionaries to keep track of trained models and their paths
        self.trained_model_paths = {}  # To keep track of trained models and their paths with era
        self.latest_trained_model_paths = {}  # To keep track of the latest trained model path per model name
        self.latest_trained_models = {}  # To keep track of the latest trained models per model name

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
        self.per_era_numerai_corr = None  # To store per-era Numerai correlations

    def _load_model(self, path):
        """Load a model from the given path and check its compatibility with sklearn."""
        print(f"Loading model from: {path}")
        with open(path, 'rb') as f:
            model = pickle.load(f)
        _check_sklearn_compatibility(model)
        return model

    def _save_model(self, model, model_name):
        """Save the trained model to the specified path."""
        if self.model_save_path is not None:
            model_path = os.path.join(self.model_save_path, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved trained model: {model_name} to {model_path}")
            return model_path
        else:
            return None

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Fits the models on provided training data and generates out-of-sample (OOF) predictions.

        Parameters:
        - X_train: DataFrame containing training feature columns and an 'era' column.
        - y_train: Series containing the training target values.
        - X_test: DataFrame containing testing feature columns and an 'era' column.
        - y_test: Series containing the test target values.
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

        # Initialize variable to keep track of test era index
        test_era_index = 0

        # Use tqdm to track progress over test eras and models
        for test_era in tqdm(eras_test, desc="Walk-forward training"):
            print(f"Processing test era: {test_era}")

            # Test data: data corresponding to the test_era
            test_data = X_test[X_test[self.era_column] == test_era].drop(
                columns=[self.era_column])  # Drop the 'era' column
            test_targets = y_test.loc[test_data.index]

            # Check if train_data and test_data are not empty
            if train_data.empty or test_data.empty:
                raise ValueError(f"Empty training or testing data for era {test_era}. Please check your data!")

            # Initialize DataFrame to collect predictions for current era
            combined_predictions = pd.DataFrame(index=test_data.index)

            # Loop through models to train and predict
            for model, model_name in zip(self.models[:len(self.model_paths)], self.model_names[:len(self.model_paths)]):
                task_count += 1

                # Define trained model name
                trained_model_name = f"{model_name}_up_to_{test_era}"

                print(f"Processing model: {trained_model_name} on test era: {test_era} ({task_count}/{total_tasks})")

                # Cache handling for model predictions
                cache_id = [train_data.shape, test_era, model_name]
                cache_hash = get_cache_hash(cache_id)
                cache_file = os.path.join(self.cache_dir,
                                          f"{trained_model_name}_{cache_hash}.pkl") if self.cache_dir else None

                if cache_file and os.path.exists(cache_file):
                    # Load cached predictions if available
                    with open(cache_file, 'rb') as f:
                        test_predictions = pickle.load(f)
                    # Ensure test_predictions has correct name
                    test_predictions.name = model_name  # Use consistent model name
                    print(
                        f"Loaded cached predictions for era {test_era} and model {trained_model_name} from {cache_file}")

                    # Update the latest trained model path if not already done
                    if trained_model_name not in self.trained_model_paths:
                        model_path = os.path.join(self.model_save_path, f"{trained_model_name}.pkl")
                        if os.path.exists(model_path):
                            self.trained_model_paths[trained_model_name] = model_path
                            self.latest_trained_model_paths[model_name] = model_path  # Update latest path
                            # Load the model if not in latest_trained_models
                            if model_name not in self.latest_trained_models:
                                with open(model_path, 'rb') as f:
                                    loaded_model = pickle.load(f)
                                self.latest_trained_models[model_name] = loaded_model
                        else:
                            # Model file does not exist, need to retrain model
                            print(f"Model file {model_path} not found. Retraining model.")
                            model.fit(train_data, train_targets)
                            model_path = self._save_model(model, trained_model_name)
                            self.trained_model_paths[trained_model_name] = model_path
                            self.latest_trained_model_paths[model_name] = model_path
                            self.latest_trained_models[model_name] = model
                    else:
                        # Update latest models if necessary
                        if model_name not in self.latest_trained_models:
                            with open(self.trained_model_paths[trained_model_name], 'rb') as f:
                                loaded_model = pickle.load(f)
                            self.latest_trained_models[model_name] = loaded_model
                else:
                    # Train the model on the training data
                    print(f"Training model: {model_name} on training data up to era {test_era}")
                    model.fit(train_data, train_targets)
                    # Save the trained model
                    model_path = self._save_model(model, trained_model_name)
                    # Keep track of the trained model path
                    self.trained_model_paths[trained_model_name] = model_path
                    self.latest_trained_model_paths[model_name] = model_path  # Update latest path
                    self.latest_trained_models[model_name] = model  # Update latest model
                    # Generate predictions for the test era
                    test_predictions = pd.Series(model.predict(test_data),
                                                 index=test_data.index, name=model_name)  # Use consistent model name
                    # Save predictions to cache if applicable
                    if cache_file:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(test_predictions, f)
                        print(
                            f"Saved predictions for era {test_era} and model {trained_model_name} to cache {cache_file}")

                # Add the current model's predictions to the combined DataFrame
                combined_predictions[model_name] = test_predictions  # Use consistent model name

            # Ensure that all OOF predictions are aligned by reindexing
            combined_predictions = combined_predictions.reindex(test_data.index)

            # Store OOF predictions and corresponding targets
            self.oof_predictions.append(combined_predictions)
            self.oof_targets.append(test_targets)
            self.eras_trained_on.append(test_era)

            # Collect original model benchmark predictions (from the first base model)
            benchmark_predictions.loc[test_data.index, 'benchmark'] = combined_predictions.iloc[:, 0]

            # After base model predictions, create meta models if meta is not None
            if self.meta is not None:
                for window_size in self.meta_eras:
                    # Check that there are enough past eras to apply the meta-model
                    # Use only past eras (not the current era)
                    if len(self.oof_predictions) > window_size:
                        print(f"Creating meta model with window size: {window_size}")

                        # Collect OOF predictions and targets from the last 'window_size' PAST eras
                        recent_oof_preds_list = self.oof_predictions[-(window_size + 1):-1]  # Exclude current test era
                        recent_oof_preds = pd.concat(recent_oof_preds_list)
                        recent_oof_targets_list = self.oof_targets[-(window_size + 1):-1]  # Exclude current test era
                        recent_oof_targets = pd.concat(recent_oof_targets_list)

                        # Check if there are any valid targets (i.e., not None)
                        if recent_oof_targets.notnull().any():
                            # Prepare base model predictions as DataFrame
                            base_models_predictions = recent_oof_preds  # DataFrame
                            true_targets = recent_oof_targets  # Series

                            # Provide the model paths corresponding to the prediction columns
                            model_name_to_path = {}
                            for col in base_models_predictions.columns:
                                model_path = self.latest_trained_model_paths.get(col)
                                if not model_path:
                                    raise ValueError(f"No trained model path found for base model {col}")
                                model_name_to_path[col] = model_path

                            # Create Meta model (clone the meta instance to avoid state carry-over)
                            meta_model = clone(self.meta)
                            meta_model.fit(base_models_predictions, true_targets, model_name_to_path, train_data,
                                           train_targets)

                            # Use consistent meta model name
                            meta_model_name = f"meta_model_{window_size}"
                            trained_meta_model_name = f"{meta_model_name}_up_to_{test_era}"

                            # Save the meta model's ensemble model
                            model_path = self._save_model(meta_model.ensemble_model, trained_meta_model_name)
                            self.trained_model_paths[trained_meta_model_name] = model_path
                            self.latest_trained_model_paths[meta_model_name] = model_path
                            self.latest_trained_models[meta_model_name] = meta_model.ensemble_model

                            print(f"Saved meta model: {trained_meta_model_name} to {model_path}")

                            # Use meta model's ensemble model to predict on current test_data
                            meta_predictions = meta_model.predict(test_data)
                            # Add meta predictions to combined_predictions using consistent name
                            combined_predictions[meta_model_name] = meta_predictions

                            # Add meta model's ensemble model to models list if not already present
                            if meta_model_name not in self.model_names:
                                self.models.append(meta_model.ensemble_model)
                                self.model_names.append(meta_model_name)
                        else:
                            print(f"Skip: window size {window_size} in era {test_era} as all true_targets are None.")

            # Append the test data and corresponding targets to the training data for future iterations
            train_data = pd.concat([train_data, test_data])
            train_targets = pd.concat([train_targets, test_targets])

            # Increment test era index
            test_era_index += 1

        # After iterating over all eras, concatenate OOF predictions and targets
        for idx, era in enumerate(self.eras_trained_on):
            self.oof_predictions[idx]['era'] = era
        self.all_oof_predictions = pd.concat(self.oof_predictions)
        self.all_oof_targets = pd.concat(self.oof_targets)

        # Store benchmark predictions
        self.benchmark_predictions = benchmark_predictions

        # Evaluate
        self.evaluate(X_test, y_test)

        return self

    def evaluate(self, X, y):
        """
        Evaluates the model predictions on the given data for each model individually.

        Parameters:
        - X: DataFrame containing the features and 'era' column
        - y: Series containing the target values
        """
        print("Starting evaluation...")

        # Initialize evaluator
        evaluator = NumeraiClassicEvaluator(metrics_list=self.metrics_list, era_col=self.era_column)

        # Filter X to only include indices for which we have OOF predictions
        oof_index = self.all_oof_predictions.index

        # Prepare OOF predictions and benchmark for the filtered eras, ensuring no duplication of 'era'
        eval_data = pd.concat([X.loc[oof_index].drop(columns=['era']), y.loc[oof_index]], axis=1)
        eval_data['target'] = y.loc[oof_index]

        # Add OOF predictions (which already include the 'era' column)
        eval_data = pd.concat([eval_data, self.all_oof_predictions.loc[oof_index]], axis=1)

        # Add benchmark predictions
        eval_data['benchmark'] = self.benchmark_predictions.loc[oof_index]

        # Get the list of prediction columns, excluding 'era'
        pred_cols = [col for col in self.all_oof_predictions.columns if col != 'era'] + ['benchmark']

        # Perform evaluation for each model's predictions (store overall and per-era results)
        self.evaluation_results = evaluator.full_evaluation(
            dataf=eval_data,
            pred_cols=pred_cols,  # Evaluate each model and the benchmark separately
            target_col='target',
        )
        self.evaluation_results = self.evaluation_results.sort_values(by='mean', ascending=False)

        # Compute Numerai correlation per era for each model and the benchmark
        per_era_numerai_corr = {}
        for model_name in pred_cols:
            per_era_numerai_corr[model_name] = evaluator.per_era_numerai_corrs(
                dataf=eval_data,
                pred_col=model_name,
                target_col='target'
            )

        # Combine them into a single DataFrame for easier comparison
        self.per_era_numerai_corr = pd.DataFrame(per_era_numerai_corr)

        print("Evaluation and per-era Numerai correlation computation completed.")

    def predict(self, X):
        """
        Predict using the latest trained models.

        Parameters:
        - X: DataFrame to predict on (same format as during training).
        """
        print("Generating predictions using the latest trained models")

        X_data = X.drop(columns=[self.era_column])

        # Collect predictions from all latest trained models
        base_model_predictions = pd.DataFrame(
            {model_name: self.latest_trained_models[model_name].predict(X_data)
             for model_name in self.latest_trained_models},
            index=X.index
        )

        return base_model_predictions
