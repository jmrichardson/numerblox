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

    def __init__(self, model_paths, horizon_eras=4, era_column="era", meta=None, meta_eras=[1, 3, 12],
                 era_models_dir='tmp/era_models', final_models_dir='tmp/final_models'):
        """
        Parameters:
        - model_paths: List of paths to pre-trained sklearn models (.pkl)
        - horizon_eras:  Number of eras of prediction time frame
        - era_column: Column name in X that contains the era indicator
        - meta: Meta ensemble instance (can be None)
        - meta_eras: List of integers specifying window sizes for meta models
        - era_models_dir: Directory to save interim models per era
        - final_models_dir: Directory to save final models (if None, do not save final models)
        """
        print("Initializing WalkForward class")

        self.model_paths = model_paths
        self.models = [self._load_model(path) for path in model_paths]  # Load models from disk with validation
        self.model_names = [os.path.basename(path).replace('.pkl', '') for path in model_paths]  # Extract model names
        self.horizon_eras = horizon_eras
        self.era_column = era_column
        self.oof_predictions = []
        self.oof_targets = []
        self.eras_trained_on = []
        self.meta = meta  # Ensemble instance passed from outside
        self.meta_eras = meta_eras  # List of window sizes for meta models
        self.era_models_dir = era_models_dir
        self.final_models_dir = final_models_dir

        # Dictionaries to keep track of trained models and their paths
        self.trained_model_paths = {}  # To keep track of trained models and their paths with era
        self.latest_trained_model_paths = {}  # To keep track of the latest trained model path per model name
        self.latest_trained_models = {}  # To keep track of the latest trained models per model name

        # Create era models directory if it doesn't exist
        os.makedirs(self.era_models_dir, exist_ok=True)

        # Create final models directory if it doesn't exist and final_models_dir is provided
        if self.final_models_dir is not None:
            os.makedirs(self.final_models_dir, exist_ok=True)
            print(f"Final models directory set to: {self.final_models_dir}")

        # To store benchmark and evaluation results
        self.evaluation_results = None
        self.benchmark_predictions = None
        self.per_era_numerai_corr = None  # To store per-era Numerai correlations

        self.meta_weights = {}

    def _load_model(self, path):
        """Load a model from the given path and check its compatibility with sklearn."""
        print(f"Loading model from: {path}")
        with open(path, 'rb') as f:
            model = pickle.load(f)
        _check_sklearn_compatibility(model)
        return model

    def _save_model(self, model, model_name, is_final=False):
        """Save the trained model to the era_models_dir or final_models_dir based on the stage."""
        if is_final:
            if self.final_models_dir is None:
                return None
            model_dir = self.final_models_dir
        else:
            model_dir = self.era_models_dir

        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model: {model_name} to {model_path}")
        return model_path

    def fit(self, X_train, y_train, X_test, y_test):
        print("Starting training and walk-forward prediction")

        # Extract unique eras from test data
        eras_test = sorted(X_test[self.era_column].unique())

        # Initialize train_data and train_targets
        train_data = X_train  # Initial training data
        train_targets = y_train

        # Benchmark DataFrame to collect predictions (only for base models, not meta models)
        benchmark_predictions = pd.DataFrame(index=X_test.index)

        # Generate benchmark predictions using the models loaded from disk (no retraining)
        for model, model_name in zip(self.models, self.model_names):
            print(f"Generating benchmark predictions for {model_name} on the entire test set.")

            # Drop the 'era' column for the test data
            test_data = X_test.drop(columns=[self.era_column])

            # Make predictions for the entire test set using the loaded model
            benchmark_predictions[f'{model_name}_benchmark'] = model.predict(test_data)

        print("Benchmark predictions generated using models loaded from disk.")

        # Initialize variables for walk-forward training
        total_tasks = len(self.models) * len(eras_test)  # Models * test eras
        task_count = 0  # To track progress

        for test_era in tqdm(eras_test, desc="Walk-forward training"):
            print(f"Processing test era: {test_era}")

            test_data = X_test[X_test[self.era_column] == test_era]
            test_targets = y_test.loc[test_data.index]

            if train_data.empty or test_data.empty:
                raise ValueError(f"Empty training or testing data for era {test_era}. Please check your data!")

            combined_predictions = pd.DataFrame(index=test_data.index)

            for model, model_name in zip(self.models, self.model_names):
                task_count += 1

                # Construct cache ID for identifying model save/load files
                cache_id = [
                    train_data.shape,  # Shape of the training data
                    sorted(train_data.columns.tolist()),  # Column names of the training data
                    train_targets.name,  # Name of the target variable
                    test_era,  # Current test era
                    model_name,  # Model name
                    self.horizon_eras,  # The horizon eras
                ]
                cache_hash = get_cache_hash(cache_id)

                trained_model_name = f"{model_name}_{test_era}_{cache_hash}"
                print(f"Processing model: {trained_model_name} on test era: {test_era} ({task_count}/{total_tasks})")

                model_path = os.path.join(self.era_models_dir, f"{trained_model_name}.pkl")

                if os.path.exists(model_path):
                    # Load the trained model from disk if it exists
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"Loaded trained model from {model_path}")

                    # Make sure the latest trained model path is updated even when loading from cache
                    self.latest_trained_model_paths[model_name] = model_path
                    self.latest_trained_models[model_name] = model
                else:
                    # Train the model if it hasn't been trained for this era yet
                    print(f"Training model: {model_name} on training data up to era {test_era}")
                    model.fit(train_data.drop(columns=[self.era_column]), train_targets)

                    # Save the trained model to the cache directory
                    self._save_model(model, trained_model_name)
                    self.trained_model_paths[trained_model_name] = model_path
                    self.latest_trained_model_paths[model_name] = model_path
                    self.latest_trained_models[model_name] = model

                # Make predictions for the current test era
                test_predictions = pd.Series(model.predict(test_data.drop(columns=[self.era_column])),
                                             index=test_data.index, name=model_name)

                combined_predictions[model_name] = test_predictions

            combined_predictions = combined_predictions.reindex(test_data.index)
            self.oof_predictions.append(combined_predictions)
            self.oof_targets.append(test_targets)
            self.eras_trained_on.append(test_era)

            # Now handle meta models, ensuring they predict each era once enough data (window) is available
            if self.meta is not None:
                for window_size in self.meta_eras:
                    # Ensure there are enough OOF predictions to form the meta model
                    if len(self.oof_predictions) >= (window_size + self.horizon_eras):
                        print(
                            f"Creating meta model with window size: {window_size} and horizon eras: {self.horizon_eras}")

                        # Get the eras up to (but not including) the last 'horizon_eras'
                        recent_oof_preds_list = self.oof_predictions[
                                                -(window_size + self.horizon_eras):-self.horizon_eras]
                        recent_oof_preds = pd.concat(recent_oof_preds_list)

                        # Get corresponding targets and eras for the same time window
                        recent_oof_targets_list = self.oof_targets[
                                                  -(window_size + self.horizon_eras):-self.horizon_eras]
                        recent_oof_targets = pd.concat(recent_oof_targets_list)

                        recent_eras_list = self.eras_trained_on[-(window_size + self.horizon_eras):-self.horizon_eras]
                        recent_eras = pd.Series(np.concatenate(
                            [[era] * len(pred) for era, pred in zip(recent_eras_list, recent_oof_preds_list)]
                        ))

                        # Exclude meta model predictions from base models
                        base_model_columns = [col for col in recent_oof_preds.columns if
                                              not col.startswith('meta_model')]
                        base_models_predictions = recent_oof_preds[base_model_columns]

                        # Proceed with fitting the meta model if targets are available
                        if recent_oof_targets.notnull().any():
                            true_targets = recent_oof_targets

                            # Collect paths of base models
                            model_name_to_path = {}
                            for col in base_model_columns:
                                model_path = self.latest_trained_model_paths.get(col)
                                if not model_path:
                                    raise ValueError(f"No trained model path found for base model {col}")
                                model_name_to_path[col] = model_path

                            meta_model = clone(self.meta)
                            meta_model_name = f"meta_model_{window_size}"
                            trained_meta_model_name = f"{meta_model_name}_{test_era}_{cache_hash}"

                            model_path = os.path.join(self.era_models_dir, f"{trained_meta_model_name}.pkl")

                            if os.path.exists(model_path):
                                print(f"Meta model cache found at {model_path}, loading model.")
                                # Load the entire meta model from the cache
                                with open(model_path, 'rb') as f:
                                    meta_model = pickle.load(f)
                            else:
                                # Fit the meta model only if no cache is found
                                meta_model.fit(base_models_predictions, true_targets, recent_eras, model_name_to_path,
                                               train_data.drop(columns=[self.era_column]), train_targets)

                                # Save the entire meta model object
                                with open(model_path, 'wb') as f:
                                    pickle.dump(meta_model, f)
                                print(f"Saved meta model: {trained_meta_model_name} to {model_path}")

                            # Store the weights of the meta model (already part of the meta_model object)
                            self.meta_weights[trained_meta_model_name] = meta_model.weights_

                            # Make predictions using the meta model for the current test era
                            meta_predictions = meta_model.predict(test_data.drop(columns=[self.era_column]))

                            # Store meta predictions in the combined predictions for each era
                            combined_predictions[meta_model_name] = meta_predictions

            # Update train data by concatenating test data after each iteration
            train_data = pd.concat([train_data, test_data])
            train_targets = pd.concat([train_targets, test_targets])

        # After the loop, all meta predictions will be present in the combined_predictions DataFrame
        # Collect all OOF predictions
        for idx, era in enumerate(self.eras_trained_on):
            self.oof_predictions[idx]['era'] = era
        self.all_oof_predictions = pd.concat(self.oof_predictions)
        self.all_oof_targets = pd.concat(self.oof_targets)

        self.benchmark_predictions = benchmark_predictions
        self.evaluate(X_test, y_test)

        return self



    def evaluate(self, X, y):
        print("Starting evaluation...")

        # Initialize evaluator
        evaluator = NumeraiClassicEvaluator(metrics_list=["mean_std_sharpe", "apy", "max_drawdown"],
                                            era_col=self.era_column)

        # Filter X to only include indices for which we have OOF predictions
        oof_index = self.all_oof_predictions.index

        # Prepare OOF predictions and benchmark for the filtered eras, ensuring no duplication of 'era'
        eval_data = pd.concat([X.loc[oof_index].drop(columns=[self.era_column]), y.loc[oof_index]], axis=1)
        eval_data['target'] = y.loc[oof_index]

        # Add OOF predictions (which already include the 'era' column)
        eval_data = pd.concat([eval_data, self.all_oof_predictions.loc[oof_index]], axis=1)

        # Add all benchmark predictions (all columns that end with "_benchmark")
        benchmark_cols = [col for col in self.benchmark_predictions.columns if col.endswith('_benchmark')]
        eval_data = pd.concat([eval_data, self.benchmark_predictions.loc[oof_index, benchmark_cols]], axis=1)

        # Clip all prediction columns between 0 and 1 - Required by evaluator
        pred_cols = [col for col in self.all_oof_predictions.columns if col != 'era'] + benchmark_cols
        eval_data[pred_cols] = eval_data[pred_cols].clip(0, 1)

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


