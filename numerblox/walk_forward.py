import os
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from tqdm import tqdm
from numerblox.evaluation import NumeraiClassicEvaluator
from numerblox.misc import get_cache_hash
import matplotlib.pyplot as plt
from datetime import datetime
import base64
from io import BytesIO


def _check_sklearn_compatibility(model):
    if not hasattr(model, "fit") or not callable(getattr(model, "fit")):
        raise ValueError(f"Model {model} does not implement a 'fit' method.")
    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise ValueError(f"Model {model} does not implement a 'predict' method.")


class WalkForward(BaseEstimator, RegressorMixin):

    def __init__(self, models_attrs, horizon_eras=4, era_column="era", meta=None,
                 era_models_dir='tmp/era_models', final_models_dir='tmp/final_models', artifacts_dir='tmp/artifacts',
                 expand_train=False):
        self.models_attrs = models_attrs
        self.horizon_eras = horizon_eras
        self.era_column = era_column
        self.oof_predictions = []
        self.oof_targets = []
        self.eras_trained_on = []
        self.meta = meta
        self.era_models_dir = era_models_dir
        self.final_models_dir = final_models_dir
        self.expand_train = expand_train
        self.artifacts_dir = artifacts_dir

        self.latest_trained_model_paths = {}

        os.makedirs(self.era_models_dir, exist_ok=True)
        if self.final_models_dir is not None:
            os.makedirs(self.final_models_dir, exist_ok=True)

        self.evaluation_results = None
        self.benchmark_predictions = None
        self.per_era_numerai_corr = None
        self.meta_weights = {}

    def fit(self, X_train, y_train, X_test, y_test):
        self.validate_arguments(X_train, y_train, X_test, y_test)

        # Initialize train_data and train_targets with X_train
        train_data = X_train.sort_values(by=self.era_column).copy()
        train_targets = y_train.loc[train_data.index].copy()

        # Sort and prepare testing data
        X_test = X_test.sort_values(by=self.era_column).copy()
        y_test = y_test.loc[X_test.index].copy()
        eras_test = sorted(X_test[self.era_column].unique())

        # Determine the starting test era to maintain the horizon gap
        min_test_era = min(eras_test)
        start_test_era = min_test_era + self.horizon_eras  # Start from era 35
        eras_to_test = [era for era in eras_test if era >= start_test_era]

        # For tracking the number of iterations
        iteration = 0

        for test_era in eras_to_test:
            # Prepare test data
            test_data = X_test[X_test[self.era_column] == test_era]
            test_targets = y_test.loc[test_data.index]

            # Skip if no valid targets
            if not test_targets.notnull().any():
                print(f"Era {test_era} has no targets; skipping.")
                continue

            # For the first iteration, compute benchmark predictions
            if iteration == 0:
                benchmark_predictions = pd.DataFrame(index=X_test.index)
                for model_name, model_attrs in self.models_attrs.items():
                    model = self._load_model(model_attrs['model_path'])
                    benchmark_predictions[f'{model_name}_benchmark'] = model.predict(
                        X_test.drop(columns=[self.era_column]))

            # Update training data after the first iteration
            if iteration > 0:
                # Remove the oldest era if expand_train is False
                if not self.expand_train:
                    eras_in_train = train_data[self.era_column].unique()
                    oldest_era = eras_in_train[0]
                    train_data = train_data[train_data[self.era_column] != oldest_era]
                    train_targets = train_targets.loc[train_data.index]

                # Add the next sequential era to train_data
                last_train_era = max(train_data[self.era_column].unique())
                next_train_era = last_train_era + 1

                # Check if the next era is in X_train or X_test
                if next_train_era in X_train[self.era_column].unique():
                    new_train_data = X_train[X_train[self.era_column] == next_train_era]
                    new_train_targets = y_train.loc[new_train_data.index]
                elif next_train_era in X_test[self.era_column].unique():
                    new_train_data = X_test[X_test[self.era_column] == next_train_era]
                    new_train_targets = y_test.loc[new_train_data.index]
                else:
                    # If the next era is not available, skip adding
                    print(f"Next train era {next_train_era} not found in data.")
                    new_train_data = pd.DataFrame()
                    new_train_targets = pd.Series()

                if not new_train_data.empty:
                    train_data = pd.concat([train_data, new_train_data])
                    train_targets = pd.concat([train_targets, new_train_targets])

            # Verify the gap between training and testing eras
            last_train_era = max(train_data[self.era_column].unique())
            gap = test_era - last_train_era - 1  # Subtract 1 because eras are inclusive
            if gap != self.horizon_eras:
                raise ValueError(
                    f"Gap between last training era ({last_train_era}) and test era ({test_era}) is {gap}, expected {self.horizon_eras}")

            # Proceed to train and predict
            print(f"Iteration {iteration + 1}")
            print(f"Training eras: {train_data[self.era_column].unique()}")
            print(f"Testing era: {test_era}")
            combined_predictions = pd.DataFrame(index=test_data.index)

            # Train models and make predictions
            for model_name, model_attrs in self.models_attrs.items():
                cache_id = [
                    train_data.shape,
                    sorted(train_data.columns.tolist()),
                    test_era,
                    model_name,
                    self.horizon_eras,
                    model_attrs,
                ]
                cache_hash = get_cache_hash(cache_id)
                trained_model_name = f"{model_name}_{test_era}_{cache_hash}"
                model_path = os.path.join(self.era_models_dir, f"{trained_model_name}.pkl")

                if os.path.exists(model_path):
                    # Load the cached model
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    self.latest_trained_model_paths[model_name] = model_path
                else:
                    # Train a new model
                    model = self._load_model(model_attrs['model_path'])
                    fit_kwargs = model_attrs.get('fit_kwargs', {}).copy()

                    # Adjust sample weights based on eras, if provided
                    sample_weights = fit_kwargs.get('sample_weight', None)
                    if sample_weights is not None:
                        train_eras_unique = train_data[self.era_column].unique()
                        if isinstance(sample_weights, pd.Series):
                            era_to_weight = {era: sample_weights.iloc[i] for i, era in enumerate(train_eras_unique)}
                        else:
                            era_to_weight = {era: sample_weights[i] for i, era in enumerate(train_eras_unique)}
                        train_weights = train_data[self.era_column].map(era_to_weight)
                        fit_kwargs['sample_weight'] = train_weights.values

                    # Clean up fit_kwargs
                    fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

                    # Fit the model
                    model.fit(train_data.drop(columns=[self.era_column]), train_targets, **fit_kwargs)
                    self._save_model(model, trained_model_name)
                    self.latest_trained_model_paths[model_name] = model_path

                # Generate predictions for the current test era
                test_predictions = pd.Series(
                    model.predict(test_data.drop(columns=[self.era_column])),
                    index=test_data.index,
                    name=model_name
                )
                combined_predictions[model_name] = test_predictions

            # Store predictions and targets
            combined_predictions = combined_predictions.reindex(test_data.index)
            self.oof_predictions.append(combined_predictions)
            self.oof_targets.append(test_targets)
            self.eras_trained_on.append(test_era)

            iteration += 1  # Increment iteration counter

        # Save final models if required
        if self.final_models_dir is not None and test_era is not None:
            for model_name, model_attrs in self.models_attrs.items():
                model = self._load_model(model_attrs['model_path'])
                final_model_name = f"{model_name}"
                self._save_model(model, final_model_name, is_final=True)
            print(f"Final models saved to {self.final_models_dir}")

        # Compile and evaluate predictions
        for idx, era in enumerate(self.eras_trained_on):
            self.oof_predictions[idx]['era'] = era
        self.all_oof_predictions = pd.concat(self.oof_predictions)
        self.all_oof_targets = pd.concat(self.oof_targets)
        self.benchmark_predictions = benchmark_predictions
        self.evaluate(X_test, y_test)

        return self

    def evaluate(self, X, y):
        evaluator = NumeraiClassicEvaluator(metrics_list=["mean_std_sharpe", "apy", "max_drawdown"],
                                            era_col=self.era_column)
        oof_index = self.all_oof_predictions.index
        eval_data = pd.concat([X.loc[oof_index].drop(columns=[self.era_column]), y.loc[oof_index]], axis=1)
        eval_data['target'] = y.loc[oof_index]
        eval_data = pd.concat([eval_data, self.all_oof_predictions.loc[oof_index]], axis=1)
        benchmark_cols = [col for col in self.benchmark_predictions.columns if col.endswith('_benchmark')]
        eval_data = pd.concat([eval_data, self.benchmark_predictions.loc[oof_index, benchmark_cols]], axis=1)
        pred_cols = [col for col in self.all_oof_predictions.columns if col != 'era'] + benchmark_cols
        eval_data[pred_cols] = eval_data[pred_cols].clip(0, 1)



        # Store evaluation results in self and save to CSV
        self.metrics = evaluator.full_evaluation(
            dataf=eval_data,
            pred_cols=pred_cols,
            target_col='target',
        )
        self.metrics = self.metrics.drop(columns=["target"]).reset_index().rename(
            columns={'index': 'model'}).sort_values(by='mean', ascending=False)

        # Calculate per era correlations and store them in self
        numerai_corr = {}
        for model_name in pred_cols:
            numerai_corr[model_name] = evaluator.per_era_numerai_corrs(
                dataf=eval_data,
                pred_col=model_name,
                target_col='target'
            )
        self.numerai_corr = pd.DataFrame(numerai_corr)

        # Save artifacts if directory is specified
        if self.artifacts_dir is not None:
            os.makedirs(self.artifacts_dir, exist_ok=True)

            # Save numerai_corr to CSV
            numerai_corr_csv_path = os.path.join(self.artifacts_dir, "numerai_corr.csv")
            self.numerai_corr.to_csv(numerai_corr_csv_path, index=True)

            # Save metrics to CSV
            metrics_csv_path = os.path.join(self.artifacts_dir, "metrics.csv")
            self.metrics.to_csv(metrics_csv_path, index=True)

            # Save benchmark predictions (with era column first, target, and all prediction columns) to CSV
            benchmark_predictions_csv_path = os.path.join(self.artifacts_dir, "benchmark_predictions.csv")

            # Combine all prediction columns (including benchmark) with era and target
            all_predictions_with_era_target = pd.concat([self.benchmark_predictions, self.all_oof_predictions], axis=1)
            all_predictions_with_era_target[self.era_column] = X.loc[oof_index, self.era_column]
            all_predictions_with_era_target['target'] = y.loc[oof_index]

            # Reorder columns to place 'era' first
            columns_order = [self.era_column] + [col for col in all_predictions_with_era_target.columns if
                                                 col != self.era_column]
            all_predictions_with_era_target = all_predictions_with_era_target[columns_order]

            all_predictions_with_era_target
            from numerblox.misc import numerai_corr_weighted
            target = all_predictions_with_era_target[all_predictions_with_era_target['era'] == '1112']['target']
            prediction = all_predictions_with_era_target[all_predictions_with_era_target['era'] == '1112']['lgb_model']
            numerai_corr_weighted(target,prediction)

            # Save to CSV
            all_predictions_with_era_target.to_csv(benchmark_predictions_csv_path, index=True)

    def _load_model(self, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        _check_sklearn_compatibility(model)
        return model

    def _save_model(self, model, model_name, is_final=False):
        model_dir = self.final_models_dir if is_final and self.final_models_dir else self.era_models_dir
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model_path

    def validate_arguments(self, X_train, y_train, X_test, y_test):
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(f"X_train must be a pandas DataFrame, got {type(X_train)} instead.")
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError(f"X_test must be a pandas DataFrame, got {type(X_test)} instead.")
        if not isinstance(y_train, pd.Series):
            raise TypeError(f"y_train must be a pandas Series, got {type(y_train)} instead.")
        if not isinstance(y_test, pd.Series):
            raise TypeError(f"y_test must be a pandas Series, got {type(y_test)} instead.")

        # Ensure y_train has no None values
        if y_train.isnull().any():
            raise ValueError("y_train contains None values, but it must not have any.")

        # Ensure y_test has at least one non-None value
        if y_test.isnull().all():
            raise ValueError("y_test contains only None values.")

        # Ensure X_train and X_test have the same number of columns
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                f"X_train and X_test must have the same number of columns. Got {X_train.shape[1]} and {X_test.shape[1]}."
            )

        # Ensure X_train and y_train have the same number of rows
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train must have the same number of rows. Got {X_train.shape[0]} and {y_train.shape[0]}."
            )

        # Ensure X_test and y_test have the same number of rows
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"X_test and y_test must have the same number of rows. Got {X_test.shape[0]} and {y_test.shape[0]}."
            )

        if self.era_column not in X_train.columns:
            raise ValueError(f"era_column '{self.era_column}' not found in X_train columns.")
        if self.era_column not in X_test.columns:
            raise ValueError(f"era_column '{self.era_column}' not found in X_test columns.")

        if not self.models_attrs or len(self.models_attrs) == 0:
            raise ValueError("No models provided. Please check models_attrs.")

        if not isinstance(self.horizon_eras, int) or self.horizon_eras <= 0:
            raise ValueError(f"horizon_eras must be a positive integer, got {self.horizon_eras}.")

        if self.meta is not None:
            if not hasattr(self.meta, 'fit'):
                raise ValueError("Meta model provided must have a 'fit' method for ensemble training.")
            if not hasattr(self.meta, 'meta_eras'):
                raise ValueError("Meta model provided must have a 'meta_eras' attribute.")

        if self.era_models_dir is None or not isinstance(self.era_models_dir, str):
            raise ValueError(f"Invalid era_models_dir. It must be a non-empty string, got {self.era_models_dir}.")

        if self.final_models_dir is not None and not isinstance(self.final_models_dir, str):
            raise ValueError(f"Invalid final_models_dir. It must be a string or None, got {self.final_models_dir}.")

        if self.artifacts_dir is not None and not isinstance(self.artifacts_dir, str):
            raise ValueError(f"Invalid artifacts_dir. It must be a string or None, got {self.artifacts_dir}.")

        # Ensure at least a 4-era difference between the last training era and the first test era
        train_eras = np.sort(X_train[self.era_column].unique())
        test_eras = np.sort(X_test[self.era_column].unique())

        if len(train_eras) < 1 or len(test_eras) < 1:
            raise ValueError("Both training and testing data must contain at least one era.")

        # Check that sample weights (if provided) are identical within each era
        for model_name, model_attrs in self.models_attrs.items():
            sample_weights = model_attrs.get('fit_kwargs', {}).get('sample_weight', None)

            if sample_weights is not None:
                # Verify that all samples in each era have the same weight
                if isinstance(sample_weights, pd.Series):
                    for era in train_eras:
                        era_weights = sample_weights[X_train[self.era_column] == era]
                        if era_weights.nunique() != 1:
                            raise ValueError(
                                f"Sample weights must be identical within each era for model {model_name}, but era {era} has varying weights."
                            )
                elif isinstance(sample_weights, np.ndarray):
                    for era in train_eras:
                        era_weights = sample_weights[X_train[self.era_column] == era]
                        if len(np.unique(era_weights)) != 1:
                            raise ValueError(
                                f"Sample weights must be identical within each era for model {model_name}, but era {era} has varying weights."
                            )
                else:
                    raise ValueError(
                        f"Sample weight for model {model_name} must be a pandas Series or numpy array, got {type(sample_weights)}."
                    )

        # If expand_train is True, ensure no sample weights are provided
        if self.expand_train:
            for model_name, model_attrs in self.models_attrs.items():
                sample_weights = model_attrs.get('fit_kwargs', {}).get('sample_weight', None)
                if sample_weights is not None:
                    raise ValueError(
                        f"Sample weights cannot be provided when expand_train is True, but model {model_name} has sample weights."
                    )

