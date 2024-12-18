import os
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm
from numerblox.evaluation import NumeraiClassicEvaluator
from numerblox.misc import get_cache_hash
import matplotlib.pyplot as plt
from datetime import datetime
import base64
from io import BytesIO
import re
import logging
import math
import copy
from . import logger


def _check_sklearn_compatibility(model):
    if not hasattr(model, "fit") or not callable(getattr(model, "fit")):
        raise ValueError(f"Model {model} does not implement a 'fit' method.")
    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise ValueError(f"Model {model} does not implement a 'predict' method.")

def _format_ranges(numbers):
    ranges = []
    start = numbers[0]

    for i in range(1, len(numbers)):
        if numbers[i] != numbers[i - 1] + 1:
            end = numbers[i - 1]
            ranges.append(f"{start}-{end}" if start != end else f"{start}")
            start = numbers[i]

    ranges.append(f"{start}-{numbers[-1]}" if start != numbers[-1] else f"{start}")
    return ", ".join(ranges)


class WalkForward(BaseEstimator, RegressorMixin):

    def __init__(self, models, purge_eras=4, era_column="era", meta=None,
                 era_models_dir='tmp/era_models', final_models_dir='tmp/final_models', artifacts_dir='tmp/artifacts',
                 cache_dir='tmp/cache', log_dir='tmp/logs', expand_train=False, evaluate_per_step=True, step_eras=1,
                 target_name="target_cyrusd_20"):
        self.models = models
        self.purge_eras = purge_eras
        self.era_column = era_column
        self.oof_predictions = []
        self.oof_targets = []
        self.eras_trained_on = []
        self.meta = meta
        self.era_models_dir = era_models_dir
        self.final_models_dir = final_models_dir
        self.expand_train = expand_train
        self.artifacts_dir = artifacts_dir
        self.cache_dir = cache_dir
        self.evaluate_per_step = evaluate_per_step
        self.step_eras = step_eras
        self.log_dir = log_dir
        self.target_name = target_name

        self.latest_trained_model_paths = {}

        os.makedirs(self.era_models_dir, exist_ok=True)
        os.makedirs(self.final_models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)


    def fit(self, X_train, y_train, X_test, y_test, meta_data=None):
        # Validate the provided data
        self.validate_arguments(X_train, y_train, X_test, y_test)
        logger.info("Starting walk forward")

        # Get unique train eras and log the details
        unique_train_eras = X_train[self.era_column].unique()
        start_train_era = min(unique_train_eras)
        end_train_era = max(unique_train_eras)
        logger.info(f"Train data: Rows: {X_train.shape[0]}, Cols: {X_train.shape[1]}, Eras: {len(unique_train_eras)}, Start era: {start_train_era}, End era: {end_train_era}")

        # Get unique test eras and log the details
        unique_test_eras = X_test[self.era_column].unique()
        start_test_era = min(unique_test_eras)
        end_test_era = max(unique_test_eras)
        logger.info(f"Test data: Rows: {X_test.shape[0]}, Cols: {X_test.shape[1]}, Eras: {len(unique_test_eras)}, Start era: {start_test_era}, End era: {end_test_era}")

        # Ensure era columns are integers and sort train and test data by era
        X_train[self.era_column] = X_train[self.era_column].astype(int)
        X_test[self.era_column] = X_test[self.era_column].astype(int)
        train_data = X_train.sort_values(by=self.era_column).copy()
        train_targets = y_train.loc[train_data.index].copy()
        X_test = X_test.sort_values(by=self.era_column).copy()
        y_test = y_test.loc[X_test.index].copy()

        # Define the eras for testing, considering the purge period
        eras_test = sorted(X_test[self.era_column].unique())
        start_test_era = min(eras_test) + self.purge_eras

        # Only test eras that start after the purge era
        eras_to_test = [era for era in eras_test if era >= start_test_era]

        iteration = 0
        predictions = pd.DataFrame(index=X_test.index)
        last_era_with_targets = None

        # Initialize lists and dictionaries for storing results
        self.oof_dfs = []
        self.latest_trained_meta_model_paths = {}
        self.meta_weights = {}
        oof_pre = []

        # Log details about the purge period and base models
        logger.info(f"Purge eras: {self.purge_eras}, Test prediction eras after purge: {_format_ranges(eras_to_test)}")
        logger.info(f"Base Models: {len(self.models)}")

        i = 0
        num_eras_to_test = len(eras_to_test)
        total_iterations = math.ceil(num_eras_to_test / self.step_eras)

        # Start the walk-forward iterations
        while i < num_eras_to_test:
            # Define the current batch of eras to test and log iteration progress
            eras_batch = eras_to_test[i:i + self.step_eras]
            current_iteration = (i // self.step_eras) + 1
            percent_complete = (current_iteration / total_iterations) * 100
            logger.info(f"Iteration {current_iteration} of {total_iterations}, Step: {self.step_eras}, Percent: {int(percent_complete)}%, Test eras batch: {_format_ranges(eras_batch)}")

            # Filter test data and targets for the current batch of eras
            test_data_batch = X_test[X_test[self.era_column].isin(eras_batch)]
            test_targets_batch = y_test.loc[test_data_batch.index]

            if iteration == 0:
                # First iteration: train base models and generate predictions for all test eras
                for model_name, model_attrs in self.models.items():
                    # Load or train the base model
                    model = self._load_model(model_attrs['model_path'])
                    cache_id = [train_data.shape, sorted(train_data.columns.tolist()), model_name, self.purge_eras, model_attrs, eras_to_test, X_test.shape]
                    cache_hash = get_cache_hash(cache_id)
                    model_path = f"{self.cache_dir}/{model_name}_{cache_hash}_model.pkl"
                    predictions_path = f"{self.cache_dir}/{model_name}_{cache_hash}_predictions.pkl"

                    # Load model from cache if available
                    if os.path.exists(model_path):
                        logger.info(f"Load base model cache: {model_name}, Train eras: {_format_ranges(train_data.era.unique())}, Num eras: {len(train_data.era.unique())}")
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        self.latest_trained_model_paths[f"{model_name}_base"] = model_path
                    else:
                        # Train the model if not cached
                        fit_kwargs = model_attrs.get('fit_kwargs', {}).copy()
                        sample_weights = fit_kwargs.get('sample_weight', None)
                        if sample_weights is not None:
                            era_weights = dict(zip(train_data[self.era_column].unique(), sample_weights))
                            train_weights = train_data[self.era_column].map(era_weights)
                            fit_kwargs['sample_weight'] = train_weights.values
                        fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

                        logger.info(f"Train base model: {model_name}, Train eras: {_format_ranges(train_data.era.unique())}, Num eras: {len(train_data.era.unique())}")
                        model.fit(train_data.drop(columns=[self.era_column]), train_targets, **fit_kwargs)
                        self.latest_trained_model_paths[f"{model_name}_base"] = model_path
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)

                    # Generate predictions for all test eras
                    test_data_filtered = X_test[X_test[self.era_column].isin(eras_to_test)]
                    if os.path.exists(predictions_path):
                        logger.info(f"Load base model predictions cache: {model_name}, Prediction eras: {_format_ranges(test_data_filtered.era.unique())}, Num eras: {len(test_data_filtered.era.unique())}")
                        with open(predictions_path, 'rb') as f:
                            predictions_filtered = pickle.load(f)
                    else:
                        logger.info(f"Predict base model: {model_name}, Prediction eras: {_format_ranges(test_data_filtered.era.unique())}, Num eras: {len(test_data_filtered.era.unique())}")
                        predictions_filtered = model.predict(test_data_filtered.drop(columns=[self.era_column]))
                        with open(predictions_path, 'wb') as f:
                            pickle.dump(predictions_filtered, f)

                    # Store predictions and handle first iteration logic
                    predictions.loc[test_data_filtered.index, f'{model_name}_base'] = predictions_filtered
                    eras_batch_predictions = predictions_filtered.reindex(test_data_filtered[test_data_filtered['era'].isin(eras_batch)].index)
                    predictions.loc[eras_batch_predictions.index, f'{model_name}'] = eras_batch_predictions

                    # Handle model-specific settings
                    if hasattr(model, "target"):
                        horizon = int(int(re.search(r'_(\d+)$', model.target).group(1)) / 5)
                        if self.purge_eras < horizon:
                            raise Exception(f"Model target {model.target} larger than purge eras: {self.purge_eras}")

                    if hasattr(model, "oof"):
                        df = model.oof.copy()
                        df.rename(columns={"predict": f"{model_name}"}, inplace=True)
                        oof_pre.append(df)

                # Combine out-of-fold predictions if available
                if oof_pre:
                    common_cols = oof_pre[0][['era', 'target']]
                    specific_cols = [df.drop(['era', 'target', 'group'], axis=1) for df in oof_pre]
                    concatenated_specific_cols = pd.concat(specific_cols, axis=1)
                    oof_pre = pd.concat([common_cols, concatenated_specific_cols], axis=1).dropna()
                    oof_pre['era'] = oof_pre.era.astype(int)

                # Create the out-of-fold DataFrame for the entire test set
                combined_predictions = predictions.loc[test_data_filtered.index]
                oof_df = combined_predictions.copy()

                # Set target values in the out-of-fold DataFrame
                test_targets_all = y_test.loc[test_data_filtered.index]
                if isinstance(test_targets_all, pd.DataFrame):
                    oof_df['target'] = test_targets_all[self.target_name].squeeze()
                else:
                    oof_df['target'] = test_targets_all.squeeze()

                oof_df['era'] = test_data_filtered[self.era_column]
                self.oof_dfs.append(oof_df)

            else:
                # Iteration for expanding or sliding window of training data
                if not self.expand_train:
                    oldest_eras = sorted(train_data[self.era_column].unique())[:self.step_eras]
                    train_data = train_data[~train_data[self.era_column].isin(oldest_eras)]
                    train_targets = train_targets.loc[train_data.index]

                # Add new eras to the training set
                last_train_era = train_data[self.era_column].max()
                next_train_eras = range(last_train_era + 1, last_train_era + 1 + self.step_eras)

                new_train_data = pd.DataFrame()
                new_train_targets = pd.Series()

                # Retrieve new train data and targets for the next eras
                for next_train_era in next_train_eras:
                    if next_train_era in X_train[self.era_column].unique():
                        era_data = X_train[X_train[self.era_column] == next_train_era]
                        era_targets = y_train.loc[era_data.index]
                    elif next_train_era in X_test[self.era_column].unique():
                        era_data = X_test[X_test[self.era_column] == next_train_era]
                        era_targets = y_test.loc[era_data.index]
                    else:
                        continue

                    new_train_data = pd.concat([new_train_data, era_data])
                    new_train_targets = pd.concat([new_train_targets, era_targets])

                if not new_train_data.empty:
                    train_data = pd.concat([train_data, new_train_data])
                    train_targets = pd.concat([train_targets, new_train_targets])

                # Make predictions for the current test batch
                combined_predictions = pd.DataFrame(index=test_data_batch.index)

                for model_name, model_attrs in self.models.items():
                    # Train or load models for the current era
                    cache_id = [train_data.shape, sorted(train_data.columns.tolist()), model_name, self.purge_eras, model_attrs, test_data_batch.shape]
                    cache_hash = get_cache_hash(cache_id)
                    era_model_name = f"{model_name}_{eras_batch[-1]}"
                    trained_model_name = f"{era_model_name}_{cache_hash}"
                    model_path = os.path.join(self.era_models_dir, f"{trained_model_name}.pkl")
                    predictions_path = os.path.join(f"{self.cache_dir}/{trained_model_name}_predictions.pkl")

                    if os.path.exists(model_path):
                        logger.info(f"Load model cache: {era_model_name}, Train eras: {_format_ranges(train_data.era.unique())}, Num eras: {len(train_data.era.unique())}")
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        self.latest_trained_model_paths[model_name] = model_path
                    else:
                        model = self._load_model(model_attrs['model_path'])
                        fit_kwargs = model_attrs.get('fit_kwargs', {}).copy()

                        sample_weights = fit_kwargs.get('sample_weight', None)
                        if sample_weights is not None:
                            era_weights = dict(zip(train_data[self.era_column].unique(), sample_weights))
                            train_weights = train_data[self.era_column].map(era_weights)
                            fit_kwargs['sample_weight'] = train_weights.values

                        fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}
                        logger.info(f"Train model: {era_model_name}, Train eras: {_format_ranges(train_data.era.unique())}, Num eras: {len(train_data.era.unique())}")

                        model.fit(train_data.drop(columns=[self.era_column]), train_targets, **fit_kwargs)

                        self._save_model(model, trained_model_name)
                        self.latest_trained_model_paths[model_name] = model_path

                    # Generate predictions for the current test batch
                    if os.path.exists(predictions_path):
                        logger.info(f"Load model predictions cache: {era_model_name}, Prediction eras: {_format_ranges(test_data_batch.era.unique())}, Num eras: {len(test_data_batch.era.unique())}")
                        with open(predictions_path, 'rb') as f:
                            test_predictions = pickle.load(f)
                    else:
                        logger.info(f"Predict model: {era_model_name}, Prediction eras: {_format_ranges(test_data_batch.era.unique())}, Num eras: {len(test_data_batch.era.unique())}")
                        test_predictions = pd.Series(model.predict(test_data_batch.drop(columns=[self.era_column])), index=test_data_batch.index, name=model_name)
                        with open(predictions_path, 'wb') as f:
                            pickle.dump(test_predictions, f)
                    combined_predictions[model_name] = test_predictions

                # Store the combined predictions
                combined_predictions = combined_predictions.reindex(test_data_batch.index)
                oof_df = combined_predictions.copy()
                if isinstance(test_targets_batch, pd.DataFrame):
                    oof_df['target'] = test_targets_batch[self.target_name].squeeze()
                else:
                    oof_df['target'] = test_targets_batch.squeeze()
                oof_df['era'] = test_data_batch[self.era_column]
                self.oof_dfs.append(oof_df)

                # If a meta model is provided, perform meta training
                if self.meta is not None:
                    oof_data = pd.concat(self.oof_dfs).groupby(level=0).first().sort_values(by='era')
                    last_target_era = eras_batch[-1] - self.purge_eras - 1

                    # Handle out-of-fold and available eras for meta training
                    if len(oof_pre) > 0:
                        oof_all = oof_data.combine_first(oof_pre)
                        oof_all = oof_all.sort_values(by='era', ascending=True)
                        available_eras = [era for era in oof_all['era'].unique() if era <= last_target_era]
                    else:
                        oof_all = oof_data.copy()
                        available_eras = [era for era in oof_data['era'].unique() if era <= last_target_era]

                    # Iterate through each window size for meta training
                    for window_size in self.meta.meta_eras:
                        if len(available_eras) >= 1 and len(self.models) > 1:
                            window_eras = available_eras[-window_size:]
                            window_oof_data = oof_all[oof_all['era'].isin(window_eras)]

                            # Filter the data for the current window size
                            era_model_names = [col for col in window_oof_data.columns if col not in ['target', 'era'] and not col.startswith('meta_model') and not col.endswith('_base')]
                            window_oof_data = window_oof_data[era_model_names + ['era', 'target']]

                            # Perform meta model fitting if there is target data available
                            if window_oof_data.target.notnull().any():
                                model_name_to_path = {col: self.latest_trained_model_paths.get(col) for col in era_model_names}
                                for col, model_path in model_name_to_path.items():
                                    if not model_path:
                                        raise ValueError(f"No trained model path found for base model {col}")

                                # All oof data must have target values and aligned
                                window_oof_data = window_oof_data.dropna(subset=['target']).copy()
                                oof_meta_data = None
                                if meta_data is not None:
                                    oof_meta_data = meta_data.reindex(window_oof_data.index)
                                    if meta_data.isna().any():
                                        raise Exception(f"Meta data not available for historical window")

                                meta_model = copy.deepcopy(self.meta)
                                meta_model_name = f"meta_model_{window_size}"
                                cache_id = [train_data.shape, sorted(train_data.columns.tolist()), window_eras, window_size, available_eras, eras_batch, self.purge_eras, self.models,
                                            meta_model_name, oof_meta_data, meta_model.meta_eras, meta_model.max_ensemble_size, meta_model.weight_factor, meta_model.metric]
                                cache_hash = get_cache_hash(cache_id)
                                trained_meta_model_name = f"{meta_model_name}_{eras_batch[-1]}_{cache_hash}"
                                model_path = os.path.join(self.era_models_dir, f"{trained_meta_model_name}.pkl")
                                predictions_path = os.path.join(f"{self.cache_dir}/{trained_meta_model_name}_predictions.pkl")

                                # Load or train the meta model
                                if os.path.exists(model_path):
                                    logger.info(f"Load meta cache {meta_model_name}: OOS Eras: {_format_ranges(window_oof_data.era.unique())}, Num OOS eras: {len(window_oof_data.era.unique())}")
                                    with open(model_path, 'rb') as f:
                                        meta_model = pickle.load(f)
                                else:
                                    logger.info(f"Meta {meta_model_name}: OOS Eras: {_format_ranges(window_oof_data.era.unique())}, Num OOS eras: {len(window_oof_data.era.unique())}")
                                    meta_model.fit(window_oof_data, model_name_to_path, oof_meta_data)
                                    with open(model_path, 'wb') as f:
                                        pickle.dump(meta_model, f)

                                # Store the meta model paths and weights
                                self.latest_trained_meta_model_paths[window_size] = model_path
                                self.meta_weights[trained_meta_model_name] = meta_model.weights_

                                # Generate meta model predictions
                                if os.path.exists(predictions_path):
                                    logger.info(f"Load meta predictions cache: {meta_model_name}, Prediction eras: {_format_ranges(test_data_batch.era.unique())}, Num eras: {len(test_data_batch.era.unique())}")
                                    with open(predictions_path, 'rb') as f:
                                        meta_predictions = pickle.load(f)
                                else:
                                    logger.info(f"Predict meta: {meta_model_name}, Prediction eras: {_format_ranges(test_data_batch.era.unique())}, Num eras: {len(test_data_batch.era.unique())}")
                                    meta_predictions = meta_model.predict(test_data_batch.drop(columns=[self.era_column]))
                                    with open(predictions_path, 'wb') as f:
                                        pickle.dump(meta_predictions, f)

                                combined_predictions[meta_model_name] = meta_predictions
                                oof_df = combined_predictions.copy()
                                if isinstance(test_targets_batch, pd.DataFrame):
                                    oof_df['target'] = test_targets_batch[self.target_name].squeeze()
                                else:
                                    oof_df['target'] = test_targets_batch.squeeze()
                                oof_df['era'] = test_data_batch[self.era_column]
                                self.oof_dfs[-1] = oof_df

                # Evaluate the predictions per step if the flag is enabled
                if self.evaluate_per_step:
                    logger.info(f"Evaluating predictions, Eras: {_format_ranges(range(start_test_era, max(eras_batch) + 1))}, Artifacts: {self.artifacts_dir}")
                    print("start")
                    self.oof_data = pd.concat(self.oof_dfs).groupby(level=0).first().sort_values(by='era')
                    print("done")
                    self.predictions = predictions
                    print("start evaluate")
                    self.evaluate(X_test, y_test)

            # Move to the next step of eras
            iteration += 1
            i += self.step_eras

        # Save final models and meta-models if final_models_dir is set
        if self.final_models_dir is not None:
            for model_name, model_attrs in self.models.items():
                model_path = self.latest_trained_model_paths.get(model_name)
                if model_path:
                    model = self._load_model(model_path)
                    final_model_name = f"{model_name}"
                    self._save_model(model, final_model_name, is_final=True)
                    logger.info(f"Final trained model: {model_path} to {self.final_models_dir}/{final_model_name}")

            if self.meta is not None:
                for window_size, meta_model_path in self.latest_trained_meta_model_paths.items():
                    meta_model = self._load_model(meta_model_path)
                    final_meta_model_name = f"meta_model_{window_size}"
                    self._save_model(meta_model, final_meta_model_name, is_final=True)
                    logger.info(f"Final meta-model: {self.final_models_dir}/{final_meta_model_name}")

        # Evaluate the final predictions if evaluation was not done per step
        if self.evaluate_per_step is False:
            self.oof_data = pd.concat(self.oof_dfs).groupby(level=0).first().sort_values(by='era')
            self.predictions = predictions
            logger.info(f"Evaluating predictions, Eras: {_format_ranges(range(start_test_era, max(eras_batch) + 1))}, Artifacts: {self.artifacts_dir}")
            self.evaluate(X_test, y_test)

        return self

    def evaluate(self, X, y):
        evaluator = NumeraiClassicEvaluator(
            metrics_list=["mean_std_sharpe", "apy", "max_drawdown"],
            era_col=self.era_column
        )

        # Ensure the oof_data and predictions are aligned with the test data
        oof_index = self.oof_data.index
        eval_data = X.loc[oof_index].drop(columns=[self.era_column]).copy()
        # eval_data['target'] = y.loc[oof_index]

        # Merge out-of-fold predictions with eval_data
        eval_data = pd.concat([eval_data, self.oof_data.loc[oof_index]], axis=1)

        # Collect base model columns and ensure they exist in both oof_data and predictions
        base_cols = [col for col in self.predictions.columns if col.endswith('_base')]
        # eval_data = pd.concat([eval_data, self.predictions.loc[oof_index, base_cols]], axis=1)

        # Collect all prediction columns (excluding 'era') to clip
        pred_cols = [col for col in self.oof_data.columns if col not in ['era', 'target']]

        # Ensure eval_data contains the prediction columns before clipping
        valid_pred_cols = eval_data.columns.intersection(pred_cols)

        # Clip predictions between 0 and 1, only for valid columns
        eval_data[valid_pred_cols] = eval_data[valid_pred_cols].clip(0, 1)

        # Store evaluation results in self and save to CSV
        self.metrics = evaluator.full_evaluation(
            dataf=eval_data,
            pred_cols=valid_pred_cols,
            target_col='target',
        ).drop(columns=["target"]).reset_index().rename(
            columns={'index': 'model'}
        ).sort_values(by='mean', ascending=False)

        # Calculate per era correlations and store them in self
        numerai_corr = {
            model_name: evaluator.per_era_numerai_corrs(
                dataf=eval_data,
                pred_col=model_name,
                target_col='target'
            ) for model_name in valid_pred_cols
        }
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

            # Combine all prediction columns (including base) with era and target
            all_predictions_with_era_target = pd.concat([self.predictions, self.oof_data], axis=1)
            all_predictions_with_era_target[self.era_column] = X.loc[oof_index, self.era_column]

            # Reorder columns to place 'era' first
            columns_order = [self.era_column] + [col for col in all_predictions_with_era_target.columns if
                                                 col != self.era_column]
            all_predictions_with_era_target = all_predictions_with_era_target[columns_order]

            # Save to CSV
            predictions_csv_path = os.path.join(self.artifacts_dir, "predictions.csv")
            all_predictions_with_era_target.to_csv(predictions_csv_path, index=True)

            # Save meta weights to CSV
            meta_weights_csv_path = os.path.join(self.artifacts_dir, "meta_weights_per_era.csv")
            meta_weights_data = pd.DataFrame(self.meta_weights).T  # Transpose to have model names as columns
            meta_weights_data.to_csv(meta_weights_csv_path, index=True)

        return self

    def _load_model(self, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def _save_model(self, model, model_name, is_final=False):
        model_dir = self.final_models_dir if is_final and self.final_models_dir else self.era_models_dir
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model_path

    def validate_arguments(self, X_train, y_train, X_test, y_test):
        # Validate that inputs are pandas DataFrames
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(f"X_train must be a pandas DataFrame, got {type(X_train)} instead.")
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError(f"X_test must be a pandas DataFrame, got {type(X_test)} instead.")

        # Validate that target inputs (y_train, y_test) are either pandas Series or DataFrames
        if not isinstance(y_train, (pd.Series, pd.DataFrame)):
            raise TypeError(f"y_train must be a pandas Series or DataFrame, got {type(y_train)} instead.")
        if not isinstance(y_test, (pd.Series, pd.DataFrame)):
            raise TypeError(f"y_test must be a pandas Series or DataFrame, got {type(y_test)} instead.")

        # Ensure X_train and y_train have the same number of rows
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"X_train and y_train must have the same number of rows. Got {X_train.shape[0]} and {y_train.shape[0]}.")

        # Ensure X_test and y_test have the same number of rows
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(f"X_test and y_test must have the same number of rows. Got {X_test.shape[0]} and {y_test.shape[0]}.")

        # Ensure the era_column is present in both X_train and X_test
        if self.era_column not in X_train.columns:
            raise ValueError(f"era_column '{self.era_column}' not found in X_train columns.")
        if self.era_column not in X_test.columns:
            raise ValueError(f"era_column '{self.era_column}' not found in X_test columns.")

        # Ensure purge_eras is a positive integer
        if not isinstance(self.purge_eras, int) or self.purge_eras <= 0:
            raise ValueError(f"purge_eras must be a positive integer, got {self.purge_eras}.")

        # Validate models are provided
        if not self.models or len(self.models) == 0:
            raise ValueError("No models provided. Please check models.")

        # Validate meta model, if provided
        if self.meta is not None:
            if not hasattr(self.meta, 'fit'):
                raise ValueError("Meta model provided must have a 'fit' method for ensemble training.")
            if not hasattr(self.meta, 'meta_eras'):
                raise ValueError("Meta model provided must have a 'meta_eras' attribute.")

        # Validate directories
        if not isinstance(self.era_models_dir, str) or not self.era_models_dir:
            raise ValueError(f"Invalid era_models_dir. It must be a non-empty string, got {self.era_models_dir}.")

        if self.final_models_dir is not None and not isinstance(self.final_models_dir, str):
            raise ValueError(f"Invalid final_models_dir. It must be a string or None, got {self.final_models_dir}.")

        if self.artifacts_dir is not None and not isinstance(self.artifacts_dir, str):
            raise ValueError(f"Invalid artifacts_dir. It must be a string or None, got {self.artifacts_dir}.")

        train_eras = np.sort(X_train[self.era_column].unique())
        test_eras = np.sort(X_test[self.era_column].unique())
        if len(train_eras) < 1 or len(test_eras) < 1:
            raise ValueError("Both training and testing data must contain at least one era.")

        # Validate sample weights per era, if provided
        for model_name, model_attrs in self.models.items():
            sample_weights = model_attrs.get('fit_kwargs', {}).get('sample_weight', None)
            if sample_weights is not None:
                if isinstance(sample_weights, pd.Series):
                    for era in train_eras:
                        era_weights = sample_weights[X_train[self.era_column] == era]
                        if era_weights.nunique() != 1:
                            raise ValueError(f"Sample weights must be identical within each era for model {model_name}, era {era} has varying weights.")
                elif isinstance(sample_weights, np.ndarray):
                    for era in train_eras:
                        era_weights = sample_weights[X_train[self.era_column] == era]
                        if len(np.unique(era_weights)) != 1:
                            raise ValueError(f"Sample weights must be identical within each era for model {model_name}, era {era} has varying weights.")
                else:
                    raise ValueError(f"Sample weight for model {model_name} must be a pandas Series or numpy array, got {type(sample_weights)}.")

        # Ensure that sample weights are not provided when expand_train is True
        if self.expand_train:
            for model_name, model_attrs in self.models.items():
                sample_weights = model_attrs.get('fit_kwargs', {}).get('sample_weight', None)
                if sample_weights is not None:
                    raise ValueError(f"Sample weights cannot be provided when expand_train is True, model {model_name} has sample weights.")

