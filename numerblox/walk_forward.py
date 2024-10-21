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
import re
import logging
import math


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

        self.evaluation_results = None
        self.predictions = None
        self.per_era_numerai_corr = None
        self.meta_weights = {}

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.DEBUG)  # Set the minimum logging level

        # Create handlers
        console_handler = logging.StreamHandler()  # Log to console

        # Get the current date and time
        log_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Log file name with date and time
        file_handler = logging.FileHandler(f'{self.log_dir}/walk_forward_{log_time}.log')

        # Set log level for each handler
        console_handler.setLevel(logging.INFO)  # Console will handle INFO and above
        file_handler.setLevel(logging.INFO)  # File will also handle INFO and above

        # Create a formatter and set it for both handlers
        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def fit(self, X_train, y_train, X_test, y_test, meta_data=None):
        self.validate_arguments(X_train, y_train, X_test, y_test)
        self.logger.info("Starting walk forward")

        unique_train_eras = X_train[self.era_column].unique()
        start_train_era = min(unique_train_eras)
        end_train_era = max(unique_train_eras)
        self.logger.info(f"Train data: Rows: {X_train.shape[0]}, Cols: {X_train.shape[1]}, Eras: {len(unique_train_eras)}, Start era: {start_train_era}, End era: {end_train_era}")

        unique_test_eras = X_test[self.era_column].unique()
        start_test_era = min(unique_test_eras)
        end_test_era = max(unique_test_eras)
        self.logger.info(f"Test data: Rows: {X_test.shape[0]}, Cols: {X_test.shape[1]}, Eras: {len(unique_test_eras)}, Start era: {start_test_era}, End era: {end_test_era}")

        X_train[self.era_column] = X_train[self.era_column].astype(int)
        X_test[self.era_column] = X_test[self.era_column].astype(int)
        train_data = X_train.sort_values(by=self.era_column).copy()
        train_targets = y_train.loc[train_data.index].copy()
        X_test = X_test.sort_values(by=self.era_column).copy()
        y_test = y_test.loc[X_test.index].copy()
        eras_test = sorted(X_test[self.era_column].unique())
        start_test_era = min(eras_test) + self.purge_eras
        if "meta" in X_test:
            meta = X_test['meta']
            X_test = X_test.drop(columns=(['meta']))
        else:
            meta = None

        eras_to_test = [era for era in eras_test if era >= start_test_era]

        iteration = 0
        predictions = pd.DataFrame(index=X_test.index)
        last_era_with_targets = None

        self.oof_dfs = []
        self.latest_trained_meta_model_paths = {}
        self.meta_weights = {}
        oof_pre = []

        self.logger.info(f"Purge eras: {self.purge_eras}, Test prediction eras after purge: {_format_ranges(eras_to_test)}")
        self.logger.info(f"Base Models: {len(self.models)}")

        i = 0
        num_eras_to_test = len(eras_to_test)
        total_iterations = math.ceil(num_eras_to_test / self.step_eras)
        while i < num_eras_to_test:
            eras_batch = eras_to_test[i:i + self.step_eras]
            current_iteration = (i // self.step_eras) + 1  # Calculate the current iteration properly
            percent_complete = (current_iteration / total_iterations) * 100  # Correct percentage calculation

            self.logger.info(f"Iteration {current_iteration} of {total_iterations}, Step: {self.step_eras}, Percent: {int(percent_complete)}%")

            test_data_batch = X_test[X_test[self.era_column].isin(eras_batch)]
            test_targets_batch = y_test.loc[test_data_batch.index]

            if iteration == 0:
                # Base models are trained here and generate predictions for all test eras
                for model_name, model_attrs in self.models.items():
                    model = self._load_model(model_attrs['model_path'])
                    cache_id = [train_data.shape, sorted(train_data.columns.tolist()), model_name,
                                self.purge_eras, model_attrs]
                    cache_hash = get_cache_hash(cache_id)
                    model_path = f"{self.cache_dir}/{model_name}_{cache_hash}_model.pkl"
                    predictions_path = f"{self.cache_dir}/{model_name}_{cache_hash}_predictions.pkl"
                    if os.path.exists(model_path):
                        self.logger.info(f"Load base model cache: {model_name}, Train eras: {_format_ranges(train_data.era.unique())}, Num eras: {len(train_data.era.unique())}")
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        self.latest_trained_model_paths[f"{model_name}_base"] = model_path
                    else:
                        fit_kwargs = model_attrs.get('fit_kwargs', {}).copy()
                        sample_weights = fit_kwargs.get('sample_weight', None)
                        if sample_weights is not None:
                            era_weights = dict(zip(train_data[self.era_column].unique(), sample_weights))
                            train_weights = train_data[self.era_column].map(era_weights)
                            fit_kwargs['sample_weight'] = train_weights.values
                        fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

                        self.logger.info(f"Train base model: {model_name}, Train eras: {_format_ranges(train_data.era.unique())}, Num eras: {len(train_data.era.unique())}")

                        model.fit(train_data.drop(columns=[self.era_column]), train_targets, **fit_kwargs)
                        self.latest_trained_model_paths[f"{model_name}_base"] = model_path
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)

                    # Generate predictions for all test eras
                    test_data_filtered = X_test[X_test[self.era_column].isin(eras_to_test)]
                    if os.path.exists(predictions_path):
                        self.logger.info(f"Load base model predictions cache: {model_name}, Prediction eras: {_format_ranges(test_data_filtered.era.unique())}, Num eras: {len(test_data_filtered.era.unique())}")
                        with open(predictions_path, 'rb') as f:
                            predictions_filtered = pickle.load(f)
                    else:

                        self.logger.info(f"Predict base model: {model_name}, Prediction eras: {_format_ranges(test_data_filtered.era.unique())}, Num eras: {len(test_data_filtered.era.unique())}")

                        predictions_filtered = model.predict(test_data_filtered.drop(columns=[self.era_column]))
                        with open(predictions_path, 'wb') as f:
                            pickle.dump(predictions_filtered, f)
                    predictions.loc[test_data_filtered.index, f'{model_name}_base'] = predictions_filtered

                    # First iteration, base models and era models are identical
                    eras_batch_predictions = predictions_filtered.reindex(test_data_filtered[test_data_filtered['era'].isin(eras_batch)].index)
                    predictions.loc[eras_batch_predictions.index, f'{model_name}'] = eras_batch_predictions

                    if hasattr(model, "target"):
                        horizon = int(int(re.search(r'_(\d+)$', model.target).group(1)) / 5)
                        if self.purge_eras < horizon:
                            raise Exception(f"Model target {model.target} larger than purge eras: {self.purge_eras}")

                    if hasattr(model, "oof"):
                        df = model.oof.copy()
                        df.rename(columns={"predict": f"{model_name}"}, inplace=True)
                        oof_pre.append(df)

                if oof_pre:
                    common_cols = oof_pre[0][['era', 'target']]
                    specific_cols = [df.drop(['era', 'target', 'group'], axis=1) for df in oof_pre]
                    concatenated_specific_cols = pd.concat(specific_cols, axis=1)
                    oof_pre = pd.concat([common_cols, concatenated_specific_cols], axis=1).dropna()
                    oof_pre['era'] = oof_pre.era.astype(int)

                # Create oof_df for the entire test set
                combined_predictions = predictions.loc[test_data_filtered.index]
                oof_df = combined_predictions.copy()

                test_targets_all = y_test.loc[test_data_filtered.index]

                if isinstance(test_targets_all, pd.DataFrame):
                    oof_df['target'] = test_targets_all[self.target_name].squeeze()
                else:
                    oof_df['target'] = test_targets_all.squeeze()

                oof_df['era'] = test_data_filtered[self.era_column]
                self.oof_dfs.append(oof_df)

            else:
                if not self.expand_train:
                    oldest_eras = sorted(train_data[self.era_column].unique())[:self.step_eras]
                    train_data = train_data[~train_data[self.era_column].isin(oldest_eras)]
                    train_targets = train_targets.loc[train_data.index]

                last_train_era = train_data[self.era_column].max()
                next_train_eras = range(last_train_era + 1, last_train_era + 1 + self.step_eras)

                new_train_data = pd.DataFrame()
                new_train_targets = pd.Series()

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

                combined_predictions = pd.DataFrame(index=test_data_batch.index)

                for model_name, model_attrs in self.models.items():
                    cache_id = [train_data.shape, sorted(train_data.columns.tolist()), model_name, self.purge_eras, model_attrs]
                    cache_hash = get_cache_hash(cache_id)
                    era_model_name = f"{model_name}_{eras_batch[-1]}"
                    trained_model_name = f"{era_model_name}_{cache_hash}"
                    model_path = os.path.join(self.era_models_dir, f"{trained_model_name}.pkl")
                    predictions_path = os.path.join(f"{self.cache_dir}/{trained_model_name}_predictions.pkl")

                    if os.path.exists(model_path):
                        self.logger.info(f"Load model cache: {era_model_name}, Train eras: {_format_ranges(train_data.era.unique())}, Num eras: {len(train_data.era.unique())}")
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
                        self.logger.info(f"Train model: {era_model_name}, Train eras: {_format_ranges(train_data.era.unique())}, Num eras: {len(train_data.era.unique())}")
                        model.fit(train_data.drop(columns=[self.era_column]), train_targets, **fit_kwargs)
                        self._save_model(model, trained_model_name)
                        self.latest_trained_model_paths[model_name] = model_path

                    if os.path.exists(predictions_path):
                        self.logger.info(f"Load model predictions cache: {era_model_name}, Prediction eras: {_format_ranges(test_data_batch.era.unique())}, Num eras: {len(test_data_batch.era.unique())}")
                        with open(predictions_path, 'rb') as f:
                            test_predictions = pickle.load(f)
                    else:
                        self.logger.info(f"Predict model: {era_model_name}, Prediction eras: {_format_ranges(test_data_batch.era.unique())}, Num eras: {len(test_data_batch.era.unique())}")
                        test_predictions = pd.Series(model.predict(test_data_batch.drop(columns=[self.era_column])), index=test_data_batch.index, name=model_name)
                        with open(predictions_path, 'wb') as f:
                            pickle.dump(test_predictions, f)
                    combined_predictions[model_name] = test_predictions

                combined_predictions = combined_predictions.reindex(test_data_batch.index)
                oof_df = combined_predictions.copy()
                if isinstance(test_targets_batch, pd.DataFrame):
                    oof_df['target'] = test_targets_batch[self.target_name].squeeze()
                else:
                    oof_df['target'] = test_targets_batch.squeeze()
                oof_df['era'] = test_data_batch[self.era_column]
                self.oof_dfs.append(oof_df)

                if self.meta is not None:
                    oof_data = pd.concat(self.oof_dfs).groupby(level=0).first().sort_values(by='era')
                    last_target_era = eras_batch[-1] - self.purge_eras - 1

                    if len(oof_pre) > 0:
                        oof_all = oof_data.combine_first(oof_pre)
                        oof_all = oof_all.sort_values(by='era', ascending=True)
                        available_eras = [era for era in oof_all['era'].unique() if era <= last_target_era]
                    else:
                        oof_all = oof_data.copy()
                        available_eras = [era for era in oof_data['era'].unique() if era <= last_target_era]

                    for window_size in self.meta.meta_eras:
                        if len(available_eras) >= 1 and len(self.models) > 1:
                            window_eras = available_eras[-window_size:]

                            window_oof_data = oof_all[oof_all['era'].isin(window_eras)]

                            era_model_names = [col for col in window_oof_data.columns if col not in ['target', 'era'] and not col.startswith('meta_model') and not col.endswith('_base')]
                            window_oof_data = window_oof_data[era_model_names + ['era', 'target']]

                            if window_oof_data.target.notnull().any():
                                model_name_to_path = {col: self.latest_trained_model_paths.get(col) for col in era_model_names}
                                for col, model_path in model_name_to_path.items():
                                    if not model_path:
                                        raise ValueError(f"No trained model path found for base model {col}")

                                meta_model = clone(self.meta)
                                meta_model_name = f"meta_model_{window_size}"
                                cache_id = [train_data.shape, sorted(train_data.columns.tolist()), window_eras, window_size,
                                            available_eras, eras_batch, self.purge_eras, self.models, meta_model_name, meta_data,
                                            meta_model.meta_eras, meta_model.max_ensemble_size, meta_model.weight_factor]
                                cache_hash = get_cache_hash(cache_id)
                                trained_meta_model_name = f"{meta_model_name}_{eras_batch[-1]}_{cache_hash}"
                                model_path = os.path.join(self.era_models_dir, f"{trained_meta_model_name}.pkl")
                                predictions_path = os.path.join(f"{self.cache_dir}/{trained_meta_model_name}_predictions.pkl")

                                if os.path.exists(model_path):
                                    self.logger.info(f"Load meta cache {meta_model_name}: OOS Eras: {_format_ranges(window_oof_data.era.unique())}, Num OOS eras: {len(window_oof_data.era.unique())}")
                                    with open(model_path, 'rb') as f:
                                        meta_model = pickle.load(f)
                                else:
                                    self.logger.info(f"Meta {meta_model_name}: OOS Eras: {_format_ranges(window_oof_data.era.unique())}, Num OOS eras: {len(window_oof_data.era.unique())}")

                                    if meta_data is not None:
                                        oof_meta_data = meta_data.reindex(window_oof_data.index)

                                    meta_model.fit(
                                        window_oof_data,
                                        model_name_to_path,
                                        oof_meta_data)
                                    with open(model_path, 'wb') as f:
                                        pickle.dump(meta_model, f)

                                self.latest_trained_meta_model_paths[window_size] = model_path
                                self.meta_weights[trained_meta_model_name] = meta_model.weights_

                                if os.path.exists(predictions_path):
                                    self.logger.info(f"Load meta predictions cache: {meta_model_name}, Prediction eras: {_format_ranges(test_data_batch.era.unique())}, Num eras: {len(test_data_batch.era.unique())}")
                                    with open(predictions_path, 'rb') as f:
                                        meta_predictions = pickle.load(f)
                                else:
                                    self.logger.info(f"Predict meta: {meta_model_name}, Prediction eras: {_format_ranges(test_data_batch.era.unique())}, Num eras: {len(test_data_batch.era.unique())}")
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

                if self.evaluate_per_step:
                    self.oof_data = pd.concat(self.oof_dfs).groupby(level=0).first().sort_values(by='era')
                    self.predictions = predictions
                    self.logger.info(f"Evaluating predictions, Eras: {_format_ranges(range(start_test_era, max(eras_batch) + 1))}, Artifacts: {self.artifacts_dir}")
                    self.evaluate(X_test, y_test)

            iteration += 1
            i += self.step_eras

        if self.final_models_dir is not None:
            for model_name, model_attrs in self.models.items():
                model_path = self.latest_trained_model_paths.get(model_name)
                if model_path:
                    model = self._load_model(model_path)
                    final_model_name = f"{model_name}"
                    self._save_model(model, final_model_name, is_final=True)
                    self.logger.info(f"Final trained model: {model_path} to {self.final_models_dir}/{final_model_name}")

            if self.meta is not None:
                for window_size, meta_model_path in self.latest_trained_meta_model_paths.items():
                    meta_model = self._load_model(meta_model_path)
                    final_meta_model_name = f"meta_model_{window_size}"
                    self._save_model(meta_model, final_meta_model_name, is_final=True)
                    self.logger.info(f"Final meta-model: {self.final_models_dir}/{final_meta_model_name}")

        if self.evaluate_per_step is False:
            self.oof_data = pd.concat(self.oof_dfs).groupby(level=0).first().sort_values(by='era')
            self.predictions = predictions
            self.logger.info(f"Evaluating predictions, Eras: {_format_ranges(range(start_test_era, max(eras_batch) + 1))}, Artifacts: {self.artifacts_dir}")
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
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(f"X_train must be a pandas DataFrame, got {type(X_train)} instead.")
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError(f"X_test must be a pandas DataFrame, got {type(X_test)} instead.")
        # if not isinstance(y_train, pd.Series):
            # raise TypeError(f"y_train must be a pandas Series, got {type(y_train)} instead.")
        # if not isinstance(y_test, pd.Series):
            # raise TypeError(f"y_test must be a pandas Series, got {type(y_test)} instead.")

        # Ensure y_train has no None values
        # if y_train.isnull().any():
            # raise ValueError("y_train contains None values, it must not have any.")

        # Ensure y_test has at least one non-None value
        # if y_test.isnull().all():
            # raise ValueError("y_test contains only None values.")

        # # Ensure X_train and X_test have the same number of columns
        # if X_train.shape[1] != X_test.shape[1]:
        #     raise ValueError(
        #         f"X_train and X_test must have the same number of columns. Got {X_train.shape[1]} and {X_test.shape[1]}."
        #     )

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

        if not self.models or len(self.models) == 0:
            raise ValueError("No models provided. Please check models.")

        if not isinstance(self.purge_eras, int) or self.purge_eras <= 0:
            raise ValueError(f"purge_eras must be a positive integer, got {self.purge_eras}.")

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
        for model_name, model_attrs in self.models.items():
            sample_weights = model_attrs.get('fit_kwargs', {}).get('sample_weight', None)

            if sample_weights is not None:
                # Verify that all samples in each era have the same weight
                if isinstance(sample_weights, pd.Series):
                    for era in train_eras:
                        era_weights = sample_weights[X_train[self.era_column] == era]
                        if era_weights.nunique() != 1:
                            raise ValueError(
                                f"Sample weights must be identical within each era for model {model_name}, era {era} has varying weights."
                            )
                elif isinstance(sample_weights, np.ndarray):
                    for era in train_eras:
                        era_weights = sample_weights[X_train[self.era_column] == era]
                        if len(np.unique(era_weights)) != 1:
                            raise ValueError(
                                f"Sample weights must be identical within each era for model {model_name}, era {era} has varying weights."
                            )
                else:
                    raise ValueError(
                        f"Sample weight for model {model_name} must be a pandas Series or numpy array, got {type(sample_weights)}."
                    )

        # If expand_train is True, ensure no sample weights are provided
        if self.expand_train:
            for model_name, model_attrs in self.models.items():
                sample_weights = model_attrs.get('fit_kwargs', {}).get('sample_weight', None)
                if sample_weights is not None:
                    raise ValueError(
                        f"Sample weights cannot be provided when expand_train is True, model {model_name} has sample weights."
                    )

