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
        self.predictions = None
        self.per_era_numerai_corr = None
        self.meta_weights = {}

    def fit(self, X_train, y_train, X_test, y_test):
        self.validate_arguments(X_train, y_train, X_test, y_test)

        X_train[self.era_column] = X_train[self.era_column].astype(int)
        X_test[self.era_column] = X_test[self.era_column].astype(int)
        train_data = X_train.sort_values(by=self.era_column).copy()
        train_targets = y_train.loc[train_data.index].copy()
        X_test = X_test.sort_values(by=self.era_column).copy()
        y_test = y_test.loc[X_test.index].copy()
        eras_test = sorted(X_test[self.era_column].unique())
        start_test_era = min(eras_test) + self.horizon_eras

        eras_to_test = [era for era in eras_test if era >= start_test_era]

        iteration = 0
        predictions = pd.DataFrame(index=X_test.index)
        last_era_with_targets = None

        self.oof_dfs = []
        self.latest_trained_model_paths = {}
        self.latest_trained_meta_model_paths = {}
        self.meta_weights = {}
        self.oof_data = None

        for test_era in tqdm(eras_to_test, desc="Processing eras"):
            test_data = X_test[X_test[self.era_column] == test_era]
            test_targets = y_test.loc[test_data.index]

            if not test_targets.notnull().any():
                print(f"Era {test_era} has no valid targets; skipping.")
                continue

            last_era_with_targets = test_era

            if iteration == 0:
                print("Generating base model predictions for all base models.")
                for model_name, model_attrs in self.models_attrs.items():
                    model = self._load_model(model_attrs['model_path'])
                    test_data_filtered = X_test[X_test[self.era_column].isin(eras_to_test)]
                    predictions_filtered = model.predict(test_data_filtered.drop(columns=[self.era_column]))
                    predictions.loc[test_data_filtered.index, f'{model_name}_base'] = predictions_filtered


            if iteration > 0:
                if not self.expand_train:
                    oldest_era = train_data[self.era_column].min()
                    train_data = train_data[train_data[self.era_column] != oldest_era]
                    train_targets = train_targets.loc[train_data.index]

                last_train_era = train_data[self.era_column].max()
                next_train_era = last_train_era + 1

                if next_train_era in X_train[self.era_column].unique():
                    new_train_data = X_train[X_train[self.era_column] == next_train_era]
                    new_train_targets = y_train.loc[new_train_data.index]
                elif next_train_era in X_test[self.era_column].unique():
                    new_train_data = X_test[X_test[self.era_column] == next_train_era]
                    new_train_targets = y_test.loc[new_train_data.index]
                else:
                    new_train_data = pd.DataFrame()
                    new_train_targets = pd.Series()

                if not new_train_data.empty:
                    train_data = pd.concat([train_data, new_train_data])
                    train_targets = pd.concat([train_targets, new_train_targets])

            last_train_era = train_data[self.era_column].max()
            gap = test_era - last_train_era

            if gap <= self.horizon_eras:
                raise ValueError(
                    f"Gap between last training era ({last_train_era}) and test era ({test_era}) is {gap}, expected > {self.horizon_eras}")

            print(f"Iteration {iteration + 1}")
            train_eras_unique = train_data[self.era_column].unique()
            print(
                f"Training eras: {train_eras_unique.min()} - {train_eras_unique.max()} ({len(train_eras_unique)} eras)")
            print(f"Testing era: {test_era}")
            combined_predictions = pd.DataFrame(index=test_data.index)

            for model_name, model_attrs in self.models_attrs.items():
                cache_id = [train_data.shape, sorted(train_data.columns.tolist()), test_era, model_name,
                            self.horizon_eras, model_attrs]
                cache_hash = get_cache_hash(cache_id)
                trained_model_name = f"{model_name}_{test_era}_{cache_hash}"
                model_path = os.path.join(self.era_models_dir, f"{trained_model_name}.pkl")

                if os.path.exists(model_path):
                    print(f"Loading cached model for {model_name} in era {test_era}")
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    self.latest_trained_model_paths[model_name] = model_path
                else:
                    print(f"Training new model {model_name} for test era {test_era}")
                    model = self._load_model(model_attrs['model_path'])
                    fit_kwargs = model_attrs.get('fit_kwargs', {}).copy()

                    sample_weights = fit_kwargs.get('sample_weight', None)
                    if sample_weights is not None:
                        era_weights = dict(zip(train_data[self.era_column].unique(), sample_weights))
                        train_weights = train_data[self.era_column].map(era_weights)
                        fit_kwargs['sample_weight'] = train_weights.values

                    fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

                    model.fit(train_data.drop(columns=[self.era_column]), train_targets, **fit_kwargs)
                    self._save_model(model, trained_model_name)
                    self.latest_trained_model_paths[model_name] = model_path

                test_predictions = pd.Series(model.predict(test_data.drop(columns=[self.era_column])),
                                             index=test_data.index, name=model_name)
                combined_predictions[model_name] = test_predictions

            combined_predictions = combined_predictions.reindex(test_data.index)
            oof_df = combined_predictions.copy()
            oof_df['target'] = test_targets
            oof_df['era'] = test_data[self.era_column]
            self.oof_dfs.append(oof_df)

            if self.meta is not None:
                self.oof_data = pd.concat(self.oof_dfs)
                last_target_era = test_era - self.horizon_eras - 1
                available_eras = [era for era in self.oof_data['era'].unique() if era <= last_target_era]
                for window_size in self.meta.meta_eras:
                    if len(available_eras) >= window_size:
                        window_eras = available_eras[-window_size:]
                        print(f"Meta model window size: {window_size}")
                        if len(window_eras) == 1:
                            print(f"Meta model window era: {window_eras[0]}")
                        else:
                            print(f"Meta model window eras: {min(window_eras)} - {max(window_eras)}")
                        print(f"Prediction test era: {test_era}")

                        window_oof_data = self.oof_data[self.oof_data['era'].isin(window_eras)]

                        base_model_columns = [col for col in window_oof_data.columns if
                                              col not in ['target', 'era'] and not col.startswith('meta_model')]
                        base_models_predictions = window_oof_data[base_model_columns]
                        true_targets = window_oof_data['target']
                        eras_series = window_oof_data['era']

                        if true_targets.notnull().any():
                            model_name_to_path = {col: self.latest_trained_model_paths.get(col)
                                                  for col in base_model_columns}
                            for col, model_path in model_name_to_path.items():
                                if not model_path:
                                    raise ValueError(f"No trained model path found for base model {col}")

                            meta_model = clone(self.meta)
                            meta_model_name = f"meta_model_{window_size}"
                            cache_id = [window_eras, window_size, test_era, self.horizon_eras]
                            cache_hash = get_cache_hash(cache_id)
                            trained_meta_model_name = f"{meta_model_name}_{test_era}_{cache_hash}"
                            model_path = os.path.join(self.era_models_dir, f"{trained_meta_model_name}.pkl")

                            if os.path.exists(model_path):
                                with open(model_path, 'rb') as f:
                                    meta_model = pickle.load(f)
                            else:
                                combined_df = base_models_predictions.copy()
                                combined_df['target'] = true_targets
                                combined_df['era'] = eras_series
                                meta_model.fit(
                                    base_models_predictions,
                                    combined_df['target'],
                                    combined_df['era'],
                                    model_name_to_path, train_data.drop(columns=[self.era_column]), train_targets)
                                with open(model_path, 'wb') as f:
                                    pickle.dump(meta_model, f)

                            self.latest_trained_meta_model_paths[window_size] = model_path
                            self.meta_weights[trained_meta_model_name] = meta_model.weights_

                            meta_predictions = meta_model.predict(test_data.drop(columns=[self.era_column]))
                            combined_predictions[meta_model_name] = meta_predictions

                            oof_df = combined_predictions.copy()
                            oof_df['target'] = test_targets
                            oof_df['era'] = test_data[self.era_column]
                            self.oof_dfs[-1] = oof_df

            iteration += 1

        if self.final_models_dir is not None and last_era_with_targets is not None:
            print(f"Saving final models to {self.final_models_dir}")
            # Save the latest retrained base models
            for model_name, model_attrs in self.models_attrs.items():
                model_path = self.latest_trained_model_paths.get(model_name)
                if model_path:
                    model = self._load_model(model_path)
                    final_model_name = f"{model_name}"
                    self._save_model(model, final_model_name, is_final=True)
                else:
                    print(f"No retrained model found for {model_name}")

            # Save the latest meta models
            if self.meta is not None:
                for window_size, meta_model_path in self.latest_trained_meta_model_paths.items():
                    meta_model = self._load_model(meta_model_path)
                    final_meta_model_name = f"meta_model_{window_size}"
                    self._save_model(meta_model, final_meta_model_name, is_final=True)

        self.oof_data = pd.concat(self.oof_dfs)
        self.predictions = predictions
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
        eval_data = pd.concat([eval_data, self.predictions.loc[oof_index, base_cols]], axis=1)

        # Collect all prediction columns (excluding 'era') to clip
        pred_cols = [col for col in self.oof_data.columns if col not in ['era', 'target']] + base_cols

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
            all_predictions_with_era_target['target'] = y.loc[oof_index]

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
            raise ValueError("y_train contains None values, it must not have any.")

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
            for model_name, model_attrs in self.models_attrs.items():
                sample_weights = model_attrs.get('fit_kwargs', {}).get('sample_weight', None)
                if sample_weights is not None:
                    raise ValueError(
                        f"Sample weights cannot be provided when expand_train is True, model {model_name} has sample weights."
                    )

