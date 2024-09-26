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
                 era_models_dir='tmp/era_models', final_models_dir='tmp/final_models', create_report=True,
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
        self.create_report = create_report
        self.expand_train = expand_train

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

        eras_test = sorted(X_test[self.era_column].unique())
        train_data = X_train.copy()
        train_targets = y_train.copy()

        benchmark_predictions = pd.DataFrame(index=X_test.index)
        test_data_no_era = X_test.drop(columns=[self.era_column])

        for model_name, model_attrs in self.models_attrs.items():
            model = self._load_model(model_attrs['model_path'])
            benchmark_predictions[f'{model_name}_benchmark'] = model.predict(test_data_no_era)

        last_era_with_targets = None

        for test_era in tqdm(eras_test, desc="Walk-forward training"):
            test_data = X_test[X_test[self.era_column] == test_era]
            test_targets = y_test.loc[test_data.index]

            if not test_targets.notnull().any():
                print(f"Era {test_era} has no targets; skipping.")
                continue

            last_era_with_targets = test_era

            if train_data.empty or test_data.empty:
                raise ValueError(f"Empty training or testing data for era {test_era}. Please check your data!")

            combined_predictions = pd.DataFrame(index=test_data.index)

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
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    self.latest_trained_model_paths[model_name] = model_path
                else:
                    model = self._load_model(model_attrs['model_path'])
                    fit_kwargs = model_attrs.get('fit_kwargs', {}).copy()

                    sample_weights = fit_kwargs.get('sample_weight', None)
                    if sample_weights is not None:
                        train_eras = train_data[self.era_column].unique()

                        if isinstance(sample_weights, pd.Series):
                            era_to_weight = {era: sample_weights.iloc[i] for i, era in enumerate(train_eras)}
                        else:
                            era_to_weight = {era: sample_weights[i] for i, era in enumerate(train_eras)}

                        train_weights = train_data[self.era_column].map(era_to_weight)
                        fit_kwargs['sample_weight'] = train_weights.values

                    fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

                    model.fit(train_data.drop(columns=[self.era_column]), train_targets, **fit_kwargs)
                    self._save_model(model, trained_model_name)
                    self.latest_trained_model_paths[model_name] = model_path

                test_predictions = pd.Series(list(model.predict(test_data.drop(columns=[self.era_column]))),
                                             index=test_data.index, name=model_name)
                combined_predictions[model_name] = test_predictions

            combined_predictions = combined_predictions.reindex(test_data.index)
            self.oof_predictions.append(combined_predictions)
            self.oof_targets.append(test_targets)
            self.eras_trained_on.append(test_era)

            if self.meta is not None:
                for window_size in self.meta.meta_eras:
                    if len(self.oof_predictions) >= (window_size + self.horizon_eras):
                        recent_oof_preds = pd.concat(
                            self.oof_predictions[-(window_size + self.horizon_eras):-self.horizon_eras])
                        recent_oof_targets = pd.concat(
                            self.oof_targets[-(window_size + self.horizon_eras):-self.horizon_eras])
                        recent_eras_list = self.eras_trained_on[
                                           -(window_size + self.horizon_eras):-self.horizon_eras]
                        recent_eras = pd.Series(np.concatenate(
                            [[era] * len(pred) for era, pred in
                             zip(recent_eras_list,
                                 self.oof_predictions[-(window_size + self.horizon_eras):-self.horizon_eras])]))

                        base_model_columns = [col for col in recent_oof_preds.columns if
                                              not col.startswith('meta_model')]
                        base_models_predictions = recent_oof_preds[base_model_columns]

                        if recent_oof_targets.notnull().any():
                            true_targets = recent_oof_targets

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
                                with open(model_path, 'rb') as f:
                                    meta_model = pickle.load(f)
                            else:
                                meta_model.fit(base_models_predictions, true_targets, recent_eras,
                                               model_name_to_path,
                                               train_data.drop(columns=[self.era_column]), train_targets)
                                with open(model_path, 'wb') as f:
                                    pickle.dump(meta_model, f)

                            self.meta_weights[trained_meta_model_name] = meta_model.weights_

                            meta_predictions = meta_model.predict(test_data.drop(columns=[self.era_column]))
                            combined_predictions[meta_model_name] = meta_predictions

            if self.expand_train:
                train_data = pd.concat([train_data, test_data])
                train_targets = pd.concat([train_targets, test_targets])
            else:
                eras_in_train = train_data[self.era_column].unique()

                if len(eras_in_train) > 0:
                    oldest_era = eras_in_train[0]
                    keep_indices = train_data[train_data[self.era_column] != oldest_era].index
                    train_data = train_data.loc[keep_indices]
                    train_targets = train_targets.loc[keep_indices]

                train_data = pd.concat([train_data, test_data])
                train_targets = pd.concat([train_targets, test_targets])

        if self.final_models_dir is not None and last_era_with_targets is not None:
            for model_name, model_attrs in self.models_attrs.items():
                model = self._load_model(model_attrs['model_path'])
                final_model_name = f"{model_name}"
                self._save_model(model, final_model_name, is_final=True)
            print(f"Final models saved to {self.final_models_dir}")

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

        self.evaluation_results = evaluator.full_evaluation(
            dataf=eval_data,
            pred_cols=pred_cols,
            target_col='target',
        )
        self.evaluation_results = self.evaluation_results.drop(columns=["target"]).reset_index().rename(
            columns={'index': 'model'}).sort_values(by='mean', ascending=False)

        per_era_numerai_corr = {}
        for model_name in pred_cols:
            per_era_numerai_corr[model_name] = evaluator.per_era_numerai_corrs(
                dataf=eval_data,
                pred_col=model_name,
                target_col='target'
            )
        self.per_era_numerai_corr = pd.DataFrame(per_era_numerai_corr)

        if self.create_report:
            self.create_html_report()

    def create_html_report(self):
        timestamp = datetime.now().strftime('%Y_%d_%m_%H_%M')
        report_dir = 'tmp/reports'
        report_path = f'{report_dir}/report_{timestamp}.html'
        os.makedirs(report_dir, exist_ok=True)
        evaluation_html = self.evaluation_results.to_html(index=False)
        per_era_corr_html = self.per_era_numerai_corr.to_html(index=True)
        plt.figure(figsize=(10, 6), dpi=150)
        self.per_era_numerai_corr.plot()
        plt.title("Per-Era Numerai Correlations", fontsize=16)
        plt.xlabel("Era", fontsize=14)
        plt.ylabel("Correlation", fontsize=14)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        img_base64_str = f"data:image/png;base64,{img_base64}"
        html_content = f"""
        <html>
        <head>
            <title>Evaluation Report - {timestamp}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                table, th, td {{
                    border: 1px solid black;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                }}
                img {{
                    display: block;
                    margin: 20px auto;
                }}
            </style>
        </head>
        <body>

            <h2>Evaluation Results</h2>
            {evaluation_html}

            <h2>Per-Era Numerai Correlation Table</h2>
            {per_era_corr_html}

            <h2>Per-Era Numerai Correlation Plot</h2>
            <img src="{img_base64_str}" alt="Per-Era Numerai Correlation">

        </body>
        </html>
        """
        with open(report_path, 'w') as file:
            file.write(html_content)

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
        if self.era_column not in X_train.columns:
            raise ValueError(f"era_column '{self.era_column}' not found in X_train columns.")
        if self.era_column not in X_test.columns:
            raise ValueError(f"era_column '{self.era_column}' not found in X_test columns.")
        if not self.models_attrs or len(self.models_attrs) == 0:
            raise ValueError("No models provided or models failed to load. Please check models_attrs.")
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

