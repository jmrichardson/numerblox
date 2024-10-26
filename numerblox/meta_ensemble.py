import numpy as np
import pandas as pd
from collections import Counter
from typing import Callable, List, Union
from sklearn.utils import check_random_state
from .misc import numerai_payout_score, mmc_score, numerai_corr_score
from . import logger


class GreedyEnsemble:
    def __init__(self,
                 max_ensemble_size: int = 10,
                 metric: Union[str, Callable] = "corr",
                 use_replacement: bool = True,
                 sorted_initialization: bool = True,
                 initial_n: int = None,
                 bag_fraction: float = 0.5,
                 num_bags: int = 20,
                 random_state: Union[int, None] = None):

        self.max_ensemble_size = max_ensemble_size
        self.use_replacement = use_replacement
        self.sorted_initialization = sorted_initialization
        self.initial_n = initial_n
        self.bag_fraction = bag_fraction
        self.num_bags = num_bags
        self.random_state = random_state

        self.weights_ = None
        self.selected_model_names_ = None

        if isinstance(metric, str):
            if metric == "corr":
                self.metric = numerai_corr_score
            elif metric == "mmc":
                self.metric = mmc_score
            elif metric == "payout":
                self.metric = numerai_payout_score
            else:
                raise ValueError("Unsupported metric string. Choose 'corr', 'mmc', or 'payout'.")
        elif callable(metric):
            self.metric = metric
        else:
            raise TypeError("Metric must be a string or a callable.")

    def fit(self,
            oof: pd.DataFrame,
            sample_weights: pd.Series = None):

        rng = check_random_state(self.random_state)
        oof_targets = oof.target
        oof_eras = oof.era
        meta_data = oof.get('meta_data', None)
        oof_predictions = oof.drop(columns=['era', 'target', 'meta_data'], errors='ignore')
        model_names = oof_predictions.columns.tolist()
        n_models = len(model_names)

        if self.max_ensemble_size < 1:
            raise ValueError("Ensemble size cannot be less than one!")

        current_ensemble_predictions = pd.Series(0.0, index=oof.index)
        ensemble_indices = []
        ensemble_scores = []
        used_model_counts = Counter()

        if self.sorted_initialization:
            model_scores = {}
            for model_name in model_names:
                predictions = oof_predictions[model_name]
                print("yo start")
                score = self.metric(oof_targets, predictions, oof_eras, meta_data=meta_data, sample_weight=sample_weights)
                print("yo endstart")
                model_scores[model_name] = score
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            if self.initial_n is None:
                N = self._determine_initial_n([score for name, score in sorted_models])
            else:
                N = self.initial_n
            for i in range(N):
                model_name = sorted_models[i][0]
                current_ensemble_predictions += oof_predictions[model_name]
                ensemble_indices.append(model_name)
                used_model_counts[model_name] += 1
            current_ensemble_predictions /= N

        for _ in range(self.max_ensemble_size):
            best_score = None
            best_model_name = None

            if self.use_replacement:
                candidate_model_names = model_names
            else:
                candidate_model_names = [name for name in model_names if name not in used_model_counts]

            if not candidate_model_names:
                break

            for model_name in candidate_model_names:
                model_predictions = oof_predictions[model_name]
                combined_predictions = (current_ensemble_predictions * len(ensemble_indices) + model_predictions) / (
                        len(ensemble_indices) + 1)

                print("start metric")
                score = self.metric(oof_targets, combined_predictions, oof_eras, meta_data=meta_data, sample_weight=sample_weights)
                print("end metric")

                if best_score is None or score > best_score:
                    best_score = score
                    best_model_name = model_name

            if best_model_name is None:
                break

            ensemble_indices.append(best_model_name)
            used_model_counts[best_model_name] += 1
            current_ensemble_predictions = (current_ensemble_predictions * (len(ensemble_indices) - 1) +
                                            oof[best_model_name]) / len(ensemble_indices)
            ensemble_scores.append(best_score)

        if ensemble_scores:
            best_index = np.argmax(ensemble_scores)
            best_ensemble_indices = ensemble_indices[:best_index + 1]
        else:
            best_ensemble_indices = []

        model_counts = Counter(best_ensemble_indices)
        total_counts = sum(model_counts.values()) or 1
        weights = pd.Series({model_name: count / total_counts for model_name, count in model_counts.items()})
        self.weights_ = weights.reindex(model_names).fillna(0.0)
        self.selected_model_names_ = self.weights_[self.weights_ > 0].index.tolist()


    def _determine_initial_n(self, scores):
        if len(scores) <= 1:
            return 1

        threshold = 0.001
        N = 1
        for i in range(1, len(scores)):
            improvement = scores[i] - scores[i - 1]
            if improvement < threshold:
                break
            N += 1
        return N

    def predict(self, base_models_predictions: pd.DataFrame) -> pd.Series:
        weighted_predictions = base_models_predictions.multiply(self.weights_, axis=1)
        ensemble_predictions = weighted_predictions.sum(axis=1)
        return ensemble_predictions

