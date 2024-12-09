import warnings
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import pandas as pd
from sklearn.base import clone
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from .misc import Logger
import os


logger = Logger(log_dir='logs', log_file='optimize.log').get_logger()


# Suppress all warnings and messages during model training
warnings.filterwarnings("ignore")


def corr_v2(predictions, targets, groups):
    ranked_preds = predictions.groupby(groups).rank(method="average", pct=True)
    ranked_preds = ranked_preds.clip(1e-10, 1 - 1e-10)
    gauss_ranked_preds = stats.norm.ppf(ranked_preds.replace([np.inf, -np.inf], np.nan).dropna())

    group_means = targets.groupby(groups).transform("mean")
    centered_targets = (targets - group_means).replace([np.inf, -np.inf], np.nan).dropna()

    preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
    target_p15 = np.sign(centered_targets) * np.abs(centered_targets) ** 1.5

    preds_df = pd.DataFrame({
        "preds_p15": preds_p15,
        "target_p15": target_p15,
        "groups": groups
    }).dropna()

    def pearson_corr(df):
        if df.empty or df["preds_p15"].std() == 0 or df["target_p15"].std() == 0:
            return 0
        return stats.pearsonr(df["preds_p15"], df["target_p15"])[0]

    group_corrs = preds_df.groupby("groups").apply(pearson_corr)
    return group_corrs.mean()


# Marcos DePrado Time Series Split
class GroupGapEmbargo(GroupKFold):
    def __init__(self, n_splits=4, gap=13):
        super().__init__(n_splits)
        self.n_splits = n_splits
        self.gap = gap
        self.embargo = 2 * gap  # Embargo period is twice the gap

    def split(self, X, y, groups):
        groups = np.array(groups)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        # For development purposes  todo: change to rows in X
        if n_groups < 100:
            self.gap = 1
            self.embargo = 2

        # Determine fold sizes ensuring all groups are covered
        fold_sizes = np.full(self.n_splits, n_groups // self.n_splits, dtype=int)
        fold_sizes[:n_groups % self.n_splits] += 1

        current = 0
        indices = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            indices.append((start, stop))
            current = stop

        all_train_indices = set()
        for i, (test_start, test_end) in enumerate(indices):
            test_groups = unique_groups[test_start:test_end]

            train_start = 0
            train_end = max(0, test_start - self.gap)
            embargo_end = min(n_groups, test_end + self.embargo)

            # Handle cases where purge and embargo exclude all groups
            if train_end == 0 and embargo_end >= n_groups:
                raise ValueError(f"Not enough groups to form a valid training set for fold {i + 1}")

            train_groups = np.concatenate((
                unique_groups[train_start:train_end],
                unique_groups[embargo_end:]
            ))

            train_indices = np.where(np.isin(groups, train_groups))[0]
            test_indices = np.where(np.isin(groups, test_groups))[0]

            all_train_indices.update(train_indices)

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class NumTreesOptimizer:
    def __init__(self, model, scoring_func=corr_v2, cv_splitter=GroupGapEmbargo, n_splits=3, gap=5, initial_trees=1000, step_size=1000, patience=3):
        self.model = model
        self.scoring_func = scoring_func
        self.cv_splitter = cv_splitter
        self.initial_trees = initial_trees
        self.step_size = step_size
        self.patience = patience
        self.cv_splitter = cv_splitter(n_splits=n_splits, gap=gap)

        self.best_score = None
        self.best_n_estimators = None
        self.history = []

    def fit(self, X, y, groups=None):
        current_trees = self.initial_trees
        step_size = self.step_size
        consecutive_non_improve = 0
        tried_halving = 0
        best_iteration = None

        logger.info(f"Starting optimization with initial_trees={self.initial_trees}, step_size={self.step_size}, patience={self.patience}")

        while True:
            logger.info(f"Evaluating with {current_trees} estimators.")
            score = self._train_and_evaluate(current_trees, X, y, groups)
            self.history.append((current_trees, score))

            if self.best_score is None or score > self.best_score:
                logger.info(f"New best score: {score:.4f} at {current_trees} trees.")
                self.best_score = score
                self.best_n_estimators = current_trees
                best_iteration = current_trees
                consecutive_non_improve = 0
                tried_halving = 0  # Reset halving attempts
                current_trees += step_size
            else:
                consecutive_non_improve += 1
                logger.info(f"Score did not improve. Consecutive non-improvements: {consecutive_non_improve}.")

                if consecutive_non_improve >= 1:  # Immediately handle worsening performance
                    if tried_halving >= self.patience:
                        logger.info(f"Stopping optimization. Best score: {self.best_score:.4f} with {self.best_n_estimators} estimators.")
                        break

                    tried_halving += 1
                    step_size = max(1, step_size // 2)
                    current_trees = best_iteration + step_size

        self.final_model = clone(self.model)
        self.final_model.set_params(n_estimators=self.best_n_estimators)
        self.final_model.fit(X, y)
        return self

    def _train_and_evaluate(self, n_estimators, X, y, groups):
        model = clone(self.model)
        model.set_params(n_estimators=n_estimators)
        scores = []

        for train_idx, val_idx in self.cv_splitter.split(X, y, groups):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            groups_val = groups.iloc[val_idx]

            model.fit(X_train, y_train)
            preds = pd.Series(model.predict(X_val), index=y_val.index)
            score = self.scoring_func(preds, y_val, groups_val)
            scores.append(score)

        return np.mean(scores)

    def plot_learning_curve(self, save_path="tmp/learning_curve.png"):
        n_estimators_list, scores = zip(*self.history)
        plt.figure(figsize=(10, 6))
        plt.plot(n_estimators_list, scores, marker='o')
        plt.axvline(self.best_n_estimators, color='red', linestyle='--', label=f'Best: {self.best_n_estimators}')
        plt.xlabel("Number of Estimators")
        plt.ylabel("Validation Score")
        plt.title("Learning Curve")
        plt.legend()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_best_model(self):
        return self.final_model

    def get_best_n_estimators(self):
        return self.best_n_estimators

    def get_best_score(self):
        return self.best_score


