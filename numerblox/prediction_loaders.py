import numpy as np
import pandas as pd
from uuid import uuid4
from pathlib import Path
from abc import abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array

from .download import NumeraiClassicDownloader

class BasePredictionLoader(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self):
        ...

    def fit(self, X=None, y=None):
        return self

    @abstractmethod
    def transform(self, X=None, y=None) -> pd.DataFrame:
        """ Return Predictions generated by model. """
        ...

    def __call__(self, X=None, y=None) -> pd.DataFrame:
        return self.predict(X=X)
    
    @abstractmethod
    def get_feature_names_out(self, input_features=None):
        """ Return feature names. """
        ...

class ExamplePredictions(BasePredictionLoader):
    """
    Load example predictions.
    :param file_name: File to download from NumerAPI.
    By default this is example predictions for v4.2 data.
    'v4.2/example_validation_predictions.parquet' by default. 
    Example predictions in previous versions:
    - v4.1. -> "v4.1/live_example_preds.parquet"
    - v4. -> "v4/live_example_preds.parquet"
    :param round_num: Optional round number. Downloads most recent round by default.
    """
    def __init__(self, file_name: str = "v4.2/live_example_preds.parquet",
                 round_num: int = None):
        super().__init__()
        self.file_name = file_name
        self.round_num = round_num

    def transform(self, X=None, y=None) -> pd.DataFrame:
        """ Return example predictions. """
        self._download_example_preds()
        example_preds = self._load_example_preds()
        self.downloader.remove_base_directory()
        return example_preds

    def _download_example_preds(self):
        data_directory = f"example_predictions_loader_{uuid4()}"
        self.downloader = NumeraiClassicDownloader(directory_path=data_directory)
        self.dest_path = f"{str(self.downloader.dir)}/{self.file_name}"
        self.downloader.download_single_dataset(filename=self.file_name,
                                                dest_path=self.dest_path,
                                                round_num=self.round_num)

    def _load_example_preds(self, *args, **kwargs):
        return pd.read_parquet(self.dest_path, *args, **kwargs)

    def get_feature_names_out(self, input_features=None):
        return [Path(self.file_name).with_suffix('').as_posix()] if not input_features else input_features