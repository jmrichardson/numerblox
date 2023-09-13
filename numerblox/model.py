import os
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from tqdm.auto import tqdm
from numerbay import NumerBay
from abc import abstractmethod
from rich import print as rich_print
from sklearn.base import BaseEstimator, RegressorMixin


from .download import NumeraiClassicDownloader
from .numerframe import NumerFrame


class BaseModel(BaseEstimator, RegressorMixin):
    """
    Setup for model prediction on a Dataset.

    :param model_directory: Main directory from which to read in models. \n
    :param model_name: Name that will be used to create column names and for display purposes.
    """
    def __init__(self, model_directory: str,
                 model_name: str = None,
                 ):
        self.model_directory = Path(model_directory)
        self.model_name = model_name if model_name else uuid.uuid4().hex
        self.prediction_col_name = f"prediction_{self.model_name}"
        self.description = f"{self.__class__.__name__}: '{self.model_name}' prediction"

    def fit(self, X: Union[pd.DataFrame, NumerFrame], y=None) -> NumerFrame:
        return self

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, NumerFrame], y=None) -> NumerFrame:
        """ Return NumerFrame with column added for prediction. """
        ...
        return NumerFrame(X)

    def get_prediction_col_names(self, pred_shape: tuple) -> list:
        """ Create multiple columns if predictions are multi-target. """
        prediction_cols = self.prediction_col_name
        if len(pred_shape) > 1:
            if pred_shape[1] > 1:
                prediction_cols = [f"{self.prediction_col_name}_{i}" for i in range(pred_shape[1])]
        return prediction_cols

    def __call__(self, dataf: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        return self.predict(dataf=dataf)


class ExternalCSVs(BaseModel):
    """
    Load external submissions and add to NumerFrame. \n
    All csv files in this directory will be added to NumerFrame.
    Make sure all external predictions are prepared and ready for submission. i.e. IDs lining up and one column named 'prediction'. \n
    :param data_directory: Directory path for retrieving external submission.
    """
    def __init__(self, data_directory: str = "external_submissions"):
        super().__init__(model_directory=data_directory)
        self.data_directory = Path(data_directory)
        self.paths = list(self.data_directory.glob("*.csv"))
        if not self.paths:
            rich_print(f":warning: WARNING: No csvs found in directory '{self.data_directory}'. :warning:")

    def predict(self, dataf: NumerFrame) -> NumerFrame:
        """ Return NumerFrame with added external predictions. """
        for path in tqdm(self.paths, desc="External submissions"):
            dataf.loc[:, f"prediction_{path.name}"] = self._get_preds(path)
        return NumerFrame(dataf)

    def _get_preds(self, path: Path) -> pd.Series:
        pred_col = pd.read_csv(path, index_col=0, header=0)['prediction']
        if not pred_col.between(0, 1).all():
            raise ValueError(f"Prediction values must be between 0 and 1. Does not hold for '{path.name}'.")
        return pred_col


class NumerBayCSVs(BaseModel):
    """
    Load NumerBay submissions and add to NumerFrame. \n
    Make sure to provide correct NumerBay credentials and that your purchases have been confirmed and artifacts are available for download. \n
    :param data_directory: Directory path for caching submission. Files not already present in the directory will be downloaded from NumerBay.
    :param numerbay_product_full_names: List of product full names (in the format of [category]-[product name]) to download from NumerBay. E.g. ['numerai-predictions-numerbay']
    :param numerbay_username: NumerBay username
    :param numerbay_password: NumerBay password
    :param numerbay_key_path: NumerBay encryption key json file path (exported from the profile page)

    """
    def __init__(self,
                 data_directory: str = "numerbay_submissions",
                 numerbay_product_full_names: list = None,
                 numerbay_username: str = None,
                 numerbay_password: str = None,
                 numerbay_key_path: str = None,
                 ticker_col: str = 'bloomberg_ticker'):
        super().__init__(model_directory=data_directory)
        self.data_directory = Path(data_directory)
        self.numerbay_product_full_names = numerbay_product_full_names
        self.numerbay_key_path = numerbay_key_path
        self._api = None
        self._get_api_func = lambda: NumerBay(username=numerbay_username, password=numerbay_password)
        self.ticker_col = ticker_col
        self.classic_number = 8
        self.signals_number = 11

    def predict(self, dataf: NumerFrame) -> NumerFrame:
        """ Return NumerFrame with added NumerBay predictions. """
        for numerbay_product_full_name in tqdm(self.numerbay_product_full_names, desc="NumerBay submissions"):
            pred_name = f"prediction_{numerbay_product_full_name}"
            if dataf.meta.era_col == "era":
                dataf.loc[:, pred_name] = \
                    self._get_preds(numerbay_product_full_name, tournament=self.classic_number)
            else:
                dataf.loc[:, pred_name] = \
                    dataf.merge(self._get_preds(numerbay_product_full_name, tournament=self.signals_number),
                                on=[self.ticker_col, dataf.meta.era_col], how='left')['signal']
        return NumerFrame(dataf)

    @property
    def api(self):
        if self._api is None:
            self._api = self._get_api_func()
        return self._api

    def _get_preds(self, numerbay_product_full_name: str, tournament: int) -> pd.Series:
        if tournament == self.signals_number: # Temporarily disable Signals
            raise NotImplementedError("NumerBay Signals predictions not yet supported.")

        # Scan for already downloaded files, allows arbitrary ext just in case (e.g. csv.zip)
        file_paths = list(self.data_directory.glob(f"{numerbay_product_full_name}.*"))
        if len(file_paths) > 0:
            file_path = file_paths[0]
        else:
            file_path = Path.joinpath(self.data_directory, f"{numerbay_product_full_name}.csv")

        # Download file if needed
        if not file_path.is_file():
            # Download to a tmp file
            tmp_path = Path.joinpath(self.data_directory, f".{numerbay_product_full_name}.tmp")
            self.api.download_artifact(
                dest_path=str(tmp_path),
                product_full_name=numerbay_product_full_name,
                key_path=self.numerbay_key_path
            )
            # Rename file after successful download (and decryption)
            os.rename(tmp_path, file_path)

        # Read downloaded file
        if file_path.is_file():
            if tournament == self.classic_number:
                pred_col = pd.read_csv(file_path, index_col=0, header=0)['prediction']
            elif tournament == self.signals_number:
                pass
            else:
                raise ValueError(f"Invalid tournament '{tournament}'.")
        else:
            raise FileNotFoundError(f"No file found in directory '{self.data_directory}' for NumerBay product "
                                    f"'{numerbay_product_full_name}'.")

        # validity check
        if not pred_col.between(0, 1).all():
            raise ValueError(f"Prediction values must be between 0 and 1. Does not hold for '{path.name}'.")
        return pred_col


class ExamplePredictionsModel(BaseModel):
    """
    Load example predictions and add to NumerFrame. \n
    :param file_name: File to download from NumerAPI.
    By default this is example predictions for v4.2 data.
    Example predictions in previous versions:
    - v4.1. -> "v4.1/live_example_preds.parquet"
    - v4. -> "v4/live_example_preds.parquet"
    'v4.2/example_validation_predictions.parquet' by default. \n
    :param data_directory: Directory path to download example predictions to or directory where example data already exists. \n
    :param round_num: Optional round number. Downloads most recent round by default.
    """
    def __init__(self, file_name: str = "v4.2/live_example_preds.parquet",
                 data_directory: str = "example_predictions_model",
                 round_num: int = None):
        super().__init__(model_directory="",
                         model_name="example",
                         )
        self.file_name = file_name
        self.data_directory = data_directory
        self.round_num = round_num

    def predict(self, X: NumerFrame) -> NumerFrame:
        """ Return NumerFrame with added example predictions. """
        self._download_example_preds()
        example_preds = self._load_example_preds()
        X.loc[:, self.prediction_col_name] = X.merge(example_preds, on='id', how='left')['prediction']
        self.downloader.remove_base_directory()
        return NumerFrame(X)

    def _download_example_preds(self):
        self.downloader = NumeraiClassicDownloader(directory_path=self.data_directory)
        self.dest_path = f"{str(self.downloader.dir)}/{self.file_name}"
        self.downloader.download_single_dataset(filename=self.file_name,
                                                dest_path=self.dest_path,
                                                round_num=self.round_num)

    def _load_example_preds(self, *args, **kwargs):
        return pd.read_parquet(self.dest_path, *args, **kwargs)
