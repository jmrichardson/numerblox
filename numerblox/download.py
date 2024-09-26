import os
import time
import glob
import json
import shutil
import concurrent
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from google.cloud import storage
from datetime import datetime as dt
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dateutil.relativedelta import relativedelta
from numerapi import NumerAPI, SignalsAPI, CryptoAPI

from .numerframe import NumerFrame


class BaseIO(ABC):
    """
    Basic functionality for IO (downloading and uploading).

    :param directory_path: Base folder for IO. Will be created if it does not exist.
    """
    def __init__(self, directory_path: str):
        self.dir = Path(directory_path)
        self._create_directory()

    def remove_base_directory(self):
        """Remove directory with all contents."""
        abs_path = self.dir.resolve()
        print(
            f"WARNING: Deleting directory for '{self.__class__.__name__}'\nPath: '{abs_path}'"
        )
        shutil.rmtree(abs_path)

    def download_file_from_gcs(self, bucket_name: str, gcs_path: str):
        """
        Get file from GCS bucket and download to local directory.
        :param gcs_path: Path to file on GCS bucket.
        """
        blob_path = str(self.dir.resolve())
        blob = self._get_gcs_blob(bucket_name=bucket_name, blob_path=blob_path)
        blob.download_to_filename(gcs_path)
        print(
            f"Downloaded GCS object '{gcs_path}' from bucket '{blob.bucket.id}' to local directory '{blob_path}'."
        )

    def upload_file_to_gcs(self, bucket_name: str, gcs_path: str, local_path: str):
        """
        Upload file to some GCS bucket.
        :param gcs_path: Path to file on GCS bucket.
        """
        blob = self._get_gcs_blob(bucket_name=bucket_name, blob_path=gcs_path)
        blob.upload_from_filename(local_path)
        print(
            f"Local file '{local_path}' uploaded to '{gcs_path}' in bucket {blob.bucket.id}"
        )

    def download_directory_from_gcs(self, bucket_name: str, gcs_path: str):
        """
        Copy full directory from GCS bucket to local environment.
        :param gcs_path: Name of directory on GCS bucket.
        """
        blob_path = str(self.dir.resolve())
        blob = self._get_gcs_blob(bucket_name=bucket_name, blob_path=blob_path)
        for gcs_file in glob.glob(gcs_path + "/**", recursive=True):
            if os.path.isfile(gcs_file):
                blob.download_to_filename(blob_path)
        print(
            f"Directory '{gcs_path}' from bucket '{blob.bucket.id}' downloaded to '{blob_path}'"
        )

    def upload_directory_to_gcs(self, bucket_name: str, gcs_path: str):
        """
        Upload full base directory to GCS bucket.
        :param gcs_path: Name of directory on GCS bucket.
        """
        blob = self._get_gcs_blob(bucket_name=bucket_name, blob_path=gcs_path)
        for local_path in glob.glob(str(self.dir) + "/**", recursive=True):
            if os.path.isfile(local_path):
                blob.upload_from_filename(local_path)
        print(
            f"Directory '{self.dir}' uploaded to '{gcs_path}' in bucket {blob.bucket.id}"
        )

    def _get_gcs_blob(self, bucket_name: str, blob_path: str) -> storage.Blob:
        """ Create blob that interacts with Google Cloud Storage (GCS). """
        client = storage.Client()
        # https://console.cloud.google.com/storage/browser/[bucket_name]
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob

    def _append_folder(self, folder: str) -> Path:
        """
        Return base directory Path object appended with 'folder'.
        Create directory if it does not exist.
        """
        dir = Path(self.dir / folder)
        dir.mkdir(parents=True, exist_ok=True)
        return dir
    
    def _get_dest_path(self, subfolder: str, filename: str) -> str:
        """ Prepare destination path for downloading. """
        dir = self._append_folder(subfolder)
        dest_path = str(dir.joinpath(filename.split("/")[-1]))
        return dest_path

    def _create_directory(self):
        """ Create base directory if it does not exist. """
        if not self.dir.is_dir():
            print(
                f"No existing directory found at '{self.dir}'. Creating directory..."
            )
            self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def get_all_files(self) -> list:
        """ Return all paths of contents in directory. """
        return list(self.dir.iterdir())

    @property
    def is_empty(self) -> bool:
        """ Check if directory is empty. """
        return not bool(self.get_all_files)


class BaseDownloader(BaseIO):
    """
    Abstract base class for downloaders.

    :param directory_path: Base folder to download files to.
    """
    def __init__(self, directory_path: str):
        super().__init__(directory_path=directory_path)

    @abstractmethod
    def download_training_data(self, *args, **kwargs):
        """ Download all necessary files needed for training. """
        ...

    @abstractmethod
    def download_live_data(self, *args, **kwargs):
        """ Download minimal amount of files needed for weekly inference. """
        ...

    @staticmethod
    def _load_json(file_path: str, verbose=False, *args, **kwargs) -> dict:
        """ Load JSON from file and return as dictionary. """
        with open(Path(file_path)) as json_file:
            json_data = json.load(json_file, *args, **kwargs)
        if verbose:
            print(json_data)
        return json_data

    def _default_save_path(self, start: dt, end: dt, backend: str):
        """ Save to downloader directory indicating backend, start date and end date as parquet file. """
        return f"{self.dir}/{backend}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"

    def __call__(self, *args, **kwargs):
        """
        The most common use case will be to get weekly inference data. So calling the class itself returns inference data.
        """
        self.download_live_data(*args, **kwargs)


class NumeraiClassicDownloader(BaseDownloader):
    """
    WARNING: Versions 1-3 (legacy data) are deprecated. Only supporting version 4+.

    Downloading from NumerAPI for Numerai Classic data. \n
    :param directory_path: Base folder to download files to. \n
    All kwargs will be passed to NumerAPI initialization.
    """
    TRAIN_DATASET_NAME = "train_int8.parquet"
    TRAIN_DATASET_NAME_5 = "train.parquet"
    VALIDATION_DATASET_NAME = "validation_int8.parquet"
    VALIDATION_DATASET_NAME_5 = "validation.parquet"
    LIVE_DATASET_NAME = "live_int8.parquet"
    LIVE_DATASET_NAME_5 = "live.parquet"
    LIVE_EXAMPLE_PREDS_NAME = "live_example_preds.parquet"
    VALIDATION_EXAMPLE_PREDS_NAME = "validation_example_preds.parquet"

    def __init__(self, directory_path: str, **kwargs):
        super().__init__(directory_path=directory_path)
        self.napi = NumerAPI(**kwargs)
        # Get all available versions available for Numerai Classic.
        self.dataset_versions = set(s.split("/")[0] for s in self.napi.list_datasets())
        self.dataset_versions.discard("signals")

    def download_training_data(
        self, subfolder: str = "", version: str = "5.0"
    ):
        """
        Get Numerai classic training and validation data.
        :param subfolder: Specify folder to create folder within base directory root.
        Saves in base directory root by default.
        :param version: Numerai dataset version.
        4 = April 2022 dataset
        4.1 = Sunshine
        4.2 = Rain
        4.3 = Midnight
        5.0 = Atlas (default)
        """
        self._check_dataset_version(version)
        if float(version) >= 5.0:
            train_val_files = [f"v{version}/{self.TRAIN_DATASET_NAME_5}",
                               f"v{version}/{self.VALIDATION_DATASET_NAME_5}"]
        else:
            print("WARNING: v4 data will only be supported until Sept. 27, 2024!!!")
            train_val_files = [f"v{version}/{self.TRAIN_DATASET_NAME}",
                            f"v{version}/{self.VALIDATION_DATASET_NAME}"]
        for file in train_val_files:
            dest_path = self._get_dest_path(subfolder, file)
            self.download_single_dataset(
                filename=file,
                dest_path=dest_path
            )

    def download_single_dataset(
        self, filename: str, dest_path: str, round_num: int = None
    ):
        """
        Download one of the available datasets through NumerAPI.

        :param filename: Name as listed in NumerAPI (Check NumerAPI().list_datasets() for full overview)
        :param dest_path: Full path where file will be saved.
        :param round_num: Numerai tournament round number. Downloads latest round by default.
        """
        print(
            f"Downloading '{filename}'."
        )
        self.napi.download_dataset(
            filename=filename,
            dest_path=dest_path,
            round_num=round_num
        )

    def download_live_data(
            self,
            subfolder: str = "",
            version: str = "5.0",
            round_num: int = None
    ):
        """
        Download all live data in specified folder for given version (i.e. minimal data needed for inference).

        :param subfolder: Specify folder to create folder within directory root.
        Saves in directory root by default.
        :param version: Numerai dataset version. 
        4 = April 2022
        4.1 = Sunshine 
        4.2 = Rain
        4.3 = Midnight
        5.0 = Atlas (default)
        :param round_num: Numerai tournament round number. Downloads latest round by default.
        """
        self._check_dataset_version(version)
        if float(version) >= 5.0:
            live_files = [f"v{version}/{self.LIVE_DATASET_NAME_5}"]
        else:
            print("WARNING: v4 data will only be supported until Sept. 27, 2024!!!")
            live_files = [f"v{version}/{self.LIVE_DATASET_NAME}"]
        for file in live_files:
            dest_path = self._get_dest_path(subfolder, file)
            self.download_single_dataset(
                filename=file,
                dest_path=dest_path,
                round_num=round_num
            )

    def download_example_data(
        self, subfolder: str = "", version: str = "5.0", round_num: int = None
    ):
        """
        Download all example prediction data in specified folder for given version.

        :param subfolder: Specify folder to create folder within base directory root.
        Saves in base directory root by default.
        :param version: Numerai dataset version.
        4 = April 2022 dataset
        4.1 = Sunshine
        4.2 = Rain
        4.3 = Midnight
        5.0 = Atlas (default)
        :param round_num: Numerai tournament round number. Downloads latest round by default.
        """
        self._check_dataset_version(version)
        example_files = [f"v{version}/{self.LIVE_EXAMPLE_PREDS_NAME}", 
                         f"v{version}/{self.VALIDATION_EXAMPLE_PREDS_NAME}"]
        for file in example_files:
            dest_path = self._get_dest_path(subfolder, file)
            self.download_single_dataset(
                filename=file,
                dest_path=dest_path,
                round_num=round_num
            )

    def get_classic_features(self, subfolder: str = "", filename="v5.0/features.json", *args, **kwargs) -> dict:
        """
        Download feature overview (stats and feature sets) through NumerAPI and load as dict.
        :param subfolder: Specify folder to create folder within base directory root.
        Saves in base directory root by default.
        :param filename: name for feature overview.
        *args, **kwargs will be passed to the JSON loader.
        :return: Feature overview dict
        """
        version = filename.split("/")[0].replace("v", "")
        self._check_dataset_version(version)
        dest_path = self._get_dest_path(subfolder, filename)
        self.download_single_dataset(filename=filename,
                                     dest_path=dest_path)
        json_data = self._load_json(dest_path, *args, **kwargs)
        return json_data

    def download_meta_model_preds(self, subfolder: str = "", filename="v4.3/meta_model.parquet") -> pd.DataFrame:
        """
        Download Meta model predictions through NumerAPI.
        :param subfolder: Specify folder to create folder within base directory root.
        Saves in base directory root by default.
        :param filename: name for meta model predictions file.
        :return: Meta model predictions as DataFrame.
        """
        version = filename.split("/")[0].replace("v", "")
        self._check_dataset_version(version)
        dest_path = self._get_dest_path(subfolder, filename)
        self.download_single_dataset(
            filename=filename,
            dest_path=dest_path,
            )
        return pd.read_parquet(dest_path)
    
    def _check_dataset_version(self, version: str):
        assert f"v{version}" in self.dataset_versions, f"Version '{version}' is not available in NumerAPI."

class NumeraiSignalsDownloader(BaseDownloader):
    """
    Support for Numerai Signals v1 parquet data.
    Downloading from SignalsAPI for Numerai Signals data. \n
    :param directory_path: Base folder to download files to. \n
    All kwargs will be passed to SignalsAPI initialization.
    """
    TRAIN_DATASET_NAME = "train.parquet"
    VALIDATION_DATASET_NAME = "validation.parquet"
    LIVE_DATASET_NAME = "live.parquet"
    LIVE_EXAMPLE_PREDS_NAME = "live_example_preds.parquet"
    VALIDATION_EXAMPLE_PREDS_NAME = "validation_example_preds.parquet"

    def __init__(self, directory_path: str, **kwargs):
        super().__init__(directory_path=directory_path)
        self.sapi = SignalsAPI(**kwargs)
        # Get all available versions available for Numerai Signals.
        self.dataset_versions = set(s.replace("signals/", "").split("/")[0] for s in self.sapi.list_datasets() if s.startswith("signals/v"))

    def download_training_data(
        self, subfolder: str = "", version: str = "1.0"
    ):
        """
        Get Numerai Signals training and validation data.
        :param subfolder: Specify folder to create folder within base directory root.
        Saves in base directory root by default.
        :param version: Numerai Signals dataset version.
        Currently only v1.0 is supported.
        """
        self._check_dataset_version(version)
        train_val_files = [f"signals/v{version}/{self.TRAIN_DATASET_NAME}",
                           f"signals/v{version}/{self.VALIDATION_DATASET_NAME}"]
        for file in train_val_files:
            dest_path = self._get_dest_path(subfolder, file)
            self.download_single_dataset(
                filename=file,
                dest_path=dest_path
            )

    def download_single_dataset(
        self, filename: str, dest_path: str
    ):
        """
        Download one of the available datasets through SignalsAPI.

        :param filename: Name as listed in SignalsAPI (Check SignalsAPI().list_datasets() for full overview)
        :param dest_path: Full path where file will be saved.
        """
        print(
            f"Downloading '{filename}'."
        )
        self.sapi.download_dataset(
            filename=filename,
            dest_path=dest_path,
        )

    def download_live_data(
            self,
            subfolder: str = "",
            version: str = "1.0",
    ):
        """
        Download all live data in specified folder (i.e. minimal data needed for inference).

        :param subfolder: Specify folder to create folder within directory root.
        Saves in directory root by default.
        :param version: Numerai dataset version. 
        Currently only v1.0 is supported.
        """
        self._check_dataset_version(version)
        live_files = [f"signals/v{version}/{self.LIVE_DATASET_NAME}"]
        for file in live_files:
            dest_path = self._get_dest_path(subfolder, file)
            self.download_single_dataset(
                filename=file,
                dest_path=dest_path,
            )

    def download_example_data(
        self, subfolder: str = "", version: str = "1.0"
    ):
        """
        Download all example prediction data in specified folder for given version.

        :param subfolder: Specify folder to create folder within base directory root.
        Saves in base directory root by default.
        :param version: Numerai dataset version.
        Currently only v1.0 is supported.
        """
        self._check_dataset_version(version)
        example_files = [f"signals/v{version}/{self.LIVE_EXAMPLE_PREDS_NAME}", 
                         f"signals/v{version}/{self.VALIDATION_EXAMPLE_PREDS_NAME}"]
        for file in example_files:
            dest_path = self._get_dest_path(subfolder, file)
            self.download_single_dataset(
                filename=file,
                dest_path=dest_path,
            )

    def _check_dataset_version(self, version: str):
        assert f"v{version}" in self.dataset_versions, f"Version '{version}' is not available in SignalsAPI."

class NumeraiCryptoDownloader(BaseDownloader):
    """
    Download Numerai Crypto data.

    :param directory_path: Base folder to download files to.
    """
    LIVE_DATASET_NAME = "live_universe.parquet"
    TRAIN_TARGETS_NAME = "train_targets.parquet"

    def __init__(self, directory_path: str, **kwargs):
        super().__init__(directory_path=directory_path)
        self.capi = CryptoAPI(**kwargs)
        self.dataset_versions = ["v1.0"]

    def download_training_data(
            self,
            subfolder: str = "",
            version: str = "1.0",
    ):
        """
        Download all training data in specified folder for given version.

        :param subfolder: Specify folder to create folder within directory root.
        Saves in directory root by default.
        :param version: Numerai dataset version. 
        Currently only v1.0 is supported.
        """
        self._check_dataset_version(version)
        training_files = [f"crypto/v{version}/{self.TRAIN_TARGETS_NAME}"]
        for file in training_files:
            dest_path = self._get_dest_path(subfolder, file)
            self.download_single_dataset(
                filename=file,
                dest_path=dest_path,
            )

    def download_live_data(
            self,
            subfolder: str = "",
            version: str = "1.0",
    ):
        """
        Download all live data in specified folder (i.e. minimal data needed for inference).

        :param subfolder: Specify folder to create folder within directory root.
        Saves in directory root by default.
        :param version: Numerai dataset version. 
        Currently only v1.0 is supported.
        """
        self._check_dataset_version(version)
        live_files = [f"crypto/v{version}/{self.LIVE_DATASET_NAME}"]
        for file in live_files:
            dest_path = self._get_dest_path(subfolder, file)
            self.download_single_dataset(
                filename=file,
                dest_path=dest_path,
            )

    def download_single_dataset(
        self, filename: str, dest_path: str
    ):
        """
        Download one of the available datasets through CryptoAPI.

        :param filename: Name as listed in CryptoAPI (Check CryptoAPI().list_datasets() for full overview)
        :param dest_path: Full path where file will be saved.
        """
        print(
            f"Downloading '{filename}'."
        )
        self.capi.download_dataset(
            filename=filename,
            dest_path=dest_path,
        )

    def _check_dataset_version(self, version: str):
        assert f"v{version}" in self.dataset_versions, f"Version '{version}' is not available in CryptoAPI."

class KaggleDownloader(BaseDownloader):
    """
    Download financial data from Kaggle.

    For authentication, make sure you have a directory called .kaggle in your home directory
    with therein a kaggle.json file. kaggle.json should have the following structure: \n
    `{"username": USERNAME, "key": KAGGLE_API_KEY}` \n
    More info on authentication: github.com/Kaggle/kaggle-api#api-credentials \n

    More info on the Kaggle Python API: kaggle.com/donkeys/kaggle-python-api \n

    :param directory_path: Base folder to download files to.
    """
    def __init__(self, directory_path: str):
        self.__check_kaggle_import()
        super().__init__(directory_path=directory_path)

    def download_live_data(self, kaggle_dataset_path: str):
        """
        Download arbitrary Kaggle dataset.
        :param kaggle_dataset_path: Path on Kaggle (URL slug on kaggle.com/)
        """
        self.download_training_data(kaggle_dataset_path)

    def download_training_data(self, kaggle_dataset_path: str):
        """
        Download arbitrary Kaggle dataset.
        :param kaggle_dataset_path: Path on Kaggle (URL slug on kaggle.com/)
        """
        import kaggle
        kaggle.api.dataset_download_files(kaggle_dataset_path,
                                          path=self.dir, unzip=True)

    @staticmethod
    def __check_kaggle_import():
        try:
            import kaggle
        except OSError:
            raise OSError("Could not find kaggle.json credentials. Make sure it's located in /home/runner/.kaggle. Or use the environment method. Check github.com/Kaggle/kaggle-api#api-credentials for more information on authentication.")

class EODDownloader(BaseDownloader):
    """
    Download data from EOD historical data. \n
    More info: https://eodhistoricaldata.com/

    Make sure you have the underlying Python package installed.
    `pip install eod`.

    :param directory_path: Base folder to download files to. \n
    :param key: Valid EOD client key. \n
    :param tickers: List of valid EOD tickers (Bloomberg ticker format). \n
    :param frequency: Choose from [d, w, m]. \n
    Daily data by default.
    """
    def __init__(self,
                 directory_path: str,
                 key: str,
                 tickers: list,
                 frequency: str = "d"):
        super().__init__(directory_path=directory_path)
        self.key = key
        self.tickers = tickers
        try: 
            from eod import EodHistoricalData
        except ImportError:
            raise ImportError("Could not import eod package. Please install eod package with 'pip install eod'")
        self.client = EodHistoricalData(self.key)
        self.frequency = frequency
        self.current_time = dt.now()
        self.end_date = self.current_time.strftime("%Y-%m-%d")
        self.cpu_count = os.cpu_count()
        # Time to sleep in between API calls to avoid hitting EOD rate limits.
        # EOD rate limit is set at 1000 calls per minute.
        self.sleep_time = self.cpu_count / 32

    def download_live_data(self):
        """ Download one year of data for defined tickers. """
        start = (pd.Timestamp(self.current_time) - relativedelta(years=1)).strftime("%Y-%m-%d")
        dataf = self.get_numerframe_data(start=start)
        dataf.to_parquet(self._default_save_path(start=pd.Timestamp(start),
                                                 end=pd.Timestamp(self.end_date),
                                                 backend="eod"))

    def download_training_data(self, start: str = None):
        """
        Download full date length available.
        start: Starting data in %Y-%m-%d format.
        """
        start = start if start else "1970-01-01"
        dataf = self.generate_full_dataf(start=start)
        dataf.to_parquet(self._default_save_path(start=pd.Timestamp(start),
                                                 end=pd.Timestamp(self.end_date),
                                                 backend="eod"))

    def get_numerframe_data(self, start: str) -> NumerFrame:
        """
        Get NumerFrame data from some starting date.
        start: Starting data in %Y-%m-%d format.
        """
        dataf = self.generate_full_dataf(start=start)
        return NumerFrame(dataf)

    def generate_full_dataf(self, start: str) -> pd.DataFrame:
        """
        Collect all price data for list of EOD ticker symbols (Bloomberg tickers).
        start: Starting data in %Y-%m-%d format.
        """
        price_datafs = []
        with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
            tasks = [executor.submit(self.generate_stock_dataf, ticker, start) for ticker in self.tickers]
            for task in tqdm(concurrent.futures.as_completed(tasks),
                             total=len(self.tickers),
                             desc="EOD price data extraction"):
                price_datafs.append(task.result())
        return pd.concat(price_datafs)

    def generate_stock_dataf(self, ticker: str, start: str) -> pd.DataFrame:
        """
        Generate Price DataFrame for a single ticker.
        ticker: EOD ticker symbol (Bloomberg tickers).
        For example, Apple stock = AAPL.US.
        start: Starting data in %Y-%m-%d format.
        """
        time.sleep(self.sleep_time)
        try:
            resp = self.client.get_prices_eod(ticker, period=self.frequency,
                                              from_=start, to=self.end_date)
            stock_df = pd.DataFrame(resp).set_index('date')
            stock_df['ticker'] = ticker
        except Exception as e:
            print(f"WARNING: Date pull failed on ticker: '{ticker}'. Exception: {e}")
            stock_df = pd.DataFrame()
        return stock_df


class SyntheticNumeraiData:
    def __init__(self,
                 n_rows_per_era: int = 300,
                 n_features: int = 5,
                 alpha: float = 0.1,
                 n_train_eras: int = 30,
                 n_test_eras: int = 15,
                 split_target: bool = True,
                 train_test_era_gap: int = 4,
                 random_state: int = 42):
        """
        Initializes the SyntheticNumeraiData generator.  Generated Synthetic data does not have same characteristics
        as real world data.

        Args:
            n_rows_per_era (int): Number of rows per era ±10% row variation.
            n_features (int): Number of features to generate.
            alpha (float): Weight of the signal in features (0-1).
            n_train_eras (int): Number of eras for training data before applying the era gap.
            n_test_eras (int): Number of eras for testing data.
            split_target (bool): If True, splits the data into X_train, y_train, X_test, and y_test.
            train_test_era_gap (int): Number of eras to remove from the end of the training set to create a gap.
            random_state (int): Seed for reproducibility.
        """
        self.n_rows_per_era = n_rows_per_era
        self.n_features = n_features
        self.alpha = alpha
        self.n_train_eras = n_train_eras
        self.n_test_eras = n_test_eras
        self.split_target = split_target
        self.random_state = random_state
        self.train_test_era_gap = train_test_era_gap

        self._validate_params()

    def _validate_params(self):
        """
        Validates the input parameters to ensure they meet the required criteria.
        """
        # n_rows_per_era validation
        if not isinstance(self.n_rows_per_era, int) or self.n_rows_per_era < 1:
            raise ValueError("n_rows_per_era must be a positive integer.")

        # n_features validation
        if not isinstance(self.n_features, int) or self.n_features < 1:
            raise ValueError("n_features must be a positive integer.")

        # alpha validation
        if not isinstance(self.alpha, (int, float)) or not (0 <= self.alpha <= 1):
            raise ValueError("alpha must be a float or integer between 0 and 1 (inclusive).")

        # n_train_eras validation
        if not isinstance(self.n_train_eras, int) or self.n_train_eras < 1:
            raise ValueError("n_train_eras must be a positive integer.")

        # n_test_eras validation
        if not isinstance(self.n_test_eras, int) or self.n_test_eras < 1:
            raise ValueError("n_test_eras must be a positive integer.")

        # Ensure the number of rows is enough for the eras
        if self.n_rows_per_era < 5:
            raise ValueError(
                "n_rows_per_era is too small. You need at least 5 rows per era for meaningful data generation.")

        # train_test_era_gap validation
        if not isinstance(self.train_test_era_gap, int) or self.train_test_era_gap < 0:
            raise ValueError("train_test_era_gap must be a non-negative integer.")

        # random_state validation
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise ValueError("random_state must be an integer.")

    def generate(self):
        """
        Generates synthetic data based on the initialized parameters.

        Returns:
            Depending on split_target, either:
            - X_train (pd.DataFrame), y_train (pd.Series), X_test (pd.DataFrame), y_test (pd.Series)
            OR
            - train_data (pd.DataFrame), test_data (pd.DataFrame)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Generate random row variations per era by ±10%
        era_row_counts = np.random.randint(int(self.n_rows_per_era * 0.9), int(self.n_rows_per_era * 1.1),
                                           size=self.n_train_eras + self.n_test_eras)

        # Generate all eras with varying row counts
        eras = np.concatenate([np.full(row_count, era_num + 1)
                               for era_num, row_count in enumerate(era_row_counts)])

        # Generate all targets in one go
        target_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        targets = np.random.choice(target_bins, size=len(eras))

        # Generate all features: signal from the target and noise
        noise = np.random.randint(0, 5, size=(len(eras), self.n_features))
        signals = (self.alpha * targets[:, None] * 4).astype(int)
        features = np.clip(signals + (1 - self.alpha) * noise, 0, 4).astype(int)

        # Create DataFrame using vectorized data
        data = pd.DataFrame(features, columns=[f'feature_x{i}' for i in range(self.n_features)])
        data['era'] = eras
        data['target'] = targets

        # Split data based on eras for train and test sets, with a gap between them
        max_train_era = self.n_train_eras - self.train_test_era_gap
        min_test_era = self.n_train_eras + 1

        if max_train_era < 1:
            raise ValueError("train_test_era_gap is too large, resulting in no training eras left.")

        train_data = data[data['era'] <= max_train_era]
        test_data = data[data['era'] >= min_test_era]

        if self.split_target:
            X_train = train_data.drop(columns=['target'])
            y_train = train_data['target']
            X_test = test_data.drop(columns=['target'])
            y_test = test_data['target']
            return X_train, y_train, X_test, y_test
        else:
            return train_data, test_data

