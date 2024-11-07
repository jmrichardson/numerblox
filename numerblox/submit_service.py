import os
import time
import pickle
import schedule
import pandas as pd
from datetime import datetime
from numerapi import NumerAPI
from numerblox.misc import Logger, Key
from numerblox.download import NumeraiClassicDownloader
from numerblox.submission import NumeraiClassicSubmitter


# Setup logger
logger = Logger(log_dir='logs', log_file='submit_service.log').get_logger()


class SubmitService:

    def __init__(self, models, public_id, secret_key, interval_minutes=5, start_hour=13, end_hour=22, tmp_dir="tmp", version="5.0", max_retries=3, sleep_time=10):
        self.models = models
        self.napi = NumerAPI(public_id, secret_key, verbosity="info")
        self.key = Key(pub_id=public_id, secret_key=secret_key)
        self.interval_minutes = interval_minutes
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.version = version
        self.tmp_dir = tmp_dir
        self.max_retries = max_retries  # Set the maximum number of retries
        self.sleep_time = sleep_time  # Set the sleep time between retries
        self.fail_delay = 60 * 60  # 60-minute fail delay before retrying submission process
        self.data_downloader = NumeraiClassicDownloader(directory_path=self.tmp_dir)
        self.submitter = NumeraiClassicSubmitter(directory_path=self.tmp_dir, key=self.key)


    def _validate_args(self):
        # Check public_id and secret_key
        if not isinstance(self.key.pub_id, str) or not isinstance(self.key.secret_key, str):
            raise ValueError("Public ID and Secret Key must be strings.")

        # Check interval_minutes
        if not isinstance(self.interval_minutes, int) or self.interval_minutes <= 0:
            raise ValueError("interval_minutes must be a positive integer.")

        # Check start_hour and end_hour
        if not (0 <= self.start_hour < 24) or not (0 <= self.end_hour < 24):
            raise ValueError("start_hour and end_hour must be integers between 0 and 23.")

        # Check max_retries and sleep_time
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer.")
        if not isinstance(self.sleep_time, int) or self.sleep_time <= 0:
            raise ValueError("sleep_time must be a positive integer.")

        # Check model names
        all_models = self.napi.get_models()
        for model_name, model_data in self.models.items():
            if model_name not in all_models:
                raise ValueError(f"Model name '{model_name}' is not found in NumerAPI models.")

        # Check model paths
        if not isinstance(self.models, dict) or not all(isinstance(model_data.get('path'), str) for model_data in self.models.values()):
            raise ValueError("Each model should be specified in a dictionary with a 'path' key pointing to the model file.")

        # Check model files
        for model_name, model_data in self.models.items():
            model_path = model_data.get('path')

            # Check if model path exists and is a .pkl file
            if not model_path or not model_path.endswith('.pkl'):
                raise ValueError(f"Model path for {model_name} must be a valid .pkl file.")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at path: {model_path}")

            # Load model and check for 'predict' method
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
                if not hasattr(model, 'predict') or not callable(getattr(model, 'predict')):
                    raise ValueError(f"Model object in {model_path} does not have a callable 'predict' method.")

    def check_new_round_with_retry(self):
        """Check for a new Numerai round with retry logic."""
        retries = 0
        while retries < self.max_retries:
            try:
                new_round = self.napi.check_new_round()
                if new_round:
                    logger.info("New Numerai round detected. Proceeding with submission process.")
                    return True
                else:
                    logger.info("No new Numerai round available at this time. Waiting for the next check.")
                    return False
            except Exception as e:
                retries += 1
                logger.error(f"Error checking for new Numerai round: {str(e)}. Attempt {retries}/{self.max_retries}. Retrying in {self.sleep_time} seconds.")
                time.sleep(self.sleep_time)

        logger.error(f"Max retry limit reached for checking new round. Waiting {self.fail_delay / 60} minutes before retrying.")
        time.sleep(self.fail_delay)
        return False

    def download_live_data_with_retry(self, live_data_path):
        """Download live data with retry logic."""
        retries = 0
        while retries < self.max_retries:
            try:
                self.data_downloader.download_live_data(live_data_path, version=self.version)
                logger.info("Live data download successful.")
                return True
            except Exception as e:
                retries += 1
                logger.error(f"Live data download error: {str(e)}. Attempt {retries}/{self.max_retries}. Retrying in {self.sleep_time} seconds.")
                time.sleep(self.sleep_time)

        logger.error(f"Max retry limit reached for live data download. Waiting {self.fail_delay / 60} minutes before retrying.")
        time.sleep(self.fail_delay)
        return False

    def submit_predictions_with_retry(self, predictions_df, model_name):
        """Submit predictions with retry logic."""
        retries = 0
        while retries < self.max_retries:
            try:
                self.submitter.full_submission(dataf=predictions_df, cols="prediction", file_name="submission.csv", model_name=model_name)
                logger.info(f"Predictions successfully submitted for model '{model_name}'.")
                return True
            except Exception as e:
                retries += 1
                logger.error(f"Error during submission for model '{model_name}': {str(e)}. Attempt {retries}/{self.max_retries}. Retrying in {self.sleep_time} seconds.")
                time.sleep(self.sleep_time)

        logger.error(f"Max retry limit reached for predictions submission for model '{model_name}'. Waiting {self.fail_delay / 60} minutes before retrying.")
        time.sleep(self.fail_delay)
        return False

    def task(self):
        """Execute the task to check, download, generate, and submit predictions."""
        if self.check_new_round_with_retry():
            current_round = self.napi.get_current_round()
            live_data_path = f"live/{current_round}"

            logger.info(f"Starting data download and submission process for Numerai round {current_round}.")
            if self.download_live_data_with_retry(live_data_path):
                live_data = pd.read_parquet(f"{self.tmp_dir}/{live_data_path}/live.parquet")
                live_features = live_data[[col for col in live_data.columns if "feature" in col]]

                for model_name, model_data in self.models.items():
                    logger.info(f"Generating predictions for model '{model_name}'.")
                    model_path = model_data['path']
                    with open(model_path, 'rb') as model_file:
                        model = pickle.load(model_file)

                    predictions = model.predict(live_features)
                    predictions_df = pd.DataFrame(predictions, index=live_data.index, columns=['prediction'])

                    if not self.submit_predictions_with_retry(predictions_df, model_name):
                        logger.error(f"Submission failed for model '{model_name}' after retries.")
            else:
                logger.error("Live data download failed after multiple retries.")
        else:
            logger.info("No new Numerai round detected or failed after retries. Submission process not initiated.")

    def daemon(self, validate=True):
        """Start the scheduled submission service, running between specified hours on weekdays."""
        if validate:
            self._validate_args()

        logger.info(f"Starting NumerAI automated submission service. Scheduled every {self.interval_minutes} minutes, active between {self.start_hour}:00 and {self.end_hour}:00 UTC.")
        schedule.every(self.interval_minutes).minutes.do(self.submit)

        while True:
            current_time = datetime.utcnow()
            if current_time.weekday() in range(1, 6) and self.start_hour <= current_time.hour < self.end_hour:
                schedule.run_pending()
            time.sleep(60)

    def submit(self, validate=True):
        logger.info("Initiating submission task.")
        if validate:
            self._validate_args()
        self.task()