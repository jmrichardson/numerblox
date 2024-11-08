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
from datetime import timedelta



# Setup logger
logger = Logger(log_dir='logs', log_file='submit_service.log').get_logger()


class SubmitService:

    def __init__(self, models, public_id, secret_key, interval_minutes=5, start_hour=13, end_hour=23, tmp_dir="tmp", version="5.0", max_retries=3, sleep_time=10, cleanup=False):
        self.models = models
        self.napi = NumerAPI(public_id, secret_key, verbosity="info")
        self.key = Key(pub_id=public_id, secret_key=secret_key)
        self.interval_minutes = interval_minutes
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.version = version
        self.tmp_dir = tmp_dir
        self.max_retries = max_retries
        self.sleep_time = sleep_time
        self.fail_delay = 60 * 60  # 60-minute fail delay before retrying submission process
        self.data_downloader = NumeraiClassicDownloader(directory_path=self.tmp_dir)
        self.submitter = NumeraiClassicSubmitter(directory_path=self.tmp_dir, key=self.key)
        self.cleanup = cleanup

        # Store the last successfully submitted round
        self.last_submitted_round = self._load_last_successful_round()


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

    def _load_last_successful_round(self):
        """Load the last successfully submitted round from disk, if it exists."""
        round_file_path = os.path.join(self.tmp_dir, 'last_successful_round.txt')
        if os.path.exists(round_file_path):
            with open(round_file_path, 'r') as file:
                try:
                    return int(file.read().strip())
                except ValueError:
                    logger.warning("Failed to read last successful round from file. Starting fresh.")
        return None

    def _save_last_successful_round(self, round_number):
        """Save the last successful round to a file."""
        round_file_path = os.path.join(self.tmp_dir, 'last_successful_round.txt')
        with open(round_file_path, 'w') as file:
            file.write(str(round_number))
        logger.info(f"Saved last successful round {round_number} to disk.")

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

    def submission_success_path(self, live_data_path):
        """Return the path to the submission success file for the round."""
        return os.path.join(self.tmp_dir, live_data_path, 'submission_success.txt')

    def save_submission_success(self, live_data_path, current_round, model_names):
        """Save a record of a successful submission to a file."""
        success_path = self.submission_success_path(live_data_path)
        timestamp = datetime.utcnow().isoformat()
        content = f"Round: {current_round}\nTimestamp: {timestamp}\nModels: {', '.join(model_names)}\n"

        with open(success_path, 'w') as file:
            file.write(content)

        logger.info(f"Submission success saved to {success_path} with details:\n{content}")
        self._save_last_successful_round(current_round)  # Update the last submitted round

    def get_current_round_with_retry(self):
        """Get the current Numerai round with retry logic."""
        retries = 0
        while retries < self.max_retries:
            try:
                current_round = self.napi.get_current_round()
                return current_round
            except Exception as e:
                retries += 1
                logger.error(f"Error getting current Numerai round: {str(e)}. Attempt {retries}/{self.max_retries}. Retrying in {self.sleep_time} seconds.")
                time.sleep(self.sleep_time)
        logger.error("Max retries reached for getting current Numerai round.")
        return None

    def task(self):
        """Execute the task to check, download, generate, and submit predictions."""
        current_round = self.get_current_round_with_retry()
        if current_round is None:
            logger.error("Failed to get current Numerai round. Skipping task.")
            return False

        # Skip processing if we've already submitted for the current round
        if self.last_submitted_round == current_round:
            logger.info(f"Already submitted for Numerai round {current_round}. Skipping.")
            return True

        if not self.check_new_round_with_retry():
            logger.info(f"No new Numerai round detected. Current round is {current_round}. Skipping submission.")
            return False

        live_data_path = f"live/{current_round}"
        logger.info(f"Starting data download and submission process for Numerai round {current_round}.")

        if self.download_live_data_with_retry(live_data_path):
            live_data = pd.read_parquet(f"{self.tmp_dir}/{live_data_path}/live.parquet")
            live_features = live_data[[col for col in live_data.columns if "feature" in col]]

            all_successful = True
            for model_name, model_data in self.models.items():
                logger.info(f"Generating predictions for model '{model_name}'.")
                model_path = model_data['path']
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)

                predictions = model.predict(live_features)
                predictions_df = pd.DataFrame(predictions, index=live_data.index, columns=['prediction'])

                if not self.submit_predictions_with_retry(predictions_df, model_name):
                    logger.error(f"Submission failed for model '{model_name}' after retries.")
                    all_successful = False
                    return False

            if all_successful:
                self.save_submission_success(live_data_path, current_round, list(self.models.keys()))
                logger.info(f"Submission successful for all models in Numerai round {current_round}.")
                self.last_submitted_round = current_round  # Update last submitted round in memory

                # Clean up temporary files
                if self.cleanup:
                    try:
                        shutil.rmtree(os.path.join(self.tmp_dir, live_data_path))
                        logger.info(f"Cleaned up temporary files for round {current_round}.")
                    except Exception as e:
                        logger.error(f"Error cleaning up temporary files: {str(e)}")
                return True
        else:
            logger.error("Live data download failed after multiple retries.")
            return False

    def daemon(self, validate=True):
        if validate:
            self._validate_args()

        logger.info(f"Starting NumerAI automated submission service. Active every {self.interval_minutes} minutes between {self.start_hour}:00 and {self.end_hour}:00 UTC.")

        while True:
            current_time = datetime.utcnow()

            # Check if the current time is within the specified hours and it's a weekday
            if current_time.weekday() in range(1, 6) and self.start_hour <= current_time.hour < self.end_hour:
                task_completed = self.task()
                if task_completed:
                    # Calculate the next start time one minute before the start hour on the next day
                    next_start_time = datetime.combine(current_time.date() + timedelta(days=1), datetime.min.time()) + timedelta(hours=self.start_hour)
                    time_to_wait = (next_start_time - current_time).total_seconds()

                    # Convert `time_to_wait` to hours and minutes for readability
                    hours, remainder = divmod(time_to_wait, 3600)
                    minutes, _ = divmod(remainder, 60)

                    logger.info(f"Task completed successfully. Sleeping until {next_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC (in {int(hours)} hours and {int(minutes)} minutes) for the next submission.")
                    time.sleep(time_to_wait)

                else:
                    # Wait for the specified interval before trying again if task was not successful
                    logger.info(f"Submission failed. Waiting {self.interval_minutes} minutes to retry.")
                    time.sleep(self.interval_minutes * 60)
            else:
                # If outside the allowed time window, calculate the wait time until the next start hour
                next_start_time = datetime.combine(current_time.date() + timedelta(days=1), datetime.min.time()) + timedelta(hours=self.start_hour)
                time_to_wait = (next_start_time - current_time).total_seconds()

                # Convert seconds to hours and minutes for readability
                hours, remainder = divmod(time_to_wait, 3600)
                minutes, _ = divmod(remainder, 60)

                logger.info(f"Outside active hours. Waiting for start time: {int(hours)} hours and {int(minutes)} minutes from now.")
                # Sleep until the calculated time
                time.sleep(time_to_wait)


    def submit(self, validate=True):
        logger.info("Initiating submission task.")
        if validate:
            self._validate_args()
        self.task()