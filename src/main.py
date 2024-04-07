import os
from pathlib import Path
from dotenv import load_dotenv
from preprocessing import preprocess_videos
from unsupervised_models import ICA_POH
from postprocessing import ppg2hr_by_window

load_dotenv()

RAW_DATA_DIR = Path(os.getenv('RAW_DATA_DIR'))
PREPROCESSED_DATA_DIR = Path(os.getenv('PREPROCESSED_DATA_DIR'))


def main():
    preprocess_videos(RAW_DATA_DIR, PREPROCESSED_DATA_DIR)


if __name__ == '__main__':
    main()