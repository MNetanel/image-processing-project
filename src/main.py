from config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR
from preprocessing import preprocess_videos

def main():
    preprocess_videos(RAW_DATA_DIR, PREPROCESSED_DATA_DIR)


if __name__ == '__main__':
    main()