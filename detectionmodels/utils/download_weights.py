import os
from google_drive_downloader import GoogleDriveDownloader as gdd


def download_file(file_id, dset_path, unzip=False):
    """Wrapper around gdd download_file_from_google_drive.

    Done so because there is a typo dest_path instead of dset_path. The
    wrapper is to prevent confusion.

    Args:

    """
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=dset_path,
                                        unzip=unzip)


def download_yolov3(save_dir):
    """Downloads the yolov3 weights + model from google drive.

    Typical Usage:
        from tensorflow.keras.models import load_model
        from detectionmodels.utils import download_yolov3

        download_yolov3("./")
        model = load_model("./yolov3.h5")
        model.summary()

    Args:
        save_dir (str): directory to save the file to

    Returns:
        null
    """
    file_id = '1SkPw81gIAQjZTpsxXW1MLUQ0RWRTWm31'
    dset_path = os.path.join(save_dir, 'yolov3.h5')
    download_file(file_id, dset_path, False)


def download_yolov3_tiny(save_dir):
    """Downloads the yolov3 weights + model from google drive.

    Typical Usage:
        from tensorflow.keras.models import load_model
        from detectionmodels.utils import download_yolov3_tiny

        download_yolov3_tiny("./")
        model = load_model("./yolov3-tiny.h5")
        model.summary()

    Args:
        save_dir (str): directory to save the file to

    Returns:
        null
    """
    file_id = '1VqJW4xMWh_hPgNSWyXX4ZwqkhNeau2Ke'
    dset_path = os.path.join(save_dir, 'yolov3-tiny.h5')
    download_file(file_id, dset_path, False)
