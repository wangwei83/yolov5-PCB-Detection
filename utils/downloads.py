

import urllib
from pathlib import Path
import requests
import torch

def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):

    from utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes} bytes"
    try:
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level<=logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg # check if the file exists and the size is greater than min_bytes
    except Exception as e: #
        if file.exists():
            file.unlink() # remove the file
        LOGGER.info(f'Error:{e} \n Re-attemp {url2 or url} to {file}') 

def attempt_download(file, repo='ultralytics/yolov5',release="v7.0"):  # from utils.downloads import attempt_download
    # Attempt file download from github
    # file: file to download
    # repo: repo to download from
    # Returns: None
    from utils.general import LOGGER

    def github_assets(repositor, version='latest'):
        if version != 'latest':
            version = f'tag/{version}'  # release
        
        response = requests.get(f'https://api.github.com/repos/{repositor}/releases/{version}') # get the release
        return response["tag_name"], [x["name"] for x in response["assets"]] # return the tag name and the assets
    
    file = Path(str(file).strip().replace("'", '')) # get the file path

    if not file.exists():
        name = Path(urllib.parse.unquote(file)).name

        if str(file).startswith(('http:/', 'https:/')):
            url=str(file).replace(':/', '://')
            file=name.split('?')[0] # get the file name
            if Path(file).is_file():
                LOGGER.info(f"Found {url} locally at {file}")
            else:
                safe_download(url, file) # download the file
            return file