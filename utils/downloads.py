
import requests
from pathlib import Path

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