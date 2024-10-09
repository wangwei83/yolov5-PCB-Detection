# This file is used to import all the necessary libraries and modules for the training script
# The libraries and modules are imported in the order of their importance

import argparse # for command line arguments
import math # for math functions
import os # for file operations
import random # for random number generation
import subprocess # for running shell commands
import sys # for system-specific parameters and functions
import time # for time functions
from copy import deepcopy # for copying objects
from datetime import datetime,timedelta # for manipulating dates and times
from pathlib import Path # for file path operations

try:
    import comet_ml # for logging to comet.ml
except ImportError:
    comet_ml = None

import numpy as np # for numerical operations
import torch # for deep learning
import torch.distributed as dist # for distributed computing
import torch.nn as nn # for neural network functions
import yaml # for YAML file operations
from torch.optim import lr_scheduler # for learning rate scheduling
from  tqdm import tqdm # for progress bars

FILE = Path(__file__).resolve() # get the current file
ROOT=FILE.parents[0] # get the directory of the file
if str(ROOT) not in sys.path:  
    sys.path.insert(0,str(ROOT)) # add the current file to the system path
ROOT=Path(os.path.relpath(ROOT,Path.cwd())) # get the relative path of the current file

import val as validate
from models.experimental import attempt_load # for loading models
from models.yolo import Model # for YOLO model
from utils.autoanchor import check_anchors # for anchor checking
from utils.autobatch import check_train_batch_size # for checking the training batch size
from utils.callbacks import Callbacks # for callbacks
from utils.dataloaders import create_dataloader # for creating data loaders 
from utils.downloads import attempt_download,is_url # for downloading files
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)

from utils.loggers import Loggers # for logging
