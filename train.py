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

from utils.loggers import LOGGERS,Loggers # for logging
from utils.loggers.comet.comet_utils import check_comet_resume # for comet resume
from utils.loss import ComputeLoss # for computing loss
from utils.metrics import fitness # for fitness computation
from utils.plots import plot_evolve # for plotting evolution
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1)) # get the local rank
RANK= int(os.getenv('RANK', -1)) # get the rank
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1)) # get the world size
GIT_INFO = check_git_status() # check the git status

def train(hyp, opt, device, callbacks): # function to train the model
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze= (
            Path(opt.save_dir),
            opt.epochs,
            opt.batch_size,
            opt.weights,
            opt.single_cls,
            opt.evolve,
            opt.data,
            opt.cfg,
            opt.resume,
            opt.noval,
            opt.nosave,
            opt.workers,
            opt.freeze,
    )
    callbacks.run("on_pretrain_routine_start") # run the pretrain routine start callback

    w=save_dir/'weights' # get the weights directory
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True) # create the weights directory
    last, best= w/'last.pt', w/'best.pt' # get the last and best weights

    if isinstance(hyp,str): # if the hyperparameters are a string
        with open(hyp,errors="ignore") as f: # open the hyperparameters file
            hyp = yaml.safe_load(f) # load the hyperparameters
    
    LOGGER.info(colorstr("hyperparameters:")+",".join(f"{k}={v}" for k,v in hyp.items())) # log the hyperparameters
    opt.hyp=hyp.copy() # copy the hyperparameters

    if not evolve: # if not evolving
        yaml_save(save_dir/'hyp.yaml',hyp)  # save the hyperparameters
        yaml_save(save_dir/'opt.yaml',vars(opt))    # save the options
    
    # Loggers
    data_dict=None # initialize the data dictionary
    if RANK in {-1,0}:
        include_loggers = list(LOGGERS) # include the loggers
        if getattr(opt,"ndjson_console",False):# if the ndjson console is enabled
            include_loggers.append("ndjson_console") # include the ndjson console
        if getattr(opt,"ndjson_file",False): # if the ndjson file is enabled
            include_loggers.append("ndjson_file")
        
        loggers = Loggers(
            save_dir=save_dir, # save directory
            weights=weights, # weights
            opt=opt, # options
            hyp=hyp, # hyperparameters
            logger=LOGGER, # logger
            include=tuple(include_loggers), # include the loggers
        )  #创建 Loggers 对象，并传递相关参数。

        for k in methods(loggers):
            callbacks.register_action(k,callback=getattr(loggers,k)) # 使用 callbacks.register_action 方法注册该方法。
        
        data_dict=loggers.remote_dataset  #获取远程数据集
        if resume: 
            weights,epochs,hyp,batch_size=opt.weights,opt.epochs,opt.hyp,opt.batch_size # get the weights, epochs, hyperparameters, and batch size
        
    # config
    plots = not evolve and not opt.noplots # whether to plot
    cuda= device.type != 'cpu' # whether to use cuda
    init_seeds(opt.seed+1+RANK,deterministic=True)  #调用 init_seeds 函数。
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict=data_dict or check_dataset(data)
    
    train_path,val_path=data_dict["train"],data_dict["val"] #
    nc=1 if single_cls else int(data_dict["nc"]) #
    names={0:"item"} if single_cls and len(data_dict["name"])!=1 else data_dict["name"]  #
    is_coco = isinstance(val_path,str) and val_path.endswith("coco/val2017.txt")

    check_suffix(weights,".pt")  
    pretrained = weights.endswitch(".pt")
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK): #上下文管理器
            weights=attempt_download(weights) 
        ckpt = torch.load(weights,map_location="cpu") #
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc,anchors=hyp.get("anchors")).to(device)
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []
        csd = ckpt["model"].float().state_dict()
        csd= intersect_dicts(csd,model.state_dict(),exclude=exclude)
        model.load_state_dict(csd,strict=False)
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")
    else:
        model=Model(cfg,ch=3,nc=nc,anchors=hyp.get("anchors")).to(device)
    amp =check_amp(model)

    #
    # freeze = 





def main(opt,callbacks=Callbacks()): # main function
    
    # Hyperparameters
    print('hello')