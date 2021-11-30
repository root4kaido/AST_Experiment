#from kuto
import argparse
import codecs
import json
import logging
import os
import random
from time import time
import requests
import subprocess
from datetime import datetime
import numpy as np
import torch
import yaml

from contextlib import contextmanager
from typing import Union, Optional
from pathlib import Path

# import mlflow
import contextlib
import ipykernel
from notebook import notebookapp
import urllib
import json

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False  # type: ignore
    
# def set_seed(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)  # type: ignore
#     torch.use_deterministic_algorithms(True)
#     torch.backends.cudnn.deterministic = True  # type: ignore
#     torch.backends.cudnn.benchmark = False  # type: ignore



class Timer:
    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' '):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2 # just added 
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    percent = 100 * (start_mem - end_mem) / start_mem
    print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))
    return df

def get_logger(out_file=None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    logger.info("logger set up")
    return logger


def load_config(path: str):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


# get git hash value(short ver.)
def get_hash(config):
    if config['globals']["kaggle"]:
        # kaggle
        hash_value = None
    else:
        # local
        cmd = "git rev-parse --short HEAD"
        hash_value = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    
    return hash_value


def get_timestamp(config):
    # output config
    if config['globals']['timestamp']=='None':
        timestamp = datetime.today().strftime("%m%d_%H%M%S")
    else:
        timestamp = config['globals']['timestamp']
    
    if config["globals"]["debug"] == True:
        timestamp = "debug"
    return timestamp


# 任意のメッセージを通知する関数
def send_slack_message_notification(message):
    webhook_url = os.environ['SLACK_WEBHOOK_URL']  
    data = json.dumps({'text': message})
    headers = {'content-type': 'application/json'}
    requests.post(webhook_url, data=data, headers=headers)

# errorを通知する関数
def send_slack_error_notification(message):
    webhook_url = os.environ['SLACK_WEBHOOK_URL']  
    # no_entry_signは行き止まりの絵文字を出力
    data = json.dumps({"text":":no_entry_sign:" + message})  
    headers = {'content-type': 'application/json'}
    requests.post(webhook_url, data=data, headers=headers)
    
def get_notebook_path():
    '''
    mlflow に notebook をそのまま保存する際に notebook のパスを取得するのに使います。
    単純にファイル名が取れないためかなりややこしいことになっています。
    使用するには jupyterlab が token ありで起動していないといけません。
    '''
    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    for server in notebookapp.list_running_servers():
        try:
            if server['token'] == '' and not server['password']:  # No token and no password, ahem...
                req = urllib.request.urlopen(server['url'] + 'api/sessions')
            elif server['token'] != '':
                req = urllib.request.urlopen(server['url'] + 'api/sessions?token=' + server['token'])
            else:
                continue
            sessions = json.load(req)
            for sess in sessions:
                if sess['kernel']['id'] == kernel_id:
                    return os.path.join(server['notebook_dir'], sess['notebook']['path'])
        except Exception:
            raise
    raise EnvironmentError('JupyterLab の token を指定するか、パスワードを無効化してください。')
    
# def log_notebook():
#     '''
#     notebook を mlflow に保存します。
#     '''
#     notebook = get_notebook_path()  # notebook のパスを取得
#     mlflow.log_artifact(notebook)  # mlflow.log_artifact(file) を呼んでるだけ