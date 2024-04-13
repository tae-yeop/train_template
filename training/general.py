import logging
from datetime import datetime
from types import MethodType
import sys
import os
import yaml
import argparse

import torch
import numpy as np
import random

try:
    import wandb
    wandb_avail = True
except ImportError:
    wandb_avail = False
    # pass

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def set_torch_backends(args):
    # Ampare architecture 30xx, a100, h100,..
    if torch.cuda.get_device_capability(0) >= (8, 0):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
    if args.inference : torch.set_grad_enabled(False)


def get_args_with_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='path to config file')
    args = parser.parse_args()
    assert args.config is not None

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument("--random_seed", type=int, help="Random seed.")
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument("--inference", action="store_true", help="Inference mode")
    parser.add_argument("--wandb", action="store_ture", help="Use wandb")

    args = parser.parse_args()
    return args

def save_args_to_yaml(args, filename='saved_config.yaml'):
    args_dict = vars(args)
    with open(filename, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

def log_eval(self, idx, loss, acc):
    self.info(f'{idx} iteration | loss : {loss} | acc : {acc}')
    
def get_logger(expname, log_path, file_log_mode='a'):
    logger = logging.getLogger(expname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)05s %(message)s \n\t--- %(filename)s line: %(lineno)d in %(funcName)s", '%Y-%m-%d %H:%M:%S')

    # 터미널 용 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # 파일 저장용 핸들러
    file_handler = logging.FileHandler(f'{log_path}/experiments.log', mode=file_log_mode)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 시작 메시지
    start_message = f"\n\n{'=' * 50}\nSession Start: {expname} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 50}"
    logger.info(start_message)

    logger.log_eval = MethodType(log_eval, logger)

    return logger


def add_project_dir_to_path(proj_root='project'):
    """
    사실 이건 필요없을듯. 이 폴더에 있는 이 함수를 임포트하려면 sys.path.append를 미리 해둬야 해서
    """
    # 현재 스크립트의 절대 경로
    # abspath 처리 하지 않으면 실행하는 스크립트가 바로 있는 위치를 working dir로 설정했을 시 빈값이 나옴
    script_dir = os.path.abspath(os.path.dirname(__file__))
    # project 디렉토리의 절대 경로를 찾기 위해 계속 상위 디렉토리로 이동
    project_dir = script_dir
    # project라는 폴더가 나올때까지 최상단으로 
    while os.path.basename(project_dir) and os.path.basename(project_dir) != proj_root:
        project_dir = os.path.dirname(project_dir)
        print(project_dir)
    
    assert os.path.basename(project_dir) == proj_root, f'Warning: {proj_root} directory not found. Current script might not work as expected.'
    # project 디렉토리를 찾지 못했을 경우 경고
    # if not os.path.basename(project_dir) or os.path.basename(project_dir) != 'project':
    #     print("Warning: 'project' directory not found. Current script might not work as expected.")
    #     return
    
    # sys.path에 추가. 기본적으로 python을 실행하면 working dir은 포함해준다.
    if project_dir not in sys.path:
        sys.path.append(project_dir)

if __name__ == '__main__':
    # logger = get_logger('exp1', './', 'a')
    # logger.info("Start")
    # try:
    #     x = 1/0
    # except Exception as e:
    #     logger.critical("error!")
    # logger.debug("debuging,...")
    # test_dict = {'name' : "John", 'age': 10}
    # # dict 출력하기
    # logger.debug('Sample dict log: %s', test_dict)
    # logger.log_eval(100, 3, 95)
    import os
    from pathlib import Path
    print('__file__', os.path.dirname(__file__)) # 현재 실행하는 파일이 있는 디렉토리 파일 # 만약 WORKING DIR에 바로 해보면 안됨
    print('2', os.path.join(os.path.dirname(__file__), '..')) # 
    print('3', os.path.dirname(os.path.abspath(__file__)))
    print('4', Path(__file__))
    print('5', os.path.abspath(os.path.dirname(__file__)))

    # 현재 작업 디렉토리의 절대 경로를 얻기
    current_dir = os.getcwd()
    # 현재 작업 디렉토리의 상위 디렉토리 경로를 구성
    parent_dir = os.path.dirname(current_dir)

    print('6', current_dir, parent_dir)

    add_project_dir_to_path('train_template')
    # sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
    sys.path.append(os.path.join(os.getcwd(), '..'))
    print(sys.path)
    from model.util import toggle_grad