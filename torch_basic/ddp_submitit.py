import os
import sys
import time

import torch

import submitit


def run():
    executor = submitit.AutoExecutor(folder="log_submitit",
                                     slurm_max_num_timeout=20)

if __name__ == '__main__':
    