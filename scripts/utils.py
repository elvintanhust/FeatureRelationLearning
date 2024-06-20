# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import logging
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
MINIMAL = 1e-6
MAXIMAL = 1e6


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :].cuda()


# ----------------------------------------------------------------------------
gauss = [1.33830625e-04, 4.43186162e-03, 5.39911274e-02, 2.41971446e-01,
         3.98943469e-01, 2.41971446e-01, 5.39911274e-02, 4.43186162e-03,
         1.33830625e-04]
gauss_dim = len(gauss)
gauss_dim_half = int(gauss_dim / 2)
min_prob = 1e-8
distance_threshold = int(1.5 * gauss_dim)


def setup_seed(seed=6):
    torch.initial_seed()
    torch.cuda.init()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger(model_dir, name, release, file_name="log.txt"):
    fmt = "%(asctime)-15s %(levelname)s %(message)s"
    date_fmt = "%a %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(fmt, date_fmt)

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    if release:
        logger_path = os.path.join(model_dir, file_name)
        fh = logging.FileHandler(logger_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def load_checkpoint(model, optimizer, lr_scheduler, file_path, logger=None, classifier=None):
    if not os.path.exists(file_path):
        logger and logger.info(f"==============> Resuming form {file_path} failed!....................")
        return 0, 1
    logger and logger.info(f"==============> Resuming form {file_path}....................")
    checkpoint = torch.load(file_path, map_location='cpu')

    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger and logger.info(msg)
    if classifier is not None and "classifier" in checkpoint:
        msg = classifier.load_state_dict(checkpoint['classifier'], strict=False)
        logger and logger.info(msg)
    if classifier is not None and "concepts" in checkpoint:
        classifier.concepts = checkpoint["concepts"]

    max_accuracy = 0.0
    epoch = 1
    if optimizer is not None and lr_scheduler is not None:
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            epoch = checkpoint['epoch']
            logger and logger.info(f"=> loaded successfully (epoch {checkpoint['epoch']})")
            if 'max_accuracy' in checkpoint:
                max_accuracy = checkpoint['max_accuracy']
    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, epoch


def save_checkpoint(epoch, model, optimizer, lr_scheduler, classifier=None,
                    max_accuracy=0.0, folder="./outpout", logger=None, save_type="base"):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch}
    if classifier is not None:
        save_state["classifier"] = classifier.state_dict()
        save_state["concepts"] = classifier.concepts

    if save_type == "best":
        save_path = os.path.join(folder, f'ckpt_best.pth')
    elif save_type == "last":
        save_path = os.path.join(folder, f'ckpt_last.pth')
    elif save_type == "pretrain":
        save_path = os.path.join(folder, f'ckpt_pretrain.pth')
    elif save_type == "epoch":
        save_path = os.path.join(folder, f'ckpt_epoch_{epoch}.pth')
    else:
        save_path = os.path.join(folder, f'{save_type}.pth')
    logger and logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger and logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def adjust_learning_rate(optimizer, epoch, batch=None, nBatch=None, total_epochs=None, lr_init=None):
    T_total = total_epochs * nBatch
    T_cur = (epoch % total_epochs) * nBatch + batch
    lr = 0.5 * lr_init * (1 + math.cos(math.pi * T_cur / T_total))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def cosine_decay(cur, total=None, base=None):
    T_cur = (cur % total)
    ret = 0.5 * base * (1 + math.cos(math.pi * T_cur / total))
    return ret


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


