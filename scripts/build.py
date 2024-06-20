# -*- coding:utf-8 -*-
import os
import subprocess
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from timm.data import Mixup
from timm.data.transforms import _pil_interp
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import random


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


class MultiStepLRScheduler(Scheduler):
    """
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 decay_t: float,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 milestones=[],
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.decay_t = decay_t
        self.decay_rate = decay_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        self.milestones = milestones
        self.last_values = None
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        t = int(t)
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if t in self.milestones:
                ind = self.milestones.index(t) + 1
                lrs = [v * (self.decay_rate ** (ind)) for v in self.base_values]
            else:
                lrs = self.last_values
        self.last_values = lrs
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


class Sparse_SGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super(Sparse_SGD, self).__init__(*args, **kwargs)
        self.weight_mask = dict()

    def update_mask(self, w, m):
        self.weight_mask[w] = m

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if p in self.weight_mask:
                        mask = self.weight_mask[p].expand_as(d_p)
                    else:
                        mask = torch.ones_like(d_p)
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p*mask).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p*mask, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.add_(d_p, alpha=-group['lr'])
        return loss


def build_optimizer(deep_model, config):
    if config.TRAIN.OPTIMIZER.NAME.upper() == "AdamW".upper():
        optimizer = torch.optim.AdamW(deep_model.parameters(), lr=config.TRAIN.BASE_LR,
                                      betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
    else:  # sgd
        optimizer = Sparse_SGD(deep_model.parameters(), lr=config.TRAIN.BASE_LR, momentum=0.9, weight_decay=1e-4)
    return optimizer


def build_scheduler(optimizer, config):
    if config.TRAIN.LR_SCHEDULER.NAME.upper() == 'linear'.upper():
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=config.TRAIN.EPOCHS,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=config.TRAIN.WARMUP_EPOCHS,
            t_in_epochs=True,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME.upper() == 'step'.upper():
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=config.TRAIN.WARMUP_EPOCHS,
            t_in_epochs=True,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME.upper() == "cosine".upper():
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=config.TRAIN.EPOCHS,
            t_mul=1.,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=config.TRAIN.WARMUP_EPOCHS,
            cycle_limit=1,
            t_in_epochs=True,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME.upper() == "plateau".upper():
        lr_scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=15, verbose=1, min_lr=config.TRAIN.MIN_LR
        )
    else:  # multi-step
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=config.TRAIN.WARMUP_EPOCHS,
            t_in_epochs=True,
            milestones=config.TRAIN.LR_SCHEDULER.MILESTONES,
        )
    return lr_scheduler


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset(config, need_aug=True):
    data_name = config.DATA.NAME
    train_data, test_data = None, None
    data_root = "../data"
    if data_name.upper() == "cifar".upper():
        train_data = CIFAR10(os.path.join(data_root, 'cifar10'), train=True, download=True,
                             transform=build_transform(need_aug, config), target_transform=None)
        test_data = CIFAR10(os.path.join(data_root, 'cifar10'), train=False, download=True,
                            transform=build_transform(False, config), target_transform=None)

    train_load = DataLoader(
        train_data, batch_size=config.DATA.BATCH_SIZE, pin_memory=config.DATA.PIN_MEMORY,
        shuffle=True, num_workers=config.DATA.NUM_WORKERS, drop_last=True)

    test_load = DataLoader(
        test_data, batch_size=config.DATA.TEST_BATCH_SIZE, pin_memory=config.DATA.PIN_MEMORY,
        shuffle=False, num_workers=config.DATA.NUM_WORKERS)

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return train_load, test_load, mixup_fn


def build_model(config):
    if config.MODEL.TYPE.upper() == "plasticity".upper():
        if config.MODEL.NAME.upper() == "vgg".upper():
            from scripts.deep_model.vgg_cifar import VGG
            deep_model = VGG(num_classes=config.MODEL.NUM_CLASSES)
        if config.MODEL.NAME.upper() == "resnet".upper():
            from scripts.deep_model.resnet import resnet18
            deep_model = resnet18(num_classes=config.MODEL.NUM_CLASSES)
    deep_model = deep_model.cuda()
    deep_model = torch.nn.DataParallel(deep_model)
    return deep_model


def mask_conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    sparse_ratio = torch.sum(conv_module.mask) / (conv_module.mask.size(0) * conv_module.mask.size(1))

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops * sparse_ratio + bias_flops

    conv_module.__flops__ += int(overall_flops)

