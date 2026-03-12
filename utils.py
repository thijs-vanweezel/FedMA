import os
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from itertools import product
import math
import copy
import time
from sklearn.metrics import confusion_matrix

# we've changed to a faster solver
#from scipy.optimize import linear_sum_assignment
import logging

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from datasets import CIFAR10_truncated, ImageNet, ImageNet_truncated
from combine_nets import prepare_uniform_weights, prepare_sanity_weights, prepare_weight_matrix, normalize_weights, get_weighted_average_pred

from model import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def parse_class_dist(net_class_config):

    cls_net_map = {}

    for net_idx, net_classes in enumerate(net_class_config):
        for net_cls in net_classes:
            if net_cls not in cls_net_map:
                cls_net_map[net_cls] = []
            cls_net_map[net_cls].append(net_idx)

    return cls_net_map

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

def partition_data(dataset, datadir, logdir, partition, n_nets, alpha, args):

    y_train, n_train = load_imagenet_data(datadir, n_clients=n_nets)

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero-dir":
        # Each client receives every n_nets-th sample starting at its index,
        # which naturally assigns interleaved label groups (see ImageNet dataset).
        net_dataidx_map = {}
        for c in range(n_nets):
            idxs = np.arange(c, n_train, n_nets)
            idxs = idxs[np.random.permutation(len(idxs))]
            net_dataidx_map[c] = idxs

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return y_train, net_dataidx_map, traindata_cls_counts

def load_imagenet_data(datadir, n_clients):
    """Return a flat y_train array (in interleaved order) and its length.

    The ImageNet dataset stores samples interleaved by client: sample at
    global index i belongs to client i % n_clients, so
    y_train[i] = ds.targets[i % n_clients][i // n_clients].
    """
    ds = ImageNet(partition="train", n_clients=n_clients)
    n = len(ds)
    y_train = np.array([
        ds.targets[i % n_clients][i // n_clients] for i in range(n)
    ])
    return y_train, n

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            out = model(x)
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())               

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)


def init_models(net_configs, n_nets, args):
    '''
    Initialise local models and build the matched-layer metadata used by the
    FedMA matching algorithm (model_meta_data, layer_type).

    For ResNet the metadata is constructed explicitly from the same layer
    ordering as pdm_prepare_full_weights_cnn:
      (initial conv weight, initial bn.bias,
       block_0 conv1 weight, block_0 bn1.bias, block_0 conv2 weight, block_0 bn2.bias,
       ...
       fc weight.T, fc bias)
    id_conv (shortcut projection) and BN gamma/running stats are excluded; they
    are handled separately during weight loading and local retraining.
    '''

    cnns = {net_i: None for net_i in range(n_nets)}
    model_meta_data = []
    layer_type = []

    if args.model == 'resnet':
        for cnn_i in range(n_nets):
            cnns[cnn_i] = ResNet()
        ref = cnns[0]
        # --- initial stem ---
        w = ref.conv.weight
        model_meta_data.append(tuple(w.shape))  # (n_out, n_in, kH, kW) — 4D needed by block_patching
        layer_type.append('conv.weight')
        model_meta_data.append(tuple(ref.bn.bias.shape))
        layer_type.append('conv.bias')
        # --- residual blocks ---
        for block in ref.layers:
            for conv_attr, bn_attr in [('conv1', 'bn1'), ('conv2', 'bn2')]:
                conv = getattr(block, conv_attr)
                bn   = getattr(block, bn_attr)
                w = conv.weight
                model_meta_data.append(tuple(w.shape))  # (n_out, n_in, kH, kW) — 4D needed by block_patching
                layer_type.append(f'{conv_attr}.weight')
                model_meta_data.append(tuple(bn.bias.shape))
                layer_type.append(f'{conv_attr}.bias')
        # --- classifier ---
        model_meta_data.append(tuple(ref.fc.weight.T.shape))
        layer_type.append('fc.weight')
        model_meta_data.append(tuple(ref.fc.bias.shape))
        layer_type.append('fc.bias')
    else:  # moderate-cnn
        for cnn_i in range(n_nets):
            cnns[cnn_i] = ModerateCNN()
        for (k, v) in cnns[0].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)

    return cnns, model_meta_data, layer_type


def save_model(model, model_index):
    logger.info("saving local model-{}".format(model_index))
    with open("trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model(model, model_index, rank=0, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, **kwargs):
    # Pop ImageNet-specific kwargs so they are not forwarded to DataLoader.
    n_clients = kwargs.pop("n_clients", None)
    assert dataidxs is None or n_clients is not None, \
        "n_clients must be provided when dataidxs is not None"
    img_kwargs = {} if n_clients is None else {"n_clients": n_clients}
    train_ds = ImageNet_truncated(dataidxs=dataidxs, partition="train", **img_kwargs)
    test_ds  = ImageNet_truncated(dataidxs=None, partition="val", **img_kwargs)
    train_dl = data.DataLoader(
        train_ds,
        batch_size=train_bs,
        drop_last=True,
        shuffle=False,
        sampler=None,
        **kwargs
    )
    test_dl = data.DataLoader(
        test_ds,
        batch_size=test_bs,
        drop_last=False,
        shuffle=False,
        sampler=None,
        **kwargs
    )
    return train_dl, test_dl


def pdm_prepare_full_weights_cnn(nets, device="cpu"):
    """
    Extract the matched-layer weight list from each network.

    For ModerateCNN the ordering follows the state_dict:
      (conv.weight_2d, conv.bias, ..., fc.weight.T, fc.bias)

    For ResNet / ResNetContainer the ordering mirrors init_models:
      (initial_conv_weight_2d, initial_bn.bias,
       block_0_conv1_weight_2d, block_0_bn1.bias,
       block_0_conv2_weight_2d, block_0_bn2.bias,
       ...
       fc.weight.T, fc.bias)
    id_conv (shortcut projection) weights and BN gamma / running stats are
    intentionally excluded – they are not directly matched by the algorithm.
    """
    def _np(t):
        """Return a numpy array regardless of CPU / GPU placement."""
        return t.detach().cpu().numpy()

    weights = []
    for net_i, net in enumerate(nets):
        net_weights = []

        if isinstance(net, (ResNet, ResNetContainer)):
            # --- initial stem ---
            w = net.conv.weight
            net_weights.append(_np(w).reshape(w.shape[0], -1))
            net_weights.append(_np(net.bn.bias))
            # --- residual blocks (main path only) ---
            for block in net.layers:
                for conv_attr, bn_attr in [('conv1', 'bn1'), ('conv2', 'bn2')]:
                    conv = getattr(block, conv_attr)
                    bn   = getattr(block, bn_attr)
                    w = conv.weight
                    net_weights.append(_np(w).reshape(w.shape[0], -1))
                    net_weights.append(_np(bn.bias))
            # --- classifier ---
            net_weights.append(_np(net.fc.weight).T)
            net_weights.append(_np(net.fc.bias))

        else:  # ModerateCNN / ModerateCNNContainer
            statedict = net.state_dict()
            for param_id, (k, v) in enumerate(statedict.items()):
                if 'fc' in k or 'classifier' in k:
                    if 'weight' in k:
                        net_weights.append(_np(v).T)
                    else:
                        net_weights.append(_np(v))
                elif 'conv' in k or 'features' in k:
                    if 'weight' in k:
                        _weight_shape = v.size()
                        if len(_weight_shape) == 4:
                            net_weights.append(_np(v).reshape(
                                _weight_shape[0],
                                _weight_shape[1] * _weight_shape[2] * _weight_shape[3]))
                    else:
                        net_weights.append(_np(v))

        weights.append(net_weights)
    return weights


def pdm_prepare_freq(cls_freqs, n_classes):
    freqs = []

    for net_i in sorted(cls_freqs.keys()):
        net_freqs = [0] * n_classes

        for cls_i in cls_freqs[net_i]:
            net_freqs[cls_i] = cls_freqs[net_i][cls_i]

        freqs.append(np.array(net_freqs))

    return freqs


def compute_ensemble_accuracy(models: list, dataloader, n_classes, train_cls_counts=None, uniform_weights=False, sanity_weights=False, device="cpu"):

    correct, total = 0, 0
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    was_training = [False]*len(models)
    for i, model in enumerate(models):
        if model.training:
            was_training[i] = True
            model.eval()

    if uniform_weights is True:
        weights_list = prepare_uniform_weights(n_classes, len(models))
    elif sanity_weights is True:
        weights_list = prepare_sanity_weights(n_classes, len(models))
    else:
        weights_list = prepare_weight_matrix(n_classes, train_cls_counts)

    weights_norm = normalize_weights(weights_list)

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            target = target.long()
            out = get_weighted_average_pred(models, weights_norm, x, device=device)

            _, pred_label = torch.max(out, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    #logger.info(correct, total)

    conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    for i, model in enumerate(models):
        if was_training[i]:
            model.train()

    return correct / float(total), conf_matrix


class ModerateCNNContainerConvBlocks(nn.Module):
    def __init__(self, num_filters, output_dim=10):
        super(ModerateCNNContainerConvBlocks, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=num_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x