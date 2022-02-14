import yaml
import torch
import torch.nn as nn

from .resnetCIFAR import *
from .LightningModelWrapper import ModelWrapper


def build_model(model='resnet20', n_classes=10, optim_params={}, loss='bce', inject_p=0.1, inject_epoch=0, norm='batch'):
    if model == 'resnet20':
        net = resnet20(n_classes, inject_p, inject_epoch, norm)
    elif model == 'resnet32':
        net = resnet32(n_classes, inject_p, inject_epoch, norm)
    elif model == 'resnet44':
        net = resnet44(n_classes, inject_p, inject_epoch, norm)
    elif model == 'resnet56':
        net = resnet56(n_classes, inject_p, inject_epoch, norm)
    else:
        model = 'resnet20'
        net = resnet20(n_classes, inject_p, inject_epoch, norm)

    print(f'\n    {model} built.')
    return ModelWrapper(net, n_classes, optim_params, loss)


def _parse_args(parser, config_parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
