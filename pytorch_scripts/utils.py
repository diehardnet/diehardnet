import platform
import re
import yaml
from timm.data import create_loader, FastCollateMixup

from .LightningModelWrapper import ModelWrapper

from pytorch_scripts.hg_noise_injector.hook_injection import Injector


def build_model(model=None, n_classes=10, optim_params={}, loss='bce', error_model='random', inject_p=0.1, inject_epoch=0,
                clip=False, nan=False):

    if model == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(weights="IMAGENET1K_V2")
    elif model == 'efficientnet':
        from torchvision.model import efficientnet_v2_s
        model = efficientnet_v2_s(weights='IMAGENET1K_V1')

    net = Injector(model, error_model, inject_p, inject_epoch, clip, nan)
    print(f'\n==> {model} built.')
    return ModelWrapper(net, n_classes, optim_params, loss)


def get_loader(data, batch_size=128, workers=4, n_classes=100, stats=None, mixup_cutmix=True, rand_erasing=0.0,
               label_smooth=0.1, rand_aug='rand-m9-mstd0.5-inc1', jitter=0.0, rcc=0.75, size=32, fp16=True):
    if mixup_cutmix:
        mixup_alpha = 0.8
        cutmix_alpha = 1.0
        prob = 1.0
        switch_prob = 0.5
    else:
        mixup_alpha = 0.0
        cutmix_alpha = 0.0
        prob = 0.0
        switch_prob = 0.0
    collate = FastCollateMixup(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, cutmix_minmax=None, prob=prob,
                               switch_prob=switch_prob, mode='batch', label_smoothing=label_smooth,
                               num_classes=n_classes)
    return create_loader(data, input_size=(3, size, size),
                         batch_size=batch_size,
                         is_training=True,
                         use_prefetcher=True,
                         no_aug=False,
                         re_prob=rand_erasing,  # RandErasing
                         re_mode='pixel',
                         re_count=1,
                         re_split=False,
                         scale=[rcc, 1.0],  #[0.08, 1.0] if size != 32 else [0.75, 1.0],
                         ratio=[3./4., 4./3.],
                         hflip=0.5,
                         vflip=0,
                         color_jitter=jitter,
                         auto_augment=rand_aug,
                         num_aug_splits=0,
                         interpolation='random',
                         mean=stats[0],
                         std=stats[1],
                         num_workers=workers,
                         distributed=True,
                         collate_fn=collate,
                         pin_memory=True,
                         use_multi_epochs_loader=False,
                         fp16=fp16)


def parse_args(parser, config_parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args


def get_default_data_root():
    node = platform.node()

    sysname = node
    if re.match(r'compute-\d+-\d+', sysname):
        sysname = 'legion'
    elif re.match(r'node\d', sysname):
        sysname = 'clustervandal'
    elif re.match(r'r\d+n\d+', sysname):
        sysname = 'marconi100'
    elif re.match(r'^(?:[fg]node|franklin)\d{2}$', sysname):
        sysname = 'franklin'

    # TODO Percorsi dei dataset sulle nostre macchine.
    paths = {
        # Lab workstations
        'demetra': '/data/lucar/datasets',
        'poseidon': '/data/lucar/datasets',
        'nike': '/home/lucar/datasets',
        'athena': '/home/lucar/datasets',
        'terpsichore': '/data/lucar/datasets',
        'atlas': '/data/lucar/datasets',
        'kronos': '/data/datasets',

        # Clusters
        'legion': '/home/lrobbiano/datasets',
        'franklin': '/projects/vandal/nas/datasets',

        # Personal (for debugging)
        'carbonite': '/home/luca/datasets'
    }

    return paths.get(sysname, None)
