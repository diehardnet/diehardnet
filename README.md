# DieHardNET

Repository for a reliable Deep Neural Network (DNN) model. DieHardNet stands for
**D**esign **i**mprov**e**d **Hard**ened neural **Net**work

[comment]: <> (TODO: Replace by two images from john mcclane one 
classified with DieHardNet and other with an error
 l![Die hard photo]&#40;/diehard.jpg&#41;)

## Directories

The directories are organized as follows:

* hg_noise_injector - A module to inject realistic errors in the training process
* eval_fault_injection_cfg - Configuration files for NVBITFI for fault injection
* pytorch_scripts - PyTorch scripts for training and inference for the used DNNs. For information, read
  the [README](/pytorch_scripts/README.md).

## Main script options

```bash
usage: main.py [-h] [--name NAME] [--mode MODE] [--ckpt CKPT] [--dataset DATASET] [--data_dir DATA_DIR] [--device DEVICE] [--loss LOSS] [--clip CLIP] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--optimizer OPTIMIZER] [--model MODEL] [--order ORDER]
               [--affine AFFINE] [--activation ACTIVATION] [--nan NAN] [--error_model ERROR_MODEL] [--inject_p INJECT_P] [--inject_epoch INJECT_EPOCH] [--wd WD] [--rand_aug RAND_AUG] [--rand_erasing RAND_ERASING] [--mixup_cutmix MIXUP_CUTMIX] [--jitter JITTER]
               [--label_smooth LABEL_SMOOTH] [--seed SEED] [--comment COMMENT]

PyTorch Training

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Experiment name.
  --mode MODE           Mode: train/training or validation/validate.
  --ckpt CKPT           Pass the name of a checkpoint to resume training.
  --dataset DATASET     Dataset name: cifar10 or cifar100.
  --data_dir DATA_DIR   Path to dataset.
  --device DEVICE       Device number.
  --loss LOSS           Loss: bce, ce or sce.
  --clip CLIP           Gradient clipping value.
  --epochs EPOCHS       Number of epochs.
  --batch_size BATCH_SIZE
                        Batch Size
  --lr LR               Learning rate.
  --optimizer OPTIMIZER
                        Optimizer name: adamw or sgd.
  --model MODEL         Network name. Resnets only for now.
  --order ORDER         Order of activation and normalization: bn-relu or relu-bn.
  --affine AFFINE       Whether to use Affine transform after normalization or not.
  --activation ACTIVATION
                        Non-linear activation: relu or relu6.
  --nan NAN             Whether to convert NaNs to 0 or not.
  --error_model ERROR_MODEL
                        Optimizer name: adamw or sgd.
  --inject_p INJECT_P   Probability of noise injection at training time.
  --inject_epoch INJECT_EPOCH
                        How many epochs before starting the injection.
  --wd WD               Weight Decay.
  --rand_aug RAND_AUG   RandAugment magnitude and std.
  --rand_erasing RAND_ERASING
                        Random Erasing propability.
  --mixup_cutmix MIXUP_CUTMIX
                        Whether to use mixup/cutmix or not.
  --jitter JITTER       Color jitter.
  --label_smooth LABEL_SMOOTH
                        Label Smoothing.
  --seed SEED           Random seed for reproducibility.
  --comment COMMENT     Optional comment.

```

# To cite this work

**The paper that describes the DieHardNet concept:**

**2022 IEEE 28th International Symposium on On-Line Testing and Robust System Design (IOLTS)**

```bibtex
@INPROCEEDINGS{diehardnetIOLTS2022,
  author={Cavagnero, Niccolò and Santos, Fernando Dos and Ciccone, Marco and Averta, 
          Giuseppe and Tommasi, Tatiana and Rech, Paolo},
  booktitle={2022 IEEE 28th International Symposium on On-Line Testing and Robust System Design (IOLTS)}, 
  title={Transient-Fault-Aware Design and Training to Enhance DNNs Reliability with Zero-Overhead}, 
  year={2022},
  pages={1-7},
  doi={10.1109/IOLTS56730.2022.9897813}
}

```

**The paper that presents the neutron beam validation of DieHardNet:**

**IEEE Transactions on Emerging Topics in Computing**
```bibtex
@article{diehardnetTETC2024,
  TITLE = {{Improving Deep Neural Network Reliability via Transient-Fault-Aware Design and Training}},
  AUTHOR = {Fernandes dos Santos, Fernando and Cavagnero, Niccol{\`o} and Ciccone, Marco and Averta, Giuseppe and Kritikakou, Angeliki and Sentieys, Olivier and Rech, Paolo and Tommasi, Tatiana},
  URL = {https://hal.science/hal-04818068},
  JOURNAL = {{IEEE Transactions on Emerging Topics in Computing}},
  PUBLISHER = {{Institute of Electrical and Electronics Engineers}},
  PAGES = {1-12},
  YEAR = {2024},
  KEYWORDS = {Deep Learning ; Reliability ; Neutrons ; GPUs ; Radiation-induced faults ✦},
  PDF = {https://hal.science/hal-04818068v1/file/tetc_2023_diehardnet.pdf},
  HAL_ID = {hal-04818068},
  HAL_VERSION = {v1},
}
```

# Neutron beam evaluations

The setup files and scripts for validating with neutron beams are available at
[diehardnetradsetup](https://github.com/diehardnet/diehardnetradsetup)
