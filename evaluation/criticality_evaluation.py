#!/usr/bin/python3

import time

import torch
import torchvision


def load_imagenet(data_dir: str, subset_size: int,
                  transform: torchvision.transforms.Compose) -> torch.utils.data.DataLoader:
    """Load imagenet from the folder <data_dir>/imagenet """
    # Get a dataset
    test_set = torchvision.datasets.imagenet.ImageNet(root=data_dir + "/imagenet", split="val", transform=transform)
    # test_set = torchvision.datasets.voc.VOCDetection(root=data_dir, download=True, transform=transform)
    subset = torch.utils.data.Subset(test_set, range(subset_size))
    test_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    return test_loader


def main() -> None:
    # Model class must be defined somewhere
    model_path = "../data/c10_resnet20_base_adamw_2-epoch=159-val_acc=0.90.ts"
    golden_model = torch.jit.load(model_path)
    k = 5
    test_loader = load_imagenet(data_dir="../data", subset_size=100,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])]))
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            # Golden execution
            model_time = time.time()
            gold_output = golden_model(image)
            model_time = time.time() - model_time

            gold_top_k_labels = torch.topk(gold_output, k=k).indices.squeeze(0)
            gold_probabilities = torch.tensor(
                [torch.softmax(gold_output, dim=1)[0, idx].item() for idx in gold_top_k_labels])

            # Print gold
            print(gold_top_k_labels)
            print(gold_probabilities)


if __name__ == '__main__':
    main()
