import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def creat_dataloader(cfg):
    dataset = datasets.ImageFolder(root=cfg.DATA_DIR,
                                   transform=transforms.Compose([
                                       transforms.Resize(cfg.IMAGE_SIZE),
                                       transforms.CenterCrop(cfg.IMAGE_SIZE),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE,
                                             shuffle=True, num_workers=cfg.NUM_WORKERS)
    return dataloader
