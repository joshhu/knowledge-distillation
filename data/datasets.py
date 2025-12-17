"""
資料載入與預處理模組
使用 CIFAR-10 資料集作為示範
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_dataloaders(
    data_dir='./data',
    batch_size=128,
    num_workers=2,
    download=True
):
    """
    取得 CIFAR-10 的訓練和測試資料載入器

    參數:
        data_dir: 資料儲存目錄
        batch_size: 批次大小
        num_workers: 資料載入的執行緒數量
        download: 是否自動下載資料集

    回傳:
        train_loader: 訓練資料載入器
        test_loader: 測試資料載入器
    """
    # 資料增強與正規化
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

    # 載入訓練資料集
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform
    )

    # 載入測試資料集
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform
    )

    # 建立資料載入器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_dataset_info():
    """
    取得 CIFAR-10 資料集的基本資訊

    回傳:
        info: 包含資料集資訊的字典
    """
    info = {
        'num_classes': 10,
        'input_shape': (3, 32, 32),
        'class_names': [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ],
        'train_size': 50000,
        'test_size': 10000
    }
    return info
