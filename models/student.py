"""
學生模型（Student Model）
使用較小的網路架構，透過知識蒸餾從教師模型學習
"""
import torch
import torch.nn as nn


class SimpleConvNet(nn.Module):
    """
    簡單的卷積神經網路作為學生模型
    參數量遠少於 ResNet18，但透過知識蒸餾可以獲得接近的效能
    """
    def __init__(self, num_classes=10):
        """
        初始化學生模型

        參數:
            num_classes: 分類類別數量
        """
        super(SimpleConvNet, self).__init__()

        # 第一個卷積區塊
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二個卷積區塊
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第三個卷積區塊
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全連接層
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        前向傳播

        參數:
            x: 輸入張量 (batch_size, 3, 32, 32)

        回傳:
            logits: 未經過 softmax 的原始輸出 (batch_size, num_classes)
        """
        x = self.conv1(x)  # (batch, 32, 16, 16)
        x = self.conv2(x)  # (batch, 64, 8, 8)
        x = self.conv3(x)  # (batch, 128, 4, 4)
        x = x.view(x.size(0), -1)  # (batch, 128*4*4)
        x = self.fc(x)  # (batch, num_classes)
        return x


class StudentModel(nn.Module):
    """
    學生模型包裝類別
    提供與教師模型一致的介面
    """
    def __init__(self, num_classes=10):
        """
        初始化學生模型

        參數:
            num_classes: 分類類別數量
        """
        super(StudentModel, self).__init__()
        self.model = SimpleConvNet(num_classes=num_classes)

    def forward(self, x):
        """
        前向傳播

        參數:
            x: 輸入張量

        回傳:
            輸出張量
        """
        return self.model(x)

    def get_num_parameters(self):
        """
        取得模型的參數數量

        回傳:
            total_params: 總參數數量
            trainable_params: 可訓練參數數量
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def create_student_model(num_classes=10, device='cuda'):
    """
    建立並初始化學生模型

    參數:
        num_classes: 分類類別數量
        device: 運算裝置 ('cuda' 或 'cpu')

    回傳:
        model: 學生模型實例
    """
    model = StudentModel(num_classes=num_classes)
    model = model.to(device)

    total_params, trainable_params = model.get_num_parameters()
    print(f"學生模型參數數量: {total_params:,} (可訓練: {trainable_params:,})")

    return model


if __name__ == "__main__":
    # 測試學生模型
    model = StudentModel(num_classes=10)
    print(model)

    # 測試前向傳播
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    print(f"\n輸入形狀: {x.shape}")
    print(f"輸出形狀: {output.shape}")

    total_params, trainable_params = model.get_num_parameters()
    print(f"\n總參數數量: {total_params:,}")
    print(f"可訓練參數數量: {trainable_params:,}")
