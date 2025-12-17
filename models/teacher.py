"""
教師模型（Teacher Model）
使用較大的 ResNet18 作為教師模型
"""
import torch
import torch.nn as nn
import torchvision.models as models


class TeacherModel(nn.Module):
    """
    教師模型 - 使用 ResNet18
    這是一個較大且效能較好的模型，用於產生軟標籤來指導學生模型
    """
    def __init__(self, num_classes=10, pretrained=False):
        """
        初始化教師模型

        參數:
            num_classes: 分類類別數量
            pretrained: 是否使用預訓練權重
        """
        super(TeacherModel, self).__init__()

        # 使用 ResNet18 作為基礎架構
        self.model = models.resnet18(pretrained=pretrained)

        # 修改第一層以適應 CIFAR-10 的 32x32 輸入
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

        # 修改最後的全連接層以適應類別數
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        前向傳播

        參數:
            x: 輸入張量 (batch_size, 3, 32, 32)

        回傳:
            logits: 未經過 softmax 的原始輸出 (batch_size, num_classes)
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


def create_teacher_model(num_classes=10, device='cuda'):
    """
    建立並初始化教師模型

    參數:
        num_classes: 分類類別數量
        device: 運算裝置 ('cuda' 或 'cpu')

    回傳:
        model: 教師模型實例
    """
    model = TeacherModel(num_classes=num_classes)
    model = model.to(device)

    total_params, trainable_params = model.get_num_parameters()
    print(f"教師模型參數數量: {total_params:,} (可訓練: {trainable_params:,})")

    return model


if __name__ == "__main__":
    # 測試教師模型
    model = TeacherModel(num_classes=10)
    print(model)

    # 測試前向傳播
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    print(f"\n輸入形狀: {x.shape}")
    print(f"輸出形狀: {output.shape}")

    total_params, trainable_params = model.get_num_parameters()
    print(f"\n總參數數量: {total_params:,}")
    print(f"可訓練參數數量: {trainable_params:,}")
