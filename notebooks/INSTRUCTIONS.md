# 如何在 Google Colab 建立知識蒸餾 Notebook

由於 Jupyter Notebook 檔案格式的限制，請按照以下步驟在 Colab 手動建立教學 notebook。

## 步驟 1: 在 Colab 建立新筆記本

1. 前往 [Google Colab](https://colab.research.google.com)
2. 點選「新增筆記本」
3. 重新命名為「知識蒸餾教學」

## 步驟 2: 設定環境

在第一個 cell 中執行：

```python
# 安裝依賴
!pip install -q torch torchvision tqdm matplotlib

# Clone 專案
!git clone https://github.com/joshhu/knowledge-distillation.git
%cd knowledge-distillation

print("✓ 環境設定完成！")
```

## 步驟 3: 匯入模組

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import get_cifar10_dataloaders, get_dataset_info
from models import TeacherModel, StudentModel
from distillation import DistillationLoss, DistillationTrainer
from utils import set_seed, compare_models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用裝置: {device}')
set_seed(42)
```

## 步驟 4: 載入資料

```python
train_loader, test_loader = get_cifar10_dataloaders(
    data_dir='./data',
    batch_size=128,
    num_workers=2
)

info = get_dataset_info()
print(f"類別數: {info['num_classes']}")
print(f"訓練樣本: {info['train_size']}")
```

## 步驟 5: 建立模型

```python
teacher_model = TeacherModel(num_classes=10).to(device)
student_model = StudentModel(num_classes=10).to(device)

compare_models(teacher_model, student_model)
```

## 步驟 6-9: 完整步驟

請參考專案根目錄的 `QUICKSTART.md` 檔案，裡面有完整的步驟說明，包括：
- 訓練教師模型
- 配置知識蒸餾
- 開始訓練
- 視覺化結果

## 或者使用現成腳本

如果你想快速開始，可以直接執行：

```python
!python example.py
```

這會執行完整的知識蒸餾流程。
