# Knowledge Distillation 專案說明

## 專案概述

這是一個完整的知識蒸餾（Knowledge Distillation）教學專案，實作了使用大型教師模型訓練小型學生模型的完整流程。

## 技術架構

### 核心元件

1. **資料模組** (`data/`)
   - `datasets.py`: CIFAR-10 資料載入與預處理

2. **模型模組** (`models/`)
   - `teacher.py`: 教師模型（ResNet18 改良版）
   - `student.py`: 學生模型（輕量 CNN）

3. **蒸餾模組** (`distillation/`)
   - `loss.py`: 蒸餾損失函數（KL 散度 + 交叉熵）
   - `trainer.py`: 訓練器實作

4. **工具模組** (`utils/`)
   - `helpers.py`: 輔助函數（種子設定、模型儲存、視覺化等）

## 使用方式

### 在 Google Colab 上運行

1. 上傳專案到 Google Drive 或直接從 GitHub clone
2. 開啟 `notebooks/distillation_tutorial.ipynb`
3. 按照 notebook 指示執行

### 本地運行

```bash
# 建立虛擬環境
python -m venv .venv
source .venv/bin/activate

# 安裝依賴
pip install -r requirements.txt

# 測試模組
python test_modules.py

# 執行範例
python example.py
```

## 核心概念

### 知識蒸餾損失

總損失 = α × 蒸餾損失 + (1-α) × 學生損失

- **蒸餾損失**: KL(Student || Teacher) × T²
- **學生損失**: CrossEntropy(Student, Labels)
- **溫度 (T)**: 軟化機率分佈，傳遞更多資訊
- **Alpha (α)**: 平衡兩種損失的權重

### 訓練流程

1. 訓練教師模型（或載入預訓練權重）
2. 凍結教師模型參數
3. 對每個訓練批次：
   - 取得教師模型的軟標籤
   - 計算蒸餾損失
   - 更新學生模型參數
4. 評估學生模型效能

## 效能指標

在 CIFAR-10 上的預期結果：

- 教師模型: ~92-94% 準確率
- 學生模型（無蒸餾）: ~85-87% 準確率
- 學生模型（有蒸餾）: ~89-91% 準確率
- 參數壓縮比: ~22x

## 開發注意事項

### 程式碼風格

- 所有註解和文件使用繁體中文
- 遵循 PEP 8 風格指南
- 使用 type hints（可選）

### 測試

執行 `test_modules.py` 確保所有模組正常運作。

### 擴展性

本專案設計為易於擴展：

1. **自訂資料集**: 修改 `data/datasets.py`
2. **自訂模型**: 繼承 `nn.Module` 並實作 `forward` 方法
3. **自訂損失**: 修改 `distillation/loss.py`
4. **多教師蒸餾**: 擴展 `trainer.py` 支援多個教師

## 未來改進方向

- [ ] 支援更多資料集（ImageNet, MNIST 等）
- [ ] 實作其他蒸餾方法（FitNet, Attention Transfer 等）
- [ ] 加入模型量化支援
- [ ] 提供預訓練權重下載
- [ ] 增加更多視覺化功能
- [ ] 支援分散式訓練

## 參考文獻

- Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
- Romero et al., "FitNets: Hints for Thin Deep Nets", 2014
- Zagoruyko and Komodakis, "Paying More Attention to Attention", 2016

## 授權

MIT License
