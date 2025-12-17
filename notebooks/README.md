# Notebook 教學目錄

## 📓 使用說明

由於 Jupyter Notebook 檔案較大，建議直接參考以下資源：

### 方法 1: 使用 Python 腳本教學

我們提供了完整的 Python 腳本教學，可以直接運行：

```bash
# 在專案根目錄執行
python tutorial_script.py
```

### 方法 2: 在 Colab 中手動建立

請參考 `../QUICKSTART.md` 中的完整步驟，在 Google Colab 中：

1. 建立新筆記本
2. 按照 QUICKSTART.md 的步驟執行
3. 每個步驟都有完整的程式碼和說明

### 方法 3: 互動式 Python 環境

如果你想要互動式體驗，可以使用 IPython 或 Jupyter：

```bash
# 安裝 Jupyter
pip install jupyter

# 啟動 Jupyter Notebook
jupyter notebook

# 然後建立新筆記本，匯入我們的模組開始實驗
```

## 📚 教學內容

完整的知識蒸餾教學包含：

1. **環境設定** - 安裝依賴套件
2. **資料載入** - 使用 CIFAR-10 資料集
3. **模型建立** - 教師模型與學生模型
4. **教師訓練** - 訓練高效能的教師模型
5. **知識蒸餾** - 配置蒸餾參數並訓練
6. **結果評估** - 視覺化與效能比較

所有步驟都在 `QUICKSTART.md` 和 `example.py` 中有完整說明。

## 🚀 快速開始

最簡單的方式是執行範例腳本：

```bash
cd ..
python example.py
```

或者查看 `QUICKSTART.md` 在 Colab 上運行。
