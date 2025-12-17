"""
輔助函數
包含常用的工具函數
"""
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os


def set_seed(seed=42):
    """
    設定隨機種子以確保可重現性

    參數:
        seed: 隨機種子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"隨機種子已設定為: {seed}")


def save_checkpoint(model, optimizer, epoch, path, **kwargs):
    """
    儲存模型檢查點

    參數:
        model: 要儲存的模型
        optimizer: 優化器
        epoch: 當前 epoch
        path: 儲存路徑
        **kwargs: 其他要儲存的資訊
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }

    torch.save(checkpoint, path)
    print(f"檢查點已儲存至: {path}")


def load_checkpoint(model, optimizer, path, device='cuda'):
    """
    載入模型檢查點

    參數:
        model: 要載入權重的模型
        optimizer: 優化器
        path: 檢查點路徑
        device: 運算裝置

    回傳:
        checkpoint: 檢查點字典
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"檢查點已載入自: {path}")
    print(f"Epoch: {checkpoint['epoch']}")

    return checkpoint


def plot_training_curves(history, save_path=None):
    """
    繪製訓練曲線

    參數:
        history: 訓練歷史記錄字典
        save_path: 圖片儲存路徑（可選）
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 訓練和測試損失
    axes[0, 0].plot(history['train_loss'], label='訓練損失', marker='o')
    axes[0, 0].plot(history['test_loss'], label='測試損失', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('損失')
    axes[0, 0].set_title('訓練與測試損失')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 訓練和測試準確率
    axes[0, 1].plot(history['train_acc'], label='訓練準確率', marker='o')
    axes[0, 1].plot(history['test_acc'], label='測試準確率', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('準確率 (%)')
    axes[0, 1].set_title('訓練與測試準確率')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 蒸餾損失和學生損失
    axes[1, 0].plot(history['distillation_loss'], label='蒸餾損失', marker='o')
    axes[1, 0].plot(history['student_loss'], label='學生損失', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('損失')
    axes[1, 0].set_title('蒸餾損失與學生損失')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 測試準確率趨勢
    axes[1, 1].plot(history['test_acc'], marker='o', color='green', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('準確率 (%)')
    axes[1, 1].set_title('測試準確率趨勢')
    axes[1, 1].grid(True)
    axes[1, 1].axhline(y=max(history['test_acc']), color='r',
                       linestyle='--', label=f"最佳: {max(history['test_acc']):.2f}%")
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"訓練曲線已儲存至: {save_path}")

    plt.show()


def count_parameters(model):
    """
    計算模型的參數數量

    參數:
        model: PyTorch 模型

    回傳:
        total: 總參數數量
        trainable: 可訓練參數數量
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compare_models(teacher_model, student_model):
    """
    比較教師模型和學生模型的大小

    參數:
        teacher_model: 教師模型
        student_model: 學生模型
    """
    teacher_total, teacher_trainable = count_parameters(teacher_model)
    student_total, student_trainable = count_parameters(student_model)

    print("=" * 60)
    print("模型參數比較")
    print("=" * 60)
    print(f"教師模型: {teacher_total:,} 個參數")
    print(f"學生模型: {student_total:,} 個參數")
    print(f"壓縮比例: {teacher_total/student_total:.2f}x")
    print(f"學生模型僅為教師模型的 {100*student_total/teacher_total:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    # 測試輔助函數
    set_seed(42)
    print("\n輔助函數測試完成!")
