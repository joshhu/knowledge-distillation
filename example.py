"""
知識蒸餾簡單範例
展示如何使用本專案進行知識蒸餾
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# 匯入專案模組
from data import get_cifar10_dataloaders, get_dataset_info
from models import TeacherModel, StudentModel
from distillation import DistillationLoss, DistillationTrainer
from utils import set_seed, plot_training_curves, compare_models


def main():
    """主函數"""
    print("=" * 70)
    print("知識蒸餾範例程式")
    print("=" * 70)

    # 1. 設定環境
    print("\n[1/7] 設定環境...")
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")

    # 2. 載入資料
    print("\n[2/7] 載入 CIFAR-10 資料集...")
    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir='./data',
        batch_size=128,
        num_workers=2
    )

    info = get_dataset_info()
    print(f"訓練樣本: {info['train_size']}, 測試樣本: {info['test_size']}")

    # 3. 建立模型
    print("\n[3/7] 建立教師模型和學生模型...")
    teacher_model = TeacherModel(num_classes=10).to(device)
    student_model = StudentModel(num_classes=10).to(device)

    print("\n模型比較:")
    compare_models(teacher_model, student_model)

    # 4. 配置知識蒸餾
    print("\n[4/7] 配置知識蒸餾參數...")
    temperature = 3.0
    alpha = 0.7
    num_epochs = 5  # 示範用，實際訓練建議 50+ epochs

    distillation_loss = DistillationLoss(
        temperature=temperature,
        alpha=alpha
    )

    optimizer = optim.SGD(
        student_model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"溫度參數 (Temperature): {temperature}")
    print(f"Alpha (蒸餾權重): {alpha}")
    print(f"訓練輪數: {num_epochs}")

    # 5. 建立訓練器
    print("\n[5/7] 建立訓練器...")
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        test_loader=test_loader,
        distillation_loss=distillation_loss,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    # 6. 開始訓練
    print("\n[6/7] 開始知識蒸餾訓練...")
    print("=" * 70)
    print("注意: 本範例僅訓練 5 個 epoch 作為示範")
    print("      實際應用建議訓練 50-100 epochs 以獲得最佳效果")
    print("=" * 70)

    history = trainer.train(num_epochs=num_epochs)

    # 7. 視覺化結果
    print("\n[7/7] 生成訓練曲線...")
    try:
        plot_training_curves(history, save_path='./training_curves.png')
        print("訓練曲線已儲存至: ./training_curves.png")
    except Exception as e:
        print(f"無法生成訓練曲線（可能缺少顯示環境）: {e}")

    # 儲存模型
    print("\n儲存訓練好的學生模型...")
    torch.save(student_model.state_dict(), 'student_model.pth')
    print("學生模型已儲存至: student_model.pth")

    # 最終總結
    print("\n" + "=" * 70)
    print("訓練完成！")
    print("=" * 70)
    print(f"最佳測試準確率: {max(history['test_acc']):.2f}%")
    print("\n後續步驟:")
    print("1. 增加訓練輪數以獲得更好的效果")
    print("2. 調整超參數（temperature, alpha）")
    print("3. 嘗試不同的模型架構")
    print("4. 在自己的資料集上應用")
    print("=" * 70)


if __name__ == "__main__":
    main()
