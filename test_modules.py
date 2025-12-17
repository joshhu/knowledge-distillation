"""
測試所有模組是否正常運作
"""
import torch
import sys

def test_imports():
    """測試模組匯入"""
    print("=" * 60)
    print("測試 1: 檢查模組匯入")
    print("=" * 60)

    try:
        from data import get_cifar10_dataloaders, get_dataset_info
        print("✓ 資料模組匯入成功")
    except Exception as e:
        print(f"✗ 資料模組匯入失敗: {e}")
        return False

    try:
        from models import TeacherModel, StudentModel
        print("✓ 模型模組匯入成功")
    except Exception as e:
        print(f"✗ 模型模組匯入失敗: {e}")
        return False

    try:
        from distillation import DistillationLoss, DistillationTrainer
        print("✓ 蒸餾模組匯入成功")
    except Exception as e:
        print(f"✗ 蒸餾模組匯入失敗: {e}")
        return False

    try:
        from utils import set_seed, compare_models
        print("✓ 工具模組匯入成功")
    except Exception as e:
        print(f"✗ 工具模組匯入失敗: {e}")
        return False

    print()
    return True


def test_models():
    """測試模型建立"""
    print("=" * 60)
    print("測試 2: 檢查模型建立")
    print("=" * 60)

    from models import TeacherModel, StudentModel

    try:
        # 建立教師模型
        teacher = TeacherModel(num_classes=10)
        teacher_params, _ = teacher.get_num_parameters()
        print(f"✓ 教師模型建立成功 (參數量: {teacher_params:,})")

        # 建立學生模型
        student = StudentModel(num_classes=10)
        student_params, _ = student.get_num_parameters()
        print(f"✓ 學生模型建立成功 (參數量: {student_params:,})")

        # 測試前向傳播
        x = torch.randn(2, 3, 32, 32)
        teacher_out = teacher(x)
        student_out = student(x)

        assert teacher_out.shape == (2, 10), "教師模型輸出形狀錯誤"
        assert student_out.shape == (2, 10), "學生模型輸出形狀錯誤"
        print("✓ 模型前向傳播測試通過")

        print()
        return True

    except Exception as e:
        print(f"✗ 模型測試失敗: {e}")
        print()
        return False


def test_distillation_loss():
    """測試蒸餾損失函數"""
    print("=" * 60)
    print("測試 3: 檢查蒸餾損失函數")
    print("=" * 60)

    from distillation import DistillationLoss

    try:
        # 建立損失函數
        loss_fn = DistillationLoss(temperature=3.0, alpha=0.7)
        print("✓ 蒸餾損失函數建立成功")

        # 測試損失計算
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))

        total_loss, dist_loss, student_loss = loss_fn(
            student_logits, teacher_logits, labels
        )

        assert total_loss.item() > 0, "總損失應為正值"
        print(f"✓ 損失計算測試通過")
        print(f"  - 總損失: {total_loss.item():.4f}")
        print(f"  - 蒸餾損失: {dist_loss.item():.4f}")
        print(f"  - 學生損失: {student_loss.item():.4f}")

        print()
        return True

    except Exception as e:
        print(f"✗ 蒸餾損失測試失敗: {e}")
        print()
        return False


def test_data_loading():
    """測試資料載入（簡化版）"""
    print("=" * 60)
    print("測試 4: 檢查資料集資訊")
    print("=" * 60)

    from data import get_dataset_info

    try:
        info = get_dataset_info()
        print(f"✓ 資料集資訊取得成功")
        print(f"  - 類別數量: {info['num_classes']}")
        print(f"  - 輸入形狀: {info['input_shape']}")
        print(f"  - 訓練樣本數: {info['train_size']}")
        print(f"  - 測試樣本數: {info['test_size']}")

        print()
        return True

    except Exception as e:
        print(f"✗ 資料集測試失敗: {e}")
        print()
        return False


def test_utils():
    """測試工具函數"""
    print("=" * 60)
    print("測試 5: 檢查工具函數")
    print("=" * 60)

    from utils import set_seed, compare_models
    from models import TeacherModel, StudentModel

    try:
        # 測試設定種子
        set_seed(42)
        print("✓ 隨機種子設定成功")

        # 測試模型比較
        teacher = TeacherModel(num_classes=10)
        student = StudentModel(num_classes=10)
        compare_models(teacher, student)
        print("✓ 模型比較功能正常")

        print()
        return True

    except Exception as e:
        print(f"✗ 工具函數測試失敗: {e}")
        print()
        return False


def main():
    """執行所有測試"""
    print("\n" + "=" * 60)
    print("知識蒸餾專案 - 模組測試")
    print("=" * 60 + "\n")

    tests = [
        ("模組匯入", test_imports),
        ("模型建立", test_models),
        ("蒸餾損失", test_distillation_loss),
        ("資料載入", test_data_loading),
        ("工具函數", test_utils),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"測試 '{test_name}' 發生錯誤: {e}")
            results.append((test_name, False))

    # 顯示總結
    print("=" * 60)
    print("測試總結")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ 通過" if result else "✗ 失敗"
        print(f"{test_name:20s} {status}")

    print("=" * 60)
    print(f"總計: {passed}/{total} 測試通過")
    print("=" * 60)

    if passed == total:
        print("\n🎉 所有測試通過！專案已準備就緒。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 個測試失敗，請檢查錯誤訊息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
