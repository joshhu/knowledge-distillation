"""
知識蒸餾訓練器
管理整個訓練和評估流程
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import time


class DistillationTrainer:
    """
    知識蒸餾訓練器
    負責訓練學生模型，使用教師模型的知識
    """
    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        test_loader,
        distillation_loss,
        optimizer,
        device='cuda',
        scheduler=None
    ):
        """
        初始化訓練器

        參數:
            teacher_model: 教師模型（已訓練好）
            student_model: 學生模型（待訓練）
            train_loader: 訓練資料載入器
            test_loader: 測試資料載入器
            distillation_loss: 蒸餾損失函數
            optimizer: 優化器
            device: 運算裝置
            scheduler: 學習率調整器（可選）
        """
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.distillation_loss = distillation_loss
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        # 將教師模型設為評估模式
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # 訓練歷史記錄
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'distillation_loss': [],
            'student_loss': []
        }

    def train_epoch(self, epoch):
        """
        訓練一個 epoch

        參數:
            epoch: 當前 epoch 數

        回傳:
            avg_loss: 平均損失
            avg_acc: 平均準確率
            avg_dist_loss: 平均蒸餾損失
            avg_student_loss: 平均學生損失
        """
        self.student_model.train()

        total_loss = 0
        total_dist_loss = 0
        total_student_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 取得教師模型的輸出（不需要梯度）
            with torch.no_grad():
                teacher_logits = self.teacher_model(inputs)

            # 取得學生模型的輸出
            student_logits = self.student_model(inputs)

            # 計算損失
            loss, dist_loss, student_loss = self.distillation_loss(
                student_logits, teacher_logits, targets
            )

            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 統計
            total_loss += loss.item()
            total_dist_loss += dist_loss.item()
            total_student_loss += student_loss.item()

            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新進度條
            pbar.set_postfix({
                'Loss': f'{loss.item():.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_dist_loss = total_dist_loss / len(self.train_loader)
        avg_student_loss = total_student_loss / len(self.train_loader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc, avg_dist_loss, avg_student_loss

    def evaluate(self):
        """
        評估模型在測試集上的表現

        回傳:
            avg_loss: 平均損失
            avg_acc: 平均準確率
        """
        self.student_model.eval()
        total_loss = 0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.student_model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.test_loader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc

    def train(self, num_epochs):
        """
        執行完整的訓練流程

        參數:
            num_epochs: 訓練的 epoch 數量

        回傳:
            history: 訓練歷史記錄
        """
        print("=" * 70)
        print("開始知識蒸餾訓練")
        print("=" * 70)

        best_acc = 0
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            # 訓練
            train_loss, train_acc, dist_loss, student_loss = self.train_epoch(epoch)

            # 評估
            test_loss, test_acc = self.evaluate()

            # 更新學習率
            if self.scheduler is not None:
                self.scheduler.step()

            # 記錄歷史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['distillation_loss'].append(dist_loss)
            self.history['student_loss'].append(student_loss)

            # 輸出結果
            print(f'\nEpoch: {epoch}/{num_epochs}')
            print(f'訓練 - 損失: {train_loss:.4f}, 準確率: {train_acc:.2f}%')
            print(f'測試 - 損失: {test_loss:.4f}, 準確率: {test_acc:.2f}%')
            print(f'蒸餾損失: {dist_loss:.4f}, 學生損失: {student_loss:.4f}')

            # 儲存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                print(f'🎯 新的最佳準確率: {best_acc:.2f}%')

            print("-" * 70)

        elapsed_time = time.time() - start_time
        print(f"\n訓練完成! 總耗時: {elapsed_time/60:.2f} 分鐘")
        print(f"最佳測試準確率: {best_acc:.2f}%")

        return self.history

    def get_history(self):
        """
        取得訓練歷史記錄

        回傳:
            history: 訓練歷史字典
        """
        return self.history
