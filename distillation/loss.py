"""
知識蒸餾損失函數
結合軟標籤損失和硬標籤損失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    知識蒸餾損失函數

    組合兩種損失:
    1. 蒸餾損失（Distillation Loss）：學生模型與教師模型的軟標籤 KL 散度
    2. 學生損失（Student Loss）：學生模型與真實標籤的交叉熵

    總損失 = α * 蒸餾損失 + (1-α) * 學生損失
    """
    def __init__(self, temperature=3.0, alpha=0.7):
        """
        初始化蒸餾損失函數

        參數:
            temperature: 溫度參數，用於軟化機率分佈
                        較高的溫度會產生更平滑的分佈，幫助傳遞更多資訊
            alpha: 蒸餾損失的權重 (0 到 1 之間)
                  alpha=1 表示只使用蒸餾損失
                  alpha=0 表示只使用學生損失
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        """
        計算蒸餾損失

        參數:
            student_logits: 學生模型的輸出 logits (batch_size, num_classes)
            teacher_logits: 教師模型的輸出 logits (batch_size, num_classes)
            labels: 真實標籤 (batch_size,)

        回傳:
            total_loss: 總損失
            distillation_loss: 蒸餾損失分量
            student_loss: 學生損失分量
        """
        # 計算軟標籤（使用溫度參數軟化）
        # 溫度越高，分佈越平滑，傳遞的「暗知識」越多
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)

        # 蒸餾損失：KL 散度
        # 需要乘以 temperature^2 來保持梯度的尺度
        distillation_loss = self.kl_div(soft_prob, soft_targets) * (self.temperature ** 2)

        # 學生損失：與真實標籤的交叉熵
        student_loss = self.ce_loss(student_logits, labels)

        # 組合損失
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss

        return total_loss, distillation_loss, student_loss


def test_distillation_loss():
    """
    測試蒸餾損失函數
    """
    # 建立測試資料
    batch_size = 8
    num_classes = 10

    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    # 測試不同的溫度和 alpha 值
    print("測試知識蒸餾損失函數")
    print("=" * 50)

    for temp in [1.0, 3.0, 5.0]:
        for alpha in [0.3, 0.5, 0.7]:
            loss_fn = DistillationLoss(temperature=temp, alpha=alpha)
            total_loss, dist_loss, student_loss = loss_fn(
                student_logits, teacher_logits, labels
            )

            print(f"\n溫度={temp}, Alpha={alpha}")
            print(f"  總損失: {total_loss.item():.4f}")
            print(f"  蒸餾損失: {dist_loss.item():.4f}")
            print(f"  學生損失: {student_loss.item():.4f}")


if __name__ == "__main__":
    test_distillation_loss()
