import torch
from torch import nn


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 人工特征分支：561 → 128 → 64
        self.feature_branch = nn.Sequential(
            nn.Linear(561, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # 时间序列分支：9×128 → 128 维
        self.signal_branch = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
        )

        # 拼接后的维度：64 + 128 = 192
        self.classifier = nn.Linear(192, 6)

    def forward(self, features, signals):
        feature_vector = self.feature_branch(features)
        signal_vector = self.signal_branch(signals)

        fused_vector = torch.cat(
            [feature_vector, signal_vector],
            dim=1,
        )

        return self.classifier(fused_vector)