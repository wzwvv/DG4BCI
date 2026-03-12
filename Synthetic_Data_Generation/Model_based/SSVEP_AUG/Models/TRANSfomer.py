import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class DimensionExpandingTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=8, num_layers=1, dim_feedforward=2048, dropout=0.1):
        """
        维度扩展的Transformer层

        参数:
            input_dim: 输入特征维度 (8Nc)
            output_dim: 输出特征维度 (16Nc)
            nhead: 多头注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络中间层维度
            dropout: Dropout概率
        """
        super().__init__()

        # 输入维度检查
        if output_dim < input_dim:
            raise ValueError("输出维度必须大于或等于输入维度以实现扩展")

        # 1. 输入投影层：将输入维度扩展到目标维度
        self.input_projection = nn.Linear(input_dim, output_dim)

        # 2. Transformer编码器层
        self.transformer_layer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=output_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True  # 使用(B, T, C)格式
            ),
            num_layers=num_layers
        )

        # 3. 输出投影层（可选，保持维度不变）
        # 如果需要进一步处理可以添加
        # self.output_projection = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        """
        前向传播

        输入: x (B, T/4, 8Nc)
        输出: (B, T/4, 16Nc)
        """
        # 1. 维度扩展投影
        x = self.input_projection(x)  # (B, T/4, 16Nc)

        # 2. 通过Transformer层
        x = self.transformer_layer(x)  # (B, T/4, 16Nc)

        # 3. 可选输出处理
        # x = self.output_projection(x)

        return x


# 使用示例
if __name__ == "__main__":
    # 假设参数
    B = 32  # 批次大小
    T_div_4 = 256  # 时间步长 T/4
    Nc = 24  # 原始通道数
    input_dim = 8 * Nc  # 8Nc
    output_dim = 16 * Nc  # 16Nc

    # 创建输入张量 (B, T/4, 8Nc)
    x3_squeezed = torch.randn(B, T_div_4, input_dim)
    print(f"输入形状: {x3_squeezed.shape}")

    # 创建维度扩展Transformer
    transformer = DimensionExpandingTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        nhead=8,  # 多头注意力头数
        num_layers=2,  # Transformer层数
        dim_feedforward=output_dim * 4  # 前馈网络中间层维度
    )

    # 前向传播
    output = transformer(x3_squeezed)
    print(f"输出形状: {output.shape}")  # 应为 (B, T_div_4, output_dim)