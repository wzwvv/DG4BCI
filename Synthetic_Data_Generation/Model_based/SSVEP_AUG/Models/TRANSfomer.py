import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class DimensionExpandingTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=8, num_layers=1, dim_feedforward=2048, dropout=0.1):
        """
        Dimension-expanding Transformer layer.

        Args:
            input_dim: Input feature dimension (8Nc).
            output_dim: Output feature dimension (16Nc).
            nhead: Number of attention heads.
            num_layers: Number of Transformer layers.
            dim_feedforward: Feed-forward hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        # Input dimension check
        if output_dim < input_dim:
            raise ValueError("output_dim must be >= input_dim for expansion")

        # Input projection: expand input to output dimension
        self.input_projection = nn.Linear(input_dim, output_dim)

        # Transformer encoder
        self.transformer_layer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=output_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True  # (B, T, C) format
            ),
            num_layers=num_layers
        )

        # Optional output projection (keep dimension unchanged)
        # Add here if further processing is needed
        # self.output_projection = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        """
        Forward pass.

        Input: x (B, T/4, 8Nc)
        Output: (B, T/4, 16Nc)
        """
        # 1. Dimension expansion projection
        x = self.input_projection(x)  # (B, T/4, 16Nc)

        # 2. Through Transformer layer
        x = self.transformer_layer(x)  # (B, T/4, 16Nc)

        # 3. Optional output processing
        # x = self.output_projection(x)

        return x


# Usage example
if __name__ == "__main__":
    # Example parameters
    B = 32  # batch size
    T_div_4 = 256  # time steps T/4
    Nc = 24  # number of channels
    input_dim = 8 * Nc  # 8Nc
    output_dim = 16 * Nc  # 16Nc

    # Create input tensor (B, T/4, 8Nc)
    x3_squeezed = torch.randn(B, T_div_4, input_dim)
    print(f"Input shape: {x3_squeezed.shape}")

    # Create dimension-expanding Transformer
    transformer = DimensionExpandingTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        nhead=8,  # attention heads
        num_layers=2,  # Transformer layers
        dim_feedforward=output_dim * 4  # feed-forward hidden dim
    )

    # Forward pass
    output = transformer(x3_squeezed)
    print(f"Output shape: {output.shape}")  # (B, T_div_4, output_dim)