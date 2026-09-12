import torch
import torch.nn as nn


class GrokTransformer(nn.Module):
    """Small transformer for multiplication tasks in GF(2^m).

    The model receives two field elements, a and b, as input tokens and
    predicts their product as a categorical distribution over field elements.

    Args:
        n_vocab: Number of field elements, i.e. 2**m.
        d_model: Transformer embedding dimension.
        n_heads: Number of attention heads.
        d_mlp: Hidden dimension of the feed-forward layer.
        n_layers: Number of transformer encoder layers.
    """

    def __init__(
        self,
        n_vocab,
        d_model=128,
        n_heads=4,
        d_mlp=512,
        n_layers=1,
    ):
        super().__init__()

        self.n_vocab = n_vocab

        self.embed = nn.Embedding(n_vocab, d_model)
        self.pos_embed = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_mlp,
            batch_first=True,
            activation="relu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.unembed = nn.Linear(
            d_model,
            n_vocab,
            bias=False,
        )

    def forward(self, a, b):
        """Predict the product of field elements a and b.

        Args:
            a: Tensor of shape (batch,) containing the first operands.
            b: Tensor of shape (batch,) containing the second operands.

        Returns:
            Logits of shape (batch, n_vocab) over possible products.
        """
        # Represent the two operands as a sequence of two tokens.
        tokens = torch.stack([a, b], dim=1)

        # Add learned positional embeddings for the two input positions.
        positions = torch.arange(
            2,
            device=a.device,
        ).unsqueeze(0).expand(a.shape[0], 2)

        x = self.embed(tokens) + self.pos_embed(positions)

        # Process the two-token sequence with the transformer.
        x = self.transformer(x)

        # Use the representation at the second position to predict a * b.
        logits = self.unembed(x[:, -1, :])

        return logits
