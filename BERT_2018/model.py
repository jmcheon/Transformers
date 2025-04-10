import torch
import torch.nn as nn
from Transformer_2017.model import Encoder


class BERT(nn.Module):
    """
    Bidirectional Encoder Representations from Transformer
    """

    def __init__(
        self,
        token_embedding,
        segment_embedding,
        position_embedding,
        encoder: Encoder,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = token_embedding
        self.segment_embedding = segment_embedding
        self.position_embedding = position_embedding
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1)
        position_ids = (
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        )

        x = (
            self.token_embedding(input_ids)
            + self.segment_embedding(segment_ids)
            + self.position_embedding(position_ids)
        )
        x = self.dropout(x)
        x = self.encoder(x, mask=None)

        return x


class MLMhead(nn.Module):
    """
    Masked Language Modeling (MLM) Head for pretraining
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.decoder(x)

        # logits over vocab
        return x


class NSPHead(nn.Module):
    """
    Next Sentence Prediction Head
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, cls_output):
        # logits for 2 classes
        return self.classifier(cls_output)
