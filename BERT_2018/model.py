import torch
import torch.nn as nn
from Transformer_2017.model import Encoder, EncoderBlock, FeedForwardBlock, MultiHeadAttentionBlock


class BERT(nn.Module):
    """
    Bidirectional Encoder Representations from Transformer
    """
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 512,
        num_heads: int = 12,
        num_layers: int = 12,
        hidden_size: int = 768,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embeddding = nn.Embedding(vocab_size, hidden_size)
        self.segment_embedding = nn.Embedding(2, hidden_size)  # segment A/B
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    MultiHeadAttentionBlock(hidden_size, num_heads, dropout),
                    FeedForwardBlock(hidden_size, d_ff, dropout),
                    dropout,
                )
            ]
            for _ in range(num_layers)
        )
        self.encoder = Encoder(hidden_size, encoder_blocks)

    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1)
        position_ids = (
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        )

        x = (
            self.token_embeddding(input_ids)
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


class BERTPretrainingModel(nn.Module):
    def __init__(self, bert: BERT, vocab_size: int):
        super().__init__()
        self.bert = bert
        self.mlm = MLMhead(bert.token_embeddding.embedding_dim, vocab_size)
        self.nsp = NSPHead(bert.token_embeddding.embedding_dim)

    def forward(self, input_ids, segment_ids):
        encoded = self.bert(input_ids, segment_ids)
        # extract CLS token (batch, hidden_size)
        cls_output = encoded[:, 0, :]
        mlm_logits = self.mlm(encoded)
        nsp_logits = self.nsp(cls_output)

        return mlm_logits, nsp_logits
