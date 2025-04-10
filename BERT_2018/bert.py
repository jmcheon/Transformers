import torch.nn as nn
from model import BERT, MLMhead, NSPHead
from Transformer_2017.model import Encoder, EncoderBlock, FeedForwardBlock, MultiHeadAttentionBlock


class BERTPretrainingModel(nn.Module):
    def __init__(self, bert: BERT, vocab_size: int):
        super().__init__()
        self.bert = bert
        self.mlm = MLMhead(bert.token_embedding.embedding_dim, vocab_size)
        self.nsp = NSPHead(bert.token_embedding.embedding_dim)

    def forward(self, input_ids, segment_ids):
        encoded = self.bert(input_ids, segment_ids)
        # extract CLS token (batch, hidden_size)
        cls_token_output = encoded[:, 0, :]
        mlm_logits = self.mlm(encoded)  # (batch, seq_len, vob_size)
        nsp_logits = self.nsp(cls_token_output)  # (batch, 2)

        return mlm_logits, nsp_logits


def build_bert(
    vocab_size: int,
    max_len: int = 512,
    num_heads: int = 12,
    num_layers: int = 12,
    hidden_size: int = 768,
    d_ff: int = 3072,
    dropout: float = 0.1,
    padding_idx: int = 0,
) -> BERT:
    token_embeddding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
    segment_embedding = nn.Embedding(2, hidden_size)  # segment A/B
    position_embedding = nn.Embedding(max_len, hidden_size)
    encoder_blocks = nn.ModuleList(
        [
            EncoderBlock(
                MultiHeadAttentionBlock(hidden_size, num_heads, dropout),
                FeedForwardBlock(hidden_size, d_ff, dropout),
                dropout,
            )
            for _ in range(num_layers)
        ]
    )
    encoder = Encoder(hidden_size, encoder_blocks)

    bert = BERT(token_embeddding, segment_embedding, position_embedding, encoder)

    # Initialize the parameters
    for p in bert.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return bert


def build_bert_for_pretraining(
    vocab_size: int,
    max_len: int = 512,
    num_heads: int = 12,
    num_layers: int = 12,
    hidden_size: int = 768,
    d_ff: int = 3072,
    dropout: float = 0.1,
) -> BERTPretrainingModel:
    bert = build_bert(
        vocab_size,
        max_len,
        num_heads,
        num_layers,
        hidden_size,
        d_ff,
        dropout,
    )

    pretraining_model = BERTPretrainingModel(bert, vocab_size)

    # Initialize the parameters
    for p in pretraining_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return pretraining_model
