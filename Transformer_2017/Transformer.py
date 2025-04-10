import torch.nn as nn
from Transformer_2017.model import (
    Decoder,
    DecoderBlock,
    Encoder,
    EncoderBlock,
    FeedForwardBlock,
    InputEmbeddings,
    MultiHeadAttentionBlock,
    PositionalEncoding,
    ProjectionLayer,
)


class Transformer(nn.Module):
    def __init__(
        self,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        encoder: Encoder,
        decoder: Decoder,
        projection_layer: ProjectionLayer,
    ):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed: InputEmbeddings,
        pos: PositionalEncoding,
        encoder: Encoder,
        projection_layer: ProjectionLayer,
    ):
        super().__init__()
        self.embed = embed
        self.pos = pos
        self.encoder = encoder
        self.projection_layer = projection_layer

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pos(x)
        return self.encoder(x, mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer_encoder(
    vocab_size: int,
    seq_len: int,
    d_model: int = 512,
    d_ff: int = 2048,
    num_layers: int = 6,
    h: int = 8,
    dropout: float = 0.1,
) -> TransformerEncoder:
    # Create the embedding layers
    embed = InputEmbeddings(vocab_size, d_model)
    # Create the positional layers
    pos = PositionalEncoding(d_model, seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(num_layers):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, vocab_size)

    # Create the transformer
    transformer = TransformerEncoder(embed, pos, encoder, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    d_ff: int = 2048,
    num_layers: int = 6,
    h: int = 8,
    dropout: float = 0.1,
) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(src_vocab_size, d_model)
    tgt_embed = InputEmbeddings(tgt_vocab_size, d_model)

    # Create the positional layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(num_layers):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(num_layers):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        src_embed, tgt_embed, src_pos, tgt_pos, encoder, decoder, projection_layer
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
