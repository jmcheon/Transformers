import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_target_pair = self.dataset[index]
        src_text = src_target_pair["translation"][self.lang_src]
        tgt_text = src_target_pair["translation"][self.lang_tgt]

        enc_input_token = self.tokenizer_src.encode(src_text).ids
        dec_input_token = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_token = self.seq_len - len(enc_input_token) - 2
        dec_num_padding_token = self.seq_len - len(dec_input_token) - 1

        if enc_num_padding_token < 0 or dec_num_padding_token < 0:
            raise ValueError("Sequence is too long")

        # Add SOS and EOS to source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_token, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_token, dtype=torch.int64),
            ]
        )

        # Add SOS to decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_token, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_token, dtype=torch.int64),
            ]
        )

        # Add EOS to label (decoder output)
        label = torch.cat(
            [
                torch.tensor(dec_input_token, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_token, dtype=torch.int64),
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
            & casual_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int64)
    return mask == 0
