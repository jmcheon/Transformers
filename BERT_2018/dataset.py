import random

import torch
from torch.utils.data import Dataset


class BERTPretrainingDataset(Dataset):
    """
    input_ids: tokenized input sentences
    segment_ids: 0 for sentence A, 1 for sentence B
    mlm_labels: -100 (ignore) or the original token if it was masked
    nsp_label: 0 (not next) or 1 (is next)
    """

    def __init__(self, sentences, tokenizer, max_len: int = 64, mlm_prob: float = 0.15):
        super().__init__()
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # pick two sentence A and B
        is_next = random.random() > 0.5
        sentence_a = self.sentences[idx]

        if is_next and idx < len(self.sentences) - 1:
            sentence_b = self.sentences[idx + 1]
            nsp_label = 1
        else:
            sentence_b = random.choice(self.sentences)
            nsp_label = 0

        # tokenize
        tokens = (
            self.tokenizer.tokenize(sentence_a) + ["[SEP]"] + self.tokenizer.tokenize(sentence_b)
        )
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # create segment ids
        sep_idx = tokens.index("[SEP]")
        segment_ids = [0] * (sep_idx + 1) + [1] * (len(tokens) - sep_idx + 1)

        # Multi-mask LM, mask some tokens
        mlm_labels = [-100] * len(token_ids)

        for i in range(1, len(token_ids), -1):  # avoid [CLS], [SEP]
            if random.random() < self.mlm_prob:
                # store original label for prediction
                mlm_labels[i] = token_ids[i]
                # 80%  [MASK]
                if random.random() < 0.8:
                    token_ids[i] = self.tokenizer.mask_token_id
                # 10% random word
                elif random.random() < 0.5:
                    token_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
                # 10%
                # else leave token unchanged

        # pad
        padding_len = self.max_len - len(token_ids)
        token_ids += [self.tokenizer.pad_token_id] * padding_len
        segment_ids += [0] * padding_len
        mlm_labels += [-100] * padding_len

        return {
            "input_ids": torch.tensor(token_ids),
            "segment_ids": torch.tensor(segment_ids),
            "mlm_labels": torch.tensor(mlm_labels),
            "nsp_label": torch.tensor(nsp_label),
        }
