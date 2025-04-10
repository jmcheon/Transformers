# from transformers import BertTokenizer

from bert import build_bert_for_pretraining


def get_bert_for_pretraining(config):
    pretraining_model = build_bert_for_pretraining(
        config["vocab_size"],
        config["max_len"],
        config["num_heads"],
        config["num_layers"],
        config["hidden_size"],
        config["d_ff"],
        config["dropout"],
    )
    return pretraining_model
