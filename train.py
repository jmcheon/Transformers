from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTranier
from torch.utils.data import ramdon_split

DATASET = "opus_books"


def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item["translation"][lang]


def build_tokenizer(config, dataset, lang):
    # config["tokenizer_path"] = "../tokenizer/tokenizer_{0}.json"
    tokenizer_path = Path(config["tokenizer_path"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTranier(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    dataset = load_dataset(DATASET, f"{config['lang_src']}-{config['lang_tgt']}", split="train")

    # Build tokenizer
    tokenizer_src = build_tokenizer(config, dataset, config["lang_src"])
    tokenizer_tgt = build_tokenizer(config, dataset, config["lang_tgt"])

    # 90% training - 10% validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = ramdon_split(dataset, [train_size, val_size])
