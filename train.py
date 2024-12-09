from pathlib import Path

import torch
import torch.nn as nn
from config import get_config, get_weights_file_path
from dataset import BilingualDataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Transformer import build_transformer

DATASET = "opus_books"


def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item["translation"][lang]


def build_tokenizer(config, dataset, lang):
    # config["tokenizer_file"] = "../tokenizer/tokenizer_{0}.json"
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    raw_dataset = load_dataset(DATASET, f"{config['lang_src']}-{config['lang_tgt']}", split="train")

    # Build tokenizer
    tokenizer_src = build_tokenizer(config, raw_dataset, config["lang_src"])
    tokenizer_tgt = build_tokenizer(config, raw_dataset, config["lang_tgt"])

    # 90% training - 10% validation
    train_size = int(0.9 * len(raw_dataset))
    val_size = len(raw_dataset) - train_size

    train_raw_data, val_raw_data = random_split(raw_dataset, [train_size, val_size])

    train_dataset = BilingualDataset(
        train_raw_data,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_dataset = BilingualDataset(
        val_raw_data,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_src_len = 0
    max_tgt_len = 0

    for item in raw_dataset:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_src.encode(item["translation"][config["lang_tgt"]]).ids
        max_src_len = max(max_src_len, len(src_ids))
        max_tgt_len = max(max_tgt_len, len(tgt_ids))

    print(f"Max lenght of source sentence: {max_src_len}")
    print(f"Max lenght of target sentence: {max_tgt_len}")

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(
        src_vocab_size, tgt_vocab_size, config["seq_len"], config["seq_len"], config["d_model"]
    )
    return model


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model = model.to(device)

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model: {model_filename}")

        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch [{epoch:02d}/{config['num_epochs']}]")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (batch, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)  # (batch, 1, seq_len, seq_len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (batch, seq_len, d_model)
            decoder_output = model.decode(
                decoder_input, encoder_output, encoder_mask, decoder_mask
            )  # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (batch, seq_len, tgt_vocab_size)

            label = batch["label"].to(device)  # (batch, seq_len)

            # (batch, seq_len, tgt_vocab_size) -> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix(loss=f"{loss.item():6.3f}")

            # Log
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            # Backpropagation
            loss.backward()

            # Update
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
