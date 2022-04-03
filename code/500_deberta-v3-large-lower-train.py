import ast
import copy
import gc
import itertools
import json
import math
import os
import pickle
import random
import re
import shutil
import string
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy as sp
import tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from torch.nn import Parameter
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

import wandb
from utils.const import CFG
from utils.dataset import TrainDataset
from utils.fix_annotation import fix_annotation
from utils.helper import (create_labels_for_scoring, get_char_probs,
                          get_logger, get_predictions, get_results, get_score,
                          seed_everything, span_micro_f1)

# %env TOKENIZERS_PARALLELISM=true
# ====================================================
# Directory settings
# ====================================================

INPUT_DIR = Path("../input/")
OUTPUT_DIR = Path("../output/500_exp")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LOGGER = get_logger()
# ====================================================
# wandb settings
# ====================================================
print("wandb settings ... ")
if CFG.wandb:

    # from getpass import getpass

    try:
        # from kaggle_secrets import UserSecretsClient
        # user_secrets = UserSecretsClient()
        # secret_value_0 = user_secrets.get_secret("wandb_api")
        # secret_value_0 = getpass()
        secret_value_0 = os.environ["WANDB_KEY"]
        wandb.login(key=secret_value_0)
        anony = None
    except:
        anony = "must"
        print(
            "If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize"
        )

    def class2dict(f):
        return dict(
            (name, getattr(f, name)) for name in dir(f) if not name.startswith("__")
        )

    run = wandb.init(
        project="NBME-Public",
        name="deberta-v3-large + 3 layers + lower",
        config=class2dict(CFG),
        group=CFG.model,
        job_type="train",
        anonymous=anony,
    )

# ====================================================
# Model
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                cfg.model, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc1 = nn.Linear(self.config.hidden_size, self.cfg.hidden_size2)
        self.fc2 = nn.Linear(self.cfg.hidden_size2, 1)
        self._init_weights(self.fc1)
        self._init_weights(self.fc2)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        feature = self.fc1(self.fc_dropout(feature))
        output = self.fc2(self.fc_dropout(feature))
        return output


# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
    valid_texts = valid_folds["pn_history"].values
    valid_labels = create_labels_for_scoring(valid_folds)

    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR / "config.pth")
    model.to(device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in model.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "lr": encoder_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if "model" not in n],
                "lr": decoder_lr,
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=CFG.encoder_lr,
        decoder_lr=CFG.decoder_lr,
        weight_decay=CFG.weight_decay,
    )
    optimizer = AdamW(
        optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas
    )

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.num_warmup_steps,
                num_training_steps=num_train_steps,
            )
        elif cfg.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.num_warmup_steps,
                num_training_steps=num_train_steps,
                num_cycles=cfg.num_cycles,
            )
        return scheduler

    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_score = -1.0

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(
            fold, train_loader, model, criterion, optimizer, epoch, scheduler, device
        )

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        predictions = predictions.reshape((len(valid_folds), CFG.max_len))

        # scoring
        char_probs = get_char_probs(valid_texts, predictions, CFG.tokenizer)
        results = get_results(char_probs, th=0.5)
        preds = get_predictions(results)
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epoch+1} - Score: {score:.4f}")
        if CFG.wandb:
            wandb.log(
                {
                    f"[fold{fold}] epoch": epoch + 1,
                    f"[fold{fold}] avg_train_loss": avg_loss,
                    f"[fold{fold}] avg_val_loss": avg_val_loss,
                    f"[fold{fold}] score": score,
                }
            )

        if best_score < score:
            best_score = score
            LOGGER.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "predictions": predictions},
                OUTPUT_DIR / f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
            )

    predictions = torch.load(
        OUTPUT_DIR / f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
        map_location=torch.device("cpu"),
    )["predictions"]
    valid_folds[[i for i in range(CFG.max_len)]] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


def get_result(oof_df):
    labels = create_labels_for_scoring(oof_df)
    predictions = oof_df[[i for i in range(CFG.max_len)]].values
    char_probs = get_char_probs(oof_df["pn_history"].values, predictions, CFG.tokenizer)
    results = get_results(char_probs, th=0.5)
    preds = get_predictions(results)
    score = get_score(labels, preds)
    LOGGER.info(f"Score: {score:<.4f}")


if __name__ == "__main__":
    pass

    transformers_path = Path("/opt/conda/lib/python3.7/site-packages/transformers")

    input_dir = Path("../input/deberta-v2-3-fast-tokenizer")

    convert_file = input_dir / "convert_slow_tokenizer.py"
    conversion_path = transformers_path / convert_file.name

    if conversion_path.exists():
        conversion_path.unlink()

    shutil.copy(convert_file, transformers_path)
    deberta_v2_path = transformers_path / "models" / "deberta_v2"

    for filename in [
        "tokenization_deberta_v2.py",
        "tokenization_deberta_v2_fast.py",
        "deberta__init__.py",
    ]:
        if str(filename).startswith("deberta"):
            filepath = deberta_v2_path / str(filename).replace("deberta", "")
        else:
            filepath = deberta_v2_path / filename
        if filepath.exists():
            filepath.unlink()

        shutil.copy(input_dir / filename, filepath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed=42)

    # ====================================================
    # Data Loading
    # ====================================================
    train = pd.read_csv(INPUT_DIR / "train.csv")
    test = pd.read_csv(INPUT_DIR / "test.csv")
    train["annotation"] = train["annotation"].apply(ast.literal_eval)
    train["location"] = train["location"].apply(ast.literal_eval)
    features = pd.read_csv(INPUT_DIR / "features.csv")

    def preprocess_features(features):
        features.loc[27, "feature_text"] = "Last-Pap-smear-1-year-ago"
        # features['feature_text'] = features['feature_text'].str.replace("-"," ")
        return features

    features = preprocess_features(features)
    patient_notes = pd.read_csv(INPUT_DIR / "patient_notes.csv")

    print(f"train.shape: {train.shape}")
    print(f"features.shape: {features.shape}")
    print(f"patient_notes.shape: {patient_notes.shape}")

    train = train.merge(features, on=["feature_num", "case_num"], how="left")
    train = train.merge(patient_notes, on=["pn_num", "case_num"], how="left")
    test = test.merge(features, on=["feature_num", "case_num"], how="left")
    test = test.merge(patient_notes, on=["pn_num", "case_num"], how="left")

    # incorrect annotation
    train = fix_annotation(train)
    # TODO: 本当に lower が効くのか検証
    train["feature_text"] = train["feature_text"].str.lower()  # とりあえず全部小文字にする
    train["pn_history"] = train["pn_history"].str.lower()
    test["feature_text"] = test["feature_text"].str.lower()  # 推論時はtestも小文字にするのを忘れずに
    test["pn_history"] = test["pn_history"].str.lower()

    # ====================================================
    # CV split
    # ====================================================
    Fold = GroupKFold(n_splits=CFG.n_fold)
    groups = train["pn_num"].values
    for n, (train_index, val_index) in enumerate(
        Fold.split(train, train["location"], groups)
    ):
        train.loc[val_index, "fold"] = int(n)
    train["fold"] = train["fold"].astype(int)
    # display(train.groupby("fold").size())
    if CFG.debug:
        train = train.sample(n=1000, random_state=0).reset_index(drop=True)

    # ====================================================
    # tokenizer
    # ====================================================
    # tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    # tokenizer.save_pretrained(OUTPUT_DIR / 'tokenizer/')
    # CFG.tokenizer = tokenizer

    from transformers.models.deberta_v2 import DebertaV2TokenizerFast

    tokenizer = DebertaV2TokenizerFast.from_pretrained(CFG.model)
    tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer/")
    CFG.tokenizer = tokenizer

    # ====================================================
    # Define max_len
    # ====================================================
    for text_col in ["pn_history"]:
        pn_history_lengths = []
        tk0 = tqdm(patient_notes[text_col].fillna("").values, total=len(patient_notes))
        for text in tk0:
            length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            pn_history_lengths.append(length)
        LOGGER.info(f"{text_col} max(lengths): {max(pn_history_lengths)}")

    for text_col in ["feature_text"]:
        features_lengths = []
        tk0 = tqdm(features[text_col].fillna("").values, total=len(features))
        for text in tk0:
            length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            features_lengths.append(length)
        LOGGER.info(f"{text_col} max(lengths): {max(features_lengths)}")

    CFG.max_len = max(pn_history_lengths) + max(features_lengths) + 3  # cls & sep & sep
    LOGGER.info(f"max_len: {CFG.max_len}")

    # ====================================================
    # main
    # ====================================================
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(OUTPUT_DIR / "oof_df.pkl")

    if CFG.wandb:
        wandb.finish()
