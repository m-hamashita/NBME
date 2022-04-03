
# https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-train
# # About this notebook
# - Deberta-base starter code
# - pip wheels is [here](https://www.kaggle.com/yasufuminakama/nbme-pip-wheels)
# - Inference notebook is [here](https://www.kaggle.com/yasufuminakama/nbme-deberta-base-baseline-inference)
#
# If this notebook is helpful, feel free to upvote :)

# # Directory settings

# ====================================================
# Directory settings
# ====================================================
import os
from pathlib import Path

INPUT_DIR = Path("../input")
OUTPUT_DIR = Path("../output/001_exp")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# # CFG
# ====================================================
# CFG
# ====================================================
class CFG:
    wandb = True
    competition = "NBME"
    _wandb_kernel = "mpeg"
    debug = False
    apex = True
    print_freq = 100
    num_workers = 4
    model = "microsoft/deberta-base"
    scheduler = "cosine"  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 5
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 12
    fc_dropout = 0.2
    max_len = 512
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    train = True


if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]



# ====================================================
# wandb
# ====================================================
if CFG.wandb:
    import wandb
    try:
        from getpass import getpass

        # from kaggle_secrets import UserSecretsClient
        # user_secrets = UserSecretsClient()
        # secret_value_0 = user_secrets.get_secret("wandb_api")
        secret_value_0 = getpass()
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
        name=CFG.model,
        config=class2dict(CFG),
        group=CFG.model,
        job_type="train",
        anonymous=anony,
    )


import ast
import gc
import itertools
import math
# # Library
# ====================================================
# Library
# ====================================================
import os
import random
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
import tokenizers
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

# get_ipython().run_line_magic("env", "TOKENIZERS_PARALLELISM=true")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Helper functions for scoring

# From https://www.kaggle.com/theoviel/evaluation-metric-folds-baseline
def micro_f1(preds, truths):
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)
def spans_to_binary(spans, length=None):
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary
def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.

    Args:
        preds (list of lists of two ints): Prediction spans.
        truths (list of lists of two ints): Ground truth spans.

    Returns:
        float: f1 score.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(
            np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0
        )
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))
    return micro_f1(bin_preds, bin_truths)
def create_labels_for_scoring(df):
    # example: ['0 1', '3 4'] -> ['0 1; 3 4']
    df["location_for_create_labels"] = [ast.literal_eval(f"[]")] * len(df)
    for i in range(len(df)):
        lst = df.loc[i, "location"]
        if lst:
            new_lst = ";".join(lst)
            df.loc[i, "location_for_create_labels"] = ast.literal_eval(
                f'[["{new_lst}"]]'
            )
    # create labels
    truths = []
    for location_list in df["location_for_create_labels"].values:
        truth = []
        if len(location_list) > 0:
            location = location_list[0]
            for loc in [s.split() for s in location.split(";")]:
                start, end = int(loc[0]), int(loc[1])
                truth.append([start, end])
        truths.append(truth)
    return truths

def get_char_probs(texts, predictions, tokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        for _, (offset_mapping, pred) in enumerate(
            zip(encoded["offset_mapping"], prediction)
        ):
            start = offset_mapping[0]
            end = offset_mapping[1]
            results[i][start:end] = pred
    return results


def get_results(char_probs, th=0.5):
    results = []
    for char_prob in char_probs:
        result = np.where(char_prob >= th)[0] + 1
        result = [
            list(g)
            for _, g in itertools.groupby(
                result, key=lambda n, c=itertools.count(): n - next(c)
            )
        ]
        result = [f"{min(r)} {max(r)}" for r in result]
        result = ";".join(result)
        results.append(result)
    return results


def get_predictions(results):
    predictions = []
    for result in results:
        prediction = []
        if result != "":
            for loc in [s.split() for s in result.split(";")]:
                start, end = int(loc[0]), int(loc[1])
                prediction.append([start, end])
        predictions.append(prediction)
    return predictions


# # Utils

# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    score = span_micro_f1(y_true, y_pred)
    return score


def get_logger(filename=OUTPUT_DIR / "train"):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)


# # Data Loading

# ====================================================
# Data Loading
# ====================================================
train = pd.read_csv(INPUT_DIR / "train.csv")
train["annotation"] = train["annotation"].apply(ast.literal_eval)
train["location"] = train["location"].apply(ast.literal_eval)
features = pd.read_csv(INPUT_DIR / "features.csv")


def preprocess_features(features):
    features.loc[27, "feature_text"] = "Last-Pap-smear-1-year-ago"
    return features


features = preprocess_features(features)
patient_notes = pd.read_csv(INPUT_DIR / "patient_notes.csv")

# print(f"train.shape: {train.shape}")
# display(train.head())
# print(f"features.shape: {features.shape}")
# display(features.head())
# print(f"patient_notes.shape: {patient_notes.shape}")
# display(patient_notes.head())


train = train.merge(features, on=["feature_num", "case_num"], how="left")
train = train.merge(patient_notes, on=["pn_num", "case_num"], how="left")
# display(train.head())


# incorrect annotation
{
    train.loc[338, "annotation"] = ast.literal_eval('[["father heart attack"]]')
    train.loc[338, "location"] = ast.literal_eval('[["764 783"]]')

    train.loc[621, "annotation"] = ast.literal_eval('[["for the last 2-3 months"]]')
    train.loc[621, "location"] = ast.literal_eval('[["77 100"]]')

    train.loc[655, "annotation"] = ast.literal_eval(
        '[["no heat intolerance"], ["no cold intolerance"]]'
    )
    train.loc[655, "location"] = ast.literal_eval(
        '[["285 292;301 312"], ["285 287;296 312"]]'
    )

    train.loc[1262, "annotation"] = ast.literal_eval('[["mother thyroid problem"]]')
    train.loc[1262, "location"] = ast.literal_eval('[["551 557;565 580"]]')

    train.loc[1265, "annotation"] = ast.literal_eval(
        "[['felt like he was going to \"pass out\"']]"
    )
    train.loc[1265, "location"] = ast.literal_eval('[["131 135;181 212"]]')

    train.loc[1396, "annotation"] = ast.literal_eval('[["stool , with no blood"]]')
    train.loc[1396, "location"] = ast.literal_eval('[["259 280"]]')

    train.loc[1591, "annotation"] = ast.literal_eval('[["diarrhoe non blooody"]]')
    train.loc[1591, "location"] = ast.literal_eval('[["176 184;201 212"]]')

    train.loc[1615, "annotation"] = ast.literal_eval('[["diarrhea for last 2-3 days"]]')
    train.loc[1615, "location"] = ast.literal_eval('[["249 257;271 288"]]')

    train.loc[1664, "annotation"] = ast.literal_eval('[["no vaginal discharge"]]')
    train.loc[1664, "location"] = ast.literal_eval('[["822 824;907 924"]]')

    train.loc[1714, "annotation"] = ast.literal_eval('[["started about 8-10 hours ago"]]')
    train.loc[1714, "location"] = ast.literal_eval('[["101 129"]]')

    train.loc[1929, "annotation"] = ast.literal_eval('[["no blood in the stool"]]')
    train.loc[1929, "location"] = ast.literal_eval('[["531 539;549 561"]]')

    train.loc[2134, "annotation"] = ast.literal_eval(
        '[["last sexually active 9 months ago"]]'
    )
    train.loc[2134, "location"] = ast.literal_eval('[["540 560;581 593"]]')

    train.loc[2191, "annotation"] = ast.literal_eval('[["right lower quadrant pain"]]')
    train.loc[2191, "location"] = ast.literal_eval('[["32 57"]]')

    train.loc[2553, "annotation"] = ast.literal_eval('[["diarrhoea no blood"]]')
    train.loc[2553, "location"] = ast.literal_eval('[["308 317;376 384"]]')

    train.loc[3124, "annotation"] = ast.literal_eval('[["sweating"]]')
    train.loc[3124, "location"] = ast.literal_eval('[["549 557"]]')

    train.loc[3858, "annotation"] = ast.literal_eval(
        '[["previously as regular"], ["previously eveyr 28-29 days"], ["previously lasting 5 days"], ["previously regular flow"]]'
    )
    train.loc[3858, "location"] = ast.literal_eval(
        '[["102 123"], ["102 112;125 141"], ["102 112;143 157"], ["102 112;159 171"]]'
    )

    train.loc[4373, "annotation"] = ast.literal_eval('[["for 2 months"]]')
    train.loc[4373, "location"] = ast.literal_eval('[["33 45"]]')

    train.loc[4763, "annotation"] = ast.literal_eval('[["35 year old"]]')
    train.loc[4763, "location"] = ast.literal_eval('[["5 16"]]')

    train.loc[4782, "annotation"] = ast.literal_eval('[["darker brown stools"]]')
    train.loc[4782, "location"] = ast.literal_eval('[["175 194"]]')

    train.loc[4908, "annotation"] = ast.literal_eval('[["uncle with peptic ulcer"]]')
    train.loc[4908, "location"] = ast.literal_eval('[["700 723"]]')

    train.loc[6016, "annotation"] = ast.literal_eval('[["difficulty falling asleep"]]')
    train.loc[6016, "location"] = ast.literal_eval('[["225 250"]]')

    train.loc[6192, "annotation"] = ast.literal_eval(
        '[["helps to take care of aging mother and in-laws"]]'
    )
    train.loc[6192, "location"] = ast.literal_eval('[["197 218;236 260"]]')

    train.loc[6380, "annotation"] = ast.literal_eval(
        '[["No hair changes"], ["No skin changes"], ["No GI changes"], ["No palpitations"], ["No excessive sweating"]]'
    )
    train.loc[6380, "location"] = ast.literal_eval(
        '[["480 482;507 519"], ["480 482;499 503;512 519"], ["480 482;521 531"], ["480 482;533 545"], ["480 482;564 582"]]'
    )

    train.loc[6562, "annotation"] = ast.literal_eval(
        '[["stressed due to taking care of her mother"], ["stressed due to taking care of husbands parents"]]'
    )
    train.loc[6562, "location"] = ast.literal_eval(
        '[["290 320;327 337"], ["290 320;342 358"]]'
    )

    train.loc[6862, "annotation"] = ast.literal_eval(
        '[["stressor taking care of many sick family members"]]'
    )
    train.loc[6862, "location"] = ast.literal_eval('[["288 296;324 363"]]')

    train.loc[7022, "annotation"] = ast.literal_eval(
        '[["heart started racing and felt numbness for the 1st time in her finger tips"]]'
    )
    train.loc[7022, "location"] = ast.literal_eval('[["108 182"]]')

    train.loc[7422, "annotation"] = ast.literal_eval('[["first started 5 yrs"]]')
    train.loc[7422, "location"] = ast.literal_eval('[["102 121"]]')

    train.loc[8876, "annotation"] = ast.literal_eval('[["No shortness of breath"]]')
    train.loc[8876, "location"] = ast.literal_eval('[["481 483;533 552"]]')

    train.loc[9027, "annotation"] = ast.literal_eval(
        '[["recent URI"], ["nasal stuffines, rhinorrhea, for 3-4 days"]]'
    )
    train.loc[9027, "location"] = ast.literal_eval('[["92 102"], ["123 164"]]')

    train.loc[9938, "annotation"] = ast.literal_eval(
        '[["irregularity with her cycles"], ["heavier bleeding"], ["changes her pad every couple hours"]]'
    )
    train.loc[9938, "location"] = ast.literal_eval('[["89 117"], ["122 138"], ["368 402"]]')

    train.loc[9973, "annotation"] = ast.literal_eval('[["gaining 10-15 lbs"]]')
    train.loc[9973, "location"] = ast.literal_eval('[["344 361"]]')

    train.loc[10513, "annotation"] = ast.literal_eval(
        '[["weight gain"], ["gain of 10-16lbs"]]'
    )
    train.loc[10513, "location"] = ast.literal_eval('[["600 611"], ["607 623"]]')

    train.loc[11551, "annotation"] = ast.literal_eval(
        '[["seeing her son knows are not real"]]'
    )
    train.loc[11551, "location"] = ast.literal_eval('[["386 400;443 461"]]')

    train.loc[11677, "annotation"] = ast.literal_eval(
        '[["saw him once in the kitchen after he died"]]'
    )
    train.loc[11677, "location"] = ast.literal_eval('[["160 201"]]')

    train.loc[12124, "annotation"] = ast.literal_eval(
        '[["tried Ambien but it didnt work"]]'
    )
    train.loc[12124, "location"] = ast.literal_eval('[["325 337;349 366"]]')

    train.loc[12279, "annotation"] = ast.literal_eval(
        '[["heard what she described as a party later than evening these things did not actually happen"]]'
    )
    train.loc[12279, "location"] = ast.literal_eval('[["405 459;488 524"]]')

    train.loc[12289, "annotation"] = ast.literal_eval(
        '[["experienced seeing her son at the kitchen table these things did not actually happen"]]'
    )
    train.loc[12289, "location"] = ast.literal_eval('[["353 400;488 524"]]')

    train.loc[13238, "annotation"] = ast.literal_eval(
        '[["SCRACHY THROAT"], ["RUNNY NOSE"]]'
    )
    train.loc[13238, "location"] = ast.literal_eval('[["293 307"], ["321 331"]]')

    train.loc[13297, "annotation"] = ast.literal_eval(
        '[["without improvement when taking tylenol"], ["without improvement when taking ibuprofen"]]'
    )
    train.loc[13297, "location"] = ast.literal_eval('[["182 221"], ["182 213;225 234"]]')

    train.loc[13299, "annotation"] = ast.literal_eval('[["yesterday"], ["yesterday"]]')
    train.loc[13299, "location"] = ast.literal_eval('[["79 88"], ["409 418"]]')

    train.loc[13845, "annotation"] = ast.literal_eval(
        '[["headache global"], ["headache throughout her head"]]'
    )
    train.loc[13845, "location"] = ast.literal_eval(
        '[["86 94;230 236"], ["86 94;237 256"]]'
    )

    train.loc[14083, "annotation"] = ast.literal_eval(
        '[["headache generalized in her head"]]'
    )
    train.loc[14083, "location"] = ast.literal_eval('[["56 64;156 179"]]')
}

train["annotation_length"] = train["annotation"].apply(len)
# display(train["annotation_length"].value_counts())


# # CV split
# ====================================================
# CV split
# ====================================================
Fold = GroupKFold(n_splits=CFG.n_fold)
# pn_num で GroupKFold
groups = train["pn_num"].values
for n, (train_index, val_index) in enumerate(
    Fold.split(train, train["location"], groups)
):
    train.loc[val_index, "fold"] = int(n)
train["fold"] = train["fold"].astype(int)
# display(train.groupby("fold").size())


if CFG.debug:
    # display(train.groupby("fold").size())
    train = train.sample(n=1000, random_state=0).reset_index(drop=True)
    # display(train.groupby("fold").size())


# # tokenizer

# ====================================================
# tokenizer
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer/")
CFG.tokenizer = tokenizer


# # Dataset
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
# Dataset
# ====================================================
def prepare_input(cfg, text, feature_text):
    inputs = cfg.tokenizer(
        text,
        feature_text,
        add_special_tokens=True,
        max_length=CFG.max_len,
        padding="max_length",
        return_offsets_mapping=False,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


def create_label(cfg, text, annotation_length, location_list):
    encoded = cfg.tokenizer(
        text,
        add_special_tokens=True,
        max_length=CFG.max_len,
        padding="max_length",
        return_offsets_mapping=True,
    )
    offset_mapping = encoded["offset_mapping"]
    ignore_idxes = np.where(np.array(encoded.sequence_ids()) != 0)[0]
    label = np.zeros(len(offset_mapping))
    label[ignore_idxes] = -1
    if annotation_length != 0:
        for location in location_list:
            for loc in [s.split() for s in location.split(";")]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx
                if (start_idx != -1) & (end_idx != -1):
                    label[start_idx:end_idx] = 1
    return torch.tensor(label, dtype=torch.float)


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.feature_texts = df["feature_text"].values
        self.pn_historys = df["pn_history"].values
        self.annotation_lengths = df["annotation_length"].values
        self.locations = df["location"].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input(
            self.cfg, self.pn_historys[item], self.feature_texts[item]
        )
        label = create_label(
            self.cfg,
            self.pn_historys[item],
            self.annotation_lengths[item],
            self.locations[item],
        )
        return inputs, label


# # Model

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
        # token ごとに分類
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

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
        output = self.fc(self.fc_dropout(feature))
        return output


# # Helpler functions


# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.8f}  ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    grad_norm=grad_norm,
                    lr=scheduler.get_lr()[0],
                )
            )
        if CFG.wandb:
            wandb.log(
                {
                    f"[fold{fold}] loss": losses.val,
                    f"[fold{fold}] lr": scheduler.get_lr()[0],
                }
            )
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to("cpu").numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step,
                    len(valid_loader),
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to("cpu").numpy())
    predictions = np.concatenate(preds)
    return predictions



# 学習のメインパート
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

    # :memo: 
    # pin_memory: 高速化用途
    # drop_last: 最後のミニバッチが不足した時、ミニバッチを削除するかどうか
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
    # pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
    #     into CUDA pinned memory before returning them.  If your data elements
    #     are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
    #     see the example below.

    # drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
    #     if the dataset size is not divisible by the batch size. If ``False`` and
    #     the size of dataset is not divisible by the batch size, then the last batch
    #     will be smaller. (default: ``False``)

    # ====================================================
    # model & optimizer
    # ====================================================
    # 実際に使用するモデルの定義
    model = CustomModel(CFG, config_path=None, pretrained=True)
    # save model config
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
    # optimizer_parameters について
    # weight_decay: L2正則化の強さ
    # betas: (0.9, 0.999)
    # add_param_group に追加していっている
    # TODO: それぞれ何を表しているかわからない
    optimizer = AdamW(
        optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas
    )

    # ====================================================
    # scheduler
    # ====================================================
    # lr のschedule 
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
        # else の場合がないなぁ(lint が怒る)
        return scheduler

    # train 時の回す回数(batchの回数(not batch_size))
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    # BCEWithLogitsLoss がある場合 sigmoid を使わない
    # 確かに使っていない
    #     self.fc = nn.Linear(self.config.hidden_size, 1)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_score = 0.0

    # 実際学習するところ
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
        # この辺 helper なのであまり気にしなくて良い（ th 以外)
        # valid_texts を token に分けて pred を埋める
        char_probs = get_char_probs(valid_texts, predictions, CFG.tokenizer)
        # th: 1 にするしきい値なので、ここで結構変わりそう(素直に0.5で良い？)
        results = get_results(char_probs, th=0.5)
        # parse するだけ
        preds = get_predictions(results)
        # micro_f1
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
    # valid_folds に prediction を埋める
    # このコードよくわかってないな
    valid_folds[[i for i in range(CFG.max_len)]] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds



if __name__ == "__main__":

    # Score 計算とか
    def get_result(oof_df):
        labels = create_labels_for_scoring(oof_df)
        predictions = oof_df[[i for i in range(CFG.max_len)]].values
        char_probs = get_char_probs(
            oof_df["pn_history"].values, predictions, CFG.tokenizer
        )
        results = get_results(char_probs, th=0.5)
        preds = get_predictions(results)
        score = get_score(labels, preds)
        LOGGER.info(f"Score: {score:<.4f}")

    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                # main part
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


