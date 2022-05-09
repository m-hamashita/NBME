# ====================================================
# CFG
# ====================================================
class CFG:
    num_workers = 4
    path = "../input/nbme-deberta-base-baseline-train/"
    config_path = path + "config.pth"
    model = "microsoft/deberta-base"
    batch_size = 24
    fc_dropout = 0.2
    max_len = 466
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]


# ====================================================
# Library
# ====================================================
import os
import gc
import ast
import random
import itertools
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import tokenizers
import transformers

print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================================================
# tokenizer
# ====================================================
CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path + "tokenizer/")


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


def get_logger(filename="inference"):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

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


# # OOF


# ====================================================
# oof
# ====================================================
oof = pd.read_pickle(CFG.path + "oof_df.pkl")

truths = create_labels_for_scoring(oof)
char_probs = get_char_probs(
    oof["pn_history"].values, oof[[i for i in range(CFG.max_len)]].values, CFG.tokenizer
)
best_th = 0.5
best_score = 0.0
for th in np.arange(0.45, 0.55, 0.01):
    th = np.round(th, 2)
    results = get_results(char_probs, th=th)
    preds = get_predictions(results)
    score = get_score(preds, truths)
    if best_score < score:
        best_th = th
        best_score = score
    LOGGER.info(f"th: {th}  score: {score:.5f}")
LOGGER.info(f"best_th: {best_th}  score: {best_score:.5f}")


# # Data Loading


# ====================================================
# Data Loading
# ====================================================
test = pd.read_csv("../input/nbme-score-clinical-patient-notes/test.csv")
submission = pd.read_csv(
    "../input/nbme-score-clinical-patient-notes/sample_submission.csv"
)
features = pd.read_csv("../input/nbme-score-clinical-patient-notes/features.csv")


def preprocess_features(features):
    features.loc[27, "feature_text"] = "Last-Pap-smear-1-year-ago"
    return features


features = preprocess_features(features)
patient_notes = pd.read_csv(
    "../input/nbme-score-clinical-patient-notes/patient_notes.csv"

test = test.merge(features, on=["feature_num", "case_num"], how="left")
test = test.merge(patient_notes, on=["pn_num", "case_num"], how="left")

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
        print(f"k: {k}, v: {v}")
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.feature_texts = df["feature_text"].values
        self.pn_historys = df["pn_history"].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input(
            self.cfg, self.pn_historys[item], self.feature_texts[item]
        )
        return inputs


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
            self.model = AutoModel.from_config(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        # hidden_size 取れるの偉いなぁ
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


# ====================================================
# inference
# ====================================================


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


test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
    drop_last=False,
)
predictions = []
for fold in CFG.trn_fold:
    model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
    state = torch.load(
        CFG.path + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(state["model"])
    prediction = inference_fn(test_loader, model, device)
    prediction = prediction.reshape((len(test), CFG.max_len))
    char_probs = get_char_probs(test["pn_history"].values, prediction, CFG.tokenizer)
    predictions.append(char_probs)
    del model, state, prediction, char_probs
    gc.collect()
    torch.cuda.empty_cache()
    predictions = np.mean(predictions, axis=0)

results = get_results(predictions, th=best_th)
submission["location"] = results
submission[["id", "location"]].to_csv("submission.csv", index=False)
