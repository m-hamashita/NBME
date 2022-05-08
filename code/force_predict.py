import re
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import itertools


def get_raw_location_annotation(
    char_probs: "list[np.ndarray]", pn_histories: "list[str]", th: float = 0.5
) -> "tuple[list[list[tuple[int,int]]], list[list[str]]]":
    """前処理なしのlocationと抜き出し"""
    locations = []
    for char_prob in char_probs:
        location = np.where(char_prob >= th)[0]
        location = [
            list(g)
            for _, g in itertools.groupby(
                location, key=lambda n, c=itertools.count(): n - next(c)
            )
        ]
        location = [(min(r), max(r) + 1) for r in location]
        locations.append(location)

    annotations = []
    for text, location in zip(pn_histories, locations):
        annotation = []
        for i, j in location:
            annotation.append(text[i:j])
        annotations.append(annotation)
    return locations, annotations


def force_predict_annotation_rate(
    char_probs: List[np.ndarray],
    pn_histories: List[str],
    feature_nums: List[str],
    th=0.5,
) -> Tuple[List[List[Tuple[int, int]]], List[List[str]]]:
    """force_predict_annotation_rate.

    Args:
        char_probs (List[np.ndarray]): char_probs
        pn_histories (List[str]): pn_histories
        feature_nums (List[str]): feature_nums
        locations (List[List[Tuple[int, int]]]): locations
        annotations (List[List[str]]): annotations
        th (float): th

    Returns:
        Tuple[List[List[Tuple[int, int]]], List[List[str]]]:
    """
    target = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
        10: [],
        11: [],
        12: [],
        100: [],
        101: [],
        102: [],
        103: [],
        104: [],
        105: [],
        106: [],
        107: [],
        108: [],
        109: [],
        110: [],
        111: [],
        112: [],
        200: [],
        201: [],
        202: [],
        203: [],
        204: [],
        205: [],
        206: [],
        207: [],
        208: [],
        209: [],
        210: [],
        211: [],
        212: [],
        213: [],
        214: [],
        215: [],
        216: [],
        300: [],
        301: [],
        302: [],
        303: [],
        304: [],
        305: [],
        306: [],
        307: [],
        308: [],
        309: [],
        310: [],
        311: [],
        312: [],
        313: [],
        314: [],
        315: [],
        400: [],
        401: [],
        402: [],
        403: [],
        404: [],
        405: [],
        406: [],
        407: [],
        408: [],
        409: [],
        500: [],
        501: [],
        502: [],
        503: [],
        504: [],
        505: [],
        506: [],
        507: [],
        508: [],
        509: [],
        510: [],
        511: [],
        512: [],
        513: [],
        514: [],
        515: [],
        516: [],
        517: [],
        600: [],
        601: [],
        602: [],
        603: [],
        604: [],
        605: ["induced asthma"],
        606: ["chest pain"],
        607: [],
        608: [],
        609: [],
        610: [],
        611: [],
        700: [],
        701: [],
        702: [],
        703: [],
        704: [],
        705: [],
        706: [],
        707: [],
        708: [],
        800: [],
        801: [],
        802: [],
        803: [],
        804: [],
        805: [],
        806: [],
        807: [],
        808: [],
        809: [],
        810: [],
        811: [],
        812: [],
        813: [],
        814: [],
        815: [],
        816: [],
        817: [],
        900: [],
        901: [],
        902: [],
        903: [],
        904: [],
        905: [],
        906: [],
        907: [],
        908: [],
        909: [],
        910: [],
        911: [],
        912: [],
        913: [],
        914: [],
        915: [],
        916: [],
    }

    for k, (pn_history, feature_num) in tqdm(
        enumerate(zip(pn_histories, feature_nums))
    ):
        if feature_num not in target:
            continue
        word_list = target[feature_num]
        for word in word_list:
            for m in re.finditer(word, pn_history.lower()):
                i, j = m.span()
                char_probs[k][i:j] = 1.0
    return get_raw_location_annotation(char_probs, pn_histories, th=th)
