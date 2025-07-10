"""Utilities for working with the SPair dataset."""

import numpy as np


SPAIR_SORTED_CATEGORIES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "train",
    "tvmonitor",
]


CATEGORY_NAME_TO_PROMPT_NAME = {
    "pottedplant": "potted plant",
    "tvmonitor": "tv-monitor",
}


ANIMAL_CATEGORIES = [
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "person",
    "sheep",
]

# aeroplane
AEROPLANE_KPT_INDEX_TO_FLIPPED_KPT_INDEX = [
    0,
    1,
    2,
    3,
    5,
    4,
    7,
    6,
    9,
    8,
    11,
    10,
    13,
    12,
    15,
    14,
    17,
    16,
    19,
    18,
    21,
    20,
    22,
    23,
    24,
]

# taken from GeoAware-SC
SPAIR_FLIP = {
    "aeroplane": [
        0,
        1,
        2,
        3,
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
        [12, 13],
        [14, 15],
        [16, 17],
        [18, 19],
        [20, 21],
        22,
        23,
        24,
    ],
    "bicycle": [0, 1, [2, 3], 4, 5, [6, 7], 8, [9, 10], 13],  # 11,12 are not used
    "bird": [0, [1, 2], 3, [4, 5], 6, [7, 8], 9, [10, 11], [12, 13], [14, 15], 16],
    "boat": [0, [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], 13],
    "bottle": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
    "bus": [
        [0, 1],
        [2, 3],
        4,
        [5, 6],
        7,
        # 8,9 are dummy
        [10, 20],
        [11, 21],
        [12, 22],
        [13, 23],
        [14, 24],
        [15, 25],
        [16, 17],
        [18, 19],
        [26, 27],
        [28, 29],
    ],
    "car": [
        [0, 1],
        [2, 3],
        4,
        5,
        [6, 7],
        8,
        9,
        [10, 20],
        [11, 21],
        [12, 22],
        [13, 23],
        [14, 24],
        [15, 25],
        [16, 17],
        [18, 19],
        [26, 27],
        [28, 29],
    ],
    "cat": [[0, 1], [2, 3], [4, 5], [6, 7], 8, [9, 10], [11, 12], 13, 14],
    "chair": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]],
    "cow": [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        8,
        [9, 10],
        [11, 12],
        13,
        14,
        [15, 16],
        [17, 18],
        [19, 20],
    ],
    "dog": [[0, 1], [2, 3], [4, 5], 6, 7, 8, [9, 10], [11, 12], 13, 14, 15],
    "horse": [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        8,
        9,
        [10, 11],
        [12, 13],
        14,
        15,
        [16, 17],
        [18, 19],
    ],
    "motorbike": [[0, 1], [2, 3], 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "person": [
        [0, 1],
        [2, 3],
        4,
        5,
        6,
        7,
        [8, 9],
        [10, 11],
        [12, 13],
        [14, 15],
        [16, 17],
        [18, 19],
    ],
    "pottedplant": [[0, 2], 1, 3, [4, 5], [6, 8], 7],
    "sheep": [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        8,
        [9, 10],
        [11, 12],
        13,
        14,
        [15, 16],
        [17, 18],
        [19, 20],
    ],
    "train": [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
        [12, 13],
        [14, 15],
        [16, 17],
    ],
    "tvmonitor": [[0, 2], [4, 6], 1, 5, [3, 7], [8, 10], [12, 14], 9, 13, [11, 15]],
}
SPAIR_KPT_INDEX_TO_FLIPPED_KPT_INDEX = {}
for category, kpt_indices in SPAIR_FLIP.items():
    flip_map = -np.ones(30, dtype=int)
    for kpt_index in kpt_indices:
        if isinstance(kpt_index, list):
            kpt_idx_1, kpt_idx_2 = kpt_index
            flip_map[kpt_idx_1] = kpt_idx_2
            flip_map[kpt_idx_2] = kpt_idx_1
        else:
            flip_map[kpt_index] = kpt_index
    SPAIR_KPT_INDEX_TO_FLIPPED_KPT_INDEX[category] = flip_map


def build_get_flipped_kpt_index(category):
    def get_flipped_kpt_index(kpt_index):
        if SPAIR_KPT_INDEX_TO_FLIPPED_KPT_INDEX[category][kpt_index] == -1:
            raise ValueError(
                f"Flipped kpt index not found for category {category} and kpt index {kpt_index}"
            )
        return SPAIR_KPT_INDEX_TO_FLIPPED_KPT_INDEX[category][kpt_index]

    return get_flipped_kpt_index
