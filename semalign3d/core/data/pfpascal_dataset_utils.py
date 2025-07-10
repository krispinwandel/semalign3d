"""Utilities for working with the SPair dataset."""

import numpy as np

PF_PASCAL_CATEGORIES = [
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
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


CATEGORY_NAME_TO_PROMPT_NAME = {
    "pottedplant": "potted plant",
    "tvmonitor": "tv-monitor",
}


# Our contribution
PF_PASCAL_FLIP = {
    "aeroplane": [0, 1, [2, 3], [4, 5], 6, 7, 8, [9, 10], 11, [12, 14], 13, 14, 15],
    "bicycle": [[0, 1], 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "bird": [[0, 1], 2, 3, 4, 5, 6, [7, 8], [9, 10], 11],
    "boat": [0, 1, 2, 3, [4, 6], [5, 7], 8, 9, 10],
    "bottle": [[0, 1], [2, 3], [4, 5], [6, 7]],
    "bus": [[0, 1], [2, 3], [4, 5], [6, 7]],
    "car": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]],
    "cat": [[0, 1], 2, [3, 4], 5, 6, 7, [8, 9], [10, 11], [12, 13], [14, 15]],
    "chair": [[0, 2], [1, 3], [4, 6], [5, 7], [8, 9]],
    "cow": [[0, 1], 2, [3, 4], 5, 6, 7, [8, 9], [10, 11], [12, 13], [14, 15]],
    "diningtable": [[0, 2], [1, 3], [4, 6], [5, 7]],
    "dog": [[0, 1], 2, [3, 4], 5, 6, 7, [8, 9], [10, 11], [12, 13], [14, 15]],
    "horse": [[0, 1], 2, [3, 4], 5, 6, 7, [8, 9], [10, 11], [12, 13], [14, 15]],
    "motorbike": [[0, 1], 2, 3, 4, 5, 6, 7, 8, 9],
    "person": [
        [0, 3],
        [1, 4],
        [2, 5],
        [6, 9],
        [7, 10],
        [8, 11],
        [12, 13],
        [14, 15],
        16,
    ],
    "pottedplant": [[0, 1], 2, 3, [4, 5]],
    "sheep": [[0, 1], 2, [3, 4], 5, 6],
    "sofa": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
    "train": [[0, 1], 2, [3, 5], [4, 6]],
    "tvmonitor": [[0, 1], [2, 3], [4, 5], [6, 7]],
}
PF_PASCAL_KPT_INDEX_TO_FLIPPED_KPT_INDEX = {}
for category, kpt_indices in PF_PASCAL_FLIP.items():
    flip_map = -np.ones(30, dtype=int)
    for kpt_index in kpt_indices:
        if isinstance(kpt_index, list):
            kpt_idx_1, kpt_idx_2 = kpt_index
            flip_map[kpt_idx_1] = kpt_idx_2
            flip_map[kpt_idx_2] = kpt_idx_1
        else:
            flip_map[kpt_index] = kpt_index
    PF_PASCAL_KPT_INDEX_TO_FLIPPED_KPT_INDEX[category] = flip_map


def build_get_flipped_kpt_index(category):
    def get_flipped_kpt_index(kpt_index):
        if PF_PASCAL_KPT_INDEX_TO_FLIPPED_KPT_INDEX[category][kpt_index] == -1:
            raise ValueError(
                f"Flipped kpt index not found for category {category} and kpt index {kpt_index}"
            )
        return PF_PASCAL_KPT_INDEX_TO_FLIPPED_KPT_INDEX[category][kpt_index]

    return get_flipped_kpt_index
