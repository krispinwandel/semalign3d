"""Data utilities for loading and processing non-processed data."""

import json
import os
from typing import List, Optional
import torch
import scipy.io as sio
from PIL import Image
import numpy as np

from semalign3d.core.data import (
    spair_dataset_utils,
    ap10k_dataset_utils,
    pfpascal_dataset_utils,
)
from semalign3d.core import data_classes
from semalign3d.utils import image_processing


def read_mat(path, obj_name):
    r"""Reads specified objects from Matlab data file, (.mat)"""
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj


def load_normalized_img_anno_spair71k(img_anno_fp):
    """
    Returns a dict with the following keys:
    - bbox: list of 4 ints
    - kps_ids: list of str
    - kps: list of list of ints, where each sublist has 2 ints (x, y)
    - filename: str
    """
    with open(img_anno_fp) as f:
        img_anno = json.load(f)
    kps = img_anno["kps"]  # Dict[str, List[int]]
    kp_ids = [int(x) for x in list(kps.keys()) if kps[x] is not None]
    kp_xy = [
        x for x in list(kps.values()) if x is not None
    ]  # null in json is None in python

    # category is name of the folder in which the img_anno_fp is stored
    category = img_anno_fp.split("/")[-2]
    fn = img_anno["filename"]

    img_anno_normalized = data_classes.ImgAnnoNormalized(
        bbox=img_anno["bndbox"],  # list of 4 ints
        kp_ids=kp_ids,
        kp_xy=kp_xy,
        filename=fn,
        img_width=img_anno["image_width"],
        img_height=img_anno["image_height"],
        category=category,
        supercategory=None,
        rel_fp=f"JPEGImages/{category}/{fn}",
    )
    return img_anno_normalized


def load_normalized_img_anno_ap10k(img_anno_fp):
    with open(img_anno_fp) as f:
        img_anno = json.load(f)
    # kps has the form [x, y, 2, x, y, 2, ..., 2, x, y, 2]
    # where the len of kps is always 50 where the 2s are visibility flags (0 if not visible, 2 otherwise)
    kps = img_anno["keypoints"]
    kps = {
        f"{i // 3}": kps[i : i + 2] if kps[i + 2] > 0 else None
        for i in range(0, len(kps), 3)
    }
    kp_ids = [int(x) for x in list(kps.keys()) if kps[x] is not None]
    kp_xy = [x for x in list(kps.values()) if x is not None]

    # category is name of the folder in which the img_anno_fp is stored
    category = img_anno_fp.split("/")[-2]
    supercategory = img_anno_fp.split("/")[-3]
    fn = img_anno["file_name"]

    x_start, y_start, dx, dy = img_anno["bbox"]
    bbox = [x_start, y_start, x_start + dx, y_start + dy]
    img_anno_normalized = data_classes.ImgAnnoNormalized(
        bbox=bbox,  # list of 4 ints
        kp_ids=kp_ids,
        kp_xy=kp_xy,
        filename=fn,
        img_width=img_anno["width"],
        img_height=img_anno["height"],
        category=category,
        supercategory=supercategory,
        rel_fp=f"JPEGImages/{supercategory}/{category}/{fn}",
    )
    return img_anno_normalized


def load_normalized_img_anno_pfpascal(img_anno_fp):
    kp_xy = read_mat(img_anno_fp, "kps")  # shape (n_kpts, 2)
    not_used = np.isnan(kp_xy).any(axis=1)
    kp_ids = np.arange(0, kp_xy.shape[0])[~not_used].astype(int).tolist()
    kp_xy = (
        kp_xy[~not_used].astype(int).tolist()
    )  # In PF-Pascal, x, y are floats. We convert them to ints.

    # category is name of the folder in which the img_anno_fp is stored
    category = img_anno_fp.split("/")[-2]
    img_fp = img_anno_fp.replace("Annotations", "JPEGImages").replace(".mat", ".jpg")
    fn = img_fp.split("/")[-1]
    # NOTE in PIL size is (width, height) instead of (height, width)
    img_shape = Image.open(img_fp).size
    w, h = img_shape
    bbox = [0, 0, w, h]
    img_anno_normalized = data_classes.ImgAnnoNormalized(
        bbox=bbox,  # list of 4 ints
        kp_ids=kp_ids,
        kp_xy=kp_xy,
        filename=fn,
        img_width=w,
        img_height=h,
        category=category,
        supercategory=None,
        rel_fp=f"JPEGImages/{category}/{fn}",
    )
    return img_anno_normalized


def infer_dataset_name_from_fp(fp: str):
    """Infer the dataset name from a file path."""
    if "71k" in fp:
        return "spair-71k"
    elif "10k" in fp:
        return "ap-10k"
    elif "pascal" in fp.lower():
        return "pf-pascal"
    else:
        raise ValueError(f"Unknown dataset name for file path: {fp}")


def load_normalized_img_anno(img_anno_fp, dataset_name: Optional[str] = None):
    """Load normalized image annotation from a JSON file."""
    if dataset_name is None:
        dataset_name = infer_dataset_name_from_fp(img_anno_fp)
    if dataset_name == "spair-71k":
        return load_normalized_img_anno_spair71k(img_anno_fp)
    elif dataset_name == "ap-10k":
        return load_normalized_img_anno_ap10k(img_anno_fp)
    elif dataset_name == "pf-pascal":
        return load_normalized_img_anno_pfpascal(img_anno_fp)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def load_normalized_img_anno_from_img_fp(img_fp, dataset_name: Optional[str] = None):
    """Load normalized image annotation from an image file path."""
    if dataset_name is None:
        dataset_name = infer_dataset_name_from_fp(img_fp)
    img_anno_fp = img_fp2anno_fp(img_fp, dataset_name)
    return load_normalized_img_anno(img_anno_fp, dataset_name)


def load_normalized_src_tgt_img_anno(pair_anno_fp: dict, dataset_name: str):
    """Load normalized source and target image annotations from a JSON file."""
    with open(pair_anno_fp) as f:
        pair_anno = json.load(f)

    if dataset_name == "spair-71k":
        # root of data dir if grandparent of pair_anno_fp
        data_dir_root = os.path.dirname(os.path.dirname(os.path.dirname(pair_anno_fp)))
        src_imname = pair_anno["src_imname"]
        trg_imname = pair_anno["trg_imname"]
        src_img_id = src_imname.split(".")[0]
        trg_img_id = trg_imname.split(".")[0]
        category = pair_anno["category"]
        src_img_anno_fp = (
            f"{data_dir_root}/ImageAnnotation/{category}/{src_img_id}.json"
        )
        trg_img_anno_fp = (
            f"{data_dir_root}/ImageAnnotation/{category}/{trg_img_id}.json"
        )
    elif dataset_name == "ap-10k":
        src_img_anno_fp = pair_anno["src_json_path"]
        trg_img_anno_fp = pair_anno["trg_json_path"]
    elif dataset_name == "pf-pascal":
        raise NotImplementedError("pf-pascal not implemented")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    src_img_anno = load_normalized_img_anno(src_img_anno_fp, dataset_name)
    trg_img_anno = load_normalized_img_anno(trg_img_anno_fp, dataset_name)
    return data_classes.SrcTrgImgAnno(src=src_img_anno, trg=trg_img_anno)


DATASET_NAME_TO_CATEGORIES = {
    "spair-71k": spair_dataset_utils.SPAIR_SORTED_CATEGORIES,
    "ap-10k": ap10k_dataset_utils.AP10K_CATEGORIES,
    "pf-pascal": pfpascal_dataset_utils.PF_PASCAL_CATEGORIES,
}


def build_get_flipped_kpt_index(dataset_name: str, category: str = ""):
    if dataset_name == "spair-71k":
        return spair_dataset_utils.build_get_flipped_kpt_index(category)
    elif dataset_name == "ap-10k":
        return ap10k_dataset_utils.build_get_flipped_kpt_index()
    elif dataset_name == "pf-pascal":
        return pfpascal_dataset_utils.build_get_flipped_kpt_index(category)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def load_kpt_img_coords_torch(
    img_files: List[str],
    dataset_name: str,
    flips: List[bool],
    category: str = "",
):
    """
    Returns:
    - all_kpt_img_coords: List[Tensor], where each tensor has shape (n_kpts, 3) and the last dimension is (x, y, kpt_index)
    """
    get_flipped_kpt_index = build_get_flipped_kpt_index(dataset_name, category)
    all_kpt_img_coords = []
    if len(flips) == 0:
        flips = [False]
    for flip in flips:
        for img_file in img_files:
            kpt_img_coords = []
            # img_anno_fp = img_file.replace("JPEGImages", "ImageAnnotation").replace(".jpg", ".json")
            img_anno = load_normalized_img_anno_from_img_fp(img_file, dataset_name)
            w, h = img_anno.img_width, img_anno.img_height
            for i, (x, y) in enumerate(img_anno.kp_xy):
                kpt_index = (
                    img_anno.kp_ids[i]
                    if not flip
                    else get_flipped_kpt_index(img_anno.kp_ids[i])
                )
                x, y = image_processing.transform_point(x, y, w, flip=flip, angle_deg=0)
                kpt_img_coords.append([x, y, kpt_index])
            all_kpt_img_coords.append(torch.tensor(kpt_img_coords))
    return all_kpt_img_coords


def img_file2category(img_filepath: str):
    """Returns the category of an image file."""
    return img_filepath.split("/")[-2]


def img_fp2anno_fp(img_fp: str, dataset_name: Optional[str] = None):
    """Returns the annotation file path given an image file path."""
    if dataset_name is None:
        dataset_name = infer_dataset_name_from_fp(img_fp)
    if dataset_name == "pf-pascal":
        return img_fp.replace("JPEGImages", "Annotations").replace(".jpg", ".mat")
    else:
        return img_fp.replace("JPEGImages", "ImageAnnotation").replace(".jpg", ".json")


def compute_n_max_kpts(all_kpt_img_coords: List[torch.Tensor]):
    dataset_n_max_kpts = 30  # each category has at most 30 keypoints FOR ALL DATASETS
    unused_kpts = []
    kpt_counts = torch.zeros(dataset_n_max_kpts, dtype=torch.int64)
    for img_idx, kpt_img_coords in enumerate(all_kpt_img_coords):
        img_kpt_labels = kpt_img_coords[:, 2]
        kpt_counts += torch.bincount(img_kpt_labels, minlength=dataset_n_max_kpts)
    unused_kpts = torch.where(kpt_counts == 0)[0]
    n_max_kpts = int(1 + torch.max(torch.where(kpt_counts > 0)[0]).item())
    unused_kpts = unused_kpts[unused_kpts < n_max_kpts]
    return n_max_kpts, unused_kpts


def get_threshold_from_annotation(img_anno: data_classes.ImgAnnoNormalized):
    """Get the threshold from the annotation.
    Args:
        img_anno: image annotation
        img_size: size of the image
    """
    img_bndbox = img_anno.bbox
    threshold = max(
        (img_bndbox[2] - img_bndbox[0]),  # / img_anno["image_width"],
        (img_bndbox[3] - img_bndbox[1]),  # / img_anno["image_height"]
    )
    # dx, dy = img_bndbox[2] - img_bndbox[0], img_bndbox[3] - img_bndbox[1]
    # threshold = (dx ** 2 + dy ** 2)**0.5
    # threshold *= img_size / max(img_anno["image_width"], img_anno["image_height"])
    return threshold
