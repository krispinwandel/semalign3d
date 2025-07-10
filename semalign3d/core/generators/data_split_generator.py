import glob
import os
import numpy as np
from semalign3d.core.data import raw_data_utils
from semalign3d.core import data_classes
from semalign3d.core.generators.generator import Generator


class DataSplitGenerator(Generator):

    def __init__(self, paths: data_classes.SemAlign3DPaths):
        self.dataset_name = paths.dataset_name
        self.spair_dir = paths.spair_data_dir
        self.out_dir = paths.img_file_splits_dir

    def generate(self, category: str):
        generate_img_file_splits(
            self.dataset_name, self.spair_dir, category, self.out_dir
        )


def generate_img_file_splits(
    dataset_name: str, spair_dir: str, category: str, out_dir: str
):
    os.makedirs(out_dir, exist_ok=True)
    splits = ["trn", "val", "test"]
    for split in splits:
        pairs = sorted(
            glob.glob(f"{spair_dir}/PairAnnotation/{split}/*:{category}.json")
        )
        files = []
        for pair in pairs:
            src_trg_img_anno = raw_data_utils.load_normalized_src_tgt_img_anno(
                pair, dataset_name
            )
            source_fp = f"{spair_dir}/{src_trg_img_anno.src.rel_fp}"
            target_fp = f"{spair_dir}/{src_trg_img_anno.trg.rel_fp}"
            files.append(source_fp)
            files.append(target_fp)
        unique_files = list(set(files))

        np.save(
            os.path.join(out_dir, f"{category}_img_files_{split}.npy"),
            np.array(unique_files),
        )
