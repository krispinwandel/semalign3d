import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from segment_anything import SamPredictor, sam_model_registry

from semalign3d.core import data_classes
from semalign3d.core.generators.generator import Generator
from semalign3d.core.data import raw_data_utils


class MaskGenerator(Generator):
    """
    This mask generator uses the ground truth keypoint positions.
    Hence, these masks are not used for inference/evaluation.
    """

    def __init__(self, paths: data_classes.SemAlign3DPaths):
        self.paths = paths

        # Load models
        print("Loading SAM model...")
        sam = sam_model_registry["vit_h"](checkpoint=paths.sam_vit_h_ckpt_path)
        sam.to("cuda")
        self.predictor = SamPredictor(sam)

        img_dir = os.path.join(self.paths.spair_data_dir, "JPEGImages")
        self.all_img_files = [
            os.path.join(subdir, file)
            for subdir, dirs, files in os.walk(img_dir)
            for file in files
            if file.endswith(".jpg")
        ]

    def get_category_img_files(self, category):
        category_img_files = [
            img_file for img_file in self.all_img_files if category in img_file
        ]
        return category_img_files

    def save_mask_data(self, mask_candidates: np.ndarray, img_fp: str, category: str):
        out_dir = os.path.join(self.paths.seg_masks_dir, category)
        os.makedirs(out_dir, exist_ok=True)
        masks_fp = f"{out_dir}/{os.path.basename(img_fp).split('.')[0]}_masks.npy"
        np.save(masks_fp, mask_candidates)

    def process_image(self, img_fp: str, category: str):
        img = np.array(Image.open(img_fp))

        img_anno = raw_data_utils.load_normalized_img_anno_from_img_fp(
            img_fp, self.paths.dataset_name
        )
        input_points = np.array(img_anno.kp_xy, dtype=float)
        input_labels = np.ones(input_points.shape[0], dtype=int)

        self.predictor.set_image(img)
        masks, mask_scores, logits = self.predictor.predict(
            point_coords=input_points[:, :], point_labels=input_labels[:]
        )
        self.save_mask_data(masks, img_fp, category)

    def generate(self, category: str):
        img_files = self.get_category_img_files(category)
        for img_idx in tqdm(range(len(img_files))):
            img_fp = img_files[img_idx]
            self.process_image(img_fp, category)
