import os
import numpy as np
import torch

from semalign3d.utils import torch_utils
from semalign3d.core.geometry import geom_filter
from semalign3d.core.data import sem3d_data_utils, keypoint_processing, raw_data_utils
from semalign3d.core.losses import focal_length_loss
from semalign3d.utils import opt_utils
from semalign3d.core import data_classes


class FocalLengthGenerator:
    def __init__(
        self,
        paths: data_classes.SemAlign3DPaths,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        opt_params={
            "n_iter": 1000,
            "lr": 0.001,
        },
    ):
        self.paths = paths
        self.dataset_name = paths.dataset_name
        self.depth_base_dir = paths.depths_dir
        self.masks_base_dir = paths.seg_masks_dir
        self.out_dir = paths.img_focal_lengths_opt_dir
        self.device = device
        self.opt_params = opt_params

    def load_data(self, img_files, category, flips):
        depths = sem3d_data_utils.load_depth(
            self.depth_base_dir, category, img_files, flips
        )
        seg_masks, _ = sem3d_data_utils.load_segmentation_masks(
            self.masks_base_dir, category, img_files, flips
        )
        kpt_img_coords = raw_data_utils.load_kpt_img_coords_torch(
            img_files=img_files,
            dataset_name=self.dataset_name,
            flips=flips,
            category=category,
        )

        return {
            "depths": depths,
            "seg_masks": seg_masks,
            "kpt_img_coords": kpt_img_coords,
        }

    def build_constants(self, seg_masks, kpt_img_coords, depths):
        # NOTE we might be able to skip this step
        # - correct keypoint coords with dilated segmentation masks
        # - extract depth values at corrected keypoint coords
        # - kpts_normalized has shape (n_imgs, n_max_kpts, 2). If a keypoint is not visible, its coords are nan
        kpts_normalized, depth_values_all = (
            keypoint_processing.correct_all_keypoint_coords_with_depths(
                img_seg_masks=seg_masks,
                kpt_img_coords=kpt_img_coords,
                depths=depths,
                show_progress_bar=False,
            )[:2]
        )

        # create img_shape tensor later used for optimization (to compute px in K_inv for reprojection)
        n_imgs = len(seg_masks)
        img_shapes = torch.zeros((n_imgs, 2))
        for i in range(n_imgs):
            img_shapes[i, :] = torch.tensor(depths[i].shape[:2])

        # for optimization, replace nan with 0 and generate vertex mask
        kpts_normalized_no_nan = kpts_normalized.clone()
        kpts_normalized_no_nan[torch.isnan(kpts_normalized_no_nan)] = 0.0
        depth_values_all_no_nan = depth_values_all.clone()
        depth_values_all_no_nan[torch.isnan(depth_values_all_no_nan)] = 0.0

        # masks
        vertex_mask = torch_utils.generate_is_not_nan_mask(kpts_normalized)
        angles_mask, ratios_mask = geom_filter.generate_angle_and_ratio_mask(
            vertex_mask
        )

        return {
            "kpt_xy_normalized": kpts_normalized_no_nan,
            "kpt_depth": depth_values_all_no_nan,
            "img_shapes": img_shapes,
            # "kpt_is_not_nan": vertex_mask,
            "angles_mask": angles_mask,
            "ratios_mask": ratios_mask,
        }

    def save_focal_lengths(self, focal_lengths_inv_opt, category):
        os.makedirs(self.out_dir, exist_ok=True)
        focal_lengths_inv_opt_file = (
            f"{self.out_dir}/{category}_focal_lengths_inv_opt{self.paths.suffix}.npy"
        )
        np.save(
            focal_lengths_inv_opt_file, torch_utils.to_np_array(focal_lengths_inv_opt)
        )

    def generate(self, img_files, category, flips=[False, True]):
        data = self.load_data(img_files, category, flips)
        depths = data["depths"]
        seg_masks = data["seg_masks"]
        kpt_img_coords = data["kpt_img_coords"]
        constants = self.build_constants(
            seg_masks=seg_masks,
            kpt_img_coords=kpt_img_coords,
            depths=depths,
        )
        loss = opt_utils.build_loss(
            loss_fun=focal_length_loss.calculate_focal_length_loss,
            constants=constants,
            compile=False,  # not worth the compilation time
            device=self.device,
        )
        focal_lengths_init = torch.full(
            (constants["img_shapes"].shape[0],), 0.2, dtype=torch.float32
        )
        data_opt, loss_val = opt_utils.opt_data(
            data={
                "focal_lengths_inv": focal_lengths_init,
            },
            opt_data_keys=["focal_lengths_inv"],
            energy_func=loss,
            logger=opt_utils.Logger("focal_length_loss"),
            device=self.device,
            lr=self.opt_params["lr"],
            n_iter=self.opt_params["n_iter"],
        )
        print("Optimization finished with loss:", loss_val.item())
        self.save_focal_lengths(data_opt["focal_lengths_inv"], category)

    def process_categories(self, categories):
        """
        Process multiple categories to generate focal lengths.
        Args:
            paths: Object containing dataset paths.
            categories: List of categories to process.
        """
        for category in categories:
            print("Processing category:", category)
            try:
                category_img_files = np.load(
                    f"{self.paths.img_file_splits_dir}/{category}_img_files_trn.npy"
                ).tolist()
                print("Number of images:", len(category_img_files))
                self.generate(
                    img_files=category_img_files,
                    category=category,
                )
            except Exception as e:
                print(f"Error processing category {category}: {e}")
