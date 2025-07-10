import os
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
from torchvision import transforms as TF

from semalign3d.core import data_classes
from semalign3d.core.data import (
    raw_data_utils,
)
from semalign3d.core.generators.generator import Generator
from semalign3d.utils import camera_intrinsics
from semalign3d.ext.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from semalign3d.ext.vggt.models.vggt import VGGT


class DepthGeneratorVggt(Generator):
    def __init__(self, paths: data_classes.SemAlign3DPaths):
        self.paths = paths
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        print("Loading model...")
        # Initialize the model and load the pretrained weights.
        # This will automatically download the model weights the first time it's run, which may take a while.
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)

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

    def save_depth_data(self, img_fp: str, category: str, depth_data: dict):
        out_dir = os.path.join(self.paths.vggt_dir, category)
        img_id_str = os.path.basename(img_fp).split(".")[0]
        os.makedirs(out_dir, exist_ok=True)
        for k, v in depth_data.items():
            torch.save(v.cpu(), f"{out_dir}/{img_id_str}_{k}.pth")

    def generate(self, category):
        category_img_files = self.get_category_img_files(category)
        for img_fp in tqdm(category_img_files):
            depth_data = generate_world_points(
                self.model,
                self.device,
                self.dtype,
                img_fp,
            )
            self.save_depth_data(
                img_fp,
                category,
                depth_data,
            )


def load_and_preprocess_images_with_bbox(image_path, bbox=None):
    """
    Args:
        bbox: A list of bounding boxes for each image in the format [x_min, y_min, x_max, y_max].
    """
    to_tensor = TF.ToTensor()
    target_size = 518

    # Open image
    img = Image.open(image_path)
    if bbox is not None:
        img = Image.fromarray(np.array(img)[bbox[1] : bbox[3], bbox[0] : bbox[2], :])
    width, height = img.size

    # Make the largest dimension 518px while maintaining aspect ratio
    if height > width:
        new_height = target_size
        new_width = (
            round(width * (new_height / height) / 14) * 14
        )  # Make divisible by 14
    else:
        # Original behavior: set width to 518px
        new_width = target_size
        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

    # Resize with new dimensions (width, height)
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img = to_tensor(img)  # Convert to tensor (0, 1)

    # height should not be larger than width, so we need to pad the width
    rect = (0, 0, new_height, new_width)
    if height > width:
        w_padding = target_size - img.shape[2]
        if w_padding > 0:
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            # Pad with white (value=1.0)
            img = torch.nn.functional.pad(
                img, (pad_left, pad_right, 0, 0), mode="constant", value=1.0
            )
            rect = (0, pad_left, target_size, target_size - pad_right)

    return img.unsqueeze(0).unsqueeze(0), rect


def generate_world_points(model, device, dtype, img_fp: str):
    img_anno = raw_data_utils.load_normalized_img_anno_from_img_fp(img_fp)
    pixel_padding = 40
    bbox_wider = [
        max(img_anno.bbox[0] - pixel_padding, 0),
        max(img_anno.bbox[1] - pixel_padding, 0),
        min(img_anno.bbox[2] + pixel_padding, img_anno.img_width),
        min(img_anno.bbox[3] + pixel_padding, img_anno.img_height),
    ]
    # NOTE we crop the image for better performance
    # NOTE for rect, the order is (y_min, x_min, y_max, x_max)
    images, rect = load_and_preprocess_images_with_bbox(img_fp, bbox_wider)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images.to(device))

    # now decrop/descale everything
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    depth_orig, K_orig = camera_intrinsics.uncrop_depth_and_intrinsics(
        predictions["depth"][0, 0][:, :, 0],
        intrinsic[0, 0],
        rect,
        bbox_wider,
        (img_anno.img_height, img_anno.img_width),
        fill_value=torch.max(predictions["depth"][0, 0, :, :, 0]),
    )
    depth_conf_orig, _ = camera_intrinsics.uncrop_depth_and_intrinsics(
        predictions["depth_conf"][0, 0, :, :],
        intrinsic[0, 0],
        rect,
        bbox_wider,
        (img_anno.img_height, img_anno.img_width),
        fill_value=0.0,
    )
    return {
        "depth_orig": depth_orig,
        "depth_conf_orig": depth_conf_orig,
        "K_orig": K_orig,
    }
