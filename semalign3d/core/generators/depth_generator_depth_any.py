import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from semalign3d.core import data_classes
from semalign3d.core.generators.generator import Generator


class DepthGeneratorDepthAny(Generator):
    def __init__(
        self,
        paths: data_classes.SemAlign3DPaths,
        model_name: str = "depth-anything/Depth-Anything-V2-Large-hf",
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.paths = paths
        print("Loading model...")
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

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

    def save_img_data(self, depth: np.ndarray, img_fp: str, category: str):
        out_dir = os.path.join(self.paths.depths_dir, category)
        os.makedirs(out_dir, exist_ok=True)
        depth_fp = f"{out_dir}/{os.path.basename(img_fp).split('.')[0]}_depth.npy"
        np.save(depth_fp, depth)

    def generate(self, category: str):
        img_files = self.get_category_img_files(category)
        for img_idx in tqdm(range(len(img_files))):
            img_fp = img_files[img_idx]
            img = Image.open(img_fp)

            # Prediction
            inputs = self.image_processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            # NOTE: Output is not normalized
            output = prediction.squeeze().cpu().numpy()

            # save
            self.save_img_data(output, img_fp, category)
