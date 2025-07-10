import sys
import os
import time
from rich.console import Console
from rich.panel import Panel

from semalign3d.core.generators import (
    geom_beta_dist_generator,
    focal_length_generator,
    sparse_pc_generator,
    dense_pc_generator,
    depth_generator_depth_any,
    depth_generator_vggt,
    mask_generator,
    data_split_generator,
)
from semalign3d.core.data import setup_paths, raw_data_utils

console = Console()

# Configuration
scripts_dir = os.path.dirname(os.path.abspath(__file__))
run_config_path = f"{scripts_dir}/run_config.yaml"
print(run_config_path)
paths = setup_paths.setup_paths(run_config_path)
depth_gen_ids = ["depth_any", "vggt"]
assert paths.depth_generator in depth_gen_ids, (
    f"Invalid depth generator: {paths.depth_generator}. "
    f"Expected one of {depth_gen_ids}."
)
use_vggt = paths.depth_generator == "vggt"

all_categories = raw_data_utils.DATASET_NAME_TO_CATEGORIES[paths.dataset_name]
categories = list(sys.argv[1:]) if len(sys.argv) > 1 else all_categories


STEPS = [
    "gen_splits",
    "gen_masks",
    "gen_depths",
    "gen_geom_stats",
    "gen_sparse_pc",
    "gen_dense_pc",
]
steps = STEPS


print("Dataset name:", paths.dataset_name)
print("Categories:", categories)
print("Suffix:", paths.suffix)
print("Using VGG-T:", use_vggt)

t_start = time.time()


# -----------------------------------------------
# 1) generate data splits
# -----------------------------------------------
if "gen_splits" in steps:
    console.print(Panel(f"(1/6) Generate Data Splits", style="bold blue"))
    data_split_gen = data_split_generator.DataSplitGenerator(paths)
    data_split_gen.process_categories(categories)


# -----------------------------------------------
# 2) generate masks
# -----------------------------------------------
if "gen_masks" in steps:
    console.print(Panel(f"(2/6) Generate Masks", style="bold blue"))
    mask_gen = mask_generator.MaskGenerator(paths)
    mask_gen.process_categories(categories)
    del mask_gen  # free memory


# -----------------------------------------------
# 3) generate depth + camera instrinsics
# -----------------------------------------------
if "gen_depths" in steps:
    console.print(Panel("(3/6) Generate Depths", style="bold blue"))
    if use_vggt:
        depth_gen_vggt = depth_generator_vggt.DepthGeneratorVggt(paths)
        depth_gen_vggt.process_categories(categories)
        del depth_gen_vggt  # free memory
    else:
        # TODO depth + focal length generation should save to the same format as vggt so
        #   that we do not need to pass the use_vggt flag
        depth_gen_depth_any = depth_generator_depth_any.DepthGeneratorDepthAny(paths)
        depth_gen_depth_any.process_categories(categories)
        del depth_gen_depth_any
        # we also need to find focal lengths when we use DepthAnything
        focal_length_gen = focal_length_generator.FocalLengthGenerator(paths)
        focal_length_gen.process_categories(categories)
        del focal_length_gen


# -----------------------------------------------
# 4) generate geom stats
# -----------------------------------------------
if "gen_geom_stats" in steps:
    console.print(Panel("(4/6) Generate Geom Stats", style="bold blue"))
    geom_beta_dist_gen = geom_beta_dist_generator.GeomBetaDistGenerator(
        paths=paths, use_vggt=use_vggt
    )
    geom_beta_dist_gen.process_categories(categories)
    del geom_beta_dist_gen  # free memory


# -----------------------------------------------
# 5) create sparse pc from geom stats
# -----------------------------------------------
if "gen_sparse_pc" in steps:
    console.print(Panel("(5/6) Generate Sparse PC", style="bold blue"))
    sparse_pc_gen = sparse_pc_generator.SparsePCGenerator(
        paths, verbose=True, use_vggt=use_vggt
    )
    sparse_pc_gen.process_categories(categories)
    del sparse_pc_gen  # free memory


# -----------------------------------------------
# 6) create dense pc from sparse pc
# -----------------------------------------------
if "gen_dense_pc" in steps:
    console.print(Panel("(8/8) Generate Dense PC", style="bold blue"))
    dense_pc_gen = dense_pc_generator.DensePCGenerator(paths=paths, use_vggt=use_vggt)
    dense_pc_gen.process_categories(categories)
    del dense_pc_gen  # free memory


t_end = time.time()
t_minutes = (t_end - t_start) // 60
t_seconds = (t_end - t_start) % 60
print(f"All categories done in {int(t_minutes)} minutes and {int(t_seconds)} seconds.")
