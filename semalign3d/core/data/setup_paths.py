import yaml
from semalign3d.core import data_classes


def setup_paths(path_to_run_config: str):
    with open(path_to_run_config, "r") as f:
        run_config = yaml.safe_load(f)

    semalign3d_data_dir = run_config["semalign3d_data_dir"]
    spair_data_dir = run_config["spair_data_dir"]
    sam_vit_h_ckpt_path = run_config["sam_vit_h_ckpt_path"]
    sam_vit_h_ckpt_path = None if sam_vit_h_ckpt_path == "" else sam_vit_h_ckpt_path

    return data_classes.SemAlign3DPaths(
        dataset_name=run_config["dataset_name"],
        semalign3d_data_dir=semalign3d_data_dir,
        spair_data_dir=spair_data_dir,
        embds_folder=run_config["embds_folder"],
        aggre_net_ckpt_path=run_config["aggre_net_ckpt_path"],
        sam_vit_h_ckpt_path=sam_vit_h_ckpt_path,
        suffix=run_config.get("suffix", "_vggt"),
        depth_generator=run_config.get("depth_generator", "vggt"),
    )
