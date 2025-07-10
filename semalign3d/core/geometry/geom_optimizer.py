import torch
from dataclasses import asdict

from semalign3d.core import data_classes
from semalign3d.core.losses import geom_loss
from semalign3d.utils import opt_utils


DEFAULT_OPT_PARAMS = {
    "lr": 1e-3,
    "n_iter": 1000,
}


class GeomOptimizer:

    def __init__(
        self,
        opt_params={},
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.opt_params = DEFAULT_OPT_PARAMS
        self.opt_params.update(opt_params)
        self.device = device

    def optimize(
        self,
        kpt_xyz_obj,
        geom_stats: data_classes.GeomRelationStatisticsBetaSimple,
        geom_relation_combinations: data_classes.GeomRelationCombinations,
    ):
        constants = {
            "geom_stats": asdict(geom_stats),
            "geom_relation_combinations": asdict(geom_relation_combinations),
        }
        initial_guess = {"kpt_xyz_obj": kpt_xyz_obj[None]}
        loss_fun = opt_utils.build_loss(
            loss_fun=lambda **args: {
                "loss": torch.sum(geom_loss.calculate_geom_loss(**args))
            },
            constants=constants,
            compile=False,
            device=self.device,
        )
        data_opt, loss_val = opt_utils.opt_data(
            data=initial_guess,
            opt_data_keys=["kpt_xyz_obj"],
            energy_func=loss_fun,
            lr=self.opt_params["lr"],
            n_iter=self.opt_params["n_iter"],
            device=self.device,
            logger=opt_utils.Logger("geom_opt"),
            disable_progress_bar=False,
        )
        return data_opt["kpt_xyz_obj"][0]  # remove batch dimension
