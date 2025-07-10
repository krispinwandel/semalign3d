import dis
from typing import Dict, List, Callable, Optional, Type
import torch
from tqdm import tqdm
from semalign3d.utils import torch_utils


# =============================
# Logger class
# =============================


class Logger:
    def __init__(self, name):
        """Initialize the logger with a name."""
        self.name = name
        self.logs = {}

    def log(self, key, message):
        """Add a log message under a specific key."""
        log_entry = message
        if isinstance(log_entry, torch.Tensor):
            log_entry = log_entry.detach().cpu()
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(log_entry)

    def get_logs(self, key=None):
        """Return all logs or logs for a specific key."""
        if key:
            return self.logs.get(key, [])
        return self.logs

    def clear_logs(self, key=None):
        """Clear logs for a specific key or all logs."""
        if key:
            if key in self.logs:
                del self.logs[key]
        else:
            self.logs = {}


# =============================
# Weight scheduler classes
# =============================


class WeightScheduler:
    def __init__(self) -> None:
        pass

    def get_weight(self, step: int):
        raise NotImplementedError


class LinearWeightScheduler(WeightScheduler):
    def __init__(self, steps: List[int], weights: List[float]) -> None:
        self.steps = steps
        self.weights = weights

    def get_weight(self, step: int):
        for i, s in enumerate(self.steps):
            if step < s:
                n_iter = self.steps[i] - self.steps[i - 1]
                n_step = step - self.steps[i - 1]
                return (
                    self.weights[i - 1]
                    + (self.weights[i] - self.weights[i - 1]) * n_step / n_iter
                )
        return self.weights[-1]


class LinearWeightSchedulerTorch(WeightScheduler):
    def __init__(
        self, steps: List[int], weights: List[float], device: torch.device
    ) -> None:
        self.steps = torch.tensor(steps, device=device)
        self.weights = torch.tensor(weights, device=device)
        self.device = device

    def get_weight(self, step: int):
        step_tensor = torch.tensor(step, device=self.device)
        for i, s in enumerate(self.steps):
            if step_tensor < s:
                n_iter = self.steps[i] - self.steps[i - 1]
                n_step = step_tensor - self.steps[i - 1]
                return (
                    self.weights[i - 1]
                    + (self.weights[i] - self.weights[i - 1]) * n_step / n_iter
                )
        return self.weights[-1]


# =============================
# Utils for optimization
# =============================


def gradify(data: Dict[str, torch.Tensor], opt_data_keys: List[str], device="cpu"):
    """Convert data to require gradients."""
    data_: Dict[str, torch.Tensor] = {}
    for key in data.keys():
        if key in opt_data_keys:
            data_[key] = data[key].clone().detach().to(device).requires_grad_(True)
        else:
            data_[key] = data[key].clone().detach().to(device).requires_grad_(False)
    return data_


# =============================
# Optimization functions
# =============================


# optimization loop
def opt_data(
    data: Dict[str, torch.Tensor],
    opt_data_keys: List[str],
    energy_func: Callable[
        [Dict[str, torch.Tensor], Logger, int], Dict[str, torch.Tensor]
    ],
    logger: Logger,
    # params
    lr: float = 0.1,
    n_iter: int = 100,
    callback_fn: Optional[
        Callable[[Dict[str, torch.Tensor], torch.Tensor, Logger], None]
    ] = None,
    device="cpu",
    use_grad_norm=False,
    normalize_grads_to_one=False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    data_internal: Optional[Dict[str, torch.Tensor]] = None,
    callback_rate: int = 1,
    disable_progress_bar=False,
    scheduler_class: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
    scheduler_params: Optional[Dict] = None,
    grad_mask: Optional[Dict[str, torch.Tensor]] = None,
    optimizer_type: str = "adam",
):
    """Optimize"""
    if data_internal is None:
        data_internal = gradify(data, opt_data_keys, device)

    if optimizer is None:
        if optimizer_type == "adam":
            optimizer = torch.optim.AdamW(
                [data_internal[key] for key in opt_data_keys], lr=lr
            )
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                [data_internal[key] for key in opt_data_keys], lr=lr
            )
        else:
            raise ValueError("optimizer type not supported")

    if scheduler_class is not None and scheduler_params is not None:
        scheduler = scheduler_class(optimizer, **scheduler_params)
    else:
        scheduler = None

    grad_mask_device: Optional[Dict[str, torch.Tensor]] = None
    if grad_mask is not None:
        grad_mask_device = {}
        for key in grad_mask:
            grad_mask_device[key] = grad_mask[key].to(device)

    callback_counter = 0

    for step in tqdm(range(n_iter), disable=disable_progress_bar):
        optimizer.zero_grad()
        total_energy = energy_func(data_internal, logger, step)["loss"]
        total_energy.backward()

        if use_grad_norm:
            grad_norm = torch.tensor(
                [torch.norm(data_internal[key].grad) for key in opt_data_keys]
            )
            grad_norm = grad_norm.mean()
            if torch.isnan(grad_norm):
                print("NaN grad norm")
                break
            grad_norm.backward()

        if normalize_grads_to_one:
            for key in opt_data_keys:
                data_internal[key].grad /= torch.norm(data_internal[key].grad) + 1e-6

        if grad_mask_device is not None:
            for key in opt_data_keys:
                data_internal[key].grad *= grad_mask_device[key]

        # for key in opt_data_keys:
        #     print(f"Key: {key}, Grad: {data_internal[key].grad.sum()}")

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            # print(f"Step: {step}, LR: {scheduler.get_last_lr()}")

        if callback_fn is not None and callback_counter % callback_rate == 0:
            callback_fn(data_internal, total_energy, logger)
        callback_counter += 1

    out: Dict[str, torch.Tensor] = {}
    for key, val in data_internal.items():
        out[key] = val.detach().clone()
    return out, total_energy


def optimize_lbfgs(
    data: Dict[str, torch.Tensor],
    opt_data_keys: list,
    energy_func: Callable[[Dict[str, torch.Tensor], Logger], torch.Tensor],
    logger: Logger,
    device,
    lr,
    n_iter,
    callback_fn=None,
    callback_rate: int = 1,
):
    data_: Dict[str, torch.Tensor] = {}

    for key in data.keys():
        if key in opt_data_keys:
            data_[key] = data[key].clone().detach().to(device).requires_grad_(True)
        else:
            data_[key] = data[key].clone().detach().to(device).requires_grad_(False)

    # LBFGS optimizer requires a closure
    optimizer = torch.optim.LBFGS([data_[key] for key in opt_data_keys], lr=lr)
    callback_counter = 0

    for _ in tqdm(range(n_iter)):

        def closure():
            optimizer.zero_grad()  # Reset gradients to zero
            total_energy = energy_func(data_, logger)  # Compute energy (loss)
            total_energy.backward()  # Backpropagation
            return total_energy  # Return loss

        optimizer.step(closure)  # Call step with the closure

        total_energy = energy_func(data_, logger)  # Calculate energy after the step

        if callback_fn is not None and callback_counter % callback_rate == 0:
            callback_fn(data_, total_energy)
        callback_counter += 1

    out: Dict[str, torch.Tensor] = {}
    for key, val in data_.items():
        out[key] = val.detach().clone()

    return out, total_energy


def build_loss(
    loss_fun,
    constants,
    weight_scheduler_dict: Dict[str, WeightScheduler] = {},
    compile=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    loss_fun = torch.compile(loss_fun) if compile else loss_fun
    constants_device = {}
    torch_utils.clone_to_device(constants, constants_device, device=device)

    def loss(data_opt, logger, step: int):
        weights = {}
        for key, scheduler in weight_scheduler_dict.items():
            weights[key] = scheduler.get_weight(step)
        return loss_fun(
            **data_opt,
            **constants_device,
            **weights,
        )

    return loss
