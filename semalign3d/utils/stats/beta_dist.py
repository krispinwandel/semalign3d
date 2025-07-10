import torch
from typing import Optional, Union, Tuple
from torch.distributions import Distribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
import torch.nn.functional as F

# import torch.special as special
from scipy.special import betainc
from scipy.stats import beta
from scipy.optimize import brentq
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import beta as beta_sp


class CustomBeta(Distribution):
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration2": constraints.positive,
    }
    support = constraints.unit_interval

    def __init__(self, concentration1, concentration2, validate_args=None):
        self.concentration1, self.concentration2 = broadcast_all(
            concentration1, concentration2
        )
        batch_shape = self.concentration1.size()
        self.concentration1_np = self.concentration1.cpu().numpy()
        self.concentration2_np = self.concentration2.cpu().numpy()
        super(CustomBeta, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        gamma1 = torch.distributions.Gamma(self.concentration1, 1).sample(shape)
        gamma2 = torch.distributions.Gamma(self.concentration2, 1).sample(shape)
        return gamma1 / (gamma1 + gamma2)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (
            (self.concentration1 - 1) * torch.log(value)
            + (self.concentration2 - 1) * torch.log1p(-value)
            - torch.lgamma(self.concentration1)
            - torch.lgamma(self.concentration2)
            + torch.lgamma(self.concentration1 + self.concentration2)
        )

    def cdf(self, value):
        # TODO we need a faster implementation here
        if self._validate_args:
            self._validate_sample(value)
        device_target = self.concentration1.device
        value_np = value.cpu().numpy()
        cdf_np = betainc(self.concentration1_np, self.concentration2_np, value_np)
        cdf = torch.tensor(cdf_np, device=device_target)
        return cdf

    def entropy(self):
        return (
            torch.lgamma(self.concentration1)
            + torch.lgamma(self.concentration2)
            - torch.lgamma(self.concentration1 + self.concentration2)
            - (self.concentration1 - 1) * torch.digamma(self.concentration1)
            - (self.concentration2 - 1) * torch.digamma(self.concentration2)
            + (self.concentration1 + self.concentration2 - 2)
            * torch.digamma(self.concentration1 + self.concentration2)
        )

    @property
    def mean(self):
        return self.concentration1 / (self.concentration1 + self.concentration2)

    @property
    def variance(self):
        total = self.concentration1 + self.concentration2
        return (self.concentration1 * self.concentration2) / (
            total.pow(2) * (total + 1)
        )

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def mode(self):
        return (self.concentration1 - 1) / (
            self.concentration1 + self.concentration2 - 2
        )


# ========================
# Other utility functions
# ========================


def compute_likelihood_beta(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    x_low: Optional[torch.Tensor] = None,
    x_high: Optional[torch.Tensor] = None,
):
    """
    Compute likelihood of x being in range [-infty, -xl] and [xr, infty],
    where xl = x - x_diff_mean and xr = x + x_diff_mean,
    where x_diff_mean = |x - x_mean|, and x_mean = alpha / (alpha + beta),
    and the beta distribution is defined by alpha and beta.
    NOTE: at x=x_mean, this value is 1.

    Args:
        - x: (D1,D2) D1=#samples, D2=#distributions (or swapped)
        - alpha: (1, D2) or (D1, 1); shape parameter of the beta distribution
        - beta: (1, D2) or (D1, 1); shape parameter of the beta distribution
    Returns:
        - likelihood: (D1, D2) tensor
    """

    # transform x to be in the range [0, 1]
    if x_low is not None and x_high is not None:
        x = (x - x_low) / (x_high - x_low)

    # Create a beta distribution with the given alpha and beta parameters
    beta_dist = CustomBeta(alpha, beta)

    # Compute likelihood of x being in range [0, xl] and [xr, 1]
    x_isnan = torch.isnan(x)
    x[x_isnan] = 0
    x_diff_mean = torch.abs(x - beta_dist.mean)
    xl = torch.clamp(beta_dist.mean - x_diff_mean, 1e-8, 1 - 1e-8)
    xr = torch.clamp(beta_dist.mean + x_diff_mean, 1e-8, 1 - 1e-8)
    # print("x", x, "mean", beta_dist.mean, "xl", xl, "xr", xr, sep="\n")
    likelihood = beta_dist.cdf(xl) + 1 - beta_dist.cdf(xr)
    likelihood[x_isnan] = torch.nan

    return likelihood


def compute_log_prob_beta(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    x_low: Optional[Union[torch.Tensor, float]] = None,
    x_high: Optional[Union[torch.Tensor, float]] = None,
):
    """
    Compute likelihood of x being in range [-infty, -xl] and [xr, infty],
    where xl = x - x_diff_mean and xr = x + x_diff_mean,
    where x_diff_mean = |x - x_mean|, and x_mean = alpha / (alpha + beta),
    and the beta distribution is defined by alpha and beta.
    NOTE: at x=x_mean, this value is 1.

    Args:
        - x: (D1,D2) D1=#samples, D2=#distributions (or swapped)
        - alpha: (1, D2) or (D1, 1); shape parameter of the beta distribution
        - beta: (1, D2) or (D1, 1); shape parameter of the beta distribution
    Returns:
        - likelihood: (D1, D2) tensor
    """
    # transform x to be in the range [0, 1]
    if x_low is not None and x_high is not None:
        x = (x - x_low) / (x_high - x_low)
    x = torch.clamp(x, 1e-6, 1 - 1e-6)

    # Create a beta distribution with the given alpha and beta parameters
    beta_dist = CustomBeta(alpha, beta)

    # Compute likelihood of x being in range [0, xl] and [xr, 1]
    # x_isnan = torch.isnan(x)
    # if torch.sum(x_isnan) > 0:
    #     print("x_isnan", x)
    # x[x_isnan] = 0
    x_pdf = beta_dist.log_prob(x)
    # x_pdf[x_isnan] = torch.nan

    return x_pdf


@torch.jit.script
def log_prob_beta(
    alpha: torch.Tensor, beta: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """
    Computes the log probability of x under a Beta distribution with parameters alpha and beta.

    Args:
    alpha (torch.Tensor): Shape parameter alpha (must be > 0).
    beta (torch.Tensor): Shape parameter beta (must be > 0).
    x (torch.Tensor): Input tensor of values in the range [0, 1].

    Returns:
    torch.Tensor: Log probability of the Beta distribution evaluated at x.
    """
    # Check the input is within the valid range [0, 1]
    # if torch.any((x <= 0) | (x >= 1)):
    #     raise ValueError("x must be in the range (0, 1)")

    # First term: (alpha - 1) * log(x)
    term1 = (alpha - 1) * torch.log(x)

    # Second term: (beta - 1) * log(1 - x)
    term2 = (beta - 1) * torch.log(1 - x)

    # Third term: - log Beta(alpha, beta)
    # log(Beta(alpha, beta)) = log(Gamma(alpha)) + log(Gamma(beta)) - log(Gamma(alpha + beta))
    log_beta = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)

    # Combine terms to compute log-probability
    log_prob = term1 + term2 - log_beta

    return log_prob


# @torch.jit.script
def compute_log_prob_beta_fast(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    x_low: torch.Tensor,
    x_high: torch.Tensor,
):
    """
    Compute likelihood of x being in range [-infty, -xl] and [xr, infty],
    where xl = x - x_diff_mean and xr = x + x_diff_mean,
    where x_diff_mean = |x - x_mean|, and x_mean = alpha / (alpha + beta),
    and the beta distribution is defined by alpha and beta.
    NOTE: at x=x_mean, this value is 1.

    Args:
        - x: (D1,D2) D1=#samples, D2=#distributions (or swapped)
        - alpha: (1, D2) or (D1, 1); shape parameter of the beta distribution
        - beta: (1, D2) or (D1, 1); shape parameter of the beta distribution
    Returns:
        - likelihood: (D1, D2) tensor
    """
    # transform x to be in the range [0, 1]
    x = (x - x_low) / (x_high - x_low)
    x = torch.clamp(x, 1e-6, 1 - 1e-6)

    # Create a beta distribution with the given alpha and beta parameters
    x_pdf_log = log_prob_beta(alpha, beta, x)

    return x_pdf_log


def find_b(alpha, beta_param, p):
    # Handling cases where mode is not defined
    if alpha <= 1 and beta_param <= 1:
        # No well-defined mode, decision based on density at boundaries
        density_at_0 = beta.pdf(0, alpha, beta_param)
        density_at_1 = beta.pdf(1, alpha, beta_param)
        x_star = 0 if density_at_0 >= density_at_1 else 1
    elif alpha <= 1:
        # Mode is at 0 when alpha <= 1 and beta > 1
        x_star = 0
    elif beta_param <= 1:
        # Mode is at 1 when alpha > 1 and beta <= 1
        x_star = 1
    else:
        # Mode is defined normally
        x_star = (alpha - 1) / (alpha + beta_param - 2)

    # Function to calculate the cumulative probability in the interval [max(x_star-b, 0), min(x_star+b, 1)]
    def cumulative_prob_diff(b):
        lower_bound = max(0, x_star - b)
        upper_bound = min(1, x_star + b)
        return (
            beta.cdf(upper_bound, alpha, beta_param)
            - beta.cdf(lower_bound, alpha, beta_param)
            - p
        )

    # The maximum possible b is the point where the interval would touch [0,1]
    b_max = max(x_star, 1 - x_star)

    # Use a root-finding method to solve for b
    b_solution = brentq(cumulative_prob_diff, 0, b_max)

    return b_solution


def compute_beta_mode_single(alpha: float, beta_param: float):
    # Find the maximum possible value of b
    if alpha <= 1 and beta_param <= 1:
        # No well-defined mode, decision based on density at boundaries
        density_at_0 = beta.pdf(0, alpha, beta_param)
        density_at_1 = beta.pdf(1, alpha, beta_param)
        x_star = 0 if density_at_0 >= density_at_1 else 1
    elif alpha <= 1:
        # Mode is at 0 when alpha <= 1 and beta > 1
        x_star = 0
    elif beta_param <= 1:
        # Mode is at 1 when alpha > 1 and beta <= 1
        x_star = 1
    else:
        # Mode is defined normally
        x_star = (alpha - 1) / (alpha + beta_param - 2)
    return float(x_star)


def compute_beta_mode_np(alpha: np.ndarray, beta_param: np.ndarray):
    # Find the maximum possible value of b
    x_star = np.zeros_like(alpha)
    mask = (alpha <= 1) & (beta_param <= 1)
    density_at_0 = beta.pdf(0, alpha[mask], beta_param[mask])
    density_at_1 = beta.pdf(1, alpha[mask], beta_param[mask])
    x_star[mask] = np.where(density_at_0 >= density_at_1, 0, 1)

    mask = (alpha <= 1) & (beta_param > 1)
    x_star[mask] = 0

    mask = (alpha > 1) & (beta_param <= 1)
    x_star[mask] = 1

    mask = (alpha > 1) & (beta_param > 1)
    x_star[mask] = (alpha[mask] - 1) / (alpha[mask] + beta_param[mask] - 2)

    return x_star.astype(float)


def compute_beta_mode_torch(alpha: torch.Tensor, beta_param: torch.Tensor):
    # Find the maximum possible value of b
    x_star = torch.zeros_like(alpha)
    mask = (alpha == 1) & (beta_param == 1)
    x_star[mask] = 0.5  # Special case when both alpha and beta are 1
    mask = (alpha < 1) & (beta_param < 1)
    density_at_0 = beta.pdf(
        0, alpha[mask].cpu().numpy(), beta_param[mask].cpu().numpy()
    )
    density_at_1 = beta.pdf(
        1, alpha[mask].cpu().numpy(), beta_param[mask].cpu().numpy()
    )
    density_at_0 = torch.from_numpy(density_at_0).to(alpha.device).float()
    density_at_1 = torch.from_numpy(density_at_1).to(alpha.device).float()
    x_star[mask] = torch.where(density_at_0 >= density_at_1, 0.0, 1.0).float()

    mask = (alpha <= 1) & (beta_param > 1)
    x_star[mask] = 0

    mask = (alpha > 1) & (beta_param <= 1)
    x_star[mask] = 1

    mask = (alpha > 1) & (beta_param > 1)
    x_star[mask] = (alpha[mask] - 1) / (alpha[mask] + beta_param[mask] - 2)

    return x_star.float()


def compute_beta_var_torch(alpha: torch.Tensor, beta_param: torch.Tensor):
    # x^2 / (4x^2 * (2x + 1)) = x^2 / (8x^3 + 4x^2)) = 1 / (8x + 4) => 0.25 as x -> 0
    total = alpha + beta_param
    return (alpha * beta_param) / (total.pow(2) * (total + 1))


def cumulative_prob_diff_single(
    b, alpha, beta_param, p, x_star: Optional[float] = None
):
    if x_star is None:
        x_star = compute_beta_mode_single(alpha, beta_param)
    # Calculate the bounds based on b
    lower_bound = max(1e-7, x_star - b)
    upper_bound = min(1 - 1e-7, x_star + b)

    # Compute the cumulative probability over the interval
    prob_diff = beta.cdf(upper_bound, alpha, beta_param) - beta.cdf(
        lower_bound, alpha, beta_param
    )

    # Return the difference from the desired probability p
    return prob_diff - p


def find_b_single(alpha, beta_param, p):
    x_star = compute_beta_mode_single(alpha, beta_param)
    b_max = max(x_star, 1 - x_star)

    # Solve the equation using brentq
    b_solution = brentq(
        cumulative_prob_diff_single, 1e-5, b_max, args=(alpha, beta_param, p, x_star)
    )

    return b_solution, x_star


def find_b_parallel(alpha, beta_param, p):
    # Parallelize the root finding for each pair of alpha and beta
    res = Parallel(n_jobs=-1)(
        delayed(find_b_single)(alpha[i], beta_param[i], p) for i in range(len(alpha))
    )
    res_arr = np.array(res)
    b_values = res_arr[:, 0]
    x_star = res_arr[:, 1]
    return np.array(b_values), np.array(x_star)


def compute_low_high_bound(alpha: np.ndarray, beta_param: np.ndarray, p: float):
    assert alpha.shape[0] == beta_param.shape[0]
    b, x_star = find_b_parallel(alpha, beta_param, p)
    low_bound = np.maximum(0, x_star - b)
    high_bound = np.minimum(1, x_star + b)
    return low_bound, high_bound, x_star, b


def fit_beta_distribution(
    x: torch.Tensor, left_bound, right_bound
) -> Tuple[float, float]:
    """Fit beta distribution to data
    Args:
            x (torch.Tensor): data
            left_bound (float): left bound of data
            right_bound (float): right bound of data
    """
    # transform to numpy
    x = torch.clamp(x, left_bound + 1e-8, right_bound - 1e-8)
    x_np = x.numpy()

    # Transform the data to fit the [0, 1] interval
    x_hat = (x_np - left_bound) / (right_bound - left_bound)

    # Fit data to a beta distribution
    params_transformed = beta_sp.fit(x_hat, floc=0, fscale=1)

    alpha, beta = params_transformed[0], params_transformed[1]
    return alpha, beta
