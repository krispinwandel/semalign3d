import torch


def log_normal_pdf(
    x: torch.Tensor, sigma: torch.Tensor, add_normalization: bool = True
) -> torch.Tensor:
    """
    Compute log-normal PDF for given x values and sigma.

    Args:
        x (torch.Tensor): Tensor of x values.
        sigma (float): Standard deviation of the underlying normal distribution.

    Returns:
        torch.Tensor: Log-normal PDF values for the input x.
    """
    res = -0.5 * (x / sigma) ** 2
    if add_normalization:
        res += -0.5 * torch.log(2.0 * torch.pi * torch.pow(sigma, 2))
    return res


def two_tail_pvalue_score(x: torch.Tensor, x_mean: torch.Tensor, x_var: torch.Tensor):
    """
    Compute two-tailed p-value score for x under a normal distribution.

    The p-value represents the probability of observing a value as extreme or more
    extreme than |x - x_mean| in either tail, assuming the distribution has mean=0
    and the given standard deviation. Higher scores indicate more typical values.

    NOTE: at x_hat=0 (x equals mean), this value is 1.

    Args:
        x: (D1,D2) D1=#samples, D2=#distributions (or swapped)
        x_mean: (D1,1) or (1,D2) depending on x
        x_var: (D1,1) or (1,D2) depending on x
    Returns:
        pvalue_score: (D1, D2) tensor
    """
    # Standard deviation is the square root of variance
    x_sd = torch.sqrt(x_var)
    x_sd[torch.isnan(x_sd)] = 1e-13
    x_sd[x_sd < 1e-9] = 1e-13

    # Create a normal distribution with mean=0 and the given standard deviation
    normal_dist = torch.distributions.Normal(0, x_sd)

    # Compute x_hat = |x - x_mean|
    x_hat = torch.abs(x - x_mean)
    x_hat_nan = torch.isnan(x_hat)
    x_hat[x_hat_nan] = 1e6

    # Compute the cumulative distribution function (CDF) for x_hat
    # Since CDF is symmetric around the mean, use the CDF of x_hat directly
    cdf_x_hat = normal_dist.cdf(x_hat)

    # The p-value score is 2 * (1 - CDF(x_hat)) for two-tailed test
    pvalue_score = 2 * (1 - cdf_x_hat)
    pvalue_score[x_hat_nan] = torch.nan

    return pvalue_score


def two_tail_pvalue_score_simple(
    x: torch.Tensor, x_mean: torch.Tensor, x_sd: torch.Tensor
):
    kpt_normal_distribution = torch.distributions.Normal(x_mean, x_sd)
    kpt_cdf = kpt_normal_distribution.cdf(x_mean + torch.abs(x - x_mean))
    kpt_likelihood = 2 * (1 - kpt_cdf)
    return kpt_likelihood
