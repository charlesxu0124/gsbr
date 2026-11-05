from typing import Tuple
from scipy import stats


def calculate_beta_posterior_percentile(
    outcomes, percentile=0.95
) -> Tuple[float, float, float]:
    """
    Calculates a specified percentile of the posterior Beta distribution.

    Args:
    outcomes: A list of successes (1) and failures (0).
    percentile: The desired percentile (between 0 and 1).
                Defaults to 0.95.

    Returns: lower, mean, upper
    The sample mean.
    The upper and lower bounds of the percentile interval.
    """
    if not 0 <= percentile <= 1:
        raise ValueError("Percentile must be between 0 and 1.")

    # Count successes and failures
    successes = sum(outcomes)
    failures = len(outcomes) - successes

    # We assume a non-informative prior (Beta(1, 1)).
    alpha_prior = 1
    beta_prior = 1

    # Calculate the posterior parameters
    alpha_posterior = alpha_prior + successes
    beta_posterior = beta_prior + failures

    # Calculate the percentile from the posterior distribution
    # The ppf function is the inverse of the cdf (cumulative distribution function)
    # and is used to find the value corresponding to a given percentile.
    lower_percentile = (1 - percentile) / 2
    upper_percentile = 1 - lower_percentile
    lower = stats.beta.ppf(lower_percentile, alpha_posterior, beta_posterior)
    mean = sum(outcomes) / float(len(outcomes))
    upper = stats.beta.ppf(upper_percentile, alpha_posterior, beta_posterior)

    return lower, mean, upper
