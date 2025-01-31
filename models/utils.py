from jax.scipy.special import logsumexp
import numpyro
from jax import numpy as jnp
from numpyro import distributions as dist


def missingness_adjustment(log_rate, missing_idx, control_idx, residual_cat_mask_idx, missing_values, dist_type, model_treated=True, dispersion=None):
    """
    Adjusts for deterministic missingness by incorporating missing data probabilities.

    Parameters:
    -----------

    log_rate : jax.numpy.ndarray
        log Predicted values for all data points.
    missing_idx : jax.numpy.ndarray
        Boolean mask indicating missing data points.
    control_idx : jax.numpy.ndarray
        Boolean mask indicating control data points.
    missing_values : jax.numpy.ndarray
        Possible values for missing data points.

    Returns:
    --------
    None
    """
    print(log_rate.shape == residual_cat_mask_idx.shape)

    if dist_type == "Poisson":
        # Calculate log probabilities of Poisson distribution for missing values
        probs = dist.Poisson(jnp.exp(log_rate)[:, None]).log_prob(missing_values[None, :])
    elif dist_type == "NB":
        if dispersion is None:
            raise Exception("Negative Binomial requires a dispersion parameter")
        print(dispersion.shape)
        print(log_rate.shape)
        probs = dist.NegativeBinomial2(jnp.exp(log_rate)[:, None], dispersion.reshape(-1)[:, None]).log_prob(missing_values[None, :])
        
    # Sum log probabilities along axis 1 (summing over the missing values)
    print(probs.shape)
    log_probs_summed = logsumexp(probs, axis=1)

    # increment log likelihood to account for missingness
    missing_factor = numpyro.factor("missing_factors", log_probs_summed[missing_idx & control_idx & ~residual_cat_mask_idx].sum())
        
    # Increment log likelihood to account for non-missingness 
    nonmissing_factor = numpyro.factor("nonmissing_factors", jnp.log(1 - jnp.exp(log_probs_summed[~missing_idx & control_idx & ~residual_cat_mask_idx])).sum())
    