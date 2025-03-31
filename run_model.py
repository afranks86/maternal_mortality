#import os
#os.chdir("birthrate_mtgp")
from jax import numpy as jnp
import numpy as np
import numpyro.distributions as dist
import jax.numpy as jnp
import numpyro
from numpyro.handlers import scope

from models.panel_nmf_model import model
from numpyro_to_draws_df_csv import dict_to_tidybayes

import pandas as pd

dist = "Poisson"
outcome_prefix = "preg_deaths_"
denom_prefix = "births_"
rank = 5
sample_disp = False
missingness=True
disp_param = 1e-4
model_treated = True
dobbs_donor_sensitivity = False
placebo_time = None
placebo_state = None
num_chains = 1
end_date = '2024-01-01'
time_resolution="quarterly"
def main(dist, time_resolution="monthly", outcome_prefix="mat_deaths_", denom_prefix='births_', rank=5, normalize_deaths=True, missingness=True, 
         disp_param=1e-4, sample_disp=False, placebo_state=None, placebo_time=None, 
         end_date='2024-01-01', dobbs_donor_sensitivity=False, model_treated=False,
         num_chains=num_chains, num_warmup=1000, num_samples=1000):
    """
    Runs the maternal mortality analysis model with specified parameters and outputs results.
    
    This function loads data, processes it according to specified parameters, runs a Bayesian
    model using NumPyro, and saves the results to CSV files.

    Exposure notes:
    - Use exposed_dm for mat_deaths and pregearly
    - Use exposed_dpa for preg_deaths   
    - Use exposed_dpal for preglate_deaths
    
    Args:
        dist (str): Distribution to use for the outcome model ("Poisson" or "NB").
        outcome_prefix (str): Prefix for outcome columns in the dataset. Default is "mat_deaths_".
        denom_prefix (str): Prefix for denominator columns in the dataset. Default is "births_".
        rank (int): Rank parameter for the matrix factorization model. Default is 5.
        normalize_deaths (bool): Whether to normalize deaths by denominators. Default is True.
        missingness (bool): Whether to adjust for missingness in the model. Default is True.
        disp_param (float): Dispersion parameter for negative binomial distribution. Default is 1e-4.
        sample_disp (bool): Whether to sample the dispersion parameter. Default is False.
        placebo_state (str): State to use for placebo analysis. Default is None.
        placebo_time (str): Time to use for placebo analysis. Default is None.
        end_date (str): End date for data inclusion. Default is '2024-01-01'.
        dobbs_donor_sensitivity (bool): Whether to adjust for Dobbs decision in donor pool. Default is False.
        model_treated (bool): Whether to model the treated units directly. Default is False.
        num_chains (int): Number of MCMC chains to run. Default is from global variable.
        num_warmup (int): Number of warmup samples for MCMC. Default is 1000.
        num_samples (int): Number of posterior samples to collect. Default is 1000.
    
    Returns:
        None: Results are saved to CSV files in the 'results' directory.
    """
    
    numpyro.set_host_device_count(num_chains)
    
    df = pd.read_csv('data/maternalmort_data_20250226.csv')
    # Ensure that both DataFrames have the 'state', 'month', and 'year' columns
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    
    from clean_birth_data import prep_data, clean_dataframe, create_unit_placebo_dataset, create_time_placebo_dataset
    
    df = clean_dataframe(df, time_resolution, outcome_prefix=outcome_prefix, denom_prefix=denom_prefix,
                         csv_filename=None, 
                         end_date=end_date, 
                         dobbs_donor_sensitivity=dobbs_donor_sensitivity)
    
    if placebo_state is not None and placebo_state != "Texas":
        df = create_unit_placebo_dataset(df, placebo_state = placebo_state)
    
    if placebo_time is not None:
        df = create_time_placebo_dataset(df, new_treatment_start = placebo_time, original_earliest_time = "2012-01-01")
    else:
        # Only use data from 2012 onwards if not using a placebo time #
        df = df[df['date'] >= pd.to_datetime('2012-01-01')]  

    data_dict_cat = prep_data(df, outcome_prefix=outcome_prefix, denom_prefix=denom_prefix)

    if(~normalize_deaths):
        data_dict_cat['denominators'] = np.ones(data_dict_cat['denominators'].shape)
    
    from jax import random
    from numpyro.infer import MCMC, NUTS, Predictive
    from statsmodels.tsa.deterministic import CalendarFourier

    # set the random seed
    rng_key = random.PRNGKey(8675309)
    # split the random key
    rng_key, rng_key_ = random.split(rng_key)
    # Setup the sampler
    kernel = NUTS(model)

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True
    )

    mcmc.run(
        rng_key_,
        y=data_dict_cat['Y'],
        denominators=data_dict_cat['denominators'],
        control_idx_array=data_dict_cat['control_idx_array'],
        missing_idx_array=data_dict_cat['missing_idx_array'],
        rank=rank,
        outcome_dist=dist,
        adjust_for_missingness=missingness,
        nb_disp = disp_param,
        sample_disp = sample_disp,
        model_treated = model_treated
    )

    samples = mcmc.get_samples(group_by_chain=True)
    predictive = Predictive(model, mcmc.get_samples(group_by_chain=False))
    rng_key, rng_key_ = random.split(rng_key)

    predictions = predictive(
        rng_key_,
        denominators=data_dict_cat['denominators'],
        control_idx_array=None, #data_dict_cat['control_idx_array'],
        missing_idx_array=None, #data_dict_cat['missing_idx_array'],
        rank=rank,
        outcome_dist=dist,
        nb_disp = disp_param,
        sample_disp = sample_disp,
        model_treated = False
    )['y_obs']
    K, D, N = data_dict_cat['denominators'].shape
    pred_mat = predictions.reshape(mcmc.num_chains, mcmc.num_samples, K, D, N)
   
    ## Take Python output and convert to draws matrix form
    params = dict_to_tidybayes({'mu': samples['mu_ctrl'], 'te': samples['te'], 'disp' : samples['disp'], 'state_te' : samples['state_treatment_effect'], 'category_te' : samples['category_treatment_effect'], 'unit_weights' : samples['unit_weight'], 'latent_factors' : samples['time_fac']})
    preds = dict_to_tidybayes({"ypred" : pred_mat})

    preds[".chain"] = params[".chain"]
    preds[".draw"] = params[".draw"]

    all_samples = params.merge(preds, left_on = ['.draw', '.chain'], right_on = ['.draw', '.chain'])
    results_df = pd.DataFrame(all_samples)

    df.to_csv('results/df_{}{}.csv'.format(outcome_prefix, time_resolution))

    results_df.to_csv(
        'results/{}_{}{}_{}.csv'.format(dist, outcome_prefix, rank, time_resolution)
    )

    
if __name__ == '__main__':
    #from clean_birth_data import subgroup_definitions
    # for cat in subgroup_definitions.keys():
    #     for rank in range(3, 9):
    #         main(cat, rank)
                
    from joblib import Parallel, delayed

    # Define the inputs for the function
    inputs = [1, 2, 3, 4, 5, 6]
    # inputs = [2, 3]
    #inputs = [7, 8, 10]
    # inputs = [7]
    # outcome_prefix = "mat_deaths_"
    outcome_prefix = "preg_deaths_"
    denom_prefix = "births_"
    dists = ['Poisson'] # Poisson or NB
    ## dists = ['NB'] # Poisson or NB
    missing_flags = [False]
    # disp_params = [1e-4, 1e-3]
    disp_params = [1e-4]
    placebo_times = [None]
    placebo_states = [None]
    sample_disp = False
    end_date = '2024-01-01'
    dobbs_donor_sensitivity = False
    normalize_deaths = True

    args = [(dist, rank, m, disp, p, tm) for dist in dists for rank in inputs 
            for m in missing_flags for disp in disp_params for p in placebo_states 
            for tm in placebo_times]
    # Run the function in parallel
    results = Parallel(n_jobs=100)(delayed(main)(dist=i[0], 
                                                 time_resolution = "biannual",
                                                 outcome_prefix=outcome_prefix,
                                                 denom_prefix=denom_prefix,
                                                rank=i[1], normalize_deaths=normalize_deaths,
                                                missingness=i[2], 
                                                disp_param=i[3],
                                                sample_disp=sample_disp, placebo_state=i[4], placebo_time = i[5], 
                                                end_date=end_date, dobbs_donor_sensitivity=dobbs_donor_sensitivity, 
                                                model_treated=True, num_chains=4, num_samples=250, num_warmup=1000) for i in args)
