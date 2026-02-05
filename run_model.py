#import os
#os.chdir("birthrate_mtgp")
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from jax import numpy as jnp
import numpy as np
import numpyro.distributions as dist
import jax.numpy as jnp
import numpyro
from numpyro.handlers import scope

from models.panel_nmf_model import model
from numpyro_to_draws_df_csv import dict_to_tidybayes
import sys

import pandas as pd

dist = "Poisson"
outcome_prefix = "pregrel_deaths_"
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
num_warmup = 1000
num_samples = 1000
end_date = '2024-01-01'
time_resolution="quarterly"
race_category = "total"
exclude_covid = True
start_date='2016-01-01'
placebo_time = "2020-04-01"
original_earliest_time = "2016-01-01"

def main(dist, time_resolution="monthly", outcome_prefix="mat_deaths_", denom_prefix='births_', 
         exclude_covid=exclude_covid,
         rank=5, 
         normalize_deaths=True, missingness=True, 
         disp_param=1e-4, sample_disp=False, placebo_state=None, placebo_time=None, 
         start_date='2016-01-01', end_date='2024-01-01', dobbs_donor_sensitivity=False, model_treated=False,
         num_chains=num_chains, num_warmup=1000, num_samples=1000, states_exclude=None,
         file_suffix = "", race_category="all"):
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
        start_date (str): Start date for data inclusion. Default is '2016-01-01'.
        end_date (str): End date for data inclusion. Default is '2024-01-01'.
        dobbs_donor_sensitivity (bool): Whether to adjust for Dobbs decision in donor pool. Default is False.
        model_treated (bool): Whether to model the treated units directly. Default is False.
        num_chains (int): Number of MCMC chains to run. Default is from global variable.
        num_warmup (int): Number of warmup samples for MCMC. Default is 1000.
        num_samples (int): Number of posterior samples to collect. Default is 1000.
        states_exclude (list): List of state names to exclude from the analysis. Default is None.
        file_suffix (str): Suffix to add to output filenames. Default is "".
        race_category (str): Race category to analyze. Options: "total" (default), "all" (joint model 
                            with all race categories), or a specific race category (e.g., "nhblack").
    
    Returns:
        None: Results are saved to CSV files in the 'results' directory.
    """
    
    numpyro.set_host_device_count(num_chains)
    
    df = pd.read_csv('data/maternalmort_data_20251022.csv')
    # Ensure that both DataFrames have the 'state', 'month', and 'year' columns
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    
    # Exclude specified states if provided
    if states_exclude is not None and len(states_exclude) > 0:
        df = df[~df['state'].isin(states_exclude)]
    
    from clean_birth_data import prep_data, clean_dataframe, create_unit_placebo_dataset, create_time_placebo_dataset
    
    # Process data based on race_category

    df = clean_dataframe(df, time_resolution, outcome_prefix=outcome_prefix, denom_prefix = "births_",
                         exclude_covid = exclude_covid,
                        race_category = race_category, csv_filename=None, end_date=end_date)

    
    if placebo_state is not None and placebo_state != "Texas":
        df = create_unit_placebo_dataset(df, placebo_state = placebo_state)
    print("-------------")
    print(placebo_time)
    print("-------------")
    if placebo_time is not None:
        df = create_time_placebo_dataset(df, time_resolution, new_treatment_start = placebo_time, original_earliest_time = start_date)
    else:
        # Filter data based on start_date
        df = df[df['date'] >= pd.to_datetime(start_date)]  

    # For "all" race categories, we need to handle joint modeling
    if race_category == "all":
        # Joint model implementation would go here
        # This might require more complex adjustments to the prep_data and model functions
        pass
        
    data_dict_cat = prep_data(df, outcome_prefix=outcome_prefix, denom_prefix=denom_prefix)

    print(data_dict_cat['Y'].shape)
    print(data_dict_cat['denominators'].shape)
    print(data_dict_cat['control_idx_array'].shape)
    print((~data_dict_cat['control_idx_array']).sum())

    if(not normalize_deaths):
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
        progress_bar=False
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

    
    if file_suffix == "":
        # If no file suffix is provided, use the default naming convention
        file_suffix = "_with_covid" if not exclude_covid else ""
    
    if placebo_state is not None:
        file_suffix += f"_placebo_state_{placebo_state}"
    
    if placebo_time is not None:
        file_suffix += f"_placebo_time_{placebo_time}"

    if states_exclude is not None and len(states_exclude) > 0 and states_exclude != ['']:
        file_suffix += f"_no_{'_'.join(states_exclude)}"

    if not normalize_deaths:
        file_suffix += "_unnormalized"
    else:
        file_suffix += "_normalized"

    # Update the filename to include race_category
    race_suffix = f"_{race_category}" if race_category != "total" else ""
    df.to_csv(f'results/df_{outcome_prefix}{time_resolution}{race_suffix}{file_suffix}.csv')

    results_df.to_csv(
        f'results/{dist}_{outcome_prefix}{rank}_{time_resolution}{race_suffix}{file_suffix}.csv'
    )

    
if __name__ == '__main__':
    #from clean_birth_data import subgroup_definitions
    # for cat in subgroup_definitions.keys():
    #     for rank in range(3, 9):
    #         main(cat, rank)
                
    from joblib import Parallel, delayed

    inputs = [1, 2, 3,]
    outcome_prefixes = ["pregrel_deaths_","preg_deaths_", "mat_deaths_"]
    denom_prefix = "births_"
    dists = ['Poisson'] # Poisson or NB
    ## dists = ['NB'] # Poisson or NB
    missing_flags = [False]
    # disp_params = [1e-4, 1e-3]
    disp_params = [1e-4]
    placebo_times = [None]
    # placebo_times = ["2020-04-01"]  
    placebo_states = [None]
    sample_disp = False
    exclude_covid= True
    #exclude_covid = False
    time_resolution = "quarterly"
    start_date = '2016-01-01'
    end_date = '2024-01-01'
    dobbs_donor_sensitivity = False
    normalization_types = [True, False]
    states_exclude = ['']
    #states_exclude = ['']
    # if states_exclude != ['']:
    #     file_suffix = "_no_{states_exclude}".format(states_exclude = "_".join(states_exclude)) if len(states_exclude) > 0 else ""
    # else:
    #     file_suffix = ""
    # if not exclude_covid:
    #     file_suffix += "_with_covid"
    
    #race_categories = ["total"]  # Options: "total", "all", or specific race categories like "nhblack", "nhwhite", etc.
    race_categories = ["all"]

    args = [(dist, rank, m, disp, p, tm, rc, op, norm) for dist in dists for rank in inputs 
            for m in missing_flags for disp in disp_params for p in placebo_states 
            for tm in placebo_times for rc in race_categories for op in outcome_prefixes
            for norm in normalization_types]
    # Run the function in parallel
    results = Parallel(n_jobs=100)(delayed(main)(dist=i[0], 
                                                 time_resolution = time_resolution,
                                                 outcome_prefix=i[7],
                                                 denom_prefix=denom_prefix,
                                                 exclude_covid=exclude_covid,
                                                 rank=i[1], normalize_deaths=i[8],
                                                 missingness=i[2], 
                                                 disp_param=i[3],
                                                 sample_disp=sample_disp, placebo_state=i[4], placebo_time=i[5], 
                                                 start_date=start_date, end_date=end_date, 
                                                 dobbs_donor_sensitivity=dobbs_donor_sensitivity, 
                                                 model_treated=True, num_chains=4, num_samples=250, num_warmup=1000,
                                                 states_exclude=states_exclude,
                                                 race_category=i[6]) for i in args)
