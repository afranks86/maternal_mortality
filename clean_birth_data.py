import numpy as np
import pandas as pd
import re


deaths_subgroup_definitions = {
        'race': ("nhwhite", "hispanic", "nhblack", "nhother"),
        'total': ("total",),
    }

deaths_residual_category_definitions = {
    'race'  : "nhother",
    'neonatal' : "nonneo",
    'congenital' : "noncon",
    'total' : None
}

def clean_dataframe(dat: pd.DataFrame, 
                    time_resolution: str,
                    outcome_prefix="mat_deaths_", denom_prefix="births_",
                    csv_filename='data/dat_quarterly.csv', end_date='2024-01-1',
                    race_category="total"):
    """
    Filters, imputes, and adds relevant columns to the dataframe
    
    Args:
        dat (pandas.DataFrame): Input data frame
        time_resolution (str): Time resolution for resampling ('monthly', 'bimonthly', 'quarterly', 'biannual')
        outcome_prefix (str): Prefix for outcome columns in the dataset
        denom_prefix (str): Prefix for denominator columns in the dataset
        csv_filename (str, optional): Filename to save output CSV. If None, no file is saved
        end_date (str): End date for data inclusion
        race_category (str): Race category to analyze ("total", "all", or specific race category)
        
    Returns:
        pandas.DataFrame: Processed dataframe
    """

    dat.rename(columns = {
        'births': 'births_total' 
    }, inplace=True)

    # Create mapping patterns for column renaming
    prefix_mapping = {
        'death_mat_nocovid': 'mat_deaths_total',
        'death_mat_all_nocovid': 'all_mat_deaths_total',
        'death_preg_nocovid': 'preg_deaths_total',
        'death_preglate_nocovid': 'late_preg_deaths_total',
        'death_pregearly_nocovid': 'early_preg_deaths_total',
        'death_wra_nocovid': 'wra_deaths_total',
    }
    
    # Generate race-specific mappings for each death category
    race_categories = {'_nhwhite': '_nhwhite', '_hispanic': '_hispanic', 
                     '_nhblack': '_nhblack', '_nhother': '_nhother'}
    
    rename_dict = {}
    # Create full mapping dictionary
    for old_prefix, new_prefix in prefix_mapping.items():
        for old_suffix, new_suffix in race_categories.items():
            if old_suffix == '':  # Handle the 'total' case which doesn't have a suffix in the original
                rename_dict[old_prefix] = new_prefix
            else:
                rename_dict[f"{old_prefix}{old_suffix}"] = f"{new_prefix.replace('_total', '')}{new_suffix}"
    
    dat.rename(columns=rename_dict, inplace=True)
    
    if outcome_prefix == "preg_deaths_":
        dat.rename(columns = {
            'exposed_dpa' : "exposed"
        }, inplace=True)
    elif outcome_prefix == "mat_deaths_" or outcome_prefix == "early_preg_deaths_":
        dat.rename(columns = {
            'exposed_dm' : "exposed"
        }, inplace=True)
    elif outcome_prefix == "late_preg_deaths_":
        dat.rename(columns = {
            'exposed_dpal' : "exposed"
        }, inplace=True)
    elif outcome_prefix == "preg_no_mat_deaths_":
        dat.rename(columns = {
            'exposed_dpa' : "exposed"
        }, inplace=True)
    
    # Handle race-specific prefixes
    elif any(rc in outcome_prefix for rc in deaths_subgroup_definitions['race']):
        # Extract the base prefix before the race category
        base_prefix = outcome_prefix.split("_")[0] + "_deaths_"
        if base_prefix == "preg_deaths_":
            dat.rename(columns = {
                'exposed_dpa' : "exposed"
            }, inplace=True)
        elif base_prefix == "mat_deaths_" or base_prefix == "early_preg_deaths_":
            dat.rename(columns = {
                'exposed_dm' : "exposed"
            }, inplace=True)
        elif base_prefix == "late_preg_deaths_":
            dat.rename(columns = {
                'exposed_dpal' : "exposed"
            }, inplace=True)
        elif base_prefix == "preg_no_mat_deaths_":
            dat.rename(columns = {
                'exposed_dpa' : "exposed"
            }, inplace=True)
        else:
            raise ValueError(f"Invalid outcome prefix: {outcome_prefix}")
    else:
        raise ValueError(f"Invalid outcome prefix: {outcome_prefix}")

    # Set time variable
    dat['date'] = pd.to_datetime(dat['date'])

    # Filter the DataFrame to keep rows with time before end_date
    dat = dat[dat['date'] < pd.to_datetime(end_date)]

    def fill_in_missing_denoms(dat):
        # Get a list of column names containing the string "pop"
        cols_with_pop = dat.filter(regex=r'pop').columns
        # Iterate over each column containing "pop"
        for col in cols_with_pop:
            # Find the row index of the maximum value for the current column in 2022
            pop_index_2022 = dat.loc[dat['year'] == 2022, col].idxmax()
            # Find the row index of the maximum value for the current column in 2021
            pop_index_2021 = dat.loc[dat['year'] == 2021, col].idxmax()
            
            # Impute missing values in the current column
            # For rows with missing values, replace with the squared value of the 2022 row divided by the 2021 row
            dat.loc[dat[col].isna(), col] = (dat.loc[pop_index_2022, col] ** 2) / dat.loc[pop_index_2021, col]
        
        # Return the modified DataFrame
        return dat

    # Hacky imputation
    # All of 2023 currently has a population of `NA`
    # We'll use a linear imputation of (pop 2022) * (pop 2022) / (pop 2021)
    # Well do so by:
    ## Group the original DataFrame by 'state'
    ## Apply the fill_in_missing_denoms function to each group
    ## Ungroup the result and reset the index
    dat = dat.groupby('state').apply(fill_in_missing_denoms).reset_index(drop=True)

    # Define the aggregation functions for each column
    agg_dict = {col: 'sum' for col in dat.columns if col.startswith(outcome_prefix) or col.startswith(denom_prefix)}
    agg_dict.update({'year': 'first', 'exposed': 'max', 
                     'banned_state': 'max', 'pop_total': 'first', 'births_total': 'sum'})
    for col in dat.columns:
        if 'deaths' in col:
            agg_dict[col] = 'sum'

    # Resample data based on the specified time resolution
    if time_resolution == "monthly":
        dat = dat.groupby('state').resample('MS', on='date').agg(agg_dict).reset_index(['state', 'date'])
    elif time_resolution == "bimonthly":
        dat = dat.groupby('state').resample('2MS', on='date').agg(agg_dict).reset_index(['state', 'date'])
    elif time_resolution == "quarterly":
        dat = dat.groupby('state').resample('QS', on='date').agg(agg_dict).reset_index(['state', 'date'])
    elif time_resolution == "biannual":
        dat = dat.groupby('state').resample('2QS', on='date').agg(agg_dict).reset_index(['state', 'date'])
    else:
        raise ValueError("Invalid time resolution. Choose from 'monthly', 'bimonthly', 'quarterly', or 'biannual'.")

    dat = dat.sort_values(['state', 'date'])

    # Select outcome columns based on race category
    if race_category == "all":
        outcome_columns = []
        for race in deaths_subgroup_definitions['race']:
            # Look for columns that include the race category in their name
            race_specific_columns = [col for col in dat.columns if col.startswith(outcome_prefix) and f"_{race}" in col]
            outcome_columns.extend(race_specific_columns)
    else:
        # For specific race category, select columns that match the prefix and contain the race category
        if race_category == "total":
            outcome_columns = [col for col in dat.columns if col.startswith(outcome_prefix) and col.endswith("_total")]
        else:
            outcome_columns = [col for col in dat.columns if col.startswith(outcome_prefix) and f"_{race_category}" in col]

    # Define the other columns you want to select
    other_columns = ['state', 'year', 'date', 'banned_state', 'pop_total', 'exposed']
    
    # Add the appropriate births column based on race_category
    if race_category == "total":
        births_column = "births_total"
        other_columns.append(births_column)
    elif race_category in deaths_subgroup_definitions['race']:
        births_column = f"births_{race_category}"
        other_columns.append(births_column)
    elif race_category == "all":
        # Include birth columns for all race categories
        for race in deaths_subgroup_definitions['race']:
            births_column = f"births_{race}"
            if births_column in dat.columns:
                other_columns.append(births_column)
    else:
        births_column = "births_total"  # Default fallback
        other_columns.append(births_column)

    # Combine the two lists and select from df
    dat = dat[other_columns + outcome_columns]

    if csv_filename is not None:
        ## Save to csv so we don't have to do this every time 
        dat.to_csv(csv_filename)
    return dat


def prep_data(dat, group=None, outcome_prefix="deaths_", denom_prefix="births_", variables=None, covariates=None):
    """
    Prepare data for analysis by creating DataFrames for births or deaths (numerators), population or births (denominators), control indices, and missing indices.

    Args:
        dat (pandas.DataFrame): Input data containing information about births, population, and other relevant variables.
        group (str, optional): Group name to use from subgroup_definitions. Default is None.
        outcome_prefix (str): Prefix for outcome columns. Default is "deaths_".
        denom_prefix (str): Prefix for denominator columns. Default is "births_".
        variables (list, optional): List of variable names. Default is None.
        covariates (list, optional): List of covariate names to include in the analysis. Default is None.

    Returns:
        dict: A dictionary containing prepared data for analysis.
    """
        
    ## Get outcomes and denoms
    outcome_columns = [col for col in dat.columns if col.startswith(outcome_prefix)]
    denom_columns = [col for col in dat.columns if col.startswith(denom_prefix)]
    
    ## Make sure suffixes match
    outcome_columns.sort()
    denom_columns.sort()

    # Rest of the function remains largely the same
    # Create an outcomes DataFrame
    Y = (
        dat[["state", "date"] + outcome_columns]
        .melt(id_vars=["state", "date"], value_vars=outcome_columns, var_name="category", value_name="outcome")
        .pivot_table(index=["category"], columns=["state", "date"], values="outcome", aggfunc="sum", fill_value=0)
    )

    denominators = (
        dat[["state", "date"] + denom_columns]
        .melt(id_vars=["state", "date"], value_vars=denom_columns, var_name="category", value_name="denominator")
        .pivot_table(index=["category"], columns=["state", "date"], values="denominator", aggfunc="sum", fill_value=0)
    ) * 10000

    num_states = len(dat.state.unique())
    total_length = denominators.shape[1]
    denominators = denominators.values.reshape((len(denom_columns), num_states, denominators.shape[1]//num_states))
    Y = Y.values.reshape((len(outcome_columns), num_states, total_length//num_states))

    control_idx_array = (
        dat[["state", "date", "exposed"] + outcome_columns]
        .melt(id_vars=["state", "date", "exposed"], value_vars=outcome_columns, var_name="category", value_name="outcome")
        .assign(ctrl_index=(lambda x: x["exposed"] == 0))
        .pivot_table(index=["category"], columns=["state", "date"], values="ctrl_index", aggfunc="sum", fill_value=0)
    ).astype(np.bool_)
    
    control_idx_array = control_idx_array.values.reshape((len(outcome_columns), num_states, total_length//num_states))

    # Create a missing index array DataFrame
    missing_idx_array = (
        dat[["state", "date", "exposed"] + outcome_columns]
        .melt(id_vars=["state", "date", "exposed"], value_vars=outcome_columns, var_name="category", value_name="outcome")
        .assign(missing_index=lambda x: x["outcome"].isna().astype(int))
        .pivot_table(index=["category"], columns=["state", "date"], values="missing_index", aggfunc="sum", fill_value=0)
    ).astype(np.bool_)
    
    missing_idx_array = missing_idx_array.values.reshape((len(outcome_columns), num_states, total_length//num_states))

    # If covariates are provided, calculate the covariates matrix
    if covariates is not None:
        D_cov = dat.groupby("state")[covariates].mean().reset_index()[covariates].values
        D_cov[np.isnan(D_cov)] = D_cov[~np.isnan(D_cov)].mean()
    else:
        D_cov = None
    
    # Return a dictionary with the calculated values
    return {
        "Y": Y,
        "denominators": denominators,
        "control_idx_array": control_idx_array,
        "missing_idx_array": missing_idx_array, 
        "variables": variables,
        "D_cov": D_cov
    }

def create_unit_placebo_dataset(df, treated_state = "Texas", placebo_state = "California"):
    """
    Create a placebo dataset for by giving Texas' treatment times to `placebo_state` state by removing the state 
    and removing Texas.

    Args:
        df (pandas.DataFrame): Input data containing information about births, population, exposure codes and other relevant variables.
        placebo_state (str): Name of the placebo state to remove from the dataset.

    Returns:
        pandas.DataFrame: Placebo dataset with the specified placebo state with treatment times of treated_state.
    """
    
    print("Creating unit-placebo dataset for {}".format(placebo_state))

    # Get the columns that start with 'exposed' (exposed_births/exposed_deaths)
    exposure_columns = df.filter(regex='^exposure').columns

    # Get the values from the rows where 'state' equals 'treated_state'
    treated_values = df.loc[df['state'] == treated_state, exposure_columns]

    # Set the values in the rows where 'state' equals 'placebo_state' to the values from the 'treated_state' rows
    df.loc[df['state'] == placebo_state, exposure_columns] = treated_values.values
    
    # Filter the DataFrame to keep rows where 'state' is not equal to the treated state
    return df[df["state"] != treated_state]

def create_time_placebo_dataset(df, new_treatment_start="2022-05-01", original_earliest_time = "2012-01-01"):
    """
    Create a time placebo dataset by shifting treatment times early and capping the end date.
    
    Args:
        df (pandas.DataFrame): Input data containing information about births, population, exposure codes and other relevant variables.
        first_treatment_start (str): Start time of the first treated unit (usually Texas). Set to "2022-05-01" by default which is the actual SB8 time.

    Returns:
        pandas.DataFrame: Modified dataset with shifted time variables.
    """
    
    print("Creating placebo-in-time dataset starting in {}".format(new_treatment_start))
    
    def round_date_to_nearest_half_year(ts: pd.Timestamp) -> pd.Timestamp:
        if 4 <= ts.month <=8:
            return pd.Timestamp(ts.year, 7, 1)
        elif ts.month >=9:
            return pd.Timestamp(ts.year+1, 1, 1)
        elif ts.month <= 3:
            return pd.Timestamp(ts.year, 1, 1)
        else:
            raise Exception("Logic error.")

    # Convert 'first_treatment_start' to datetime
        
    new_treatment_start = pd.to_datetime(new_treatment_start)
    original_treatment_start = df.loc[df['exposure_code'] == 1, 'time'].min()

    end_date = df.loc[df['exposure_code'] == 1, 'time'].max()
    
    new_end = new_treatment_start + (end_date - original_treatment_start)
    
    original_time_length = end_date - pd.to_datetime(original_earliest_time)
    new_start = (new_end - original_time_length)
    if new_start < df["time"].min():
        new_start = df["time"].min()
    
    new_start = round_date_to_nearest_half_year(new_start)
    new_end = round_date_to_nearest_half_year(new_end)
    
    new_time_length = new_end - new_start

    # Get the columns that start with 'exposure_code'
    exposure_code_values = df.loc[(df['time'] >= end_date - new_time_length), ['exposure_code']]

    df = df[(df['time'] <= new_end) & (df['time'] >= new_start)]

    if len(exposure_code_values) == len(df):
        df.loc[:, "exposure_code"] = exposure_code_values.values
    else:
        raise ValueError("The length of new exposure_code values does not match the number of rows in df")

    return df

if __name__ == '__main__':
    import argparse
    import pandas as pd
    import pickle 
    import os
    import gzip 
    
    parser = argparse.ArgumentParser(description='Script for cleaning and parsing quarterly births. We will assume that the file name is in the format quarterly_fertility_mortality_MMDDYY.csv where MM is month,DD is day, and YY is year.')
    parser.add_argument('filename', type=str, help='Name of the input file')
    parser.add_argument('--save-dict', action='store_true', help='Create and save the dictionary used for model training as a pkl file')
    parser.add_argument("--group", help="Subgroups to create a dictionary for", default="total")

    args = parser.parse_args()
    # Create a directory to hold the parsed files
    directory_name = 'model_data/' + args.filename.split('/')[-1].split('.')[0]
    os.makedirs(directory_name, exist_ok=True)
    dat = clean_dataframe(pd.read_csv(args.filename))
    # Save the dataframe
    dat.to_csv(directory_name + '/data_frame.csv')

    if args.save_dict:
        data_dict = prep_data(dat, subgroup_definitions[outcome_type][args.group])
        # Specify the filename for the gzipped pickle file
        dict_filename = 'births_{}_dict.pkl.gz'.format(args.group)

        # Open the file in binary write mode using gzip
        with gzip.open('{}/{}'.format(directory_name, dict_filename), 'wb') as f:
            # Use pickle to dump the data dictionary into the file
            pickle.dump(data_dict, f)