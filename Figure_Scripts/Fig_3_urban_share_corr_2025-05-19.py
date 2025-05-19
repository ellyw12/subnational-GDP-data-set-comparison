''' Update 2025-04-18: using C2022 and WS2022 0.001 degree aggregations '''
'''Modified paths to repo paths and edited line 230 to fix a dictionary reading error (BW 2025-05-19)'''
###Note - this version does not include the error bars.###

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

""" 1. Load and merge data from different sets """

# Define paths
data_path               =   './Data/'
graphics_path           =   './Figures/'
deflator_path           =    data_path +'deflator/'
pickle_path             =    data_path +'pickle/'
WS_path                =    data_path +'modelled_data/'

# Define files
dose_v2   = 'DOSE_V2.10.csv'
pkl_files = ['C2022_data.pkl', 'K2025_data.pkl',
             'Z2024_data.pkl']
WS2022 = 'GRP_WangSun_aggregated(new).csv'

data = pd.read_csv(data_path+dose_v2)
WS_data = pd.read_csv(WS_path+WS2022)

# Load each .pkl file and merge it with the data dataframe
for pkl_file in pkl_files:
    file_path = os.path.join(pickle_path, pkl_file)
    with open(file_path, 'rb') as file:
        pkl_data = pickle.load(file)
        # Drop unnecessary columns before merging
        pkl_data = pkl_data.drop(columns=['GID_0'], errors='ignore')
        data = pd.merge(data, pkl_data, on=['GID_1', 'year'], how='outer')

# print(data.columns)

data = data.drop(
    columns=data.filter(regex='^(ag_|man_|serv_)').columns.tolist()
    + ['cpi_2015', 'deflator_2015', 'T_a', 'P_a'], errors='ignore')

data['WS2022'] = pd.merge(data, WS_data, on=['GID_1', 'year'], how='left')['grp']

""" 2. Prepare GRP conversion and deflation for comparison """

# Load World Bank GDP deflator data
deflators = pd.read_excel(deflator_path+'2022_03_30_WorldBank_gdp_deflator.xlsx', sheet_name='Data',
                            index_col=None, usecols='B,E:BM', skiprows=3).set_index('Country Code')
deflators = deflators.dropna(axis=0, how='all')

# Create two dataframes from the deflators, with 2015 and 2005 as ref years
deflators_2017 = deflators.copy()
column2017 = deflators_2017.columns.get_loc(2017)
for i in range(len(deflators_2017)):
    deflators_2017.iloc[i, :] = (deflators_2017.iloc[i, :] / deflators_2017.iloc[i, column2017]) * 100

deflators_2005 = deflators.copy()
column2005 = deflators_2005.columns.get_loc(2005)
for i in range(len(deflators_2005)):
    deflators_2005.iloc[i, :] = (deflators_2005.iloc[i, :] / deflators_2005.iloc[i, column2005]) * 100

# Make separate dataframes for the US and local GDP deflators, both stacked
deflators_2017_us = pd.DataFrame(deflators_2017.loc[deflators_2017.index=='USA'].stack()
                                 ).rename(columns={0:'deflator_2017_us'}
                                          ).reset_index().set_index('level_1')
deflators_2017 = pd.DataFrame(deflators_2017.stack()).rename(columns={0:'deflator_2017'})

deflators_2005_us = pd.DataFrame(deflators_2005.loc[deflators_2005.index=='USA'].stack()
                                 ).rename(columns={0:'deflator_2005_us'}
                                          ).reset_index().set_index('level_1')
deflators_2005 = pd.DataFrame(deflators_2005.stack()).rename(columns={0:'deflator_2005'})

# Add the deflators to the 'data' dataframe
data['deflator_2017'] = np.nan
data['deflator_2017_us'] = np.nan
data['deflator_2005'] = np.nan
data['deflator_2005_us'] = np.nan

# print(data.columns)

data = data.set_index(['GID_0','year'])
data.update(deflators_2017)
data.update(deflators_2005)
data = data.reset_index('GID_0')
data.update(deflators_2017_us)
data.update(deflators_2005_us)

# Load PPP conversion factors for both relevant years ...
# ... and add both sets of factors for each country to 'data'
ppp_data = pd.read_excel(deflator_path+'ppp_data_all_countries.xlsx')
ppp_data_2005 = ppp_data.loc[ppp_data.year==2005]
ppp_data_2017 = ppp_data.loc[ppp_data.year==2017]
d2005 = dict(zip(ppp_data_2005.iso_3, ppp_data_2005.PPP))
d2017 = dict(zip(ppp_data_2017.iso_3, ppp_data_2017.PPP))
data['ppp_2005'] = data['GID_0'].apply(lambda x: d2005.get(x))
data['ppp_2017'] = data['GID_0'].apply(lambda x: d2017.get(x))
data.loc[data.GID_0 == 'USA', 'ppp_2005'] = 1
data.loc[data.GID_0 == 'USA', 'ppp_2017'] = 1
data['ppp_2005'] = pd.to_numeric(data['ppp_2005'], errors='coerce')
data['ppp_2017'] = pd.to_numeric(data['ppp_2017'], errors='coerce')

#Create PPP column and print head
data['PPP'] = pd.to_numeric(data.PPP, errors='coerce')
# data.sample(60)

# Load MER conversion factors for 2015 and add them for each country to 'data'
fx_data = pd.read_excel(deflator_path+'fx_data_all_countries.xlsx')
fx_data = fx_data.loc[fx_data.year==2005]
d = dict(zip(fx_data.iso_3, fx_data.fx))
data['fx_2005'] = data['GID_0'].apply(lambda x: d.get(x))
data.loc[data.GID_0=='USA','fx_2005'] = len(data.loc[data.GID_0=='USA']) * [1]

""" 3. Perform GRP pc conversion and deflation calculations (restricted by population data in DOSE)"""

# DOSE (current lcu) to current PPP, 2017 LCU, 2005 PPP, and 2017 PPP
data['grp_pc_ppp'] = data['grp_pc_lcu'] / data.PPP
# grp_pc_usd already exists in DOSE
data['grp_pc_lcu_2017'] = data['grp_pc_lcu'] * 100 / data['deflator_2017']
data['grp_pc_ppp_2005'] = data['grp_pc_ppp'] * 100 / data['deflator_2005_us']
data['grp_pc_ppp_2017'] = data['grp_pc_ppp'] * 100 / data['deflator_2017_us']

# Z2024 (current usd) to current PPP and 2017 LCU
data['Z2024_grp_pc_ppp'] = data['Z2024_pc'] * data.fx / data.PPP
data['Z2024_grp_pc_lcu_2017'] = data['Z2024_pc'] * data.fx * 100 / data['deflator_2017']

# K2025 (2017 int USD) to 2017 LCU and current PPP
data['K2025_grp_pc_lcu_2017'] = data['K2025_pc'] * data['ppp_2017']
data['K2025_grp_pc_ppp'] = data['K2025_grp_pc_lcu_2017'] / 100 * data['deflator_2017'] / data.PPP

# C2022 (2017 int USD) to current PPP
data['C2022_grp_pc_ppp'] = data['C2022_pc'] / 100 * data['deflator_2017_us']

# W&S in 2005 ppp to current PPP
data['WS2022_grp_pc_ppp'] = data['WS2022'] / data['pop'] / 100 * data['deflator_2005_us']

urban                   =   pd.read_csv(data_path+'urban_areas.csv')
urban = urban.rename(columns={'var': 'urban_share'})

# Merge urban data with data
urban_data = data.merge(urban[['GID_1', 'year', 'urban_share']], on=['GID_1', 'year'], how='left')

from scipy.stats import pearsonr

# update 2025-05-19: bootstrap the correlation values

# List of target variables to analyze
target_variables = ['K2025_grp_pc_lcu_2017', 'Z2024_pc', 'WS2022_grp_pc_ppp', 'C2022_grp_pc_ppp']
dose_variables = ['grp_pc_lcu_2017', 'grp_pc_usd', 'grp_pc_ppp', 'grp_pc_ppp']

# Step 0: Compute shared subset for decile calculation
dose_variables_set = set(dose_variables)
required_columns = list(dose_variables_set.union({'urban_share'}))

# Subset of rows with non-null values in *all* dose_variables + urban_share
urban_decile_base = urban_data.dropna(subset=required_columns)

# Create urban share deciles once from the shared base
urban_decile_base = urban_decile_base.copy()
urban_decile_base['urbanshare_band'] = pd.qcut(
    urban_decile_base['urban_share'], q=10, duplicates='drop'
)

# Store urbanshare_band per index
urbanshare_band_map = urban_decile_base['urbanshare_band']

def bootstrap_pearsonr(x, y, n_bootstrap=1000, ci=95, random_state=None):
    """Perform bootstrap on Pearson correlation."""
    rng = np.random.default_rng(seed=random_state)
    boot_corrs = []
    
    data = np.array([x, y]).T
    valid_data = data[~np.isnan(data).any(axis=1)]
    
    if len(valid_data) < 2:
        return np.nan, (np.nan, np.nan)

    for _ in range(n_bootstrap):
        sample = rng.choice(valid_data, size=len(valid_data), replace=True)
        r, _ = pearsonr(sample[:, 0], sample[:, 1])
        boot_corrs.append(r)
    
    boot_corrs = np.array(boot_corrs)
    lower = np.percentile(boot_corrs, (100 - ci) / 2)
    upper = np.percentile(boot_corrs, 100 - (100 - ci) / 2)
    mean_corr = np.mean(boot_corrs)
    
    return mean_corr, (lower, upper)

def analyze_urban_share_bootstrap(
    urban_data,
    target_variable,
    dose_variable,
    urbanshare_band_map,
    n_bootstrap=1000,
    ci=95,
    random_state=None
):
    # Filter rows with non-null values for both variables
    urban_decile_base = urban_data[
        urban_data[target_variable].notnull() &
        urban_data[dose_variable].notnull()
    ].copy()

    # Merge precomputed urban share bands
    urban_decile_base['urbanshare_band'] = urbanshare_band_map

    # Drop rows with missing bands
    urban_decile_base = urban_decile_base.dropna(subset=['urbanshare_band'])

    bands = []
    bootstrapped_results = []

    for band in urban_decile_base['urbanshare_band'].cat.categories:
        band_data = urban_decile_base[urban_decile_base['urbanshare_band'] == band]
        x = band_data[dose_variable].values
        y = band_data[target_variable].values

        if len(band_data) > 1:
            mean_corr, (lower_ci, upper_ci) = bootstrap_pearsonr(
                x, y, n_bootstrap=n_bootstrap, ci=ci, random_state=random_state
            )
        else:
            mean_corr, lower_ci, upper_ci = np.nan, np.nan, np.nan

        bands.append(band)
        bootstrapped_results.append({
            'mean_correlation': mean_corr,
            'ci_lower': lower_ci,
            'ci_upper': upper_ci,
            'n': len(band_data)
        })

    return bands, bootstrapped_results

results = {}

for target_variable, dose_variable in zip(target_variables, dose_variables):
    print(f"\nBootstrapping {target_variable} vs {dose_variable}...\n")
    bands, boot_results = analyze_urban_share_bootstrap(
        urban_data,
        target_variable,
        dose_variable,
        urbanshare_band_map,
        n_bootstrap=1000,
        ci=95,
        random_state=42
    )
    results[target_variable] = boot_results

    for band, res in zip(bands, boot_results):
        print(f"Urban share band {band}: "
              f"Mean r = {res['mean_correlation']:.2f}, "
              f"95% CI = ({res['ci_lower']:.2f}, {res['ci_upper']:.2f}), "
              f"n = {res['n']}")


# Define custom names for the legend
custom_legend_names = {
    'C2022_grp_pc_ppp': 'C2022',
    'Z2024_pc': 'Z2024',
    'K2025_grp_pc_lcu_2017': 'K2025',
    'WS2022_grp_pc_ppp': 'WS2022'
}
# plot the bootstrapped correlations

# Define colors for each target variable
colors = ['blue', 'orange', 'green', 'red']  # Match your previous variable colors

plt.figure(figsize=(12, 6))

for i, (target_variable, boot_results) in enumerate(results.items()):
    # Extract data for plotting
    mean_corrs = [res['mean_correlation'] for res in boot_results]
    lower_ci = [res['ci_lower'] for res in boot_results]
    upper_ci = [res['ci_upper'] for res in boot_results]

    # Plot mean correlation with error bars (95% CI)
    plt.errorbar(
        range(len(bands)),
        mean_corrs,
        yerr=[np.array(mean_corrs) - np.array(lower_ci), np.array(upper_ci) - np.array(mean_corrs)],
        fmt='o',
        capsize=4,
        label=custom_legend_names[target_variable],
        color=colors[i],
        alpha=0.8
    )

    # Optional: Add horizontal dashed line for overall correlation
    plt.axhline(
        y=overall_correlations[target_variable],
        color=colors[i],
        linestyle='--',
        linewidth=1.5,
        alpha=0.5
    )

# Create decile labels
decile_labels = [
    'Lowest 10%' if i == 0 else
    'Highest 10%' if i == 9 else
    f'{i+1}th decile'
    for i in range(len(bands))
]

plt.xlabel('Urban Area Share Deciles', fontsize=14)
plt.ylabel('Bootstrapped Pearson Correlation', fontsize=14)
plt.xticks(range(len(bands)), decile_labels, rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=12)
plt.tight_layout()

# Save and show the plot
plot_path = os.path.join(graphics_path, 'urban_corr_bootstrapped_2025-05-19.png')  # update with today's date
plt.savefig(plot_path, dpi=300)
plt.show()

# Function to calculate metrics for each decile group
def calculate_metrics(urban_data, target_variable, dose_variable):
    # Filter out rows with missing values
    urban_filtered = urban_data[urban_data[target_variable].notnull() & urban_data[dose_variable].notnull() & urban_data['urban_share'].notnull()]
    urban_filtered = urban_filtered.copy()  # Avoid SettingWithCopyWarning
    urban_filtered['urbanshare_band'] = pd.qcut(urban_filtered['urban_share'], q=10, duplicates='drop')

    # Define decile groups
    decile_groups = {
        '1st': urban_filtered[urban_filtered['urbanshare_band'] == urban_filtered['urbanshare_band'].cat.categories[0]],
        '10th': urban_filtered[urban_filtered['urbanshare_band'] == urban_filtered['urbanshare_band'].cat.categories[-1]],
        '2nd - 8th': urban_filtered[~urban_filtered['urbanshare_band'].isin([urban_filtered['urbanshare_band'].cat.categories[0], urban_filtered['urbanshare_band'].cat.categories[-1]])],
        'Overall': urban_filtered  # All data as one group
    }

    # Initialize results dictionary
    results = []

    # Calculate metrics for each group
    for group_name, group_data in decile_groups.items():
        if len(group_data) > 0:
            x = group_data[dose_variable]
            y = group_data[target_variable]

            # Calculate metrics
            percent_over = (y > x).mean() * 100
            percent_under = (y < x).mean() * 100
            mean_relative_diff = ((y - x) / x).mean()
            mean_absolute_relative_diff = ((y - x).abs() / x).mean()

            # Append results
            results.append({
                'Urban share decile': group_name,
                '% Over': round(percent_over, 2),
                '% Under': round(percent_under, 2),
                'Mean Rel Diff': round(mean_relative_diff, 4),
                'Mean Abs Rel Diff': round(mean_absolute_relative_diff, 4)
            })

    # Convert results to a DataFrame and sort rows
    return pd.DataFrame(results).set_index('Urban share decile').reindex(['1st', '2nd - 8th', '10th', 'Overall'])

# Perform analysis for each pair of target and dose variables
summary_tables = {}
for target_variable, dose_variable in zip(target_variables, dose_variables):
    print(f"\nAnalyzing {target_variable} vs {dose_variable}...\n")
    summary_table = calculate_metrics(urban_data, target_variable, dose_variable)
    summary_tables[f"{target_variable} vs {dose_variable}"] = summary_table

    # Pretty print the summary table
    print(summary_table.to_string())

# Define the output Excel file path
output_excel_path = os.path.join(graphics_path, 'urban_share_analysis_results.xlsx')

# Export summary tables to an Excel file
with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
    for key, table in summary_tables.items():
        # Use the key (e.g., "C2022_grp_pc_ppp vs grp_pc_ppp") as the sheet name
        sheet_name = key[:31]  # Excel sheet names have a 31-character limit
        table.to_excel(writer, sheet_name=sheet_name)

print(f"Results have been exported to {output_excel_path}")

