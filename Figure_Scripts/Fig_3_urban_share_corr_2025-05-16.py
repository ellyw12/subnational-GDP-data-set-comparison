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

def analyze_urban_share(urban_data, target_variable, dose_variable):
    # Filter out rows with missing values
    urban_filtered = urban_data[urban_data[target_variable].notnull() & urban_data[dose_variable].notnull() & urban_data['urban_share'].notnull()]

    # Step 1: Create urban share bands w/ equal number of data points (q = number of bands)
    urban_filtered = urban_filtered.copy()  # Create a copy (avoids warning)
    urban_filtered['urbanshare_band'] = pd.qcut(urban_filtered['urban_share'], q=10, duplicates='drop')

    # Step 2: Calculate Pearson Correlation Coefficient for each band and count data points
    urb_correlations = []
    bands = []

    for band in urban_filtered['urbanshare_band'].cat.categories:
        band_data = urban_filtered[urban_filtered['urbanshare_band'] == band]
        if len(band_data) > 1:  # Ensure there are enough data points to calculate correlation
            x = band_data[dose_variable]
            y = band_data[target_variable]
            correlation, p_value = pearsonr(x, y)  # Calculate correlation and p-value
        else:
            correlation = np.nan  # Not enough data points to calculate correlation
        urb_correlations.append({'correlation': correlation, 'p_value': p_value})
        bands.append(band)

    return bands, urb_correlations

# List of target variables to analyze
target_variables = ['K2025_grp_pc_lcu_2017', 'Z2024_pc', 'WS2022_grp_pc_ppp', 'C2022_grp_pc_ppp']
dose_variables = ['grp_pc_lcu_2017', 'grp_pc_usd', 'grp_pc_ppp', 'grp_pc_ppp']

# Dictionary to store correlation values for each target variable
results = {}

# Define custom names for the legend
custom_legend_names = {
    'C2022_grp_pc_ppp': 'C2022',
    'Z2024_pc': 'Z2024',
    'K2025_grp_pc_lcu_2017': 'K2025',
    'WS2022_grp_pc_ppp': 'WS2022'
}

# Run the analysis for each target variable
for target_variable, dose_variable in zip(target_variables, dose_variables):
    print(f"\nAnalyzing {target_variable} against {dose_variable}...\n")
    bands, correlations = analyze_urban_share(urban_data, target_variable, dose_variable)
    results[target_variable] = correlations

    # Print correlation values for each band
    for band, correlation in zip(bands, correlations):
        print(f"Urban share band {band}: Pearson Correlation = {correlation['correlation']:.2f}, P-Value = {correlation['p_value']:.4f}")

# Calculate overall correlations for each target and dose variable pair
overall_correlations = {}
for target_variable, dose_variable in zip(target_variables, dose_variables):
    # Filter out rows with missing values
    valid_data = urban_data[urban_data[target_variable].notnull() & urban_data[dose_variable].notnull()]
    x = valid_data[dose_variable]
    y = valid_data[target_variable]
    overall_correlation = np.corrcoef(x, y)[0, 1]  # Pearson Correlation Coefficient
    overall_correlations[target_variable] = overall_correlation

# Define colors for each target variable (same as used in the scatter plot)
colors = ['blue', 'orange', 'green', 'red']  # Adjust these to match your plot's colors

# Plot all comparisons on the same plot
plt.figure(figsize=(12, 6))
for i, (target_variable, correlations) in enumerate(results.items()):
    correlation_values = [c['correlation'] for c in correlations] #add this line and modified below scatter command due to dictionary reading error (BW 19.5.2025)
    plt.scatter(range(len(bands)), correlation_values, label=custom_legend_names[target_variable], color=colors[i])
    # Add horizontal dotted line for the overall correlation
    plt.axhline(y=overall_correlations[target_variable], color=colors[i], linestyle='--', linewidth=1.5, alpha=0.5)

# Create decile labels
decile_labels = [f'Lowest 10%' if i == 0 else f'{i+1}nd decile' if i == 1 else f'{i+1}rd decile' if i == 2 else f'Highest 10%' if i == 9 else f'{i+1}th decile' for i in range(len(bands))]

# # Plot all comparisons on the same plot
# plt.figure(figsize=(12, 6))
# for target_variable, correlations in results.items():
#     plt.scatter(range(len(bands)), correlations, label=custom_legend_names[target_variable])

plt.xlabel('Urban area share decile')
plt.ylabel('Pearson Correlation Coefficient')
plt.title('GRP per capita correlations by urban area share:\nDOSE vs. global modelled data sets')
plt.xticks(range(len(bands)), decile_labels, rotation=45)
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.tight_layout()
plt.savefig(graphics_path+'urban_share_corr_2025-05-19.png') ## update with today's date
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

