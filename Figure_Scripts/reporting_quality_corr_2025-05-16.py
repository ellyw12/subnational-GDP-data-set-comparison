import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

""" 1. Load and merge data from different sets """

# Define paths
data_path               =   '../Data/'
graphics_path           =   '../Graphics/'
deflator_path           =    data_path +'deflator/'
pickle_path             =    data_path +'pickles/'

# Define files
dose_v2   = 'DOSE_V2.10.csv'
pkl_files = ['C2022_data_outer.pkl', 'K2025_data_outer.pkl',
             'Z2024_data_outer.pkl']
WS2022 = 'WS2022_total_grp_0p001.csv'

data = pd.read_csv(data_path+dose_v2)
WS_data = pd.read_csv(data_path+WS2022)

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

spi_file                   =   'SPI_data.csv'
        # --> data downloaded from https://github.com/worldbank/SPI/tree/master/03_output_data

spi_pillar52 = pd.read_csv(data_path + spi_file,
                           usecols=['iso3c', 'date', 'SPI.D5.2.1.SNAU',
                                'SPI.D5.2.2.NABY', 'SPI.D5.2.3.CNIN',
                                'SPI.D5.2.4.CPIBY', 'SPI.D5.2.5.HOUS',
                                'SPI.D5.2.6.EMPL', 'SPI.D5.2.7.CGOV',
                                'SPI.D5.2.8.FINA', 'SPI.D5.2.9.MONY',
                                'SPI.D5.2.10.GSBP'])

# Select only DOSE countries for 2016 from the SPI data
spi_52_mean = spi_pillar52[spi_pillar52['iso3c'].isin(data['GID_0'].unique())
                           & (spi_pillar52['date'] == 2016)
                           ].drop(columns=['date'])

spi_52_mean['spi_5.2_2016'] = spi_52_mean.loc[:, 'SPI.D5.2.1.SNAU':'SPI.D5.2.10.GSBP'
                                              ].fillna(0).mean(axis=1)

# Calculate quintiles based on the values of 'spi_5.2_2016'
spi_52_mean = spi_52_mean.dropna(subset=['spi_5.2_2016'])
spi_52_mean['quintile'] = pd.qcut(spi_52_mean['spi_5.2_2016'], 5,
                                  labels=['BotQuint', '2ndQuint', '3rdQuint',
                                          '4thQuint', 'TopQuint'])

# Calculate quintiles for full SPI data
spi_52_mean_full = spi_pillar52[spi_pillar52['date'] == 2016
                                ].drop(columns=['date']).dropna(how='any')

spi_52_mean_full['spi_5.2_2016_full'] = spi_52_mean_full.loc[:, 'SPI.D5.2.1.SNAU':'SPI.D5.2.10.GSBP'
                                                             ].fillna(0).mean(axis=1)

spi_52_mean_full['quintile'] = pd.qcut(spi_52_mean_full['spi_5.2_2016_full'], 5,
                                  labels=['BotQuint', '2ndQuint', '3rdQuint',
                                          '4thQuint', 'TopQuint'])

# Initialize the dictionary to store the country codes for each quintile group
init_groups = {
    'BotQuint': [],
    '2ndQuint': [],
    '3rdQuint': [],
    '4thQuint': [],
    'TopQuint': [],
}

# Assign the country codes to each quintile group
for quintile in init_groups.keys():
    init_groups[quintile] = spi_52_mean_full[spi_52_mean_full['quintile'] == quintile]['iso3c'].tolist()

# Display the grouping, then copy below
print(init_groups)

spi_2016_quintiles = {
    'BotQuint': ['ARG', 'BHS', 'BOL', 'ETH', 'GTM', 'IRN', 'LAO', 'MAR', 'MOZ',
                 'NPL', 'PAK', 'PAN', 'PER', 'PHL', 'LKA', 'URY', 'VNM'],
    '2ndQuint': ['BRA', 'CHN', 'COL', 'ECU', 'HND', 'IND', 'IDN', 'KEN', 'MYS',
                 'MEX', 'MNG', 'NGA', 'MKD', 'PRY', 'ROU', 'TZA', 'ARE', 'UZB'],
    '3rdQuint': ['AZE', 'BIH', 'CHL', 'HRV', 'EGY', 'GEO', 'KAZ', 'NZL', 'RUS',
                 'SRB', 'ZAF', 'THA', 'TUR', 'UKR'],
    '4thQuint': ['ALB', 'BLR', 'BEL', 'BGR', 'CAN', 'CZE', 'DEU', 'GRC', 'IRL',
                 'JPN', 'KOR', 'KGZ', 'NOR', 'ESP', 'CHE', 'GBR'],
    'TopQuint': ['AUS', 'AUT', 'DNK', 'EST', 'FIN', 'FRA', 'HUN', 'ITA', 'LVA',
                 'LTU', 'NLD', 'POL', 'PRT', 'SVK', 'SVN', 'SWE', 'USA'],
}

spi_2016_full_quintiles = {
    'BotQuint': ['DZA', 'ATG', 'BRB', 'BEN', 'BTN', 'BOL', 'BFA', 'CPV', 'CAF',
                 'TCD', 'COM', 'CIV', 'DJI', 'GNQ', 'ERI', 'ETH', 'GAB', 'GMB',
                 'GRD', 'GTM', 'GIN', 'GNB', 'HTI', 'JOR', 'KIR', 'LAO', 'LBN',
                 'LBR', 'LBY', 'MWI', 'MLI', 'MHL', 'FSM', 'NRU', 'NPL', 'NER',
                 'PLW', 'PAN', 'PNG', 'SLE', 'SLB', 'SOM', 'SSD', 'KNA', 'LCA',
                 'SDN', 'SYR', 'TGO', 'TON', 'TUN', 'TKM', 'TUV', 'URY', 'VUT',
                 'VEN', 'VNM'],
    '2ndQuint': ['BLZ', 'BWA', 'BRN', 'COG', 'FJI', 'IRN', 'KWT', 'MDG', 'MRT',
                 'MNE', 'MAR', 'MMR', 'NAM', 'NIC', 'PER', 'QAT', 'STP', 'SAU',
                 'SEN', 'TTO', 'YEM'], # 22 in SPI list
    '3rdQuint': ['AFG', 'AGO', 'ARG', 'BHS', 'BHR', 'BGD', 'BDI', 'KHM', 'CMR',
                 'CHN', 'DMA', 'SLV', 'SWZ', 'GHA', 'GUY', 'HND', 'IRQ', 'JAM',
                 'KEN', 'LSO', 'MYS', 'MDV', 'MNG', 'MOZ', 'NGA', 'MKD', 'OMN',
                 'PAK', 'PRY', 'PHL', 'RWA', 'WSM', 'SMR', 'LKA', 'VCT', 'SUR',
                 'UZB', 'ZMB', 'ZWE'],
    '4thQuint': ['ALB', 'ARM', 'AZE', 'BLR', 'BIH', 'BRA', 'CHL', 'COL', 'CRI',
                 'HRV', 'CYP', 'DOM', 'ECU', 'EGY', 'GEO', 'IND', 'IDN', 'KAZ',
                 'KOR', 'MLT', 'MUS', 'MEX', 'NZL', 'RUS', 'SRB', 'SYC', 'SGP',
                 'ZAF', 'TJK', 'TZA', 'THA', 'TUR', 'UGA', 'UKR', 'ARE'],
    'TopQuint': ['AUS', 'AUT', 'BEL', 'BGR', 'CAN', 'CZE', 'DNK', 'EST', 'FIN',
                 'FRA', 'DEU', 'GRC', 'HUN', 'ISL', 'IRL', 'ISR', 'ITA', 'JPN',
                 'KGZ', 'LVA', 'LTU', 'LUX', 'MDA', 'NLD', 'NOR', 'POL', 'PRT',
                 'SVK', 'SVN', 'ESP', 'SWE', 'CHE', 'GBR', 'USA']} # 38 in SPI list

spi_colors = {
    'BotQuint': '#ff9f1c',
    '2ndQuint': '#f1dc76',
    '3rdQuint': '#f1dc76',
    '4thQuint': '#acece7',
    'TopQuint': '#2ec4b6'
}

# Select quintile groups
#spi_countries = spi_2016_quintiles
spi_countries = spi_2016_full_quintiles

def analyze_spi_groups_with_pvalues(data, target_variable, dose_variable, spi_countries):
    # Dictionary to store correlation results, p-values, and counts for each SPI group
    spi_results = {}

    # Iterate over each SPI group
    for group_name, country_list in spi_countries.items():
        # Filter data for the current SPI group
        group_data = data[data['GID_0'].isin(country_list)]

        # Count the number of observations
        observation_count = len(group_data)

        # Ensure there are enough data points for correlation analysis
        if observation_count > 1:
            x = group_data[dose_variable]
            y = group_data[target_variable]

            # Filter out rows with missing values
            valid_data = group_data[x.notnull() & y.notnull()]
            if len(valid_data) > 1:
                # Calculate correlation and p-value
                correlation, p_value = pearsonr(valid_data[dose_variable], valid_data[target_variable])
            else:
                correlation, p_value = np.nan, np.nan
        else:
            correlation, p_value = np.nan, np.nan

        # Store the correlation result, p-value, and observation count
        spi_results[group_name] = {'correlation': correlation, 'p_value': p_value, 'count': observation_count}

    return spi_results

# Run the analysis for each target variable
results_with_pvalues = {}

for target_variable, dose_variable in zip(target_variables, dose_variables):
    print(f"\nAnalyzing {target_variable} against {dose_variable}...\n")
    correlations_with_pvalues = analyze_spi_groups_with_pvalues(data, target_variable, dose_variable, spi_countries)
    results_with_pvalues[target_variable] = correlations_with_pvalues

    # Print correlation values, p-values, and counts for each SPI group
    for group_name, result in correlations_with_pvalues.items():
        print(f"SPI group {group_name}: Pearson Correlation = {result['correlation']:.2f}, "
              f"P-Value = {result['p_value']:.4f}, Observations = {result['count']}")

# Plot the results
plt.figure(figsize=(12, 6))
colors = ['blue', 'orange', 'green', 'red']  # Colors for each target variable
group_names = list(spi_countries.keys())

# Define custom names for the legend
custom_legend_names = {
    'C2022_grp_pc_ppp': 'C2022',
    'Z2024_pc': 'Z2024',
    'K2025_grp_pc_lcu_2017': 'K2025',
    'WS2022_grp_pc_ppp': 'WS2022'
}

# Calculate the number of observations for each group
group_counts = {group: sum(data['GID_0'].isin(countries)) for group, countries in spi_countries.items()}

# Create x-tick labels with observation counts
xtick_labels = [
    f"{label}\n(n = {group_counts[group]})"
    for label, group in zip(['Lowest 20%', '2nd quintile', '3rd quintile', '4th quintile', 'Highest 20%'], group_names)
]

for i, (target_variable, correlations) in enumerate(results.items()):
    # Extract correlation values in the order of SPI groups
    correlation_values = [correlations[group] for group in group_names]
    plt.scatter(group_names, correlation_values, label=custom_legend_names[target_variable], color=colors[i], s=100)

# Add labels and title
plt.xlabel('Reporting quality quintiles (World Bank SPI Groups)')
plt.ylabel('Pearson Correlation Coefficient')
plt.title('GRP per capita correlations by reporting quality group')
plt.xticks(ticks=range(len(group_names)), labels=xtick_labels, rotation=0)
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.tight_layout()
plt.savefig(graphics_path + 'spi_group_corr_2025-04-21.png')  # Update with today's date
plt.show()

