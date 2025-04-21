"""
Using the pre-processed and merged data from the prep files, this script derives PCC for each modelled data set compared to reported values.
It is calculated following three steps:


1. Calculate PCC for modelled data sets and reported GRP
2. Deflate DOSE and Z2024 to prepare for growth rate calculations
3. Calculating GRP volume growth rates
4. Appendix: redundant code for beginning of step 4

Author: LS
"""

import os
import pickle

import numpy as np
import pandas as pd


""" 1. Load and merge data from different sets """

# Define paths

data_path = './Data/'
graphics_path = './Figures/'
deflator_path           =    data_path +'deflator/'

data_dose_pop = 'merged_dose_pop.csv'
dose_pop_path = data_path + data_dose_pop
dose_pop_data = pd.read_csv(dose_pop_path)

data = dose_pop_data


""" 1. Calculate PCC for modelled data sets and reported GRP """

corr_variables = ['grp_ppp_2017', 'grp_lcu2017_ppp', 'grp_pc_ppp_2017',
                  'grp_pc_lcu2017_ppp', 'grp_ppp_2005', 'grp_lcu2005_ppp',
                  'grp_pc_ppp_2005', 'grp_pc_lcu2005_ppp', 'grp_usd',
                  'grp_pc_usd', 'C2022', 'C2022_pc', 'K2025', 'K2025_pc',
                  'WS2022', 'WS2022_pc', 'Z2024', 'Z2024_pc']

data[corr_variables] = data[corr_variables].apply(pd.to_numeric, errors='coerce')

# DOSE vs C2022 total, method 1
valid_rows_C2022_tot1 = data[['grp_ppp_2017', 'C2022']].dropna()
corr_C2022_tot1 = valid_rows_C2022_tot1['grp_ppp_2017'].corr(valid_rows_C2022_tot1['C2022'])

# DOSE vs C2022 total, method 2
valid_rows_C2022_tot2 = data[['grp_lcu2017_ppp', 'C2022']].dropna()
corr_C2022_tot2 = valid_rows_C2022_tot2['grp_lcu2017_ppp'].corr(valid_rows_C2022_tot2['C2022'])

# DOSE vs C2022 pc, method 1
valid_rows_C2022_pc1 = data[['grp_pc_ppp_2017', 'C2022_pc']].dropna()
corr_C2022_pc1 = valid_rows_C2022_pc1['grp_pc_ppp_2017'].corr(valid_rows_C2022_pc1['C2022_pc'])

# DOSE vs C2022 pc, method 2
valid_rows_C2022_pc2 = data[['grp_pc_lcu2017_ppp', 'C2022_pc']].dropna()
corr_C2022_pc2 = valid_rows_C2022_pc2['grp_pc_lcu2017_ppp'].corr(valid_rows_C2022_pc2['C2022_pc'])

# DOSE vs K2025 total, method 1
valid_rows_K2025_tot1 = data[['grp_ppp_2017', 'K2025']].dropna()
corr_K2025_tot1 = valid_rows_K2025_tot1['grp_ppp_2017'].corr(valid_rows_K2025_tot1['K2025'])

# DOSE vs K2025 total, method 2
valid_rows_K2025_tot2 = data[['grp_lcu2017_ppp', 'K2025']].dropna()
corr_K2025_tot2 = valid_rows_K2025_tot2['grp_lcu2017_ppp'].corr(valid_rows_K2025_tot2['K2025'])

# DOSE vs K2025 pc, method 1
valid_rows_K2025_pc1 = data[['grp_pc_ppp_2017', 'K2025_pc']].dropna()
corr_K2025_pc1 = valid_rows_K2025_pc1['grp_pc_ppp_2017'].corr(valid_rows_K2025_pc1['K2025_pc'])

# DOSE vs K2025 pc, method 2
valid_rows_K2025_pc2 = data[['grp_pc_lcu2017_ppp', 'K2025_pc']].dropna()
corr_K2025_pc2 = valid_rows_K2025_pc2['grp_pc_lcu2017_ppp'].corr(valid_rows_K2025_pc2['K2025_pc'])

# DOSE vs WS2022 total, method 1
valid_rows_WS2022_tot1 = data[['grp_ppp_2005', 'WS2022']].dropna()
corr_WS2022_tot1 = valid_rows_WS2022_tot1['grp_ppp_2005'].corr(valid_rows_WS2022_tot1['WS2022'])

# DOSE vs WS2022 total, method 2
valid_rows_WS2022_tot2 = data[['grp_lcu2005_ppp', 'WS2022']].dropna()
corr_WS2022_tot2 = valid_rows_WS2022_tot2['grp_lcu2005_ppp'].corr(valid_rows_WS2022_tot2['WS2022'])

# DOSE vs WS2022 pc, method 1
valid_rows_WS2022_pc1 = data[['grp_pc_ppp_2005', 'WS2022_pc']].dropna()
corr_WS2022_pc1 = valid_rows_WS2022_pc1['grp_pc_ppp_2005'].corr(valid_rows_WS2022_pc1['WS2022_pc'])

# DOSE vs WS2022 pc, method 2
valid_rows_WS2022_pc2 = data[['grp_pc_lcu2005_ppp', 'WS2022_pc']].dropna()
corr_WS2022_pc2 = valid_rows_WS2022_pc2['grp_pc_lcu2005_ppp'].corr(valid_rows_WS2022_pc2['WS2022_pc'])

# DOSE vs Z2024 total
valid_rows_Z2024_tot = data[['grp_usd', 'Z2024']].dropna()
corr_Z2024_tot = valid_rows_Z2024_tot['grp_usd'].corr(valid_rows_Z2024_tot['Z2024'])

# DOSE vs Z2024 pc
valid_rows_Z2024_pc = data[['grp_pc_usd', 'Z2024_pc']].dropna()
corr_Z2024_pc = valid_rows_Z2024_pc['grp_pc_usd'].corr(valid_rows_Z2024_pc['Z2024_pc'])

# Print the correlation results
print(f"Results for pc and total grp correlation:")

print(f"Correlation (DOSE vs C2022 total, method 1): {corr_C2022_tot1:.4f}, Data points: {len(valid_rows_C2022_tot1)}")
print(f"Correlation (DOSE vs C2022 total, method 2): {corr_C2022_tot2:.4f}, Data points: {len(valid_rows_C2022_tot2)}")
print(f"Correlation (DOSE vs C2022 pc, method 1): {corr_C2022_pc1:.4f}, Data points: {len(valid_rows_C2022_pc1)}")
print(f"Correlation (DOSE vs C2022 pc, method 2): {corr_C2022_pc2:.4f}, Data points: {len(valid_rows_C2022_pc2)}")

print(f"Correlation (DOSE vs K2025 total, method 1): {corr_K2025_tot1:.4f}, Data points: {len(valid_rows_K2025_tot1)}")
print(f"Correlation (DOSE vs K2025 total, method 2): {corr_K2025_tot2:.4f}, Data points: {len(valid_rows_K2025_tot2)}")
print(f"Correlation (DOSE vs K2025 pc, method 1): {corr_K2025_pc1:.4f}, Data points: {len(valid_rows_K2025_pc1)}")
print(f"Correlation (DOSE vs K2025 pc, method 2): {corr_K2025_pc2:.4f}, Data points: {len(valid_rows_K2025_pc2)}")

print(f"Correlation (DOSE vs WS2022 total, method 1): {corr_WS2022_tot1:.4f}, Data points: {len(valid_rows_WS2022_tot1)}")
print(f"Correlation (DOSE vs WS2022 total, method 2): {corr_WS2022_tot2:.4f}, Data points: {len(valid_rows_WS2022_tot2)}")
print(f"Correlation (DOSE vs WS2022 pc, method 1): {corr_WS2022_pc1:.4f}, Data points: {len(valid_rows_WS2022_pc1)}")
print(f"Correlation (DOSE vs WS2022 pc, method 2): {corr_WS2022_pc2:.4f}, Data points: {len(valid_rows_WS2022_pc2)}")

print(f"Correlation (DOSE vs Z2024 total): {corr_Z2024_tot:.4f}, Data points: {len(valid_rows_Z2024_tot)}")
print(f"Correlation (DOSE vs Z2024 pc): {corr_Z2024_pc:.4f}, Data points: {len(valid_rows_Z2024_pc)}")



""" 2. Deflate DOSE and Z2024 to prepare for growth rate calculations """

# Convert Z2024 from current USD to 2017 PPP-USD
data['Z2024_grp_pc_lcu'] = data['Z2024_pc'] * data.fx
data['Z2024_grp_pc_lcu_2017'] = data['Z2024_grp_pc_lcu'] * 100 / data['deflator_2017']
data['Z2024_grp_pc_lcu2017_ppp'] = data['Z2024_grp_pc_lcu_2017'] / data['ppp_2017']

# Add total GRP column
data['Z2024_grp_lcu2017_ppp'] = data['Z2024_grp_pc_lcu2017_ppp'] * data['pop']

# print(data[['GID_1', 'year', 'Z2024_grp_pc_lcu', 'Z2024_grp_pc_lcu_2017', 'Z2024_grp_pc_lcu2017_ppp', 'deflator_2017', 'ppp_2017']].sample(60))


""" 3. Calculating GRP volume growth rates """

# Ensure the data is sorted by year within each GID_1 group
data = data.sort_values(by=['GID_1', 'year'])

volume_columns = ['grp_lcu2017_ppp', 'grp_pc_lcu2017_ppp', 'C2022',
                  'C2022_pc', 'K2025', 'K2025_pc', 'Z2024', 'Z2024_pc',
                  'WS2022', 'WS2022_pc']

for vcol in volume_columns:
    new_gcol = vcol + '_growth'
    data[new_gcol] = data.groupby('GID_1')[vcol].transform(
        lambda x: (np.log(x.where(x > 0)) - np.log(x.shift(1).where(x.shift(1) > 0))) * 100
    )

# data.columns
# print(data[['GID_1', 'year', 'grp_pc_lcu2017_ppp_growth', 'C2022_pc_growth', 'K2025_pc_growth', 'Z2024_pc_growth', 'WS2022_pc_growth', 'deflator_2017', 'ppp_2017']].sample(60))

# List printed growth columns
growth_columns = ['grp_lcu2017_ppp_growth', 'grp_pc_lcu2017_ppp_growth',
                  'C2022_growth', 'C2022_pc_growth', 'K2025_growth',
                  'K2025_pc_growth', 'Z2024_growth', 'Z2024_pc_growth',
                  'WS2022_growth', 'WS2022_pc_growth']

data[growth_columns] = data[growth_columns].apply(pd.to_numeric, errors='coerce')

# DOSE vs C2022 total
valid_rows_C2022_tot = data[['grp_lcu2017_ppp_growth', 'C2022_growth']].dropna()
corr_C2022_tot = valid_rows_C2022_tot['grp_lcu2017_ppp_growth'].corr(valid_rows_C2022_tot['C2022_growth'])

# DOSE vs C2022 pc
valid_rows_C2022_pc = data[['grp_pc_lcu2017_ppp_growth', 'C2022_pc_growth']].dropna()
corr_C2022_pc = valid_rows_C2022_pc['grp_pc_lcu2017_ppp_growth'].corr(valid_rows_C2022_pc['C2022_pc_growth'])

# DOSE vs K2025 total
valid_rows_K2025_tot = data[['grp_lcu2017_ppp_growth', 'K2025_growth']].dropna()
corr_K2025_tot = valid_rows_K2025_tot['grp_lcu2017_ppp_growth'].corr(valid_rows_K2025_tot['K2025_growth'])

# DOSE vs K2025 pc
valid_rows_K2025_pc = data[['grp_pc_lcu2017_ppp_growth', 'K2025_pc_growth']].dropna()
corr_K2025_pc = valid_rows_K2025_pc['grp_pc_lcu2017_ppp_growth'].corr(valid_rows_K2025_pc['K2025_pc_growth'])

# DOSE vs WS2022 total
valid_rows_WS2022_tot = data[['grp_lcu2017_ppp_growth', 'WS2022_growth']].dropna()
corr_WS2022_tot = valid_rows_WS2022_tot['grp_lcu2017_ppp_growth'].corr(valid_rows_WS2022_tot['WS2022_growth'])

# DOSE vs WS2022 pc
valid_rows_WS2022_pc = data[['grp_pc_lcu2017_ppp_growth', 'WS2022_pc_growth']].dropna()
corr_WS2022_pc = valid_rows_WS2022_pc['grp_pc_lcu2017_ppp_growth'].corr(valid_rows_WS2022_pc['WS2022_pc_growth'])

# DOSE vs Z2024 total
valid_rows_Z2024_tot = data[['grp_lcu2017_ppp_growth', 'Z2024_growth']].dropna()
corr_Z2024_tot = valid_rows_Z2024_tot['grp_lcu2017_ppp_growth'].corr(valid_rows_Z2024_tot['Z2024_growth'])

# DOSE vs Z2024 pc
valid_rows_Z2024_pc = data[['grp_pc_lcu2017_ppp_growth', 'Z2024_pc_growth']].dropna()
corr_Z2024_pc = valid_rows_Z2024_pc['grp_pc_lcu2017_ppp_growth'].corr(valid_rows_Z2024_pc['Z2024_pc_growth'])


# Print the correlation results

print(f"Results for growth rates:")

print(f"Correlation (DOSE vs C2022 total): {corr_C2022_tot:.4f}, Data points: {len(valid_rows_C2022_tot)}")
print(f"Correlation (DOSE vs C2022 pc): {corr_C2022_pc:.4f}, Data points: {len(valid_rows_C2022_pc)}")

print(f"Correlation (DOSE vs K2025 total): {corr_K2025_tot:.4f}, Data points: {len(valid_rows_K2025_tot)}")
print(f"Correlation (DOSE vs K2025 pc): {corr_K2025_pc:.4f}, Data points: {len(valid_rows_K2025_pc)}")

print(f"Correlation (DOSE vs WS2022 total): {corr_WS2022_tot:.4f}, Data points: {len(valid_rows_WS2022_tot)}")
print(f"Correlation (DOSE vs WS2022 pc): {corr_WS2022_pc:.4f}, Data points: {len(valid_rows_WS2022_pc)}")

print(f"Correlation (DOSE vs Z2024 total): {corr_Z2024_tot:.4f}, Data points: {len(valid_rows_Z2024_tot)}")
print(f"Correlation (DOSE vs Z2024 pc): {corr_Z2024_pc:.4f}, Data points: {len(valid_rows_Z2024_pc)}")



""" 4. Appendix: redundant code for beginning of step 2 """

# Load World Bank GDP deflator data
deflators = pd.read_excel(deflator_path+'2022_03_30_WorldBank_gdp_deflator.xlsx', sheet_name='Data',
                            index_col=None, usecols='B,E:BM', skiprows=3).set_index('Country Code')
deflators = deflators.dropna(axis=0, how='all')

# Deflate the two data sets reporting GRP at current prices
deflators_2017 = deflators.copy()
column2017 = deflators_2017.columns.get_loc(2017)
for i in range(len(deflators_2017)):
    deflators_2017.iloc[i, :] = (deflators_2017.iloc[i, :] / deflators_2017.iloc[i, column2017]) * 100

# Make separate dataframes for the US and local GDP deflators, both stacked
deflators_2017_us = pd.DataFrame(deflators_2017.loc[deflators_2017.index=='USA'].stack()
                                 ).rename(columns={0:'deflator_2017_us'}
                                          ).reset_index().set_index('level_1')
deflators_2017 = pd.DataFrame(deflators_2017.stack()).rename(columns={0:'deflator_2017'})

# Add the deflators to the 'data' dataframe
data['deflator_2017'] = np.nan
data['deflator_2017_us'] = np.nan

data = data.set_index(['GID_0','year'])
data.update(deflators_2017)
data = data.reset_index('GID_0')
data.update(deflators_2017_us)

data[data['GID_0'] == 'COL'].head(60)

# Load PPP conversion factors for both relevant years ...
# ... and add both sets of factors for each country to 'data'
ppp_data = pd.read_excel(deflator_path+'ppp_data_all_countries.xlsx')
ppp_data_2017 = ppp_data.loc[ppp_data.year==2017]
d2017 = dict(zip(ppp_data_2017.iso_3, ppp_data_2017.PPP))
data['ppp_2017'] = data['GID_0'].apply(lambda x: d2017.get(x))
data.loc[data.GID_0 == 'USA', 'ppp_2017'] = 1
data['ppp_2017'] = pd.to_numeric(data['ppp_2017'], errors='coerce')

# data.columns
data.reset_index(inplace=True)
# print(data[['GID_1', 'year', 'grp_lcu', 'grp_pc_lcu', 'deflator_2017', 'ppp_2017']].sample(60))

# Convert DOSE from current LCU to 2017 PPP-USD
data['grp_pc_lcu_2017'] = data['grp_pc_lcu'] * 100 / data['deflator_2017']
data['grp_pc_lcu2017_ppp'] = data['grp_pc_lcu_2017'] / data['ppp_2017']

# Add total GRP column
data['grp_lcu2017_ppp'] = data['grp_pc_lcu2017_ppp'] * data['pop']

# print(data[['GID_1', 'year', 'grp_pc_lcu2017_ppp', 'grp_lcu2017_ppp', 'deflator_2017', 'ppp_2017']].sample(60))