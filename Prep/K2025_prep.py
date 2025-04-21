"""
K2025_conversion.py

With this script GRP data (Admin 1) by Kummu et al. 2025 are converted to 2015
constant USD variables to enable comparison with other data sets.

Author: Luuk Staal
"""

import numpy as np
import pandas as pd

data_path               =   './Data/'
deflator_path           =    data_path +'deflator/'
kummu_path              =    data_path +'modelled_data'

dose_v2                 =   'DOSE_V2.10.csv'
kummu_2025_adm1_pc      = 'tabulated_adm1_gdp_perCapita.csv'
    # Data downloaded from https://zenodo.org/records/13943886
dose                    =   pd.read_csv(data_path+dose_v2)


# Define paths
data_path               = r'C:\Users\31629\Documents\HU INRM\WiSe 2024-2025\PIK Study project\Coding\SP_dose\DOSE replication files\DOSE replication files\Data'
deflator_path           = data_path+'deflator'


# Define files
dose_v2                 = 'DOSE_V2.10.csv'
kummu_2025_adm1_pc      = 'tabulated_adm1_gdp_perCapita.csv'
        # Data downloaded from https://zenodo.org/records/13943886

# Load DOSE data
data                    = pd.read_csv(data_path+dose_v2)

# Load and melt Kummu data
kummu_2025 = pd.read_csv(kummu_path+kummu_2025_adm1_pc)
kummu_2025 = kummu_2025.melt(id_vars=['GID_nmbr', 'iso3', 'Country', 'Subnat'],
                             var_name='year', value_name='K2025_pc',
                             value_vars=[str(year) for year in range(1992, 2022)]
                             ).sort_values(by=['iso3', 'Subnat', 'year'])

""" Match subnational regions """

# Create dictionary with unique regions in K2025 for countries present in DOSE
baseline1 = kummu_2025.copy().set_index('Subnat')

#A = baseline1.copy().reset_index()
#A = A.loc[A.iso3.isin(list(data.GID_0.unique()))]

#regions_kummu2025 = A.drop_duplicates(subset=['iso3', 'Subnat'])[['iso3', 'Subnat']]

# Make dictionary of unique DOSE identifiers and region names
#d = dict(zip(zip(data.drop_duplicates(['GID_0', 'GID_1']).GID_0, data.drop_duplicates(['GID_0', 'GID_1']).region),
#             data.drop_duplicates(['GID_0', 'GID_1']).GID_1))

# Add GID_1 to regions_kummu2025 where 'region' matches 'Subnat'
#regions_kummu2025['GID_1'] = regions_kummu2025.apply(lambda x: d.get((x['iso3'], x['Subnat'])), axis=1)

#regions_kummu2025.to_excel(kummu_path  + 'kummu_regions_identifiers.xlsx',
#                           sheet_name='Data', index=False)

# Match manually edited list with GID_1 data
B = pd.read_excel(kummu_path+'kummu_regions_identifiers_manually_edited.xlsx')
B['helper'] = B.iso3+'_'+B.Subnat
d = dict(zip(B.helper, B.GID_1))

# Print number of matching regions and countries
print(B['GID_1'].notna().sum())
print(B[B['GID_1'].notna()]['iso3'].nunique())

A = baseline1.copy().reset_index()
A['year'] = A['year'].astype('int32')
A['helper'] = A.iso3+'_'+A.Subnat
A['GID_1'] = A['helper'].apply(lambda x: d.get(x))
A = A.loc[A.GID_1.notna()]
A = A.drop_duplicates(subset=['GID_1', 'year'], keep='first')

num_duplicates = A.duplicated(['GID_1', 'year']).sum()
print(f"Number of duplicates: {num_duplicates}")  # Print number of duplicates

A = A.set_index(['GID_1', 'year'])[['K2025_pc']]

# Matching with DOSE data using merge with outer join
data = data.drop_duplicates(subset=['GID_1', 'year'])
data = data.set_index(['GID_1', 'year']).reset_index()

# Perform outer join
data = pd.merge(data, A, on=['GID_1', 'year'], how='outer')

# Add total GRP
data['K2025'] = data['K2025_pc'] * data['pop']

# Print number of compatible data points
print((data['K2025_pc'].notna() & (data['K2025_pc'] != 0)).sum())  # 26,657 -> 37,980
print((data['K2025'].notna() & (data['K2025'] != 0)).sum())  # 25,681 -> 25,681

# Display a sample of the merged data
print(data.sample(20))



""" Conversion to 2015 constant USD at PPP and MER """

# Load World Bank GDP deflator data
deflators = pd.read_excel(deflator_path+
                          '2022_03_30_WorldBank_gdp_deflator.xlsx',
                          sheet_name='Data', index_col=None, usecols='B,E:BM',
                          skiprows=3).set_index('Country Code')
deflators = deflators.dropna(axis=0, how='all')

# Create two versions of the deflators, one with 2015 and one with 2017 as ref year
deflators_2015 = deflators.copy()
column2015 = deflators_2015.columns.get_loc(2015)
for i in range(len(deflators_2015)):
    deflators_2015.iloc[i, :] = (deflators_2015.iloc[i, :] / deflators_2015.iloc[i, column2015]) * 100

deflators_2017 = deflators.copy()
column2017 = deflators_2017.columns.get_loc(2017)
for i in range(len(deflators_2017)):
    deflators_2017.iloc[i, :] = (deflators_2017.iloc[i, :] / deflators_2017.iloc[i, column2017]) * 100

# Make separate dataframes for the US and local GDP deflators, both stacked
deflators_2015_us = pd.DataFrame(deflators_2015.loc[deflators_2015.index=='USA'].stack()
                                 ).rename(columns={0:'deflator_2015_us'}
                                          ).reset_index().set_index('level_1')
deflators_2015 = pd.DataFrame(deflators_2015.stack()).rename(columns={0:'deflator_2015'})

deflators_2017_us = pd.DataFrame(deflators_2017.loc[deflators_2017.index=='USA'].stack()
                                 ).rename(columns={0:'deflator_2017_us'}
                                          ).reset_index().set_index('level_1')
deflators_2017 = pd.DataFrame(deflators_2017.stack()).rename(columns={0:'deflator_2017'})

# Add the deflators to the 'data' dataframe
data['deflator_2015'] = np.nan
data['deflator_2015_us'] = np.nan
data['deflator_2017'] = np.nan
data['deflator_2017_us'] = np.nan

print(data.columns)

data = data.set_index(['GID_0','year'])
data.update(deflators_2015)
data.update(deflators_2017)
data = data.reset_index('GID_0')
data.update(deflators_2015_us)
data.update(deflators_2017_us)

# Add ppp conversion factors for 2015 and add them for each country to 'data'
ppp_data = pd.read_excel(deflator_path+'ppp_data_all_countries.xlsx')
ppp_data_2015 = ppp_data.loc[ppp_data.year==2015]
d = dict(zip(ppp_data_2015.iso_3, ppp_data_2015.PPP))
data['ppp_2015'] = data['GID_0'].apply(lambda x: d.get(x))
data.loc[data.GID_0=='USA','ppp_2015'] = len(data.loc[data.GID_0=='USA']) * [1]

# Repeat for 2017
ppp_data_2017 = ppp_data.loc[ppp_data.year==2017]
d = dict(zip(ppp_data_2017.iso_3, ppp_data_2017.PPP))
data['ppp_2017'] = data['GID_0'].apply(lambda x: d.get(x))
data.loc[data.GID_0=='USA','ppp_2017'] = len(data.loc[data.GID_0=='USA']) * [1]

# Load MER conversion factors for 2015 and add them for each country to 'data'
fx_data = pd.read_excel(deflator_path+'fx_data_all_countries.xlsx')
fx_data_2015 = fx_data.loc[fx_data.year==2015]
d = dict(zip(fx_data_2015.iso_3, fx_data_2015.fx))
data['fx_2015'] = data['GID_0'].apply(lambda x: d.get(x))
data.loc[data.GID_0=='USA','fx_2015'] = len(data.loc[data.GID_0=='USA']) * [1]

# Repeat for 2017
fx_data = pd.read_excel(deflator_path+'fx_data_all_countries.xlsx')
fx_data_2017 = fx_data.loc[fx_data.year==2015]
d = dict(zip(fx_data_2017.iso_3, fx_data_2017.fx))
data['fx_2017'] = data['GID_0'].apply(lambda x: d.get(x))
data.loc[data.GID_0=='USA','fx_2017'] = len(data.loc[data.GID_0=='USA']) * [1]

data.reset_index(inplace=True)
data[['GID_0', 'GID_1', 'year', 'fx', 'PPP']].sample(20)

# First transform to current lcu, assuming method 2 original conversion
data['K2025_pc_lcu_2017'] = data['K2025_pc'] * data['ppp_2017']
data['K2025_pc_lcu'] = data['K2025_pc_lcu_2017'] * data['deflator_2017'] / 100

# Convert to 2015 USD at PPP: METHOD 1
data['K2025_pc_ppp_2015'] = (data['K2025_pc_lcu'] 
                             / data.PPP # To current USD at PPP
                             / data['deflator_2015_us'] * 100) # To 2015 USD at PPP

# Convert to 2015 USD at PPP: METHOD 2
data['K2025_pc_lcu2015_ppp'] = (data['K2025_pc_lcu']
                                / data['deflator_2015'] * 100 # To 2015 lcu
                                / data['ppp_2015']) # To 2015 USD at PPP

# Convert to 2015 USD at MER: METHOD 1
data['K2025_pc_usd_2015'] = (data['K2025_pc_lcu']
                             / data.fx # To current USD at PPP
                             / data['deflator_2015_us'] * 100) # To 2015 USD at PPP

# Convert to 2015 USD at MER: METHOD 2
data['K2025_pc_lcu2015_usd'] = (data['K2025_pc_lcu']
                                / data['deflator_2015'] * 100 # To 2015 lcu
                                / data['fx_2015']) # To 2015 USD at MER

print(data[['GID_0', 'year', 'K2025_pc', 'K2025_pc_ppp_2015',
            'K2025_pc_lcu2015_ppp', 'K2025_pc_usd_2015',
            'K2025_pc_lcu2015_usd']].sample(20))
print(data[['GID_1', 'deflator_2015', 'deflator_2015_us', 'K2025_pc']].head(60))

data = data.reset_index()
data.sample(20)

pickle_path = data_path + 'pickle/'

data[['GID_0', 'GID_1', 'year', 'K2025', 'K2025_pc', 'K2025_pc_ppp_2015',
      'K2025_pc_lcu2015_ppp', 'K2025_pc_usd_2015', 'K2025_pc_lcu2015_usd']
      ].to_pickle(pickle_path+'K2025_data_outer.pkl')

# # Select for Egypt and save separately
# egypt_data = data[data['GID_0'] == 'EGY']

# egypt_data[['GID_0', 'GID_1', 'year', 'K2025_pc', 'K2025_pc_ppp_2015',
#             'K2025_pc_lcu2015_ppp', 'K2025_pc_usd_2015',
#             'K2025_pc_lcu2015_usd']].to_pickle(
#             pickle_path+'K2025_EGY_data.pkl')