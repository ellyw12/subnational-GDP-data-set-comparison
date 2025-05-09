import numpy as np
import pandas as pd
import os

# Paths:

data_path               =   '../Data/'
graphics_path           =   '../Graphics/'
deflator_path           =    data_path +'deflator/'
chen_path               =    data_path +'chen2022/'

# File names:
dose_v2p10               =   'DOSE_V2.10.csv'
# C2022 at 0.25 and 0.50 degree aggregations
chen2022_0p25                =   'chen_grp_0p25_new.csv'
chen2022_0p50                =   'chen_total_gdp_subnat.csv'
chen2022_0p001               =   'chen_total_grp_0p001.csv'

dose                    =   pd.read_csv(data_path+dose_v2p10)
chen                    =   pd.read_csv(chen_path+chen2022_0p001) # specify aggregation level

# Merge dc: correcting units to GRP per capita

# Merge data frames 
chen = chen.rename(columns={'grp':'var'}) # use this if using the 0.001 degree aggregation
dc = dose.merge(chen[['GID_1', 'year', 'var']], on=['GID_1', 'year'], how='outer')

# correct units: GRP in millions of 2017 PPP-USD to GRP in 2017 PPP-USD
dc['C2022'] = (dc['var'] * 1000000)

"""
Conversion to 2015 constant USD at PPP and MER
"""

# Load World Bank GDP deflator dc
deflators = pd.read_excel(deflator_path+'2022_03_30_WorldBank_gdp_deflator.xlsx',
                          sheet_name='Data', index_col=None, usecols='B,E:BM',
                          skiprows=3).set_index('Country Code')
deflators = deflators.dropna(axis=0, how='all')

# Change the reference year in the GDP deflator dc to 2015 (i.e. 2015 = 100)
# (in the original, the reference year varies by country)
column = deflators.columns.get_loc(2015)
for i in list(range(0,len(deflators))):
    deflators.iloc[i,:] = (deflators.iloc[i,:]/deflators.iloc[i,column])*100

# Make separate dataframes for the US and all countries, both in 'long' format
deflators_us = pd.DataFrame(deflators.loc[deflators.index=='USA'].stack()
                            ).rename(columns={0:'deflator_2015_us'}
                                     ).reset_index().set_index('level_1')
deflators = pd.DataFrame(deflators.stack()).rename(columns={0:'deflator_2015'})

# Add the deflators to the 'dc' dataframe
dc['deflator_2015'] = np.nan
dc['deflator_2015_us'] = np.nan
dc = dc.set_index(['GID_0','year'])
dc.update(deflators)
dc = dc.reset_index('GID_0')
dc.update(deflators_us)

# Load PPP conversion factors for both relevant years ...
# ... and add both sets of factors for each country to 'dc'
ppp_data = pd.read_excel(deflator_path+'ppp_data_all_countries.xlsx')
ppp_data_2015 = ppp_data.loc[ppp_data.year==2015]
#ppp_data_2017 = ppp_data.loc[ppp_data.year==2017]
d2015 = dict(zip(ppp_data_2015.iso_3, ppp_data_2015.PPP))
#d2017 = dict(zip(ppp_data_2017.iso_3, ppp_data_2017.PPP))
dc['ppp_2015'] = dc['GID_0'].apply(lambda x: d2015.get(x))
#dc['ppp_2017'] = dc['GID_0'].apply(lambda x: d2017.get(x))
dc.loc[dc.GID_0 == 'USA', 'ppp_2015'] = len(dc.loc[dc.GID_0=='USA']) * [1]
#dc.loc[dc.GID_0 == 'USA', 'ppp_2017'] = 1
dc['ppp_2015'] = pd.to_numeric(dc['ppp_2015'], errors='coerce')
#dc['ppp_2017'] = pd.to_numeric(dc['ppp_2017'], errors='coerce')

# Load MER conversion factors for 2015 and add them for each country to 'dc'
fx_data = pd.read_excel(deflator_path+'fx_data_all_countries.xlsx')
fx_data = fx_data.loc[fx_data.year==2015]
d = dict(zip(fx_data.iso_3, fx_data.fx))
dc['fx_2015'] = dc['GID_0'].apply(lambda x: d.get(x))
dc.loc[dc.GID_0=='USA','fx_2015'] = len(dc.loc[dc.GID_0=='USA']) * [1]

# Change reference year (at PPP) from 2017 to 2015: METHOD 1 + 2 resp.
dc['C2022_grp_ppp_2015'] = dc['C2022'] * 100 / dc['deflator_2015_us']
dc['C2022_grp_lcu_2015'] = dc['C2022'] * dc.PPP / dc.fx

# Convert from 2017 USD at PPP to 2015 USD at MER: METHOD 1
dc['C2022_grp_usd_2015'] = (
    dc['C2022'] * dc['deflator_2015_us'] / 100 * # To current USD
    dc.PPP / dc.fx / # To current lcu and back to current USD, now at MER
    dc['deflator_2015_us'] * 100 # To 2015 USD at MER
)

# Convert from 2017 USD at PPP to 2015 USD at MER: METHOD 2
dc['C2022_grp_lcu2015_usd'] = (
    dc['C2022'] * dc['deflator_2015_us'] / 100 * # To current USD
    dc.PPP / dc['deflator_2015'] * 100 / # To current lcu and back to current USD, now at MER
    dc['fx_2015'] # To 2015 USD at MER
)

# conversions to per capita values
dc['C2022_pc'] = dc['C2022'] / dc['pop']
dc['C2022_grp_pc_ppp_2015'] = dc['C2022_grp_ppp_2015'] / dc['pop']
dc['C2022_grp_pc_lcu2015_ppp'] = dc['C2022_grp_lcu_2015'] / dc['pop']
dc['C2022_grp_pc_usd_2015'] = dc['C2022_grp_usd_2015'] / dc['pop']

dc = dc.reset_index()

# if you want to select particular country, otherwise comment out
country = 'EGY'
filtered_dc = dc.loc[dc['GID_0'] == country, ['GID_0', 'GID_1', 'year', 'C2022',
                                            'C2022_grp_ppp_2015', 'C2022_grp_usd_2015',
                                            'C2022_grp_lcu2015_usd', 'C2022_grp_lcu_2015',
                                            'C2022_pc', 'C2022_grp_pc_ppp_2015',
                                            'C2022_grp_pc_lcu2015_ppp', 'C2022_grp_pc_usd_2015']]
filtered_dc.to_pickle(f"{chen_path}C2022_{country}_data.pkl")

dc[['GID_0', 'GID_1', 'year', 'C2022',
    'C2022_grp_ppp_2015', 'C2022_grp_usd_2015',
    'C2022_grp_lcu2015_usd', 'C2022_grp_lcu_2015',
    'C2022_pc', 'C2022_grp_pc_ppp_2015',
    'C2022_grp_pc_lcu2015_ppp', 'C2022_grp_pc_usd_2015']].to_pickle(chen_path+'C2022_data.pkl')

# Note: to import other data this way:
# imported_data = pd.read_pickle(chen_path+'chen_data.pkl')