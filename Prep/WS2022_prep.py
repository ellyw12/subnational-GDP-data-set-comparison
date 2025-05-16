import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string

data_path               =   './Data/'

deflator_path           =    data_path +'deflator/'
wangsun_path              =    data_path +'modelled_data/'


# File names:
dose_v2 = 'DOSE_V2.10.csv'
wangsun = 'GRP_WangSun_aggregated(new).csv'

# Load data
dose = pd.read_csv(data_path+dose_v2)
wang = pd.read_csv(wangsun_path+wangsun)

data = dose

# Total GDP and total GRP: merge data and set 'wangsun' to appropriate level of measurement
dose_with_wangsun = data.merge(wang[['GID_1', 'year', 'grp']], on=['GID_1', 'year'], how='outer')

dose_with_wangsun['GID_0'] = dose_with_wangsun.apply(lambda row: row['GID_1'].split('.')[0] if pd.isna(row['GID_0']) else row['GID_0'], axis=1)

# Rename 'var' column to 'WangGDPpc' (although not per capita value)
dose_with_wangsun['WS2022'] = dose_with_wangsun['grp']

print(dose_with_wangsun.head())

#Pickle creation
data=dose_with_wangsun
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


deflators_2005 = deflators.copy()
column2005 = deflators_2005.columns.get_loc(2005)
for i in range(len(deflators_2005)):
    deflators_2005.iloc[i, :] = (deflators_2005.iloc[i, :] / deflators_2005.iloc[i, column2005]) * 100


# Make separate dataframes for the US and local GDP deflators, both stacked
deflators_us_2015 = pd.DataFrame(deflators_2015.loc[deflators_2015.index=='USA'].stack()
                                 ).rename(columns={0:'deflator_2015_us'}
                                          ).reset_index().set_index('level_1')
deflators_2015 = pd.DataFrame(deflators_2015.stack()).rename(columns={0:'deflator_2015'})


deflators_us_2005 = pd.DataFrame(deflators_2005.loc[deflators_2005.index=='USA'].stack()
                                 ).rename(columns={0:'deflator_2005_us'}
                                          ).reset_index().set_index('level_1')
deflators_2005 = pd.DataFrame(deflators_2005.stack()).rename(columns={0:'deflator_2005'})


# Add the deflators to the 'data' dataframe
data['deflator_2015'] = np.nan
data['deflator_2015_us'] = np.nan
data['deflator_2005'] = np.nan
data['deflator_2005_us'] = np.nan


print(data.columns)


data = data.set_index(['GID_0','year'])
data.update(deflators_2015)
data.update(deflators_2005)
data = data.reset_index('GID_0')
data.update(deflators_us_2015)
data.update(deflators_us_2005)


# Add ppp conversion factors for 2015
ppp_data = pd.read_excel(deflator_path+'ppp_data_all_countries.xlsx')
ppp_data = ppp_data.loc[ppp_data.year==2015]
d = dict(zip(ppp_data.iso_3, ppp_data.PPP))


data['ppp_2015'] = data['GID_0'].apply(lambda x: d.get(x))
data.loc[data.GID_0=='USA','ppp_2015'] = len(data.loc[data.GID_0=='USA']) * [1]


# Load MER conversion factors for 2015 and add them for each country to 'data'
fx_data = pd.read_excel(deflator_path+'fx_data_all_countries.xlsx')
fx_data = fx_data.loc[fx_data.year==2015]
d = dict(zip(fx_data.iso_3, fx_data.fx))
data['fx_2015'] = data['GID_0'].apply(lambda x: d.get(x))
data.loc[data.GID_0=='USA','fx_2015'] = len(data.loc[data.GID_0=='USA']) * [1]


# First inflate kummu: from constant 2005 to current USD at PPP
data['WS2022_ppp'] = data['WS2022'] * data['deflator_2005_us'] / 100


# Convert to 2015 USD at PPP: METHOD 1 + 2 respectively
data['WS2022_ppp_2015'] = data['WS2022_ppp'] / data['deflator_2015_us'] * 100
data['WS2022_lcu2015_ppp'] = ((data['WS2022_ppp'] * data.PPP) # To current lcu
                                    / data.fx # To current USD, now at MER
                                    / data['deflator_2015'] * 100 # To 2015 lcu
                                    / data['ppp_2015']) # To 2015 USD at PPP


# Convert to 2015 USD at MER: METHOD 1
data['WS2022_usd_2015'] = (data['WS2022_ppp'] * data.PPP # To current lcu
                                 / data.fx # To current USD, now at MER
                                 / data['deflator_2015_us'] * 100) # To 2015 USD at MER


# Convert to 2015 USD at MER: METHOD 2
data['WS2022_lcu2015_usd'] = ((data['WS2022_ppp'] * data.PPP) # To current lcu
                                    / data['deflator_2015'] * 100 # To 2015 lcu
                                    / data['fx_2015']) # To 2015 USD at MER


print(data[['GID_0', 'WS2022', 'WS2022_ppp_2015',
            'WS2022_lcu2015_ppp', 'WS2022_usd_2015',
            'WS2022_lcu2015_usd']].sample(20))
print(data[['GID_1', 'deflator_2015', 'deflator_2015_us', 'WS2022']].head(60))


# Add per capita values
data['WS2022_pc'] = data['WS2022'] / data['pop']
data['WS2022_pc_ppp_2015'] = data['WS2022_ppp_2015'] / data['pop']
data['WS2022_pc_lcu2015_ppp'] = data['WS2022_lcu2015_ppp'] / data['pop']
data['WS2022_pc_usd_2015'] = data['WS2022_usd_2015'] / data['pop']
data['WS2022_pc_lcu2015_usd'] = data['WS2022_lcu2015_usd'] / data['pop']

data = data.reset_index()

pickle_path = data_path + 'pickle/'

data.rename(columns={'year': 'year'}, inplace=True)
data[['GID_0', 'GID_1', 'year', 'WS2022', 'WS2022_ppp_2015',
            'WS2022_lcu2015_ppp', 'WS2022_usd_2015',
            'WS2022_lcu2015_usd', 'WS2022_pc', 'WS2022_pc_ppp_2015',
            'WS2022_pc_lcu2015_ppp', 'WS2022_pc_usd_2015',
            'WS2022_pc_lcu2015_usd']].to_pickle(pickle_path+
            'WS2022_data.pkl')
