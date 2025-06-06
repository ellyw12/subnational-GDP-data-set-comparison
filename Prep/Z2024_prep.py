"""
Z2024_prep.py

With this script GRP data (Admin 1) by Zhang et al. 2024 are converted to enable comparison with other data sets.

Author: Brielle Wells 
"""



import pandas as pd
import numpy as np
import pickle

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Zhang data downloaded from: https://doi.org/10.6084/m9.figshare.24024597.v3. 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# Paths and files:

data_path               =   './Data/'
graphics_path           =   '../Graphics/'
deflator_path           =    data_path +'deflator/'
Zhang_path              =    data_path +'modelled_data/'

dose_v2                 =   'DOSE_V2.10.csv'
Zhang_og                =   'Global_sub-national_GDP-pc_with_input_output.csv'

dose                    =   pd.read_csv(data_path+dose_v2)
zhang                   =   pd.read_csv(Zhang_path+Zhang_og, low_memory=False)


# Toggle to remove either '2013' or '2013b' duplicates
# Set to True to remove '2013b' duplicates, False to remove '2013' duplicates
remove_2013b = True #higher correlation when set to True 

if remove_2013b:
    zhang_clean = zhang[~((zhang['year_str'] == '2013b') & (zhang['year'] == 2013))]
    removed_duplicates = zhang[(zhang['year_str'] == '2013b') & (zhang['year'] == 2013)]
    # removed_duplicates.to_csv(data_path + 'zhang_duplicates1.csv', index=False)

else:
    zhang_clean = zhang[~((zhang['year_str'] == '2013') & (zhang['year'] == 2013))]
    removed_duplicates = zhang[(zhang['year_str'] == '2013') & (zhang['year'] == 2013)]
    # removed_duplicates.to_csv(data_path + 'zhang_duplicates2.csv', index=False)

duplicates_removed = len(removed_duplicates)
print(f"Number of duplicates removed: {duplicates_removed}")

# add Zhang files to DOSE 
merge_zhang_with_dose = pd.merge(dose, zhang_clean[['year', 'GID_1','CPGDP', 'MLPPred_lGDP_mean', 'MLPPred_lGDP_var']],
                     on=['year', 'GID_1'], how='outer')
merge_zhang_with_dose = merge_zhang_with_dose.rename(columns={'MLPPred_lGDP_mean': 'zoutput', 'MLPPred_lGDP_var': 'zvar'})
merge_zhang_with_dose['zmean'] = np.exp(merge_zhang_with_dose['zoutput'])

data = merge_zhang_with_dose
# data['GID_0'] = data['GID']
print(data.columns)

# Print the number of rows in the data DataFrame
num_rows = len(data)
print(f"Number of rows in data: {num_rows}")

# Load deflator data
deflators = pd.read_excel(deflator_path + '2022_03_30_WorldBank_gdp_deflator.xlsx', sheet_name='Data', 
                          index_col=None, usecols='B,E:BM', skiprows=3).set_index('Country Code')
deflators = deflators.dropna(axis=0, how='all')

# Normalize deflators to the year 2015
column = deflators.columns.get_loc(2015)
for i in range(len(deflators)):
    deflators.iloc[i, :] = (deflators.iloc[i, :] / deflators.iloc[i, column]) * 100
deflators_us = deflators.loc[deflators.index == 'USA'].stack().reset_index()
deflators_us.columns = ['Country Code', 'year', 'deflator_2015_us']
deflators_us = deflators_us.set_index('year')
deflators = deflators.stack().reset_index()
deflators.columns = ['Country Code', 'year', 'deflator_2015']
deflators = deflators.set_index(['Country Code', 'year'])

# Update data with deflators
data = data.set_index(['GID_0', 'year'])
data['deflator_2015'] = deflators['deflator_2015']
data = data.reset_index()
data = data.set_index('year')
data['deflator_2015_us'] = deflators_us['deflator_2015_us']
data = data.reset_index()

# Load PPP data
ppp_data = pd.read_excel(deflator_path + 'ppp_data_all_countries.xlsx')
ppp_dict = dict(zip(ppp_data.iso_3, ppp_data.PPP))
data['ppp'] = data['GID_0'].apply(lambda x: ppp_dict.get(x))
data.loc[data.GID_0 == 'USA', 'ppp'] = 1
data['ppp'] = pd.to_numeric(data['ppp'], errors='coerce')

ppp_data = ppp_data.loc[ppp_data.year==2015]
d = dict(zip(ppp_data.iso_3, ppp_data.PPP))

data['ppp_2015'] = data['GID_0'].apply(lambda x: d.get(x))
data.loc[data.GID_0=='USA','ppp_2015'] = len(data.loc[data.GID_0=='USA']) * [1]

# Load FX data and create a dictionary
fx_data = pd.read_excel(deflator_path + 'fx_data_all_countries.xlsx')
fx_dict = fx_data.set_index(['iso_3', 'year'])['fx'].to_dict()
data['fx'] = data.apply(lambda row: fx_dict.get((row['GID_0'], row['year']), np.nan), axis=1)
data.loc[data['GID_0'] == 'USA', 'fx'] = 1

# Normalize zmean (Zhang Output) to constant 2015 USD
data['Z2024_GRP_pc_usd_2015'] = data['zmean'] * 100 / data['deflator_2015_us']

fx_data = fx_data.loc[fx_data.year==2015]
d = dict(zip(fx_data.iso_3, fx_data.fx))
data['fx_2015'] = data['GID_0'].apply(lambda x: d.get(x))

data['Z2024_GRP_pc_lcu'] = data['zmean'] * data['fx']
data['Z2024_GRP_pc_lcu_2015'] = data['Z2024_GRP_pc_lcu']* 100 / data['fx_2015']
data['Z2024_GRP_pc_lcu2015_usd'] = data['Z2024_GRP_pc_lcu_2015'] / data['deflator_2015']

data['Z2024_GRP_pc_lcu2015_ppp'] = data['Z2024_GRP_pc_lcu'] / data['deflator_2015'] * 100 / data['ppp_2015']

# Normalize zmean (Zhang Output) to PPP 2015
#data['zmean_lcu'] = data['zmean'] * data['fx']
data['Z2024_GRP_pc_ppp'] = data['Z2024_GRP_pc_lcu'] / data['ppp']
data['Z2024_GRP_pc_ppp_2015'] = data['Z2024_GRP_pc_ppp'] * 100 / data['deflator_2015_us']

data = data.rename(columns={'zmean': 'Z2024_pc'})
data = data.sort_values(by=['GID_0', 'GID_1'])
data = data.reset_index(drop=True)
# print(data[['GID_0', 'GID_1', 'year', 'Z2024_GRP_pc_current_usd', 'Z2024_GRP_pc_ppp_2015', 'Z2024_GRP_pc_usd_2015', 'Z2024_GRP_pc_lcu2015_usd', 'Z2024_GRP_pc_lcu2015_ppp']].head(20))

# Iterate over columns and create total GRP versions by multiplying by population
data['Z2024'] = data['Z2024_pc'] * data['pop']
for col in data.columns:
    if '_pc_' in col:  # Check if the column name contains '_pc_'
        new_col = col.replace('_pc_', '_')  # Create the new column name
        # Multiply by 'pop', allowing NaN to propagate
        data[new_col] = data[col] * data['pop']

unique_gid_1 = data['GID_1'].unique()
print(f"Unique GID_1 values ({len(unique_gid_1)} total):")
print(unique_gid_1)

num_rows2 = len(data)
print(f"Number of rows in data after conversions: {num_rows2}")
data = data.drop_duplicates(subset=['year', 'GID_1'], keep='first')
print(f"Number of rows after removing duplicate year and GID_1 combinations: {len(data)}")

pickle_path = data_path + 'pickle/'

# Save the DataFrame to a pickle file
data[['GID_0', 'GID_1', 'year', 'Z2024_pc','Z2024_GRP_pc_lcu', 'Z2024_GRP_pc_lcu_2015','Z2024_GRP_pc_lcu2015_usd', 'Z2024_GRP_pc_ppp_2015', 'Z2024_GRP_pc_lcu2015_ppp','Z2024','Z2024_GRP_lcu', 'Z2024_GRP_lcu_2015','Z2024_GRP_lcu2015_usd', 'Z2024_GRP_ppp_2015', 'Z2024_GRP_lcu2015_ppp']].to_pickle(pickle_path + 'Z2024_data.pkl')


# Filter the data for GID_0 = 'EGY'
egy_data = data[data['GID_0'] == 'EGY']

# Save the filtered data to a separate pickle file in the 'EGY' folder
egy_pickle_path = pickle_path + 'EGY/'

egy_data[['GID_0', 'GID_1', 'year', 'Z2024_pc', 'Z2024_GRP_pc_lcu', 'Z2024_GRP_pc_lcu_2015',
          'Z2024_GRP_pc_lcu2015_usd', 'Z2024_GRP_pc_ppp_2015', 'Z2024_GRP_pc_lcu2015_ppp',
          'Z2024', 'Z2024_GRP_lcu', 'Z2024_GRP_lcu_2015', 'Z2024_GRP_lcu2015_usd',
          'Z2024_GRP_ppp_2015', 'Z2024_GRP_lcu2015_ppp']
         ].to_pickle(egy_pickle_path + 'Z2024_data_EGY.pkl')
