import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from   scipy.stats import pearsonr
import numpy as np


data_path               =   r'C:\Users\Leon Ließem\OneDrive\Desktop\Master INRM\DOSE replication files\DOSE replication files\Data'
graphics_path           =   r'C:\Users\Leon Ließem\OneDrive\Desktop\Master INRM\DOSE replication files\DOSE replication files\Graphics'
deflator_path           =    data_path +'\\deflator/'
suleiman_path           =    data_path +'\\suleiman/'  # Add the path to the new dataset folder
gadm_path               =    data_path + '\\spatial data/'


# File names:
dose_v2                 =   'DOSE_V2.9.csv'
gennaioli               =   '10887_2014_9105_MOESM1_ESM.xlsx'
suleiman_data           =   'Illuminating_the_Nile_NTL_GDP__1992_2021.xlsx'  # Add the filename of the new dataset
egypt_population        =   'Egypt_population_2010_2019.xlsx'
map_data                =   'gadm_custom_merged.shp'


# Load the datasets
dose                    =   pd.read_csv(data_path+'\\'+dose_v2)
suleiman                =   pd.read_excel(suleiman_path + '\\' + suleiman_data, engine='openpyxl').rename(columns={'GDP_current_EGP':'S2024'})  
egypt_population_data   =   pd.read_excel(data_path + '\\' + egypt_population , sheet_name='Panel Data')
maps                    =   gpd.read_file(gadm_path+'\\'+map_data)


data=dose


# Set index to 'GID_1' for easy merging
maps = maps.set_index('GID_1')


custom_regions = {'EGY': ['EGY']}


# Extract region maps (e.g. Egypt)
region_maps = pd.DataFrame()
for region, countries in custom_regions.items():
    region_data = data[data['GID_0'].isin(countries)]
    region_gids = region_data['GID_1'].unique()
    region_maps = maps[maps.index.isin(region_gids)].copy()

# --- Fix and streamline population merging and S2024_pc calculation ---


# Merge population data into DOSE (adds pop where missing)
merged_data = pd.merge(dose, egypt_population_data[['region', 'year', 'pop']],
                       on=['region', 'year'], how='left', suffixes=('', '_pop'))


# Update the 'pop' column in DOSE only where it's NaN
dose['pop'] = dose['pop'].combine_first(merged_data['pop_pop'])


# Filter to Egypt after population is properly handled
dose = dose[dose['GID_0'] == 'EGY'].copy()


# Drop any potential duplicates by GID_1 + year
dose = dose.drop_duplicates(subset=['GID_1', 'year'])


# Set index for merging
dose.set_index(['GID_1', 'year'], inplace=True)


# Add a new S2024 column and update with Suleiman values
dose['S2024'] = np.nan
suleiman_series = suleiman.set_index(['GID_1', 'year'])['S2024']
dose.update(suleiman_series)


# Calculate per capita S2024
dose['S2024_pc'] = dose['S2024'] / dose['pop']


# Reset index for further processing
dose.reset_index(inplace=True)


# Store cleaned-up Egypt data in `data` for the rest of the script
data = dose.copy()



import pickle


folder_path = r'C:\Users\Leon Ließem\OneDrive\Desktop\Master INRM\DOSE replication files\DOSE replication files\Data\Egypt intercomp'
pkl_files = ['C2022_EGY_data.pkl', 'K2025_EGY_data.pkl', 'Z2024_EGY_data.pkl', 'WS2022_EGY_data.pkl']


# Load each .pkl file and merge it with the data dataframe
for pkl_file in pkl_files:
    file_path = os.path.join(folder_path, pkl_file)
    with open(file_path, 'rb') as file:
        pkl_data = pickle.load(file)
        # Drop unnecessary columns before merging
        pkl_data = pkl_data.drop(columns=['GID_0'], errors='ignore')
        data = pd.merge(data, pkl_data, on=['GID_1', 'year'], how='left')


data = data.drop(columns=['GID_0_x', 'GID_0_y'], errors='ignore')


# Load World Bank GDP deflator data
deflators = pd.read_excel(deflator_path+'2022_03_30_WorldBank_gdp_deflator.xlsx', sheet_name='Data',
                            index_col=None, usecols='B,E:BM', skiprows=3).set_index('Country Code')
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
   
deflators_2005 = deflators.copy()
column2005 = deflators_2005.columns.get_loc(2005)
for i in range(len(deflators_2005)):
    deflators_2005.iloc[i, :] = (deflators_2005.iloc[i, :] / deflators_2005.iloc[i, column2005]) * 100


# Make separate dataframes for the US and local GDP deflators, both stacked
deflators_2015_us = pd.DataFrame(deflators_2015.loc[deflators_2015.index=='USA'].stack()
                                 ).rename(columns={0:'deflator_2015_us'}
                                          ).reset_index().set_index('level_1')
deflators_2015 = pd.DataFrame(deflators_2015.stack()).rename(columns={0:'deflator_2015'})


deflators_2017_us = pd.DataFrame(deflators_2017.loc[deflators_2017.index=='USA'].stack()
                                 ).rename(columns={0:'deflator_2017_us'}
                                          ).reset_index().set_index('level_1')
deflators_2017 = pd.DataFrame(deflators_2017.stack()).rename(columns={0:'deflator_2017'})


deflators_2005_us = pd.DataFrame(deflators_2005.loc[deflators_2005.index=='USA'].stack()
                                 ).rename(columns={0:'deflator_2005_us'}
                                          ).reset_index().set_index('level_1')
deflators_2005 = pd.DataFrame(deflators_2005.stack()).rename(columns={0:'deflator_2005'})


# Add the deflators to the 'data' dataframe
data['deflator_2015'] = np.nan
data['deflator_2015_us'] = np.nan
data['deflator_2017'] = np.nan
data['deflator_2017_us'] = np.nan
data['deflator_2005'] = np.nan
data['deflator_2005_us'] = np.nan


print(data.columns)


data = data.set_index(['GID_0','year'])
data.update(deflators_2015)
data.update(deflators_2017)
data.update(deflators_2005)
data = data.reset_index('GID_0')
data.update(deflators_2015_us)
data.update(deflators_2017_us)
data.update(deflators_2005_us)
print(data.columns)


# Load PPP conversion factors for both relevant years ...
# ... and add both sets of factors for each country to 'data'
ppp_data = pd.read_excel(deflator_path+'\\'+'ppp_data_all_countries.xlsx')
ppp_data_2015 = ppp_data.loc[ppp_data.year==2015]
ppp_data_2017 = ppp_data.loc[ppp_data.year==2017]
ppp_data_2005 = ppp_data.loc[ppp_data.year==2005]
d2015 = dict(zip(ppp_data_2015.iso_3, ppp_data_2015.PPP))
d2017 = dict(zip(ppp_data_2017.iso_3, ppp_data_2017.PPP))
d2005 = dict(zip(ppp_data_2005.iso_3, ppp_data_2005.PPP))
data['ppp_2015'] = data['GID_0'].apply(lambda x: d2015.get(x))
data['ppp_2017'] = data['GID_0'].apply(lambda x: d2017.get(x))
data['ppp_2005'] = data['GID_0'].apply(lambda x: d2005.get(x))
data.loc[data.GID_0 == 'USA', 'ppp_2015'] = 1
data.loc[data.GID_0 == 'USA', 'ppp_2017'] = 1
data.loc[data.GID_0 == 'USA', 'ppp_2005'] = 1
data['ppp_2015'] = pd.to_numeric(data['ppp_2015'], errors='coerce')
data['ppp_2017'] = pd.to_numeric(data['ppp_2017'], errors='coerce')
data['ppp_2005'] = pd.to_numeric(data['ppp_2005'], errors='coerce')


# Load MER conversion factors for 2015 and add them for each country to 'data'
fx_data = pd.read_excel(deflator_path+'\\'+'fx_data_all_countries.xlsx')
fx_data = fx_data.loc[fx_data.year==2015]
d = dict(zip(fx_data.iso_3, fx_data.fx))
data['fx_2015'] = data['GID_0'].apply(lambda x: d.get(x))
data.loc[data.GID_0=='USA','fx_2015'] = len(data.loc[data.GID_0=='USA']) * [1]

#Conversion of dose values for comparison
data['grp_pc_ppp'] = data['grp_pc_lcu'] / data.PPP


data['grp_pc_ppp_2017'] = data['grp_pc_ppp'] *100 / data.deflator_2017_us
data['grp_pc_lcu2017_ppp'] = data.grp_pc_lcu *100 / data.deflator_2017 / data.ppp_2017


data['grp_pc_ppp_2005'] = data['grp_pc_ppp'] *100 / data.deflator_2005_us
data['grp_pc_lcu2005_ppp'] = data.grp_pc_lcu *100 / data.deflator_2005 / data.ppp_2005


# Fill missing values in C2022_pc using C2022 / pop
data["C2022_pc"] = data.apply(
    lambda row: row["C2022"] / row["pop"] if pd.isna(row["C2022_pc"]) else row["C2022_pc"], axis=1
)


# Fill missing values in WS2022_pc using WS2022 / pop
data["WS2022_pc"] = data.apply(
    lambda row: row["WS2022"] / row["pop"] if pd.isna(row["WS2022_pc"]) else row["WS2022_pc"], axis=1
)

# Define correlation pairs
correlation_pairs = {
("DOSE", "K2025"): ("grp_pc_lcu2017_ppp", "K2025_pc"),
("DOSE", "Z2024"): ("grp_pc_usd", "Z2024_pc"),
("DOSE", "C2022"): ("grp_pc_lcu2017_ppp", "C2022_pc"),
("DOSE", "WS2022"): ("grp_pc_lcu2005_ppp", "WS2022_pc"),
("DOSE", "S2024"): ("grp_pc_lcu", "S2024_pc")
}


# New dictionary: stores metrics per pair and region
all_results = {pair: {} for pair in correlation_pairs}



# Define metric calculation again (if missing)
def calculate_metrics(x, y):
    if len(x) > 1 and len(y) > 1:
        pearson_r, _ = pearsonr(x, y)
        with np.errstate(divide='ignore', invalid='ignore'):
            denominator = (x + y) / 2
            rel_diff_sq = ((x - y) / denominator) ** 2
            rmspd = np.sqrt(np.nanmean(rel_diff_sq)) * 100
        return pearson_r, rmspd
    else:
        return np.nan, np.nan


for (dataset1, dataset2), (col1, col2) in correlation_pairs.items():
    print(f"\nProcessing: {col1} vs {col2}")
   
    for gid, group in data.groupby('GID_1'):
        if col1 in group.columns and col2 in group.columns:
            # Drop rows with NaNs in either column
            valid_data = group[[col1, col2]].dropna()
           
            if len(valid_data) > 1:  # Minimum 2 rows required for correlation
                x = valid_data[col1].values
                y = valid_data[col2].values
               
                try:
                    pearson_r, rmspd = calculate_metrics(x, y)
                    all_results[(dataset1, dataset2)][gid] = {
                        'pearson_r': pearson_r,
                        'rmspd': rmspd
                    }
                except Exception as e:
                    print(f"[ERROR] Correlation failed for {gid} in {col1} vs {col2}: {e}")
            else:
                print(f"[SKIP] {gid} has insufficient data for {col1} vs {col2}")
        else:
            print(f"[MISSING] {gid} is missing one or both columns: {col1}, {col2}")
#Pearson R


import matplotlib.image as mpimg


# Define datasets to plot and image path
custom_order = [("DOSE", "S2024"), ("DOSE", "K2025"), ("DOSE", "Z2024"),
                ("DOSE", "C2022"), ("DOSE", "WS2022")]
image_path = os.path.join(graphics_path, "Egypt_pop_density.png")  # Replace with actual filename


# Set up 3x2 subplot grid
fig, axes = plt.subplots(3, 2, figsize=(12, 14))  # Wider for spacing
axes = axes.flatten()


# Plot the 5 correlation maps
for ax, (dataset1, dataset2) in zip(axes[:5], custom_order):
    key = (dataset1, dataset2)
    plot_column = region_maps.index.map(
        lambda gid: all_results[key].get(gid, {}).get('pearson_r', np.nan)
    )


    region_maps.assign(pearson_r=plot_column).plot(
        column='pearson_r',
        cmap='Blues',
        linewidth=0.5,
        ax=ax,
        edgecolor='0.6',
        legend=True,
        legend_kwds={'shrink': 0.8, 'label': 'Pearson R'},
        missing_kwds={"color": "#f0f0f0"},
        vmin=-1,
        vmax=1
    )


    ax.set_title(f"{dataset2} vs. DOSE", fontsize=13)
    ax.axis('off')  # Hide axis ticks and borders for clean look


# Show the external PNG image on the last subplot
img = mpimg.imread(image_path)
axes[5].imshow(img)
axes[5].axis('off')
axes[5].set_title("Map - Population density", fontsize=13)


# Final formatting and save
plt.suptitle("Pearson R | DOSE vs. Each Dataset (Egypt)", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(graphics_path, "EGY_pearson_r_all_datasets_for_paper.png"), dpi=300, bbox_inches='tight')
plt.close()




# Alternate for RMSPD:

#RMSPD


# Define datasets to plot and image path
custom_order = [("DOSE", "S2024"), ("DOSE", "K2025"), ("DOSE", "Z2024"),
                ("DOSE", "C2022"), ("DOSE", "WS2022")]
image_path = os.path.join(graphics_path, "Egypt_pop_density.png")  # Replace with actual filename


# Set up 3x2 subplot grid
fig, axes = plt.subplots(3, 2, figsize=(12, 14))  # Wider for spacing
axes = axes.flatten()


# Plot the 5 correlation maps
for ax, (dataset1, dataset2) in zip(axes[:5], custom_order):
    key = (dataset1, dataset2)
    plot_column = region_maps.index.map(
        lambda gid: all_results[key].get(gid, {}).get('rmspd', np.nan)
    )


    region_maps.assign(rmspd=plot_column).plot(
        column='rmspd',
        cmap='OrRd',
        linewidth=0.5,
        ax=ax,
        edgecolor='0.6',
        legend=True,
        legend_kwds={'shrink': 0.8, 'label': 'RMSPD (%)'},
        missing_kwds={"color": "#f0f0f0"},
        vmin=0,
        vmax=100
    )


    ax.set_title(f"{dataset2} vs. DOSE", fontsize=13)
    ax.axis('off')  # Hide axis ticks and borders for clean look


# Show the external PNG image on the last subplot
img = mpimg.imread(image_path)
axes[5].imshow(img)
axes[5].axis('off')
axes[5].set_title("Map - Population density", fontsize=13)


# Final formatting and save
plt.suptitle("RMSPD | DOSE vs. Each Dataset (Egypt)", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(graphics_path, "EGY_rmspd_all_datasets_for_paper.png"), dpi=300, bbox_inches='tight')
plt.close()


