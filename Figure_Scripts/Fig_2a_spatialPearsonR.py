"""
To expand spatial coverage, the ISIMIP population data aligns GRP and GRP per capita values across the data sets.
Calculations consider the correlation pairs as described in the currency conversion section, and they produce
the mean of the available correlation pairs for that sub-national region. 

Author: BW 
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

data_path = './Data/'

data_dose_pop = 'merged_dose_pop.csv'
dose_pop_path = data_path + data_dose_pop
dose_pop_data = pd.read_csv(dose_pop_path)

data_isimip_pop = 'merged_isimip_pop.csv'
isimip_pop_path = data_path + data_isimip_pop
isimip_pop_data = pd.read_csv(isimip_pop_path)



'''Toggle between the datasets by commenting in/out the desired one'''

# data = isimip_pop_data
data = dose_pop_data

print(data.columns)
print("Current index:", data.index)


# Load spatial data (shapefile) for subnational regions
gadm_path = data_path + 'spatial data/'
map_data = 'gadm_custom_merged.shp'
maps = gpd.read_file(gadm_path + map_data)

# Set index to 'GID_1' for easy merging
maps = maps.set_index('GID_1')

# Define correlation pairs
'''Select GRP per capita or total GRP by commenting in/out the desired one. Note: Rename file if you want to save both.'''

##GRP per capita##
# correlation_pairs = {
#     ("DOSE", "K2025"): ("grp_pc_lcu_2017", "K2025_grp_pc_lcu_2017"),
#     ("DOSE", "Z2024"): ("grp_pc_usd", "Z2024_pc"),
#     ("DOSE", "C2022"): ("grp_pc_ppp", "C2022_grp_pc_ppp"),
#     ("DOSE", "WS2022"): ("grp_pc_ppp", "WS2022_grp_pc_ppp"),
#     # ("K2025", "Z2024"): ("K2025_grp_pc_lcu_2017", "Z2024_grp_pc_lcu_2017"),
#     # ("K2025", "C2022"): ("K2025_pc", "C2022_pc"),
#     # ("K2025", "WS2022"): ("K2025_grp_pc_ppp", "WS2022_grp_pc_ppp"),
#     # ("Z2024", "C2022"): ("Z2024_grp_pc_ppp", "C2022_grp_pc_ppp"),
#     # ("Z2024", "WS2022"): ("Z2024_grp_pc_ppp", "WS2022_grp_pc_ppp"),
#     # ("C2022", "WS2022"): ("C2022_grp_pc_ppp", "WS2022_grp_pc_ppp")
# }

 ##GRP total##
correlation_pairs = { 
    ("DOSE", "K2025"): ("grp_lcu_2017", "K2025_grp_lcu_2017"),
    ("DOSE", "Z2024"): ("grp_usd", "Z2024"),
    ("DOSE", "C2022"): ("grp_ppp", "C2022_grp_ppp"),
    ("DOSE", "WS2022"): ("grp_ppp", "WS2022_grp_ppp"),
    # ("K2025", "Z2024"): ("K2025_grp_pc_lcu_2017", "Z2024_grp_pc_lcu_2017"),
    # ("K2025", "C2022"): ("K2025_pc", "C2022_pc"),
    # ("K2025", "WS2022"): ("K2025_grp_pc_ppp", "WS2022_grp_pc_ppp"),
    # ("Z2024", "C2022"): ("Z2024_grp_pc_ppp", "C2022_grp_pc_ppp"),
    # ("Z2024", "WS2022"): ("Z2024_grp_pc_ppp", "WS2022_grp_pc_ppp"),
    # ("C2022", "WS2022"): ("C2022_grp_pc_ppp", "WS2022_grp_pc_ppp")
}

# Calculate Pearson R for each correlation pair within each subnational region
pearson_results = []

for gid, group in data.groupby('GID_1'):
    pearson_values = []
    for (dataset1, dataset2), (col1, col2) in correlation_pairs.items():
        if col1 in group.columns and col2 in group.columns:
            # Filter out rows with missing values for the relevant columns
            valid_data = group[[col1, col2]].dropna()
            
            if not valid_data.empty:
                # Compute Pearson correlation
                correlation_value = valid_data[col1].corr(valid_data[col2])
                pearson_values.append(correlation_value)
    
    if pearson_values:
        # Compute the average Pearson R for the region
        avg_pearson = np.mean(pearson_values)
        pearson_results.append({'GID_1': gid, 'avg_pearson': avg_pearson})

# Convert Pearson R results to DataFrame
pearson_df = pd.DataFrame(pearson_results)

# Merge Pearson R results with spatial data
maps = maps.reset_index()
maps = maps.merge(pearson_df, on='GID_1', how='left')
maps = maps.set_index('GID_1')

# Define a diverging colormap (blue for positive, red for negative)
# cmap = plt.cm.bwr_r

# Define a custom colormap with more intermediate colors
colors = ["darkred", "red", "pink", "white", "lightblue", "blue", "darkblue"]  # Only include color names
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
cmap = custom_cmap
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)


# Create a normalization object for Pearson R (-1 to 1)
# norm = Normalize(vmin=-1, vmax=1)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot each region with the appropriate color
for idx, row in maps.iterrows():
    avg_pearson = row['avg_pearson']
    if pd.isna(avg_pearson):
        color = 'grey'  # Grey for missing data
    else:
        color = cmap(norm(avg_pearson))  # Automatically normalize Pearson R (-1 to 1)
    gpd.GeoSeries([row.geometry]).plot(ax=ax, color=color, edgecolor='black', linewidth=0.1)

# Set title and display
# ax.set_title("Average Pearson R Across Datasets")
ax.text(0.01, 0.98, 'A', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
ax.axis('off')

# Create a color bar with the diverging colormap
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5)  # Adjust the shrink parameter to make the legend smaller
cbar.set_label('Average Pearson R')

# Create grey label for "No Data"
no_data_patch = mpatches.Patch(color='grey', label='No Data')
ax.legend(handles=[no_data_patch], loc='lower left', fontsize=10, frameon=True)

# Save the plot as a .png file
graphics_path = './Figures/'
if ("C2022", "WS2022") in correlation_pairs:
    output_file = f"{graphics_path}/spatial_pearson_r.png" #all correlation pairs 
else:
    output_file = f"{graphics_path}/DOSE_only_spatial_pearson_r.png" #DOSE only correlation pairs
plt.savefig(output_file, format='png', bbox_inches='tight')

plt.show()

# print gid_1 with the lowest average Pearson R to better specify 
# lowest_correlating_regions = pearson_df.sort_values(by='avg_pearson', ascending=True).head(120)
# print("30 Lowest Correlating GID_1 Regions:")
# print(lowest_correlating_regions[['GID_1', 'avg_pearson']])
# lowest_correlating_regions[['GID_1', 'avg_pearson']].to_csv('./Data/lowest_correlating_regions.csv', index=False)
