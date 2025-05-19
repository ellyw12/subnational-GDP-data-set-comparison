"""
Same as Fig 2a but for the spatial RMSPD rather than Pearson R. 

Author: BW 
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib.colors import Normalize

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

# Load spatial data (shapefile) for subnational regions
gadm_path = data_path + 'spatial data/'
map_data = 'gadm_custom_merged.shp'
maps = gpd.read_file(gadm_path + map_data)

# Set index to 'GID_1' for easy merging
maps = maps.set_index('GID_1')

'''Select GRP per capita or total GRP by commenting in/out the desired one. Note: Rename file if you want to save both.'''

# Define correlation pairs

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


'''RMSPD Calculation'''

def calculate_rmspd(x, y):
    #elly changed syntax becauc this is a numpy array to fit with the rest of the code
    x = x.flatten() if x.ndim > 1 else x
    y = y.flatten() if y.ndim > 1 else y

    valid = ~np.isnan(x) & ~np.isnan(y) & (x + y != 0)
    x = x[valid]
    y = y[valid]
    rel_diff_sq = ((x - y) / ((x + y) / 2)) ** 2
    return np.sqrt(rel_diff_sq.mean()) * 100


# Calculate RMSPD for each correlation pair within each subnational region
rmspd_results = []

for gid, group in data.groupby('GID_1'):
    rmspd_values = []
    for (dataset1, dataset2), (col1, col2) in correlation_pairs.items():
        if col1 in group.columns and col2 in group.columns:
            valid_data = group[[col1, col2]].dropna()
            if not valid_data.empty:
                x = valid_data[col1].values.reshape(-1, 1)
                y = valid_data[col2].values
                if len(x) > 1 and len(y) > 1:
                    rmspd = calculate_rmspd(x, y)
                    rmspd_values.append(rmspd)
    if rmspd_values:
        avg_rmspd = np.mean(rmspd_values)
        rmspd_results.append({'GID_1': gid, 'avg_rmspd': avg_rmspd})

# Convert RMSPD results to DataFrame
rmspd_df = pd.DataFrame(rmspd_results)

print(f"RMSPD Range: {rmspd_df['avg_rmspd'].min()} to {rmspd_df['avg_rmspd'].max()}")

# Calculate the full range and normalize for colormap
rmspd_min = rmspd_df['avg_rmspd'].min()
rmspd_max = rmspd_df['avg_rmspd'].max()
print(f"RMSPD Range: {rmspd_min} to {rmspd_max}")
norm = Normalize(vmin=rmspd_min, vmax=rmspd_max)



# Merge RMSPD results with spatial data
maps = maps.reset_index()
maps = maps.merge(rmspd_df, on='GID_1', how='left')
maps = maps.set_index('GID_1')

# Define a colormap
cmap = plt.cm.Reds_r  # Use the reversed Reds colormap - low rmspd will be dark, high will be light

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot each region with the appropriate color
for idx, row in maps.iterrows():
    avg_rmspd = row['avg_rmspd']
    if pd.isna(avg_rmspd):
        color = 'grey'
    else:
        color = cmap(norm(avg_rmspd)) 
        # color = cmap(avg_rmspd / 100)  # Normalize to [0, 1] for colormap
    gpd.GeoSeries([row.geometry]).plot(ax=ax, color=color, edgecolor='black', linewidth=0.10)


#Plot details 
ax.axis('off')
# ax.set_title("Average RMSPD Across Datasets")
ax.text(0.01, 0.98, 'B', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
norm = Normalize(vmin=rmspd_min, vmax=rmspd_max)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5)  # Adjust the shrink parameter to make the legend smaller
cbar.set_label('Average RMSPD')

no_data_patch = mpatches.Patch(color='grey', label='No Data')
ax.legend(handles=[no_data_patch], loc='lower left', fontsize=10, frameon=True)

# Save the plot as a .png file
graphics_path = './Figures/'
if ("C2022", "WS2022") in correlation_pairs:
    output_file = f"{graphics_path}/spatial_rmspd.png" #all correlation pairs 
else:
    output_file = f"{graphics_path}/DOSE_only_rmspd_GRP_total.png" #DOSE only correlation pairs
plt.savefig(output_file, format='png', bbox_inches='tight')

plt.show()