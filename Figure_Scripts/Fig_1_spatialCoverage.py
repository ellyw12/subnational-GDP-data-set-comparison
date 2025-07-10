"""
Using the pre-processed and merged data from the prep files, this script determines the spatial coverage of 
each dataset by plotting the regions where data is available.

This includes manually entered regions from Kummu et al. 2025 that are narrower than admin level 1. 
Note also that Zhang et al. contains GID_1 regions (MAR, PHL, NPL) that are not able to be converted to
DOSE regions and are therefore not included in the final plot.

Author: BW 
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib import colormaps
import pandas as pd


data_path = './Data/'
graphics_path = './Figures/'

data_dose_pop = 'merged_dose_pop.csv'
dose_pop_path = data_path + data_dose_pop
dose_pop_data = pd.read_csv(dose_pop_path)

data_isimip_pop = 'merged_isimip_pop.csv'
isimip_pop_path = data_path + data_isimip_pop
isimip_pop_data = pd.read_csv(isimip_pop_path)


'''Toggle between the datasets by commenting in/out the desired one'''

data = dose_pop_data
# data = isimip_pop_data

# Load spatial data (shapefile) for subnational regions
gadm_path = data_path + 'spatial data/'
map_data = 'gadm_custom_merged.shp'
maps = gpd.read_file(gadm_path + map_data)
maps = maps.set_index('GID_1')


# Define the columns to check for data presence
data_columns = [
    ('grp_lcu'),       # DOSE original unit 
    ('K2025_pc'),      # Z2024 and K2025 are given in per capita values while WS and C are total grp
    ('Z2024_pc'),
    ('WS2022'),
    ('C2022')
]

######################
"""
#Kummu has values spatially narrower than admin level 1 - we do not use these narrower values since they do not 
compare to reported data.

In order to show they exist, we insert values for these specific regions so that they will appear on the map. 
"""

# List of GID_1 prefixes for regions narrower than admin level 1
narrower_regions = ['AUT', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 
                    'IRL', 'ITA', 'MAR', 'NGA', 'NLD', 'NPL', 'POL', 'SRB', 'USA']

# Add an arbitrary value (123) to the Kummu_pc column for regions starting with narrower_regions
data['GID_1'] = data['GID_1'].fillna('')  # Replace NaN with empty strings
data.loc[data['GID_1'].str.startswith(tuple(narrower_regions)), 'K2025_pc'] = 123

########################

# Create a figure with layout (2 rows, 2 columns)
fig = plt.figure(figsize=(20, 12)) 
gs = fig.add_gridspec(2,2)  # 2 row, 2 columns

axes = [
    fig.add_subplot(gs[0, :]),  # DOSE
    fig.add_subplot(gs[1, 0]),  # Kummu
    fig.add_subplot(gs[1, 1])   # Global datasets
]
plt.subplots_adjust(hspace=0.000001)  # Reduce vertical space between rows

# Add labels 'a', 'b', 'c' to the panels
axes[0].text(-0.05, 1.05, 'A', transform=axes[0].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
axes[1].text(-0.05, 1.05, 'B', transform=axes[1].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
axes[2].text(-0.05, 1.05, 'C', transform=axes[2].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

panel_titles = [
    "DOSE v2 - Spatial Coverage of Reported Data",
    "Kummu et al., 2025 - K2025",
    "Global Data Sets Z2024, C2022, WS2022"
]

# Colors for the first two plots
panel_colors = ['Reds', 'Oranges']

# Iterate over the first two datasets (DOSE and Kummu) to create the plots
for i, (col, ax, color, title) in enumerate(zip(data_columns[:2], axes[:2], panel_colors, panel_titles[:2])):
    # Add a column to maps indicating whether the dataset has data for each region
    maps[f'{col}_has_data'] = maps.index.map(lambda gid: data[data['GID_1'] == gid][col].notna().any())
    
    cmap = colormaps.get_cmap(color) 
    color_total = cmap(0.8)  
    lighter_orange = cmap(0.4)  # Define a lighter orange color for specific regions

    # For the Kummu plot, shade specific regions with a lighter orange
    if col == 'K2025_pc':  # Apply this logic only to the Kummu plot
        # List of GID_1 prefixes for regions narrower than admin level 1
        narrower_regions = ['AUT', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 
                            'IRL', 'ITA', 'MAR', 'NGA', 'NLD', 'NPL', 'POL', 'SRB', 'USA']
        
        data = data[data['GID_1'].notna()]
        narrower_data_regions = data['GID_1'].str.startswith(tuple(narrower_regions))
        narrower_gids = data.loc[narrower_data_regions, 'GID_1'].unique()
        
        maps['narrower_than_admin1'] = maps.index.isin(narrower_gids)
        
        # Plot regions narrower than admin level 1 in lighter orange
        maps[maps['narrower_than_admin1']].plot(
            color=lighter_orange,
            linewidth=0.15,
            edgecolor='black',
            ax=ax,
            legend=False
        )
    else:
        maps['narrower_than_admin1'] = False
    
    # Plot the main dataset regions in the default color
    maps[maps[f'{col}_has_data'] & ~maps['narrower_than_admin1']].plot(
        color=color_total,
        linewidth=0.15,
        edgecolor='black',
        ax=ax,
        legend=False
    )
    
    # Plot missing regions in light grey
    missing_data = maps[~maps[f'{col}_has_data']]
    if not missing_data.empty:  # Check if there are any missing regions to plot
        missing_data.plot(
            color='lightgrey',
            linewidth=0.15,
            edgecolor='black',
            ax=ax
        )
    
    # ax.set_title(title, fontsize=16)
    ax.axis('off')

    
# Define legends for Panel A and Panel B
    legend_patches = []
    if col == 'grp_lcu':  # Panel A: DOSE
        legend_patches = [
            mpatches.Patch(color=color_total, label="DOSE lcu data available"),
            mpatches.Patch(color='lightgrey', label="No Data")
        ]
    elif col == 'K2025_pc':  # Panel B: Kummu
        legend_patches = [
            mpatches.Patch(color=color_total, label="K2025"),
            mpatches.Patch(color=lighter_orange, label="K2025 Narrower than GID_1"),
            mpatches.Patch(color='lightgrey', label="No Data")
        ]
    
    # Add the legend to the axis
    ax.legend(handles=legend_patches, loc='lower left', fontsize=20, frameon=True, handlelength=1.2, borderpad=0.5, labelspacing=0.4)

# Create the third plot for global datasets
global_ax = axes[2]

# Add a column to maps indicating the presence of data in global datasets
maps['global_data'] = maps.index.map(lambda gid: (
    data[data['GID_1'] == gid]['Z2024_pc'].notna().any(),
    data[data['GID_1'] == gid]['C2022'].notna().any(),
    data[data['GID_1'] == gid]['WS2022'].notna().any()
))

# Define colors for each condition
def get_global_color(has_z2024, has_c2022, has_ws2022):
    if has_z2024 and has_c2022 and has_ws2022:
        return 'purple'  # All three datasets
    elif has_z2024 and has_c2022:
        return 'yellow'  # Z2024 and C2022
    elif has_z2024 and has_ws2022:
        return 'yellow'  # Z2024 and WS2022
    elif has_c2022 and has_ws2022:
        return 'green'  # C2022 and WS2022
    elif has_z2024:
        return 'pink'  # Only Z2024
    elif has_c2022:
        return 'pink'  # Only C2022
    elif has_ws2022:
        return 'pink'  # Only WS2022
    else:
        return 'lightgrey'  # No data


maps['global_color'] = maps['global_data'].apply(lambda x: get_global_color(*x))
maps.plot(
    color=maps['global_color'],
    linewidth=0.15,
    edgecolor='black',
    ax=global_ax,
    legend=False
)

# global_ax.set_title(panel_titles[2], fontsize=12)
global_ax.axis('off')

global_legend_patches = [
    mpatches.Patch(color='purple', label='Z2024, C2022, and WS2022'),
    mpatches.Patch(color='green', label='WS2022 and C2022'),
    # mpatches.Patch(color='pink', label='Only Z2024, C2022, or WS2022'),
    mpatches.Patch(color='lightgrey', label='No data')
]
global_ax.legend(handles=global_legend_patches, loc='lower left', fontsize=20, frameon=True, handlelength=1.2, borderpad=0.5, labelspacing=0.4)

output_file = f"{graphics_path}data_coverage_plot_3panels.png"
plt.savefig(output_file, format='png', bbox_inches='tight', dpi=300)