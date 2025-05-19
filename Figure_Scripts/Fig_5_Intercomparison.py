'''
In this script pre-processed data from selected sets are combined, and
agreement among them is measured and displayed into a heatmap.


NEW VERSION containing GRP total and p.c., measured both in pearson R and RMSPD and with different colour scales

Authors: Leon Liessem, Matthias Zarama Giedion, Luuk Staal, Joshua Arky
'''

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_path = './Data/'

data_dose_pop = 'merged_dose_pop.csv'
dose_pop_path = data_path + data_dose_pop
dose_pop_data = pd.read_csv(dose_pop_path)

data_loglog = 'merged_loglog.csv'
dose_log_data = pd.read_csv(data_path+data_loglog)

data_isimip_pop = 'merged_isimip_pop.csv'
isimip_pop_path = data_path + data_isimip_pop
isimip_pop_data = pd.read_csv(isimip_pop_path)

'''Toggle between the datasets by commenting in/out the desired one'''

# data = isimip_pop_data
#data = dose_log_data
data = dose_pop_data


# """ 3. Perform GRP conversion and deflation calculations """




# DOSE (current lcu)
data['grp_pc_ppp_2017'] = data['grp_pc_lcu'] * 100 / data['deflator_2017'] / data['ppp_2017']
data['grp_pc_ppp_2005'] = data['grp_pc_lcu'] * 100 / data['deflator_2005'] / data['ppp_2005']
# grp_pc_usd already exists in DOSE
data['grp_pc_lcu_2017'] = data['grp_pc_lcu'] * 100 / data['deflator_2017']




# Z2024 (current usd)
data['Z2024_grp_pc_ppp'] = data['Z2024_pc'] * data.fx / data.PPP
data['Z2024_grp_pc_lcu_2017'] = data['Z2024_pc'] * data.fx * 100 / data['deflator_2017']




# K2025 (2017 int USD) to
data['K2025_grp_pc_lcu_2017'] = data['K2025_pc'] * data['ppp_2017']
data['K2025_grp_pc_ppp'] = data['K2025_grp_pc_lcu_2017'] / 100 * data['deflator_2017'] / data.PPP




# C2022 (2017 int USD)
data['C2022_grp_pc_ppp'] = data['C2022_pc'] / 100 * data['deflator_2017_us']




# W&S in 2005 ppp
data['WS2022_grp_pc_ppp'] = data['WS2022_pc'] / 100 * data['deflator_2005_us']


#NOW the same only for total values:




# DOSE (current lcu)
data['grp_ppp_2017'] = data['grp_lcu'] * 100 / data['deflator_2017'] / data['ppp_2017']
data['grp_ppp_2005'] = data['grp_lcu'] * 100 / data['deflator_2005'] / data['ppp_2005']
data['grp_usd'] = data['grp_lcu'] / data.fx
data['grp_lcu_2017'] = data['grp_lcu'] * 100 / data['deflator_2017']




# Z2024 (current usd)
data['Z2024_grp_ppp'] = data['Z2024'] * data.fx / data.PPP
data['Z2024_grp_lcu_2017'] = data['Z2024'] * data.fx * 100 / data['deflator_2017']




# K2025 (2017 int USD) to
data['K2025_grp_lcu_2017'] = data['K2025'] * data['ppp_2017']
data['K2025_grp_ppp'] = data['K2025_grp_lcu_2017'] / 100 * data['deflator_2017'] / data.PPP




# C2022 (2017 int USD)
data['C2022_grp_ppp'] = data['C2022'] / 100 * data['deflator_2017_us']




# W&S in 2005 ppp
data['WS2022_grp_ppp'] = data['WS2022'] / 100 * data['deflator_2005_us']



# Dataset names
dataset_names = ["DOSE", "K2025", "Z2024", "C2022", "WS2022"]


# Define the column pairs
correlation_pairs = {
    ("DOSE", "K2025"): ("grp_pc_ppp_2017", "K2025_pc"),
    ("DOSE", "Z2024"): ("grp_pc_usd", "Z2024_pc"),
    ("DOSE", "C2022"): ("grp_pc_ppp_2017", "C2022_pc"),
    ("DOSE", "WS2022"): ("grp_pc_ppp_2005", "WS2022_pc"),


    ("K2025", "Z2024"): ("K2025_grp_pc_lcu_2017", "Z2024_grp_pc_lcu_2017"),
    ("K2025", "C2022"): ("K2025_pc", "C2022_pc"),
    ("K2025", "WS2022"): ("K2025_grp_pc_ppp", "WS2022_grp_pc_ppp"),


    ("Z2024", "C2022"): ("Z2024_grp_pc_ppp", "C2022_grp_pc_ppp"),
    ("Z2024", "WS2022"): ("Z2024_grp_pc_ppp", "WS2022_grp_pc_ppp"),


    ("C2022", "WS2022"): ("C2022_grp_pc_ppp", "WS2022_grp_pc_ppp")
}


# Helper functions
def get_total_column_name(colname):
    if colname.endswith('_pc'):
        return colname[:-3]
    return colname.replace('_pc_', '_')


def rmspd(x, y):
    valid = x.notna() & y.notna() & (x + y != 0)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return np.nan
    return np.sqrt((((x - y) / ((x + y) / 2)) ** 2) .mean()) * 100


# Matrices to hold values and labels
n = len(dataset_names)
corr_values = np.full((n, n), np.nan)
label_matrix = pd.DataFrame('', index=dataset_names, columns=dataset_names)


# Fill in values
for (d1, d2), (col1_pc, col2_pc) in correlation_pairs.items():
    i, j = dataset_names.index(d1), dataset_names.index(d2)


    # PC values (lower triangle)
    if col1_pc in data.columns and col2_pc in data.columns:
        pc_corr = data[col1_pc].corr(data[col2_pc])
        pc_rmspd = rmspd(data[col1_pc], data[col2_pc])
        corr_values[i, j] = pc_corr
        label_matrix.iloc[i, j] = f"{pc_corr:.2f}\n({pc_rmspd:.1f}%)"


    # Total values (upper triangle)
    col1_total = get_total_column_name(col1_pc)
    col2_total = get_total_column_name(col2_pc)
    if col1_total in data.columns and col2_total in data.columns:
        total_corr = data[col1_total].corr(data[col2_total])
        total_rmspd = rmspd(data[col1_total], data[col2_total])
        corr_values[j, i] = total_corr
        label_matrix.iloc[j, i] = f"{total_corr:.2f}\n({total_rmspd:.1f}%)"


# Fill diagonals
for i in range(n):
    corr_values[i, i] = 1
    label_matrix.iloc[i, i] = "1.00"


# Create masks
lower_mask = np.tril(np.ones_like(corr_values, dtype=bool), k=-1)
upper_mask = np.triu(np.ones_like(corr_values, dtype=bool), k=1)


# Plotting
fig, ax = plt.subplots(figsize=(10, 8))


# Lower triangle (Total): cool colormap
lower = np.where(lower_mask, corr_values, np.nan)
im1 = ax.imshow(lower, cmap="Blues", vmin=0, vmax=1)


# Upper triangle (PC): warm colormap
upper = np.where(upper_mask, corr_values, np.nan)
im2 = ax.imshow(upper, cmap="Greens", vmin=0, vmax=1)


# Add colorbar for upper triangle (PC - Greens)
cbar_ax2 = fig.add_axes([0.92, 0.55, 0.02, 0.3])
cbar2 = fig.colorbar(im2, cax=cbar_ax2)
cbar2.set_label("GRP p.c. Correlation", fontsize=16)


# Add colorbar for lower triangle (Total - Blues)
cbar_ax1 = fig.add_axes([0.92, 0.15, 0.02, 0.3])
cbar1 = fig.colorbar(im1, cax=cbar_ax1)
cbar1.set_label("Total GRP Correlation", fontsize=16)


# Axis labels
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(dataset_names, rotation=45, ha="right", fontsize=14)
ax.set_yticklabels(dataset_names, fontsize=14)


# Add text values
for i in range(n):
    for j in range(n):
        label = label_matrix.iloc[i, j]
        if label:
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=14)


# Add title and layout
ax.set_title("Intercomparison Matrix (DOSE-restricted): Correlation (upper) and RMSPD (in parentheses)\n"
             "Total GRP values (bottom-left) vs per capita values (top-right)", fontsize=18)
plt.tight_layout()
plt.show()
