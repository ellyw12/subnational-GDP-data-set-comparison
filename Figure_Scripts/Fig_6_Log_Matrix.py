
"""
In this script pre-processed data from selected sets are combined, and
agreement among them is measured and displayed into a heatmap.


NEW VERSION containing GRP total and p.c., measured both in pearson R and RMSPD and with different colour scales


Authors: Leon Liessem, Matthias Zarama Giedion, Luuk Staal, Joshua Arky
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data_path = './Data/'
graphics_path = './Figures/'
deflator_path           =    data_path +'deflator/'

data_loglog = 'merged_loglog.csv'
dose_log_data = pd.read_csv(data_path+data_loglog)

data = dose_log_data


# Dataset names
dataset_names = ["DOSE", "K2025", "Z2024", "C2022", "WS2022"]


# Define the column pairs
correlation_pairs = {
    ("DOSE", "K2025"): ("log_grp_pc_ppp_2017", "log_K2025_pc"),
    ("DOSE", "Z2024"): ("log_grp_pc_usd", "log_Z2024_pc"),
    ("DOSE", "C2022"): ("log_grp_pc_ppp_2017", "log_C2022_pc"),
    ("DOSE", "WS2022"): ("log_grp_pc_ppp_2005", "log_WS2022_pc"),


    ("K2025", "Z2024"): ("log_K2025_grp_pc_lcu_2017", "log_Z2024_grp_pc_lcu_2017"),
    ("K2025", "C2022"): ("log_K2025_pc", "log_C2022_pc"),
    ("K2025", "WS2022"): ("log_K2025_grp_pc_ppp", "log_WS2022_grp_pc_ppp"),


    ("Z2024", "C2022"): ("log_Z2024_grp_pc_ppp", "log_C2022_grp_pc_ppp"),
    ("Z2024", "WS2022"): ("log_Z2024_grp_pc_ppp", "log_WS2022_grp_pc_ppp"),


    ("C2022", "WS2022"): ("log_C2022_grp_pc_ppp", "log_WS2022_grp_pc_ppp")
}


# Helper functions
def get_total_column_name(colname):
    if colname.endswith('_pc'):
        return colname[:-3]
    return colname.replace('_pc_', '_')
def rmspd(x, y):
    valid = x.notna() & y.notna() & (x + y != 0) & np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return np.nan
    return np.sqrt((((x - y) / ((x + y) / 2)) ** 2).mean()) * 100




# Matrices to hold values and labels
n = len(dataset_names)
corr_values = np.full((n, n), np.nan)
label_matrix = pd.DataFrame('', index=dataset_names, columns=dataset_names)


# Fill in values
for (d1, d2), (col1_pc, col2_pc) in correlation_pairs.items():
    i, j = dataset_names.index(d1), dataset_names.index(d2)


    # PC values (lower triangle)
    if col1_pc in data.columns and col2_pc in data.columns:
        x_pc = data[col1_pc]
        y_pc = data[col2_pc]
        valid_pc = x_pc.notna() & y_pc.notna() & np.isfinite(x_pc) & np.isfinite(y_pc) & (x_pc != 0) & (y_pc != 0) #control for infinites, nans and zeros
        if valid_pc.any():
            pc_corr = x_pc[valid_pc].corr(y_pc[valid_pc])
            pc_rmspd = rmspd(x_pc, y_pc)
            corr_values[i, j] = pc_corr
            label_matrix.iloc[i, j] = f"{pc_corr:.2f}\n({pc_rmspd:.1f}%)"


    # Total values (upper triangle)
    col1_total = get_total_column_name(col1_pc)
    col2_total = get_total_column_name(col2_pc)
    if col1_total in data.columns and col2_total in data.columns:
        x_total = data[col1_total]
        y_total = data[col2_total]
        valid_total = x_total.notna() & y_total.notna() & np.isfinite(x_total) & np.isfinite(y_total) & (x_total != 0) & (y_total != 0) #control for infinites, nans and zeros
        if valid_total.any():
            total_corr = x_total[valid_total].corr(y_total[valid_total])
            total_rmspd = rmspd(x_total, y_total)
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
ax.set_title("Intercomparison Matrix (DOSE-restricted): log_log Correlation (upper) and RMSPD (in parentheses)\n"
             "Total GRP values (bottom-left) vs per capita values (top-right)", fontsize=18)
plt.tight_layout()
plt.show()

