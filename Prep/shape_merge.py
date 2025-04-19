'''COPIED DIRECTLY FROM DOSE REPLICATION'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:44:19 2023

@author: robertcarr
"""
import geopandas as gpd
import pandas as pd

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
This code  can be used to combine the GADM level 1 global shapefile with our
own custom shapefile. In this way, users can work with one comprehensive 
shapefile with adminstrative boundaries that match the regions in DOSE
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#local path with folder where the downloaded shapefiles are stored 
#(both GADM and the custom one)

gadm_path = './Data/spatial data/'

# Read shapefiles

gadm = gpd.read_file(gadm_path+'gadm36_1.shp')
# has to be downloaded from https://gadm.org/download_world36.html; follow instructions in readme

custom = gpd.read_file(gadm_path+'all_non_GADM_regions.shp')

#list of GADM countries whose data is not needed because we provide it with the custom file
unneeded_list = ["KAZ","MKD","NPL","PHL","LKA"]

#remove geometry for these countries from GADM
gadm_trim = gadm[~gadm.GID_0.isin(unneeded_list)]

# Merge/Combine multiple shapefiles into one
gadm_custom = gpd.pd.concat([gadm_trim, custom])

#Export merged geodataframe into shapefile
gadm_custom.to_file(gadm_path+'gadm_custom_merged.shp')