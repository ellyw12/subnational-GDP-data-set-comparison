
'''
This repository is built to aid in recreation of the figures from the paper 'Comparing subnational GDP data sets based on reporting and on night-time light'. 

Authors: Josh Arky, Leon Liessem, Matthias Zarama Giedion, Luuk Staal, Brielle Wells

To begin please complete the following preparation in this order: 

    1. Download the necessary collection of data files from the Zenodo folder. Link:  __________________. 
    Original Files origins: 

        C2022: https://figshare.com/articles/dataset/Global_1_km_1_km_gridded_revised_real_gross_domestic_product_and_electricity_consumption_during_1992-2019_based_on_calibrated_nighttime_light_data/17004523/1

        WS2022: https://zenodo.org/records/5880037

        Z2024: https://figshare.com/articles/journal_contribution/Code_and_data_for_the_paper_Developing_an_Annual_Global_Sub-National_Scale_Economic_Data_from_1992_to_2021_Using_Nighttime_Lights_and_Deep_Learning/24024597?file=47716642

        K2025: https://zenodo.org/records/13943886

        S2024: recieved directly from Author. 

    2. Download GADM 3.6 files for spatial map curration. A comprehensive instruction guide for this is available in the READme file of the DOSE Replication files. Link: https://zenodo.org/records/7659600. 


    3. Aggregate global gridded products. This requires extra computing power. Reccomended to utilize a SLURM cluster or equivalent. 
        - C2022_aggregate.py  
        - WS2022_aggregate.py 
        
        Execute the above files with the .sh job file (
        details of computing capacity utilitzed are in this file, #of nodes, etc.)
        - run_aggregation.sh

    3. Run prep files: 
        - Z2024_prep.py
        - K2025_prep.py
        - WS2022_prep.py
        - C2022_prep.py
        - S2024_prep.py

        - shape_merge.py
        - load_merge.ipynb 

    4. Following the completion of the prep files, it is then possible to recreate the figures. Further detail explaining each script is available at the top of each script file. 
        

'''
