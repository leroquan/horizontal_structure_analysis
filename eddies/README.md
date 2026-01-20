Anne Leroquais 2026 01 20

The scripts contained in this folder are used to post-process the results of the swirl_toolbox scripts. 
The expected inputs are "lvl0" and "lake_characteristics" csv files for each timesteps.

Process:
1-Create MITgcm inputs with preprocessing-mitgcm repository
2-Run MITgcm 
3-Run main.py in swirl_toolbox repository
4-Run lvl0_concat_csv.ipynb to obtain only one big csv for lvl0 and lake_characteristic over the entire duration. This way, the ids for lvl0 are coherents.
5-Run lvl1_aggregate_over_depth.ipynb to get one lvl1 csv with eddies aggregated over depth.
6-Run lvl2a_aggregate_over_time.ipynb to get one lvl2a csv with eddies aggregated over time.