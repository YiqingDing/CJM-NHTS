import numpy as np
import pandas as pd
from math import *
from datetime import date
import func, collections, utils, importlib, ujson, pathlib, os

###################### Input #######################

# Input data file
current_path = pathlib.Path(os.getcwd()) #Get the current working directory
raw_trip_file = 'trippub_top2k.csv' #File name of the 2k data
trip_ls  = func.trip_ls_input(raw_trip_file,'w') #Generate the day trips for dataset

# # Create output folder
# output_path = str(current_path.parent)+ '/Results/Baseline_LabMachine/'+str(date.today()) #Output file path
# pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) #Create the folder (and parent folder) if not exists yet 

# Raw distance dictionary
######## Compute and save distances between raw journeys ########
dist_dict_file_path = 'dist_dict_baseline.json'
dist_dict0 = utils.cal_cross_dist(trip_ls, trip_ls)
utils.dict2json(dist_dict_file_path, dist_dict0)

# utils.json2dict(dist_dict_file_path)[0]