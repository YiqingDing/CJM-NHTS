import pandas as pd
from math import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import func, collections, utils, importlib, ujson, os

# Input: Local data files from "output" folder with names similar as "FinalResult0.csv"
# 
result = pd.DataFrame() 
folder_name = 'output'
for root,dirs,files in os.walk(folder_name):
	for file in files:
		if file.startswith("FinalResult"): #Check if the file starts with the name FinalResult
			df = pd.read_csv(folder_name+'/'+file)
			result = result.append(df.tail(1))

final = result.sort_values('Score',ascending=False)
final.to_csv(folder_name+'/sorted_final_result.csv')


# Visualize the raw data
# Extract raw data
# file_trip = '/Users/irislab/Google Drive/CJM Code & Data/Data/trippub_top2k.csv' # Lab machine data
file_trip  ='/Users/yichingding/Google Drive/School/Stanford/Research/IRIS/Journey Map/App Approach Paper/CJM Code & Data/Data/trippub_top2k.csv'
trip_ls  = func.data_processing(file_trip) #Generate the day trips for dataset

#Plot
# test_lines = (((1,1),(2,2)), ((1,2),(2,2)))
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
for trip_ind in trip_ls:
	plt.plot(trip_ind[0],trip_ind[1])
plt.show()
