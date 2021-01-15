import numpy as np
import pandas as pd
from math import *
from datetime import date
import func, collections, utils, importlib, ujson, pathlib, os

###################### Input #######################

# Input data file
current_path = pathlib.Path(os.getcwd()) #Get the current working directory
data_file = 'trippub_top2k.csv' #File name of the 2k data
file_trip = str(current_path.parent.parent)+'/Data/'+data_file #Data file location+name
trip_ls  = func.data_processing(file_trip) #Generate the day trips for dataset

#Create output folder
output_path = str(current_path.parent)+ '/LabMachineResults/'+str(date.today()) #Output file path
if not os.path.exists(output_path): #Create output folder if not exists
    os.makedirs(output_path)
#Read raw distance dictionary
f = open('output/dist_dict0.json','r') #Required input for this file - distance dictionary to reduce computation complexity
dist_dict0 = ujson.loads(f.read()) #Load the file
dist_dict0 = func.dict_key2tuple(dist_dict0) #Convert the file to tuples (original format before saving to .json)

###################### Input #######################
#Generate initial population (a set of CJMs each as an individual)
m = 10 # Maximum length of CJs in initial ppl CJMs (also determines the number of ini ppl CJMs = m-1)
n = 5 # Number of CJs in initial ppl CJMs
gen_max = 50 #Number of generations
file_no_tot = 20 #Number of trials (each trial produces a separate output file)
###################### New #######################
for file_no in range(file_no_tot):
	ppl, top_n = func.ini_ppl_gen(trip_ls, m, n) #Generate the initial populations
	center_dict_all = {} #Create an empty dictionary for assignment
	cjm_score = collections.defaultdict(list)
	record = [['Key','Best CJM','Score']] #List of [best CJM key, current best CJM, current best score]
	filename= output_path + '/FinalResult' + str(file_no+1) + '.csv' #Create name of the output folder
	func.save_ls2csv(record, 'w' , filename) #Saves the header 'record'

	for t in range(gen_max):
		#Assign CJs to current ppl and evaluate current population (CJMs)
		for key, center_ls in ppl.items(): #key is identifier in [2, top_num], center_ls is a CJM (a list of centers/CJs in the CJM)
			#Note that for initial ppl, key is length of individual journey (i in CJM_i).
			dist_dict1 = utils.cal_cross_dist(trip_ls, center_ls, dist_dict0) #Compute the distances between all trips and current centers given dist_dict0
			center_dict_all[key] = func.cjm_assign(trip_ls, center_ls, dist_dict1) #Assign each journey to the closest centers/cluster in CJM_i (returns a dictionary)
			score = func.cjm_eval(trip_ls, center_ls, center_dict_all[key], dist_dict1) #Compute score for the current CJM
			cjm_score[key].append(score) #Saves the score for current CJM and generation (each cjm_score[key] is a list of scores of history)
		cjm_sort_ls = sorted(cjm_score, key = lambda x: cjm_score[x][-1], reverse = 1) #Sort the CJM by their latest score in descending order
		print('The best CJM at time ' + str(t) + ' is CJM #' + str(cjm_sort_ls[0]) +' and its score is '+ str(cjm_score[cjm_sort_ls[0]][t]))
		#Saves [key, current best CJM, current best score] to a list
		record = [str(cjm_sort_ls[0]), str(ppl[cjm_sort_ls[0]]), str(cjm_score[cjm_sort_ls[0]][t])]
		func.save_ls2csv(record,'a' , filename)
		ppl = func.ga_CJM(ppl, cjm_score, top_n) #Genetic operations

	# Final evaluation
	for key, center_ls in ppl.items(): #center_ls is a CJM (a list of centers/CJs in the CJM), key is identifier in [2, top_num]
		dist_dict1 = utils.cal_cross_dist(trip_ls, center_ls, dist_dict0) #Compute the distances between all trips and current centers
		center_dict_all[key] = func.cjm_assign(trip_ls, center_ls, dist_dict1) #Assign each journey to the closest centers/cluster in CJM_i
		score = func.cjm_eval(trip_ls, center_ls, center_dict_all[key], dist_dict1) #Compute score for the current CJM
		cjm_score[key].append(score)
	cjm_sort_ls = sorted(cjm_score, key = lambda x: cjm_score[x][-1], reverse = 1) #Sort the CJM by their latest score in descending order
	print('The best CJM at time ' + str(t) + ' of trial '+ str(file_no)+ ' is CJM #' + str(cjm_sort_ls[0]) +' and its score is '+ str(cjm_score[cjm_sort_ls[0]][t]))
	record = [str(cjm_sort_ls[0]), str(ppl[cjm_sort_ls[0]]), str(cjm_score[cjm_sort_ls[0]][t])]
	func.save_ls2csv(record, 'a' , filename)
	print('The final CJM of trial '+ str(file_no)+ ' is #'+ str(cjm_sort_ls[0]) +': '+str(ppl[cjm_sort_ls[0]]))

###################### Test #######################
# importlib.reload(func)
# importlib.reload(utils)
# print('the current map is ' + str(key) + ' and assignment is completed')