import numpy as np
import pandas as pd
from math import *
from datetime import date
import func, collections, utils, importlib, ujson, pathlib, os, time, openpyxl

###################### Input #######################
gen_max = int(input('Please enter the number of generations(min 1, default 20): ') or 20) #Number of generations
trial_no_tot = int(input('Please enter the total number of trials(min 1, defult 5): ') or 5) #Number of trials (each trial produces a separate output file)
m = 10 # Maximum length of CJs in initial ppl CJMs (also determines the number of ini ppl CJMs = m-1)
n = 5 # Number of CJs in initial ppl CJMs

# Input data file
current_path = pathlib.Path(os.getcwd()) #Get the current working directory
raw_trip_file = 'trippub_top2k.csv' #File name of the 2k data
trip_ls  = func.trip_ls_input(raw_trip_file,'w') #Generate the day trips for dataset

# Output excel file setup
workbook_path = str(current_path.parent)+ '/Results/Baseline_LabMachine/'+str(date.today())+' '+os.environ.get('USER') +'.xlsx' #Output file path
pathlib.Path(os.path.split(workbook_path)[0]).mkdir(parents=True, exist_ok=True) #Create the folder (and parent folder) if not exists yet 
if os.path.exists(workbook_path): #Remove workbook if it already exists
	os.remove(workbook_path)

workbook = openpyxl.Workbook() #Create a workbook with openpyxl
worksheet_0 = workbook.active #Get the first worksheet
worksheet_0.title = 'General Results' #First worksheet to save overall results
worksheet_0.append(['Number of Generations:',str(gen_max),'Number of Trial:',str(trial_no_tot)]) #Append the 1st header row
worksheet_0.append(['Trial Number', 'Key Identifier', 'Best CJM in Trial','Score of Best CJM']) #Append a header row
workbook.save(workbook_path) #First save the workbook before any results

# Raw distance dictionary
dist_dict_file_path = 'output/dist_dict_baseline.json' #File path for writing/reading

if os.path.exists(dist_dict_file_path):
	######## Read and load distances between raw journeys ########
	print('Reading an existing distance file!!!')
	dist_dict0 = utils.json2dict(dist_dict_file_path)[0] #existing distance dictionary 
	dist_dict0 = utils.dict_key2tuple(dist_dict0) #Convert the file to tuples (original format before saving to .json)
else:
	######## Compute and save distances between raw journeys ########
	print('Computing and saving distances between raw journeys!!!')
	dist_dict0 = utils.cal_cross_dist(trip_ls, trip_ls)  #Compute the distances
	utils.dict2json(dist_dict_file_path, dist_dict0) #Save the distances

print('Trials start! # of Trials =',trial_no_tot,'Max Generation',gen_max)
print('***********************************************************************************************')
###################### Main Loop #######################
start_time = time.time()
# The purpose of the following loop is to generate populations (i.e., cluster centers) that have the highest score
for trial_no in range(trial_no_tot):
	print('----------------Execution starts for Trial No',trial_no,'----------------')
	last_time = time.time()
	# Iterate over different number of trials to avoid local minimum
	ppl, top_n = func.ini_ppl_gen(trip_ls, m, n) #Generate the initial populations for cluster centers
	center_dict_all = {} #Initialize the dictionary for assignment
	cjm_score = collections.defaultdict(list) #Initialize the score dictionary
	
	workbook = openpyxl.load_workbook(workbook_path) #Realod workbook
	worksheet_0 = workbook.active #Retrieve the first worksheet
	worksheet_1 = workbook.create_sheet('Trial '+str(trial_no)) #Create the new sheet for current trial 
	worksheet_1.append(['Key','Best CJM','Score']) #Append the 2nd header row

	for t in range(gen_max):
		#Assign CJs to current population and evaluate current population's score (CJMs)
		for key, center_ls in ppl.items(): 
			# Iterate over each CJM in testing
			# Note:
				# Center_ls is a CJM (a list of centers/CJs in the CJM)
				# key is a numerical identifier that belong to [2, top_num]
					# For initial ppl, key is length of individual journey (i in CJM_i).
			dist_dict1 = utils.cal_cross_dist(trip_ls, center_ls, dist_dict0) #Compute the distances between all trips and current centers given dist_dict0
			#Assign each journey to the closest centers/cluster in CJM_i (returns a dictionary - see details in cjm_assign function)
			center_dict_all[key] = func.cjm_assign(trip_ls, center_ls, dist_dict1) # Each item of center_dict_all is a dict for ONE CJM

			score = func.cjm_eval(trip_ls, center_ls, center_dict_all[key], dist_dict1) #Compute score for the current CJM
			cjm_score[key].append(score) #Saves the score for current CJM and generation (each cjm_score[key] is a list of scores of history)
		
		# Sort the CJM by their latest score in descending order
		cjm_sort_ls = sorted(cjm_score, key = lambda x: cjm_score[x][-1], reverse = 1) 
		print('The best CJM at time ' + str(t) + ' is CJM #' + str(cjm_sort_ls[0]) +' and its score is '+ str(cjm_score[cjm_sort_ls[0]][t]))

		# Saves [key, current best CJM, current best score] for current trial to csv
		record = [str(cjm_sort_ls[0]), str(ppl[cjm_sort_ls[0]]), str(cjm_score[cjm_sort_ls[0]][t])]
		worksheet_1.append(record) #Saves to csv

		# Performs genetic operations
		ppl = func.ga_CJM(ppl, cjm_score, top_n) 

	# Evaluation for the last round of genetic operations
	for key, center_ls in ppl.items(): #center_ls is a CJM (a list of centers/CJs in the CJM), key is identifier in [2, top_num]
		dist_dict1 = utils.cal_cross_dist(trip_ls, center_ls, dist_dict1) #Compute the distances between all trips and current centers
		center_dict_all[key] = func.cjm_assign(trip_ls, center_ls, dist_dict1) #Assign each journey to the closest centers/cluster in CJM_i
		score = func.cjm_eval(trip_ls, center_ls, center_dict_all[key], dist_dict1) #Compute score for the current CJM
		cjm_score[key].append(score)
	cjm_sort_ls = sorted(cjm_score, key = lambda x: cjm_score[x][-1], reverse = 1) #Sort the CJM by their latest score in descending order
	
	record = [str(cjm_sort_ls[0]), str(ppl[cjm_sort_ls[0]]), str(cjm_score[cjm_sort_ls[0]][t])]
	worksheet_1.append(record) #Saves to sheet
	worksheet_0.append([str(trial_no), str(cjm_sort_ls[0]), str(ppl[cjm_sort_ls[0]]), str(cjm_score[cjm_sort_ls[0]][t])])

	# Final Printings
	print('The best CJM at time ' + str(t+1) + ' is CJM #' + str(cjm_sort_ls[0]) +' and its score is '+ str(cjm_score[cjm_sort_ls[0]][t]))
	print('Execution time of this trial is',time.time()-last_time,'and current total time taken is',time.time()-start_time)
	worksheet_1.append(['Time Taken for Current Trial',time.time() - last_time])
	workbook.save(workbook_path)
###################### Test #######################
# importlib.reload(func)
# importlib.reload(utils)
# print('the current map is ' + str(key) + ' and assignment is completed')