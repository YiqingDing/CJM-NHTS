import utils, collections, csv, random, os, pathlib
import pandas as pd
from math import *
import numpy as np

def data_processing(file):
	# Read the data file and return a list of day trips
	# Output:
	# trip: List of day trips, each item is a 2-tuple, with 1st item being timestamps, 2nd item being events
	# final_trip.csv: A csv file with all the day tips, every 2 lines form a day trip

	# file = 'NHTS/trippub.csv'
	raw_data = pd.read_csv(file) #Read the data and return a pandas df
	previous_id = 0 #Initialize the ID of the day trip
	trip = []  #Create an empty trip segment list
	csv_trip = open('output/final_trip.csv','w') #Create an empty output csv file
	csv_writer = csv.writer(csv_trip, delimiter = ',') #Create a writer object for output
	# n = 10 #Test variable for limit row
	for row in raw_data.itertuples(): #Iterate over namedtuple for each row (1st row as index)
		# if idx < n:  #Test maximum row number
		TDCASEID = row.TDCASEID #Get the unique ID of the trip segment

		new_id = str(int(row.TDCASEID))[:-2] #Get the ID of day trip for current trip segment (HOUSEID+PERSONID)
		if row.WHYTO > 0: #The trip segment is valid with a WHY2 variable
			if new_id != previous_id: #Not the same day trip as previous day trip
				# Start a new day trip
				
				# Append the last day trip to the trip list (this can't be done until last day trip is over)
				if previous_id != 0: #Not the first trip	
					current_trip = tuple([tuple(i) for i in current_trip]) #Convert last day trip to a tuple of tuples
					trip.append(current_trip) #Append the last day trip to the trip catalog
					csv_writer.writerow(current_trip[0]) #Write the last day trip's time to csv
					csv_writer.writerow(current_trip[1]) #Write the last day trip's location to csv
				previous_id = new_id #Assign the new id 
				
				# Find the start and end time of the current trip segment, round to hours
				time0 = int(row.STRTTIME) #Start time 0000-2359
				time0round = int( floor(time0/100) + round(time0/100%1/0.6,0) ) #Convert to interger hours (round to the nearest hour)
				time1 = int(row.ENDTIME) #End time 0000-2359
				time1round = int(floor(time1/100) + round(time1/100%1/0.6,0))
				if time1round == time0round:
					time1round += 0.5 #if the start and end at the same hour, add a 30m interval
				current_trip = [[time0round, time1round], [row.WHYFROM, row.WHYTO]] #Rebuild a new current day trip log
			 
			else:#Continue the current day trip with the new trip segment
				#Find the end hour of this segment
				time1 = int(row.ENDTIME)
				time1round = int(floor(time1/100) + round(time1/100%1/0.6,0))

				# Append the segment to current trip log
				if time1round <= current_trip[0][-1]: #If the current trip end time is less or equal to the previous one, add 30 min
					time1round =current_trip[0][-1]+0.5
				current_trip[0].append(time1round)
				current_trip[1].append(row.WHYTO)

	#Write the current day trip (since no more new day trip, we will write this separately)
	csv_writer.writerow(current_trip[0])
	csv_writer.writerow(current_trip[1])
	
	current_trip = tuple([tuple(i) for i in current_trip])
	trip.append(current_trip)
	#return a list of day trips, each day trip is in the format of tuple(time tuple, location tuple)
	return trip

def ini_ppl_gen(trip, m, n = 10):
	# generate initial population (a list of journey maps = A list of lists of journeys) and top_n list
	# The initial population is a set of CJMs at length from 2 to m each with n most occuring patterns 
	# There are (m-1) CJMs and each with n journeys inside (m, n given by user)
	# Within each CJM, the n journeys are the n most occuring journeys at length i where i is from 2 to m
	# trip: List of journeys, each entry is a journey
	# m: Maximum length for a journey in a CJM
	# n: Number of journeys in each CJM
	# Output 
	# ini_ppl: A dictionary of CJMs each with different length of journeys within. The key is length of journey, item is list of journeys (CJM)
	# top_n: Top trips within the range (all of them are filled in ini_ppl)

	# Count (m-1)*n journeys - max number of journeys for initial population
	num_max = (m-1)*n #Maximum number of journeys for initial population
	trip_cnt = collections.Counter(trip) #convert the trip list to a counter (each individual trip is a tuple of tuples)
	ini_ppl_cnt = trip_cnt.most_common() #find the top population for initial population
	
	ini_ppl = collections.defaultdict(list) #Create an empty dictionary
	i = 1 #Assignment number
	top_n = [] #Create an empty list
	# The following loop assigns each most_common trip to its respecitive bucket/CJM
	for trip_set in ini_ppl_cnt: #Iterate over the most common entries
		#Each trip_set entry is a tuple of (trip, count of trip occurance)
		trip_ind = trip_set[0] #Extract the individual trip - ((time), (activities))
		ind_l = len(trip_ind[0]) #Individual trip length
		if i<= num_max: #If total assignment number hasn't reach max yet
			if ind_l <= m and (n - len(ini_ppl[ind_l])) > 0: #Length of journey falls within range and there is space in the bucket
				ini_ppl[ind_l].extend([trip_ind]) #Add the trip to the bucket (bucket list length < n)
				top_n.append(trip_ind) #Add the trip to the Top_n list
				i += 1 #Increase new assignment number
		else: #The total assignment reached max
			break #If reaches the maximum number of assignment, break out of the loop
	
	# Check if any of the CJMs are not filled (bucket length < n) and print it out
	for idx, center_ls in ini_ppl.items():
		if len(center_ls) < n:
			print('The CJM for length '+ str(idx) + ' is not filled (missing '+ str(n - len(center_ls))+ ' entries)! Reduce n or expand dataset!')
	
	return ini_ppl, top_n

def cjm_assign(trip_ls, centers, dist_dict = {}):
	# Assigns indiviaul entries to centers/clusters based on relative distances to given centers
	# Input:
	# trip_ls: Input list of individual journeys
	# centers: Input list of centers (each is an individual journey) - can have same members as trip_ls
	# dist_dict: Input distance dictionary, default empty
	# Output:
	# center_dict: Ouput a dictionary with structure detailed at the end of function
	center_dict = collections.defaultdict(list)
	
	# n = 100 #sample limit 
	for ind_trip in trip_ls:
		# for i in range(n):
		if ind_trip not in centers:#The individual trip is not the top trip
			dist_min = float('inf')
			for ind_top in centers:
				# Calculate the distance between current trip and current top trip (read if dist_dict available)
				new_dist = dist_dict[ind_trip][ind_top] if bool(dist_dict) else utils.cal_dist(ind_trip, ind_top)
				if new_dist < dist_min: #If new distance is less than current distance
					center_dict[ind_trip] = ind_top #Change the individual trip's center to the current top trip
					dist_min = new_dist #Update the min distance
			
			# Add the individual trip to the top trip's list
			ind_center = center_dict[ind_trip] #Identify the individual trip's center
			center_dict[ind_center].append(ind_trip) #Add the individual trip to its center's cluster list
		else: #If the individual trip is in the top trip list (can be easily assigned)
			center_dict[ind_trip].insert(0, ind_trip) #Insert the top/individual trip to the 1st element of its list
	
	# center_dict has the following structure:
	# center_dict[trip not a center] = center (a tuple)
	# center_dict[trip that is a center] = [trip_1, trip_2 ......] (a list of tuples)
	return center_dict

def cjm_eval(trip_ls, center_ls, center_dict, dist_dict = {}):
	# Evaluate each CJM based on different criterias
	fitness = utils.cal_fitness(trip_ls, center_dict, dist_dict) #Calculate fitness function (the higher the better)
	# print('Fitness score is '+ str(fitness))

	# Calculate Silhouette index (the higher the better)
	ShCoeff = utils.cal_Silhouette(center_ls, center_dict, dist_dict) 
	# print('Silhouette score is ' + str(ShCoeff['mean']))
	score = 0 + fitness + ShCoeff['mean']
	return score

def ga_CJM(ppl, cjm_score, top_n):
	# Genetic operations of the representative journeys (centers in center_dict, other journeys are constants)
	# center_ls: List of representatives journeys 
	elite_no = 1 #Size of elite population (which will be kept)
	ga_action_no = [1]*5 #Number of transformation for each action (5 actions in total)
	
	prob_action = 0.1 #Probability threshold of action being taken

	cjm_sort_ls = sorted(cjm_score, key = lambda x: cjm_score[x][-1], reverse = 1) #Sort the CJM by their latest score in descending order
	cjm_elite_key = cjm_sort_ls[:elite_no] #Get the key of elite individuals (CJMs) in population
	cjm_ga_key = cjm_sort_ls[elite_no:] #Get the key of non-elite individuals (CJMs) in population

	# Extract all genes (CJs) from elite individuals (CJMs) into a list
	cjm_elite_ls = [ind_gene for key in cjm_elite_key for ind_gene in ppl[key]] 

	for cjm_key in cjm_ga_key: #Iterate over keys of all non-elite individuals (CJMs)
		cjm_current = ppl[cjm_key] #Extract current non-elite individuals (CJMs)

		#Rebuild ppl[cjm_key] using current CJM (processed by GA)
		ppl[cjm_key] =  utils.ga_operations(cjm_current, cjm_elite_ls, top_n, ga_action_no, prob_action)
	return ppl

def cal_mutual_dist_para(data_0, data_1, dist_dict =  collections.defaultdict(utils.dd)):
	# Parallel computing: calculate the distances between different pairs of data entries and load them into dist_dict
	# Input: 
	# Let's think of overall data to be a combination of different data_0, and data_1 is the set start from data_0 to the end (including data_0)
	# data_0: A data list specific to this process in parallel computing
	# data_1: The data starts from data_0 in the entire dataset to the end of the dataset (including data_0)
	# dist_dict: Existing distance dictionary, default = empty 2d dictionary
	# Output: Dictionary of dstances between different pairs of data entries, dist_dict[item1][item2]
	for item_1 in data_0:
		for item_2 in data_1:
			dist = utils.cal_dist(item_1, item_2)
			dist_dict[item_1][item_2] = dist
			dist_dict[item_2][item_1] = dist #Symmetry
			# if item1 not in dist_dict: #If the current dictionary doesnt have this entry
			# 	dist_dict[item_1] = {item_2: dist}
			# 	dist_dict[item_2] = {item_1: dist}
			# else: #If the current dictionary has this entry
			# 	dist_dict[item_1].update({})
	return dist_dict

def dict_key2tuple(dict0):
	# Convert dictionaries with string keys to tuple keys (works for dictionary of dictionary)
	new_dict = collections.defaultdict(lambda: collections.defaultdict(int)) #Create empty dictionary 
	for key, val in dict0.items():
		if type(val) is dict: #If lower level is dict, pass it to itself
			new_val = dict_key2tuple(val) #Lower level dict is converted
			new_dict.update({eval(key): new_val}) #Update the new_dict with converted lower level dict
		else: #If lower level is not dictionary 
			new_dict.update({eval(key):val})
	return new_dict

def save_ls2csv(ls,writetype = 'w' , file_name='output/results.csv'):
	# Saves a list to a new csv file with file_name 
	# Input:
	# ls: A list
	# file_name: Name of csv file
	# Output:
	# Saves the ls to a csv file with file_name, depends on the writetype
	with open(file_name,writetype) as f:
		fwrite = csv.writer(f)
		if writetype == 'w': #Write a new file
			fwrite.writerows(ls)
		else: #Append to an existing file
			fwrite.writerow(ls)

def data_sort_labmachine(folder_path, data_name_format, file_name = 'result_sorted.csv'):
	#Input file folder and name format, analyze and save output analysis
	#folder_path: Path of data file folder
	#data_name_format: Name format of data files (start with)
	#Output: A dataframe of best results, each as a row 
	result = pd.DataFrame() #Empty dataframe to save best trips
	for root,dirs,files in os.walk(folder_path): #Iterate over all data files in path
		for file in files:
			if file.startswith(data_name_format): #Check if the file starts with the name FinalResult
				df = pd.read_csv(folder_path+'/'+file) #Read the file
				result = result.append(df.tail(1)) #Append the last line (final optimized trip) to the result list

	result.insert(0,'Trial No', range(1, result.shape[0]+1)) #Add the trial number 
	final = result.sort_values('Score',ascending=False) #Sort the result list based on final score
	output_path = folder_path+'/' + file_name
	final.to_csv(output_path, index=False) #Save the result list to csv

	#Return final result list (each row is an optimization result for a trial) and output file path (in case of future use)
	return final, output_path

def data_translate_labmachine(result_file, simplified_activities = True, keep_origin = True):
	#Default book
	NHTS_book = {1: 'Regular home activities (chores, sleep)', 2: 'Work from home (paid)', 3: 'Work', 4: 'Work-related meeting / trip', 
	5: 'Volunteer activities (not paid)', 6: 'Drop off /pick up someone', 7: 'Change type of transportation', 
	8: 'Attend school as a student', 9: 'Attend child care', 10: 'Attend adult care', 11: 'Buy goods (groceries, clothes, appliances, gas)',
	12: 'Buy services (dry cleaners, banking, service a car, pet care)', 13: 'Buy meals (go out for a meal, snack, carry-out)',
	14: 'Other general errands (post office, library)', 15: 'Recreational activities (visit parks, movies, bars, museums)',
	16: 'Exercise (go for a jog, walk, walk the dog, go to the gym)', 17: 'Visit friends or relatives', 18: 'Health care visit (medical, dental, therapy)',
	19: 'Religious or other community activities', 97: 'Something else', 99: 'Nothing',}

	df_data = pd.read_csv(result_file, converters={'Best CJM': eval}) #Read the csv file and treat list as value
	result_translated = pd.DataFrame() #Create an empty dataframe

	for idx, row in df_data.iterrows():
		trip_ls = row['Best CJM'] #The bset CJMs for current trial
		trial_current = pd.concat([row]*len(trip_ls), ignore_index=True, axis = 1) #Make a dataframe with # of col equal to number of clusters
		trial_current = trial_current.transpose() #Transpose the dataframe so each row is a cluster
		activity_ls = [] #An empty list of activities for one cluster center

		for trip_ind in trip_ls: #Iterate over one trip/cluster center
			activity_ls.append(utils.trip_translator(input_trip = trip_ind, book=NHTS_book, single_col = simplified_activities)) #Append a activity list
		trial_current.insert(3, 'Activity List', activity_ls)
		trial_current.insert(3, 'Current Trip', trip_ls)
		
		result_translated = result_translated.append(trial_current)
	if keep_origin == False:
		del result_translated['Best CJM']
		del result_translated['Key']
	result_translated.to_csv(str(pathlib.Path(result_file).parent)+'/result_translated.csv',  index=False)

	return result_translated