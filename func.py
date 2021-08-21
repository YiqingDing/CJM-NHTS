import utils, collections, csv, random, os, pathlib, ast, uuid, time, copy, itertools, sys
import pandas as pd
from math import *
import numpy as np
from bidict import bidict
import matplotlib.pyplot as plt, matplotlib.lines as ml
# from matplotlib.gridspec import GridSpec
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages

np.seterr('raise')

def trip_data_processing(raw_trip_path, processed_file_name = 'final_trip.csv', save_file = True):
	# Read the raw trip data file and return a list of day trips and save the output trip file
	# Output:
		# trip: List of day trips, each item is a 2-tuple, with 1st item being timestamps, 2nd item being events
		# final_trip.csv: A csv file with all the day tips, every 2 lines form a day trip
	raw_data = pd.read_csv(raw_trip_path) #Read the data and return a pandas df
	previous_id = 0 #Initialize the ID of the day trip
	trip = []  #Create an empty trip segment list
	processed_trip_path = 'output/'+ processed_file_name #Default output folder
	if save_file:
		csv_trip = open(processed_trip_path,'w') #Create an empty output csv file
		csv_writer = csv.writer(csv_trip, delimiter = ',') #Create a writer object for output
	# n = 10 #Test variable for limit row
	for row in raw_data.itertuples(): #Iterate over namedtuple for each row (1st row as index)
		# if idx < n:  #Test maximum row number
		TDCASEID = row.TDCASEID #Get the unique ID of the trip segment

		new_id = str(int(row.TDCASEID))[:-2] #Get the ID of day trip for current trip segment (HOUSEID+PERSONID) - This only changes with a new individual
		if row.WHYTO > 0: #The trip segment is valid with a WHY2 variable
			if new_id != previous_id: #Not the same day trip as previous day trip (new individual)
				# Start a new day trip
				
				# Record-keeping: Append the last day trip to the final trip list (this can't be done until last day trip is over)
				# This has nothing to do with the current day trip!
				if previous_id != 0: #Not the first individual/day trip in the dataset
					current_trip = tuple([tuple(i) for i in current_trip]) #Convert last day trip to a tuple of tuples
					trip.append(current_trip) #Append the last day trip to the trip catalog
					if save_file:
						csv_writer.writerow(current_trip[0]) #Write the last day trip's time to csv
						csv_writer.writerow(current_trip[1]) #Write the last day trip's location to csv
				
				previous_id = new_id #Assign the new id 
				# Find the start and end time of the current trip segment, round to hours
				time0 = int(row.STRTTIME) #Start time 0000-2359
				time0float = (time0/100)%1/0.6 + floor(time0/100) #Get the float representation for time0
				time1 = int(row.ENDTIME) #End time 0000-2359
				time1float = (time1/100)%1/0.6 + floor(time1/100)  #Get the float representation for time1
				
				time0round = round(time0float * 2)/2 #Round starting time to nearest 30min
				time1round = round(time1float * 2)/2 #Round ending time to nearest 30min
				
				# print(time0, time0float, time0round) #Debug
				# print(time1, time1float, time1round) #Debug

				# # Original approach on CodingNotes (Current Result - middle column)
				# time0round = int( floor(time0/100) + round(time0/100%1/0.6,0) ) #Convert to interger hours (round to the nearest hour)
				# time1round = int(floor(time1/100) + round(time1/100%1/0.6,0)) #Possible 00 - 24
				# print(time0round, time1round) #Debug

				# break
				if time1round == time0round: #Cases where rounded times are the same
					if (time1float - time1round) >= (time1round - time1float):
						#If the ending time is farther to the rounded .5 hour than starting time
						time1round += 0.5 #Add 30 min to ending time
					elif (time1float - time1round) < (time1round - time1float):
						#If the starting time is farther to the rounded .5 hour than ending time
						time0 -= 0.5 #Subtract 30 min from starting time
					# Original approach
					# time1round += 0.5 #if the start and end at the same hour, add a 30m interval to end time
				current_trip = [[time0round, time1round], [row.WHYFROM, row.WHYTO]] #Rebuild a new current day trip log
			 
			else:#Continue the current day trip with the new trip segment
				#Find the end hour of this segment
				time1 = int(row.ENDTIME)
				time1round = round(time1/100%1/0.6 * 2)/2 #Round ending time to nearest 30min
				# time1round = int(floor(time1/100) + round(time1/100%1/0.6,0)) #Original round to hour approach

				# Append the segment to current trip log
				if time1round <= current_trip[0][-1]: #If the current trip ending time is less or equal to the previous one
					time1round =current_trip[0][-1]+0.5 #Add 30 mins to the last ending time
				current_trip[0].append(time1round)
				current_trip[1].append(row.WHYTO)

	#Write the current day trip (since no more new day trip, we will write this separately)
	if save_file:
		csv_writer.writerow(current_trip[0])
		csv_writer.writerow(current_trip[1])
	
	current_trip = tuple([tuple(i) for i in current_trip])
	trip.append(current_trip)
	#return a list of day trips, each day trip is in the format of tuple(time tuple, location tuple)
	return trip

def trip_ls_input(file_name, mode = 'w', save_file = True):
	# Given raw file name and mode, return either newly processed data or data stored in existing file
	# The file_name input is always the raw file name, depends on mode:
		# If 'w': raw file is processed:
			# If save_file=True: Processed file is saved
	# 	If 'r': processed file name is inferred and read.
	current_path = pathlib.Path(os.getcwd()) #Get the current working directory
	if mode == 'w': #Data writing mode: Process raw inputs and return and save generated data 
		processed_file_name = file_name.split('.csv')[0]+'_processed'+'.csv'
		raw_trip_path = str(current_path.parent.parent)+'/Data/'+file_name #Raw data file path
		trip_ls = trip_data_processing(raw_trip_path, processed_file_name, save_file)
		trip_ls = utils.container_conv(trip_ls, tuple)
	elif mode == 'r': #Data reading mode: Read the existing processed file
		# Generate processed data file name from given file_name
		processed_file_name = file_name.split('.csv')[0]+'_processed'+'.csv'
		processed_trip_path = 'output/'+ processed_file_name #Default output folder

		trip_ls_raw = utils.csv_read(processed_trip_path, output_type = list) #Read the file into a list of rows
		trip_ls = utils.ls2trip_ls(trip_ls_raw, tuple)
	return trip_ls
#################################################################################################################################################
def ini_ppl_gen(trip, m, n = 10):
	# Generate initial population (a list of journey maps = A list of lists of journeys) and top_n list
	# The initial population is a set of CJMs at length from 2 to m each with n most occuring patterns 
	# There are (m-1) CJMs and each with n journeys inside (m, n given by user)
	# Within each CJM, the n journeys are the n most occuring journeys at length i where i is from 2 to m
	# Input:
		# trip: List of journeys, each entry is a journey
		# m: Maximum length for a journey in a CJM
		# n: Number of journeys in each CJM
	# Output: 
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
		ind_trip = trip_set[0] #Extract the individual trip - ((time), (activities))
		ind_l = len(ind_trip[0]) #Individual trip length
		if i<= num_max: #If total assignment number hasn't reach max yet
			if ind_l <= m and (n - len(ini_ppl[ind_l])) > 0: #Length of journey falls within range and there is space in the bucket
				ini_ppl[ind_l].extend([ind_trip]) #Add the trip to the bucket (bucket list length < n)
				top_n.append(ind_trip) #Add the trip to the Top_n list
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

def cal_mutual_dist_baseline(data_0, data_1, dist_dict =  collections.defaultdict(utils.dd)):
	# Parallel computing: calculate the distances between different pairs of data entries and load them into dist_dict
	# Input: 
	# Let's think of the complete dataset to be a combination of different datasets. data_0 is one of them (the target one), and data_1 is the set start from data_0 to the end (including data_0) of the complete dataset 
	# data_0: A data list specific to this process in parallel computing
	# data_1: The data starts from data_0 in the entire dataset to the end of the complete dataset (including data_0)
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

def data_sort_labmachine(folder_path, data_name_format, output_file_name = 'result_sorted.csv'):
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
	output_path = folder_path+'/' + output_file_name
	final.to_csv(output_path, index=False) #Save the result list to csv

	#Return final result list (each row is an optimization result for a trial) and output file path (in case of future use)
	# final is a dataframe
	return final

def data_translate_labmachine(result_file, simplified_activities = True, keep_origin = True):
	# Given a result file and traslate it into a df using default book and save it to CSV
	# Input:
		# result_file: Full file path for read, each row is a CJM
		# simplified_activities: Whether the result would be a single column dataframe or multiple columns
		# keep_origin: If original Best CJM and keys are kept
	# Output:
		# result_translated

	NHTS_book = utils.NHTS()

	df_data = pd.read_csv(result_file, converters={'Best CJM': eval}) #Read the sorted csv file and treat CJM list as value(rather than string)
	result_translated = pd.DataFrame() #Create an empty dataframe

	for idx, row in df_data.iterrows(): #Iterate over each set of cluster centers for each trial(row)
		trip_ls = row['Best CJM'] #The bset CJMs for current trial as a list, each entry is a list of individual trips

		for ind_trip in trip_ls: #Iterate over each trip/cluster center in current set and translate the trip 
			trip_current_df = row.copy().to_frame().transpose() #Create a copy of the current row
			trip_current_df.insert(3, 'Current Trip', [ind_trip]) #Add the current trip to the row
			trip_current_df.reset_index(drop=True, inplace=True) #Reset the index
			# single_col determines the format of the translated result
				# If true, a single cell dataframe containing list of activities is generated (time ignored)
				# If false, a row dataframe where each cell contains an activity for that 30 min interval (49 lists in total)
			trip_translated_df = utils.trip_translator(input_trip = ind_trip, book=NHTS_book, single_col = simplified_activities)
			trip_current_df = pd.concat([trip_current_df,trip_translated_df],axis = 1) #Concat translated result with current trip

			result_translated = result_translated.append(trip_current_df) #Append the current trip to translated result

	if keep_origin == False: #If the original Best CJM and keys are kept
		del result_translated['Best CJM']
		del result_translated['Key']
	
	name_extension = 'simplified' if simplified_activities else 'full'
	result_translated.to_csv(str(pathlib.Path(result_file).parent)+'/result_translated_'+ name_extension +'.csv',  index=False) #Save to CSV and ignore index

	#Output: A csv file with each row as a cluster center for a specific trial in the trial column
	return result_translated

# def raw_translate(raw_trip_ls, simplified_activities = True):
# 	#Default book
# 	NHTS_book = utils.NHTS()
	
# 	result_translated = pd.DataFrame() #Create an empty dataframe

# 	for ind_trip in raw_trip_ls: #Iterate over each trip/cluster center in current set and translate the trip 
		
# 		trip_current_df = pd.DataFrame([[ind_trip]], columns = ['Current Trip'] ) #Create an empty df
# 		trip_current_df.reset_index(drop=True, inplace=True)
# 		# single_col determines the format of the translated result
# 			# If true, a single cell dataframe containing list of activities is generated (time ignored)
# 			# If false, a row dataframe where each cell contains an activity for that 30 min interval (49 lists in total)
# 		trip_translated_df = utils.trip_translator(input_trip = ind_trip, book=NHTS_book, single_col = simplified_activities)
# 		trip_current_df = pd.concat([trip_current_df,trip_translated_df],axis = 1) #Concat translated result with current trip

# 		result_translated = result_translated.append(trip_current_df) #Append the current trip to translated result
	
# 	name_extension = 'simplified' if simplified_activities else 'full'
# 	result_translated.to_csv('result_translated_raw_'+name_extension+ '.csv',  index=False) #Save to CSV and ignore index

# 	#Output: A csv file with each row as a cluster center for a specific trial in the trial column
# 	return result_translated

def most_frequent_activities(result_translated, result_file_path):
	col_names = utils.col_names_30min() #Get names of the columns (hours)
	sort_dict = {} #A dictionary of sorted activities (key is the time range, value is another dictionary where key is activity, value is number of appearances)
	sort_ls = [] #A list of most frequent activities at each hour (len = len(col_names)) excluding 'Nothing'
	freq1 = [] #A list of frequencies, freq1 refers to freq including 'Nothing'
	freq2 = [] #A list of frequencies, freq2 refers to freq excluding 'Nothing'
	# sort_ls_raw = [] #A list of most frequent activities at each hour (len = len(col_names)) including 'Nothing'
	for time_range in col_names: # Iterate over 30-min time range
		sort_dict_df = result_translated[time_range].value_counts() #Count the number of appearances of activities for this time range
		sort_dict[time_range] = sort_dict_df.to_dict() #Create a dictionary: key is time range, value is another dictionary (key is activities, value is # of appearances)

		activities_ls = list(sort_dict_df.index.values) #List of activities filled at this hour, in the order of number of appearances
		sort_ls.append('Nothing')
		freq1.append(1) #Default 'Nothing' freq1 = 1, freq1 refers to freq including 'Nothing'
		freq2.append(1) #Default 'Nothing' freq2 = 1, freq2 refers to freq excluding 'Nothing'
		# sort_ls_raw.append(activities_ls[0])
		for activity in activities_ls: #The 1st activity that is not 'Nothing' would be the most appearing one
			# if activity != 'Nothing' and sort_dict_df[activity]>= 50: #Only records if appeared over 50 times
			if activity != 'Nothing':
				sort_ls[-1] = activity #Replace 'Nothing' with another activity and stop
				freq1[-1] = sort_dict_df[activity]/( sum(sort_dict[time_range].values()) ) # Change freq to the freq with 'Nothing'
				freq2[-1] = sort_dict_df[activity]/( sum(sort_dict[time_range].values()) - sort_dict[time_range]['Nothing']) # Change freq to the freq without 'Nothing'
				break

	pd.DataFrame([sort_ls,freq1, freq2], columns = col_names).to_csv(str(pathlib.Path(result_file_path).parent)+'/frequent_activities.csv',  index=False)
	return sort_dict, sort_ls

#################################################################################################################################################
def tripls2df(trip_ls, t_interval):
	# Convert trip_ls to a df with columns for each time interval given by the time interval
	# Input:
		# trip_ls: List of lists, in which each entry is a tuple of trip: ((time), (activities))
		# t_interval: Time window width (individual trip will be assigned to based on window)
	# Output:
		# trip_df: A df with each row as the activity df for each trip

	####################################################
	trip_df = pd.DataFrame()

	for idx, ind_trip in enumerate(trip_ls):
		# We re-use the trip_translator fn to produce the df with column names as time windows and values are activities in that interval
		activity_df = utils.trip_translator(input_trip = ind_trip, t_interval = t_interval, default_val= 0)
		activity_df.iloc[0,:] = utils.data2mc(activity_df.values.tolist()[0]) #Performs zero padding for all the inter-activity entries
		trip_df = pd.concat([trip_df,activity_df]) #Append to the existing df
	return trip_df

def tripdf2mcls(trip_df, mc_len):
	# Given a trip df and MC length, generate data lists for the specific window length over the entire df
	# Input:
		# trip_df: A dataframe of trips, where each row is a trip and activities are assigned to columns of corresponding times (empty columns are assigned default_val in tripls2df's call of utils.trip_translator)
		# mc_len: Integer, defines how long a MC is, or how long the window to crop from
	# Output:
		# mc_crop_dict: A dictionary of list, keyed by window index and valued by list of Markov chains - cropped out of trip_df
			# A list of Markov chains is a list of list, in which each entry is a Markov chains
				# A Markov chain is a list of activities 
		# col_name_ls: A list of list, in which each entry is a list of column names, each corresponding to the list in mc_crop_ls
	#####################################################
	i_max = trip_df.shape[1]-1 #Maximum index of trip_df column
	mc_crop_ls = [] #Initialize the output mc list
	# for i in np.arange(0,trip_df.shape[1]-1,mc_len):
	col_names = list(trip_df.columns) #Complete list of names
	col_name_ls = [] #Output list of names for each entry in mc_crop_ls
	# Warnings for transition number/mc_len
	if mc_len < 1:
		raise Exception('Min of mc_len is 1')
	if mc_len > trip_df.shape[1]-1: 
		raise Exception('The number of transitions desired is larger than max transitions available')

	# Iterate over each time window
	# mc_crop_dict= {} #Initialize a dictionary
	mc_crop_dict = collections.defaultdict(list) #Initialize a dictionary with default value empty list
	for i in range(trip_df.shape[1]-mc_len): #Iterate over column (window) indices (starting from 0)
		# The last starting index is always (trip_df.shape[1]-mc_len-1)
		i_end = min(i+mc_len, i_max) #Find out the end index for cropping (avoid out of index with min fn)
		ind_df = trip_df.iloc[:,int(i):i_end+1] #Crop the specific columns out of trip_df
		ind_ls = ind_df.values.tolist() #Convert the ind_df to list of item, where each item is a row in the original df
		ind_mc_ls = datals2mcls(ind_ls) #Convert the data list to mc list - preprocessing: zero padding & index replacing 
		if ind_mc_ls: #Only appends if it's not completely empty
			ind_mc_ls_new = [mc_ls[np.nonzero(mc_ls)[0][0]:] for mc_ls in ind_mc_ls] #Remove leading zeros in the MC
			mc_crop_dict[i] = ind_mc_ls_new #Add the new mc_ls to the mc_crop
			col_name_ls.append(col_names[i:i_end+1])
	return mc_crop_dict, col_name_ls

def datals2mcls(data_ls):
	# Process all the chains in a list (data_ls) to a list of Markov chains
	# Input:
		# data_ls: A list of data, where each data is a chain of activities
	# Output:
		# mc_ls: A list of Markov chains, where each MC is a chain of activities under operations
	# This function performs utils.data2mc to all the items in data_ls. See utils.data2mc for more details.
	# Note: Zero padding performed in data2mc here maybe unnecessary if tripls2df has performed zero padding for inter-activity entries 
	#####################################################
	mc_ls = [] #Initizalize the mc list
	for data in data_ls:
		mc = utils.data2mc(data) #Datapoint to mc using data2mc: zero padding & index replacing
		if sum(mc) != 0: #Drop the MC with all zeros
			mc_ls.append(mc) #Append to mc_ls
	return mc_ls

def bayesian_clustering(mc_ls, alpha, s, prior_input = ['uniform'], **kwargs):
	# Given a list of data, perform Bayesian clustering 
	# input:
		# mc_ls: A list of list, where each list is a MC generated from an individual data point
		# alpha: Global precision
		# s: Number of states for Markov chain 
		# prior_input: A list of data for prior generation. The 1st element, prior_input[0], is always the prior type
	# Output:
		# cluster_result: A dictionary that contains all relevant results keyed as follows:
			# cluster_ls: See below
			# trans_ls: List of transitional matrices, where each entry is a transitional matrix
			# cluster_ls_id: Same format as cluster_ls, but each count matrix is translated using id_dict
	# Comments:
		# 1. Print statements ending with comment '--PrintStatement' are used for showing progress of clustering
		# 2. Please see below for the comments
	################### Important Variables #####################
	# 	cluster_ls: List of clusters, each entry is a list of count matrices (in list format). Can convert to count_ls.
	# 	count_ls: List of count matrices, each entry is a count matrix combined from the corresponding cluster (list of count matrices)
	# 	prior_ls: List of prior matrices, each entry is a count matrix (generated from uniform_prior function)
	# 	id_dict: ID bidirectional dictionary, key is unique id, value is hased count matrix (tuple format)
	# 	dist_dict_mat: Distance dictionary, key is a pair of IDs for count matrices (from id_dict), value is the distance. This dict contains all past values (a repository)
	# 	dist_rank: Distance rank for current count_ls, in a tuple format (key_pair, distance) and ascending.
	# 	mc_temp: A single cluster, composed of several existing count matrices
	###################### Initialization #######################
	# last_time = time.time() #Timepoint of last step
	clustering_result = collections.defaultdict(list)
	KL_dict = kwargs['KL_dict'] if 'KL_dict' in kwargs.keys() else {} #A dictionary for KL_distance_input from kwargs

	ini_count_ls = mcls2mat(mc_ls, s)[1] #Get the initial count_ls (duplicates exist)
	
	cluster_ls = utils.initial_cluster_ls(ini_count_ls) #Generate the initial cluster list (duplicates allowed in count_ls)
	count_ls = utils.cluster_ls2count_ls(cluster_ls)[0] #Generate the list of count matrix (no duplicates)
	# Note: 
		# The new count_ls can have duplicates, but not those in original count_ls since they are merged. len(count_ls) = len(cluster_ls) <= len(ini_count_ls) 
	########## Create/Read ini_id_dict ##########
	suffix_default = ''
	suffix = (KL_dict['id_suffix'] if 'id_suffix' in KL_dict.keys() else '')+suffix_default #Read suffix of dictionary from kwargs if given (then add suffix_default)
	suffix = (str(suffix) if str(suffix).startswith('_') else '_' + str(suffix)) if suffix else '' #Add underscore if there isn't
	dict_file_path = 'output/idDict'+suffix+'.json'
	ini_id_dict = bidict(utils.dict_val2tuple(utils.json2dict(dict_file_path)[0])) if os.path.isfile(dict_file_path) else None #Reads ini_id_dict if it exists (with the same name), else None
	ini_id_dict = id_modifier(new_val_ls = ini_count_ls,id_dict = ini_id_dict, save_dict = False) #Updates ini_id_dict with entries from ini_count_ls (we will save it later on if meaningful clusters are generated)
	###################### Initialization #######################
	
	# Compute distances and ids for unique count matrices in the new count_ls
	id_dict, dist_dict_mat = KL_distance_input(count_ls = count_ls,id_dict = ini_id_dict, **KL_dict) #Create id_dict from ini_id_dict with count_ls and compute distances between count_ls
	# Sort dist_dict_mat
	dist_rank = sorted(dist_dict_mat.items(), key = lambda x: x[1]) #Sort the dictionary based on distances between count_ls - output a tuple of (idx pair, distance)
	# print('Initial distane computed and ranked!') #--PrintStatement
	
	# Generate a prior_ls for count_ls, one prior for each count matrix or cluster
	# Prior type given by prior_input[0], prior_data generated from prior_input
	if prior_input[0] == 'dev': #dev prior case
		# Parameter: prior_input = ['dev', mc_ls_prior]
		# Parameter: prior_data = [mc_ls_prior, s, prior_ratio, cluster_len]
		prior_ratio = alpha/s * (1/len(prior_input[1])) #Prior ratio should be alpha/(m_prior*s)
		prior_data = [prior_input[1], s, prior_ratio, len(cluster_ls)]
	else: #Default uniform case
		# Parameter: prior_input = ['uniform']
		# Parameter: prior_data = [cluster_ls, alpha]
		prior_data = [cluster_ls, alpha]
	prior_ls = prior_generator(prior_data, type = prior_input[0]) #Generate prior_ls

	p_new = posterior_Bayesian(cluster_ls, prior_ls) #Compute the initial posterior
	p_old = float('-inf') #Initial old posterior

	# print('The initial number of clusters is',len(cluster_ls)) #--PrintStatement
	###################### Loop #######################
	# print('Clustering Starts!') #--PrintStatement
	run_no1 = 0 #Debug index for outer loop
	while p_new > p_old: #Continue loop if if the previous run generated a better posterior
		p_old = p_new #Replace p_old with p_new, the best posterior from previous run
		idx = 0 #Index for inner loop (dist_rank) - restart for every loop
		###########################Test Variable###########################	
		run_no2 = 0 #Debug index for inner loop
		run_no1 += 1 #Debug index for outer loop
		# print('----------------------------')#--PrintStatement
		# print('External loop no.',run_no1, end = '')#--PrintStatement
		# print('External loop no.',run_no1)
		# last_time = time.time() #Timepoint of last step
		########################### Notes for Loop###########################
		# The following loop goes through every possible merging in current dist_rank setting
		# The following loop will produce a best posterior p_new but since dist_rank will be updated with clusters, this p_new may not be the best p_new (thus the outer loop)
		# In the inner loop, we simply loop over all the items in the dist_rank
		# If a merging happens, we would change the clusters (so is the dist_rank)
		########################### Notes for Loop###########################
		# print(' And the number of clusters is', len(cluster_ls))#--PrintStatement
		while idx <= len(dist_rank)-1: #dist_rank is dynamically updated
			run_no2 += 1
			key_pair = dist_rank[idx][0] #Get the key pair
			###################### Debug #######################
			# print('The current run_no2 is', run_no2) 

			# key_pair_check = list(zip(*dist_rank))[0]
			# key_ls_check = list(set(item for sublist in key_pair_check for item in sublist)) #All the keys in dist_rank
			# count_ls_check = [utils.id2count(key_temp, id_dict) for key_temp in key_ls_check] #Corresponding count mat from dist_rank
			# key_ls = [utils.count2id(item, id_dict) for item in count_ls] #Lookup all the keys in id_dict for count mat in count_ls
			
			# if not utils.check_item_in_ls(key_ls, key_ls_check): #Chech if all the keys in dist_rank are also in count_ls
			# # if not utils.check_item_in_ls(count_ls, count_ls_check): #Chech if all matrices in dist_rank are also in count_ls
			# 	print('This is breaking No. 0 happens at run_no2',run_no2)
			# 	break
			###################### Debug #######################
			# Find the clusters to be merged given by key_pair
			id1 = count_ls.index(utils.container_conv(id_dict[key_pair[0]], list)) #Find index (of the count matrix) in count_ls for the 1st matrix referred by the key pair
			id2 = count_ls.index(utils.container_conv(id_dict[key_pair[1]], list)) #Find index (of the count matrix) in count_ls for the 2nd matrix referred by the key pair
			# Note for id1&id2: The count matrix referred can have duplicates in count_ls, count_ls.index only returns the 1st index

			# Merge the clusters and produce new cluster_ls and 
			cluster_ls_temp, cluster_temp = merge_cluster(cluster_ls, id1, id2) #Merge the cluster in cluster_ls and produce a temporary cluster list and generated mc
			prior_ls_temp = merge_count(prior_ls, id1, id2) #Merge two clusters' priors and produce a temporary prior list

			p_temp = posterior_Bayesian(cluster_ls_temp, prior_ls_temp) #Compute the temporary posterior using the temp cluster_ls and prior_ls     
			
			if p_temp > p_new: #If the merged clusters (temp) have a higher posterior, we would accept
				###################### Debug #######################
				# print('The current run_no2 is', run_no2, end = '') #--PrintStatement
				# print(' The original p_new is',p_new, 'and the new p_new is',p_temp, 'and the number of new cluster centers is', len(cluster_ls_temp)) #--PrintStatement

				# a_mat = utils.cluster2count(cluster_ls[id1])
				# b_mat = utils.cluster2count(cluster_ls[id2])
				# a_id = utils.count2id(a_mat, id_dict) #id for the cluster that was merged (original)
				# b_id = utils.count2id(b_mat, id_dict)
				# print('Merged clusters have keys',a_id, b_id)
				###################### Debug #######################

				#Assign the new cluster data to existing cluster data
				cluster_ls = cluster_ls_temp 
				prior_ls = prior_ls_temp
				count_ls = utils.cluster_ls2count_ls(cluster_ls)[0] #Produce the new count_ls
				p_new = p_temp		

				#Iterate over all dist_rank to remove any entries with merged cluster
				dist_rank_temp = [dist_pair for dist_pair in dist_rank if not bool(set(key_pair).intersection(dist_pair[0]))]
				# print('The number of units removed from dist_rank is' ,(len(dist_rank) - len(dist_rank_temp)))
				dist_rank = dist_rank_temp #The dist_rank_temp is for printing purpose

				# Update the id_dict with the newly generated cluster (a list of count mat)
				count_temp = utils.cluster2count(cluster_temp) #Convert the newly merged cluster to a count mat
				
				if utils.container_conv(count_temp, tuple) not in id_dict.inverse:
					# if count_temp not in id_dict (the new cluster/count mat hasn't been encountered before):
					id_dict = id_modifier(new_val_ls = [count_temp], id_dict = id_dict) #Update the original id dictionary with the new count mat
				id_temp = id_dict.inverse[utils.container_conv(count_temp, tuple)] #Get id for the new cluster

				# Compute the distance between newly generated cluster and the cluster_ls (cluster_ls), save it to a temporary dictionary 
				dist_dict_mat_temp = calc_MC_distance(mat_ls1 = [count_temp], mat_ls2 = count_ls, dist_dict_mat = collections.defaultdict(float), id_dict = id_dict) #Generate the dist dict between count_temp and cluster_ls
				
				###################### Debug #######################
				# if run_no2 == 2:
				# 	print(dist_dict_mat_temp)
				# if utils.check_item_in_ls(count_ls,[a_mat, b_mat], False): #Check if original count mat (a,b) in new count mat
				# for key_pair_temp in dist_dict_mat_temp.keys():
				# 	if a_id in key_pair_temp or b_id in key_pair_temp:
					# print('Error for count_ls!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

				# if utils.check_dist_rank_keys(list(dist_dict_mat_temp.items()), a_id,b_id):
				# 	print('Checkpoint 1 for dist_rank')
				###################### Debug #######################

				# Merge the newly computed distance with the existing distance dictionary
				dist_rank_new = list(dist_dict_mat_temp.items()) #Build a liast of dist_rank from the temp dictionary
				dist_rank.extend(dist_rank_new) #Merge the new dist_rank with original one
				dist_rank = sorted(dist_rank, key=lambda x: x[1]) #Re-sort the merged dist_rank by distance (last step of merging doesn't sort)
				
				# print('The length of added dist_rank is',len(dist_rank_new), 'and the length of the new dist_rank',len(dist_rank), 'while the length of count_ls is',len(count_ls))
				
				dist_dict_mat = utils.merge_dict(dist_dict_mat, dist_dict_mat_temp) #Merge the two dictionary for future purpose
				# Note: dist_dict_mat and dist_rank are not equilvalent: dist_dict_mat is a repository of all distances while dist_rank is a ranked list of current mc
				# break #Break for inner loop
			idx += 1
			# break #Break for outer loop
		# utils.dict2json('temp/cluster_ls_'+str(run_no1)+'.json',cluster_ls)
		# utils.dict2json('temp/count_ls_'+str(run_no1)+'.json',count_ls)
	count_ls = utils.cluster_ls2count_ls(cluster_ls)[0] #Convert final clusters to list of count mat
	trans_ls = []
	if len(cluster_ls) > 1: #Save ini_id_dict if meaningful clusters are generated
		# Note: the ini_id_dict is saved if meaningful clusters are generated and it updates the previous version of ini_id_dict file by deleting origin file and saving the new one
		id_modifier(new_val_ls = [], id_dict = ini_id_dict, save_dict = True, **KL_dict) #Use an empty new_val_ls to save ini_id_dict
	for nmat in count_ls: #Convert list of count matrices to list of transitional matrices
		trans_ls.append(utils.count2trans(nmat)) #Convert count matrix and append to list
	# Output
	clustering_result['cluster_ls'] = cluster_ls
	clustering_result['trans_ls'] =trans_ls
	clustering_result['cluster_ls_id'] = [[id_dict.inverse[utils.container_conv(count_mat, tuple)] for count_mat in cluster] for cluster in cluster_ls]
	return clustering_result

def prior_generator(prior_data, type = 'uniform'):
	# Incomplete
	# Generate priors based on the types of priors needed and data provided
	# Input:
		# type: Type of priors
		# prior_data: A dataset provided for prior generation, varies according to type
			# uniform prior: prior_data = [cluster_ls, alpha]
			# dev prior: prior_data = [mc_ls_prior, s, prior_ratio, cluster_len]
	if type == 'uniform':
		cluster_ls = prior_data[0]
		alpha = prior_data[1] 
		prior_ls = utils.uniform_prior_ls(cluster_ls, alpha)
	elif type == 'dev':
		# Using data from the prior dataset to generate prior
		# prior_data = [mc_ls_prior, s, prior_ratio, cluster_len]
		mc_ls_prior = prior_data[0] #Prior dataset
		s = prior_data[1] #State number needed to produce nmat
		prior_ratio = prior_data[2] #Ratio for the prior to be adjusted
		cluster_len = prior_data[3] #Length of cluster_ls to generate prior_ls (1 prior mat per cluster)

		ini_count_ls = mcls2mat(mc_ls_prior, s)[1] #Generate the initial count_ls
		# Sum up count mat in ini_count_ls and normalize it 
		prior_mat = utils.dev_prior_mat(ini_count_ls, s, prior_ratio) 
		
		prior_ls = [prior_mat]*cluster_len

	return prior_ls

def mcls2mat(mc_ls, s):
	# Given a list of mc, convert all of them to a list of transitional matrices
	# input:
	# 	mc_ls: List of markov chains(processed), in which each entry is a list of activities
	# output: 
	# 	trans_ls: A list of transitional matrices, index corresponds to mc_ls
	# 	count_ls: A list of count matrices, index corresponds to mc_ls
	trans_ls = []
	count_ls = []
	m0 = len(mc_ls) #This is the raw m > real m since there are repetitive count mat
	
	for mc in mc_ls:
		pmat, nmat = utils.mc2mat(mc, s) #Generate nmat and pmat
		trans_ls.append(pmat)
		count_ls.append(nmat)
	# count_ls = utils.unique_ls(count_ls) #Get list of unique count matrices 
	# trans_ls = utils.unique_ls(trans_ls) #Get list of unique trans matrices 
	return trans_ls, count_ls

def id_modifier(new_val_ls, id_dict = None, f_hash = utils.container_conv, save_dict = False, **kwargs):
	# Modifies a dictionary (if not given, create a new bidict) of ids with the new input variable
	# Input:
		# new_val_ls: List of data points that need to be added to id_dict (can have duplicates in new_val_ls or existing entries in id_dict)
		# id_dict: ID dictionary, default empty bidict
		# f_hash: Function to change input hashable (default list of lists to tuple using container_conv)
		# save_dict: If to save dictionary
		# kwargs:
			# id_suffix: Suffix for saving id_dict
			# id_dict_path: Parent folder for id_dict
	# Output:
	# 	id_dict: Dictionary of ids where key is the id#, value is the data (in case of hashable)
	if not id_dict: #If not bidict supplied, default empty bidict
		id_dict = bidict()

	for data in new_val_ls:
		if f_hash(data, tuple) not in id_dict.inverse: #Check if the new value already in dictionary
			id_dict[uuid.uuid4().hex] = f_hash(data, tuple) #Change the data to hashable(tuple) for dict keys
			# raise Exception('Input value already in id_dict!!!\n')
	if save_dict: #Save dict with suffix (from kwargs if given)
		# suffix_default = time.strftime("%Y-%m-%d-%H-%M") #Default suffix (we are using time of day for special purposes, usually this is empty) 
		suffix_default = ''
		suffix = (kwargs['id_suffix'] if 'id_suffix' in kwargs.keys() else '')+suffix_default #Read suffix of dictionary from kwargs if given (then add suffix_default)
		suffix = (str(suffix) if str(suffix).startswith('_') else '_' + str(suffix)) if suffix else '' #Add underscore if there isn't
		id_dict_path = kwargs['id_dict_path'] if 'id_dict_path' in kwargs.keys() else 'output/idDict/'
		dict_file_path = ''.join([id_dict_path, 'idDict', suffix, '.json'])
		utils.dict2json(dict_file_path, dict(id_dict)) #Save the dist dictionary and id dictionary to a json file
	return id_dict

def KL_distance_input(count_ls, id_dict = None, save_dict = False, **kwargs):
	# Given a list of count matrix, initialize/update their IDs in id_dict and create distance dictionary
	# Input:
		# count_ls: List of count matrix, note that there can be duplicates in the count_ls
		# id_dict: Initial bidirectional dict that to be updated, default empty
		# save_dict: If saving both id_dict and dist_dict_mat
		# kwargs: Keyword arguments
			# KL_suffix: Suffix for saving of both dictionary
	# Output:
		# id_dict: Bidirectional id dictionary that includes both entries from input id_dict and count_ls (removing any duplicates)
		# dist_dict_mat: Dictionary of distances, keyed by pair of ids of matrices from count_ls (not all mat from id_dict are included)
	# Note: 
		# dist_dict_mat only contains matrices in count_ls, not everything in id_dict. 
		# id_dict can have an initial entries and will be merged with entries from count_ls
	##################################################
	# # The input object is a string, thus treat it as the file name for reading
	# id_dict, dist_dict_mat = utils.json2dict('output/'+input_obj)
	# id_dict = bidict(utils.dict_val2tuple(id_dict)) #When onverting to dict from bidict (to save to json), the tuples are converted to ls, thus needs to be converted back to tuple for bidict
	# dist_dict_mat = utils.dict_key2tuple(dist_dict_mat) #The keys of dist_dict_mat are in str format, thus needs to be converted
	# return id_dict, dist_dict_mat
	##################################################
	# count_ls_unique = utils.unique_ls(count_ls) #Get list of unique count matrices (remove duplicates)
	id_dict = id_modifier(new_val_ls = count_ls, id_dict=id_dict) #Create/Update id_dict from input matrices
	dist_dict_mat = calc_MC_distance(count_ls, count_ls, id_dict, dist_dict_mat = collections.defaultdict(float), mat_type = 'count') #Compute the distance dictionary for entries in count_ls
	if save_dict: #Save both dicts with suffix (from kwargs if given)
		suffix_default = time.strftime("%Y-%m-%d-%H-%M") #Default suffix (we are using time of day for special purposes, usually this is empty) 
		suffix = (kwargs['KL_suffix'] if 'KL_suffix' in kwargs.keys() else '')+suffix_default #Read suffix of saved distance dictionary from kwargs if given (and add suffix_default)
		suffix = (str(suffix) if str(suffix).startswith('_') else '_' + str(suffix)) if suffix else '' #Add underscore if there isn't
		utils.dict2json('output/ini_dist_dict_Bayesian'+suffix+'.json', dict(id_dict), dist_dict_mat) #Save the dist dictionary and id dictionary to a json file
	return id_dict, dist_dict_mat

def calc_MC_distance(mat_ls1, mat_ls2, id_dict, f_hash=utils.container_conv ,dist_dict_mat = collections.defaultdict(float), mat_type = 'count', p_out = False):
	# Given two lists of count matrices, compute the KL distances between their count matrices and save them to a dictionary
	# Input:
	# 	mat_ls1, mat_ls2: Lists of matrices, each element is an np array, they can be count (will be converted to trans) or trans matrices (given by mat_type)
	# 	id_dict: Dictionary for count matrices
	# 	f_hash: Function to change data point hashable (default list of lists to tuple) for id_dict
	# 	dist_dict_mat: Dictionary of distances, default 0.
	# 	mat_type: Type of matrices mat_ls are ('count' or 'trans'). Must be same across id_dict and dist_dict_mat keys
	# Output:
	# 	dist_dict_mat:
	# 	-Dictionary of distances, key is a tuple of ids
	
	if mat_type == 'count': #Generate the transitional matrices if input mats are count matrices
		trans_ls1 = [utils.count2trans(nmat) for nmat in mat_ls1]
		trans_ls2 = [utils.count2trans(nmat) for nmat in mat_ls2]
	else: #Input is transitional matrix list
		trans_ls1 = mat_ls1
		trans_ls2 = mat_ls2
	
	for idx1, mat1 in enumerate(mat_ls1): #Iterate over 1st matrix list (use original mat_ls than trans_ls for id purposes)
		for idx2, mat2 in enumerate(mat_ls2): #Iterate over 2nd matrix list
			if not np.array_equal(mat1, mat2):
				# Computes dist only if two matrices are different 
				id1 = id_dict.inverse[f_hash(mat1, tuple)] #Generate id for current data from id_dict
				id2 = id_dict.inverse[f_hash(mat2, tuple)]
				if dist_dict_mat[(id1, id2)] == 0:
					# And there are no existing distances stored in dictionary
					trans1 = trans_ls1[idx1] #Find the transitional matrix from list
					trans2 = trans_ls2[idx2]
					KL_dist = utils.mat_KL_dist(trans1, trans2) #Distance compute fn input has to be trans matrices
					dist_dict_mat[(id1, id2)] = KL_dist
					dist_dict_mat[(id2, id1)] = KL_dist
					if p_out:
						print(id1, id2)
	return dist_dict_mat

def posterior_Bayesian(cluster_ls, prior_ls, mode = 'log'):
	# Compute the posteriori
	count_ls, m_ls = utils.cluster_ls2count_ls(cluster_ls) #Compute count mat list (a list of count matrices, each for one cluster) from the cluster list (each entry is a list of count matrices)
	f1 = utils.f1_comp(cluster_ls, prior_ls, mode)	
	f2 = utils.f2_comp(count_ls, prior_ls, mode)

	if mode == 'log':
		# print('Exporting log(posterior)')
		posterior = f1 + f2 #Computed f1 and f2 are actually log(f1) and log(f2), thus return log(posterior)
	else:
		# print('Exporting real posterior')
		posterior = f1*f2 #Real f1 and f2 return real posterior

	# print('f1 = '+str(f1))
	# print('f2 = '+str(f2))
	return posterior

def merge_cluster(cluster_ls, *idx):
	# Given a list of lists, combine two lists given by idx within the original list 
	if len(idx) != 2:
		raise Exception("Only tuple of 2 can be accepted!")

	new_cluster_ls = copy.deepcopy(cluster_ls) #Create a copy of original list
	id1 = min(idx)
	id2 = max(idx)
	new_cluster_ls[id1] = new_cluster_ls[id1] + new_cluster_ls[id2] #Combine the merged data to form a new list of length = sum of length
	del new_cluster_ls[id2] #Remove the one being merged
	# print('Merged cluster at',idx,'and resulted cluster is',new_cluster_ls[id1])
	
	# Return the new cluster list and the newly generated cluster (merged cluster - a new list of count mat)
	return new_cluster_ls, new_cluster_ls[id1]

def merge_count(count_ls, *idx):
	# Given a list of matrices, combine two values given by idx by summing them up

	if len(idx) != 2:
		raise Exception("Only tuple of 2 can be accepted!")

	new_count_ls = copy.deepcopy(count_ls) #Create a copy of original list
	val1 = new_count_ls[idx[0]]
	val2 = new_count_ls[idx[1]]
	new_count_ls[idx[0]] =  np.add(np.asarray(val1), np.asarray(val2)).tolist()
	del new_count_ls[idx[1]] #Remove the one being merged

	return new_count_ls

#################################################################################################################################################
def plot_vec_centers(trip_data_df, result_loc, raw_trip_ls, top_n = float("inf")):
	# Given set of centers in df format, plot and save all the centers
	# Input:
	# 	trip_data_df: Dataframe of trip centers with four columns [Trial No, Key, Best CJM, Score]
	# 	result_loc: Location of current file
	img_folder_name = 'IMG' #Name of folder to save images to
	img_folder_path =  result_loc +'/'+img_folder_name
	pathlib.Path(img_folder_path).mkdir(parents=True, exist_ok=True) #Create the image folder (and parent folder) if not exists yet 
	# if not os.path.exists(img_folder_path):
	# 	os.mkdir(img_folder_path)
	key = 'Best CJM'
	for index, ind_trip in trip_data_df[key].reset_index(drop = True).iteritems():
		if index < top_n:
			if isinstance(ind_trip,str):
				ind_trip = ast.literal_eval(ind_trip)
			fig = utils.vec_plot(ind_trip,[], raw_trip_ls)
			img_file_path = img_folder_path + '/img_'+str(index+1)
			fig.savefig(img_file_path)
			plt.close()

def plot_freq_centers_bar(sort_dict, result_loc):
	NHTS_book = utils.NHTS()
	activity_label_dict = {v: k for k, v in NHTS_book.items()}
	activity_label_raw = list(activity_label_dict.keys())[:-1]

	time_label_raw = utils.col_names_30min()

	img_folder_name = 'IMG' #Name of folder to save images to

	img_folder_path =  result_loc +'/'+img_folder_name

	pathlib.Path(img_folder_path).mkdir(parents=True, exist_ok=True) #Create the image folder (and parent folder) if not exists yet 
	# if not os.path.exists(img_folder_path):
	# 	os.mkdir(img_folder_path)
	time, activity, freq0, dt, da, df = [],[],[],[],[],[]
	for key0, value0 in sort_dict.items(): #Iterate over time intervals
		for key1, value1 in value0.items(): #Iterate over activities available
			if key1 != 'Nothing': #'Nothing" is not counted
				time.append(time_label_raw.index(key0)+1) #Append the location of times (add 1 for idx starts from 1)
				activity.append(activity_label_raw.index(key1)+1) #Append the activity index
				df.append(value1)
	freq0 = np.zeros(len(time))
	dt = np.ones(len(time))
	da = np.ones(len(time))

	x, y, z, dx, dy, dz = time, activity, freq0, dt, da, df
	xlabel = time_label_raw
	ylabel = activity_label_raw

	fig, ax = utils.ax3d_plot(x,y,z,dx,dy,dz,xlabel, ylabel,7)
	fig.savefig(img_folder_path+'/FrequencyBar.png')
	plt.close()

def plot_freq_centers_heat(sort_dict, result_loc):
	NHTS_book = utils.NHTS()
	activity_label_dict = {v: k for k, v in NHTS_book.items()}
	activity_label_raw = list(activity_label_dict.keys())[:-1]

	time_label_raw = utils.col_names_30min()

	img_folder_name = 'IMG' #Name of folder to save images to
	img_folder_path =  result_loc +'/'+img_folder_name
	pathlib.Path(img_folder_path).mkdir(parents=True, exist_ok=True) #Create the image folder (and parent folder) if not exists yet 
	# if not os.path.exists(img_folder_path):
	# 	os.mkdir(img_folder_path)

	# xlabel = time_label_raw #x label
	xlabel = activity_label_raw #x label

	# ylabel = activity_label_raw #y label
	ylabel = time_label_raw #y label
	freq = np.zeros([len(xlabel), len(ylabel)])
	for key0, value0 in sort_dict.items(): #Iterate over time intervals
		for key1, value1 in value0.items(): #Iterate over activities available
			if key1 != 'Nothing': #'Nothing" is not counted
				time_idx = time_label_raw.index(key0)
				activity_idx = activity_label_raw.index(key1)
				
				# freq[time_idx][activity_idx] = value1
				freq[activity_idx][time_idx] = value1

	fig, ax1 = utils.ax3d_plot_heat(freq, ylabel, xlabel)
	fig.savefig(img_folder_path+'/FrequencyHeat.png')
	plt.close()

#################################################### DataAnalysis ############################################################
def processed_data_generator(dataFilePath, baselineFilePath, resultNo, s = 21, func_type = 'Write'):
	# Mix the baseline result with the clustered result and generate a new excel (with multiple sheets)
	# Input:
		# dataFilePath: Path for clustered result data
		# baselineFilePath: Path for baseline result data (unclustered)
		# resultNo; List of transitions where there are meaningful clustering results
	# Output:
		# processedFilePath
	baselineFile = pd.ExcelFile(baselineFilePath) #Generate a ExcelFile object for raw file
	processedFilePath = dataFilePath.split('.xlsx')[0]+'_processed.xlsx' #Generate a processed file path
	if func_type == 'Write':
		processedFileWriter = pd.ExcelWriter(processedFilePath) #Create a processed file writer

		for transitionNo in resultNo: #Input results only for those with meaningful result
			sheetName = baselineFile.sheet_names[transitionNo] #Get the current sheet name
			baselineT = baselineFile.parse(sheet_name = sheetName, header = None) #Read the baseline sheet
			SpecificT = pd.read_excel(dataFilePath, sheet_name = transitionNo,header = None) #Read the result sheet
			newT =  pd.DataFrame(np.nan, index= baselineT.index , columns=SpecificT.columns) #Create new table/sheet with appropriate size
			newT.loc[:,:] = baselineT #Fill the new table with the baseline data and then edit this new table

			windowArray = ast.literal_eval(SpecificT.loc[0,2]) #Get the list of meaningful time window indices (which time windows within the current transitional # have meaningful results), this index is 1-indexed

			# We want to replace data for indices in windowArray in baselineT with the data in SpecificT
			rowRangeArr = utils.calcRow(windowArray,s, ttype = 'Result')
			baselineRangeArr = utils.calcRow(windowArray,s, ttype = 'Baseline')
			for idx, rowRange in enumerate(rowRangeArr):
				baselineRange = baselineRangeArr[idx] #Get the ranges for baseline file
				temp_df = SpecificT.loc[list(range(*rowRange))+[rowRange[1]]] #The df that will replace origin
				temp_df.index = newT.loc[list(range(*baselineRange))+[baselineRange[1]]].index #Re-index the df so it can be filled in newT
				newT.loc[list(range(*baselineRange))+[baselineRange[1]]] = temp_df #Fill the rows in newT with temp_df
			newT.to_excel(processedFileWriter,sheet_name = sheetName, header = False, index = False) #Save newT to a new sheet
	
		processedFileWriter.close()
	elif func_type != 'Read':
		raise Exception('No such function type! Input Read or Write')
	return processedFilePath

# def raw_data_generator(rawFilePath, s= 21):
# 	# Given rawFilePath, return the needed data 
# 	# Input:
# 		# rawFilePath: File path for the raw data file
# 	# Output:
# 		# 
# 	rawFile = pd.ExcelFile(rawFilePath) #Generate a ExcelFile object for raw file
# 	for sheetName in baselineFile.sheet_names:


def node_col2layout(state_valid):
	# Unused!!!
	# Both shifts below keep the remainder of (state/s) the same and thus keeps state labels the same
	col1 = (np.asarray(state_valid)+2*s*chainNo).tolist() #1st column, shifts by 2*s*chainNo to avoid repetition
	col2 = (col1 + s).tolist() #Shifts s from column to avoid repetition
	node_tot[0].append(col1)
	node_tot[1].append(col2)
	return None

def pmat2dict(pmat,plot_type,start_state = 1):
	# Given a transitional matrix, converts it to a dictionary keyed by edge pairs and valued by trans prob
	# Input:
		# pmat: Transitional matrix with state as intergers from [start_state,start_state+s], with s = pmat.shape[0]
		# plot_type: Type of processing model:
			# 'step': We will duplicate nodes and treat the trans mat as one step forward transitions from original state to the duplicated nodes. 
			# 'homogeneous': We will use the original nodes.
		# start_state: An integer that the states start from.
	# Output:
		# mc_dict: A dictionary keyed by edge pairs in tuples and valued by trans prob.
	# Note: Since we want the nodes to be distributed along two sides, we will duplicate the states and final number of nodes will be 2*valid_states
	pmat = np.asarray(pmat) if not isinstance(pmat, np.ndarray) else pmat #Convert pmat to np array if it's not
	s = pmat.shape[0] #Get number of states
	mc_dict = collections.defaultdict(float) #Initialize the dictionary
	edges = np.argwhere(pmat)+start_state #Get the raw edges (without modification to indices)
	if plot_type == 'step': #We will indices of the end of edges
		edges[:, 1] += s #Modify indices of end of edge/column
	edges = [tuple(i) for i in edges.tolist()] #Convert edges to list of tuples
	probs = pmat[np.nonzero(pmat)].tolist() #Get all the nonzero values in pmat
	mc_dict = dict(zip(edges,probs))  #Zip edge and prob and create the dict
	return mc_dict

def simulate_mc_sheet(pmat_sheet, n_steps = 20000, **kwargs):
	# UNUSED!!!
	# Given a sheet of Markov chain transitional matrices, simulate its transitions and record states
	# Input:
		# pmat_sheet: A list of list that contains all the trans mat for current sheet
			# Each entry of pmat_sheet is pmat_window, which contains all the pmat for a single time window
			# Each entry of pmat_window is a pmat
		# n_steps
		# initial_state
	# Output:
		# states
		# 
	
	# Extract titles for figs and axes from kwargs if there are any
	titles_dict = {'title_sheet': kwargs['title_sheet']} if 'title_sheet' in kwargs.keys() else {'title_sheet': 'Markov Chain Simulation'}
	titles_dict['title_win'] = kwargs['title_win']+' '+str(n_steps)+' steps' if 'title_win' in kwargs.keys() else str(n_steps)+' steps'
	# Generate the figs and axes
	figs, axs = fig_generator(fig_num = 1, ax_num = [1],titles_dict = titles_dict) #Create the figs and axes

	pmat_flatten = [pmat for pmat_window in pmat_sheet for pmat in pmat_window] #Get a flattend list of pmats
	for pmat in pmat_flatten:
		states, dist_steps = utils.simulate_mc(pmat, n_steps = 20000, plot_simulation = True, **kwargs)

	file_path = kwargs['file_path'] if 'file_path' in kwargs.keys() else '/MCsimulation.pdf' #Extract the file path
	fig2pdf(file_path,fig_num = 'all') #Save all figures to PDF	

######### Plotting in DataAnalysis #############
def plot_mc_sheet(mc_sheet, titles_dict, transCountArr, plot_type = 'homogeneous', fig_type = 'multiple', s = 21, save_pdf = False, 
                  fig_kw = {}, plot_kw = {}, **kwargs):
	# Given all MCs for time windows within one sheet, plot them depends on fig_type:
		# fig_type='single': Plot each time window on one column with one figure contains all time windows
		# fig_type='multiple': Plot each time window on a single figure, and produce multiple figures
	# Input:
		# mc_sheet: A dictionary contains: mc_data_mat, cluster_size
			# mc_data_mat: A list of list that contains all the nodes&edges, and structured as follows:
				# Each entry of mc_data_mat is a list, mc_window, contains all the nodes&edges for the time window
				# Each entry of mc_window is a container, mc_data, for one MC within the window
			# cluster_size_ls: A list of cluster sizes (a list of list) for current number of transitions
				# Each entry of cluster_size_ls is cluster_size, a list of numbers of datapoints for 1 time window
		# titles_dict: A dictionary contains titles for figures and axes:
			# 'title_win': Title for each time window
			# 'title_sheet': Title for each sheet (contains all MCs of the same transition #)
		# transCountArr: A list of # of trans mat in each time window (each entry is a number)
		# plot_type: The type of plot to be generated, this determines type of container in mc_window
			# 'step' or 'homogeneous': mc_data is a dictionary where keys are edges (node pairs) and values are edge weights
			# 'heatmap' or 'simulation-bar' or 'simulation-line' or 'chord': mc_data is a transitional matrix
		# fig_type: 'single' or 'multiple' that determines the # of figures to be generated
		# s: Number of states
		# save_pdf: Save figure(s) as PDF or not:
			# True: Save PDF without showing figs
			# False: Showing figs without saving as PDF
		# kwargs:
			# resultFolderPath
			# resultAffix
	# Note: This function primarily deals with axes and figure level settings. 
		# To modify plotting within the axes, check out utils.plot_mc_dict.
	################################################
	if plot_type.startswith('simulation-bar'): #If plot type is simulation bar plot
	# We will add a translation figure at the beginning to translate state number to labels
		fig, ax = plt.subplots(num = -2, figsize = [10,10])
		fig.suptitle('Table for State Labels')
		ax.set_axis_off()
		table = ax.table(cellText = list(utils.NHTS_new().items()), 
		                 colLabels = ['State Number','State Label'],
		                 fontsize = 100,
		                 cellLoc ='center', 
		                 loc ='center',
		                 colColours = ['palegreen']*2)
		table.scale(1.1, 2)
	elif plot_type == 'chord': #Settings for chord diagram
		# Only 1 chord graph per figure
		fig_type = 'individual' #Override fig_type to 'individual' so 1 chord per figure
		fig_kw['fig_size'] = [15]*2 #Override any figure size in fig_kw

	if fig_type == 'single':
		fig_num = 1
		# If there is only 1 fig, we will create a gridspace with unit rows
		ax_num = [transCountArr, len(transCountArr)] # Since only 1 fig, only 1 element in ax_num with row and column number of axes in figure
		print(fig_num, ax_num)
	elif fig_type =='multiple':
		# If there are multiple figures:
			#Number of fig = number of windows = # of columns = len(transCountArr)
			#Each fig has number of axes = number of trans mat in the time window=transCountArr
		ax_num, fig_num = [transCountArr, len(transCountArr)]
		# print('ax_num =',ax_num,'fig_num =',fig_num)
	elif fig_type == 'individual':
		# If one plot (multiple plots in a window) has 1 figure
		fig_num = sum(transCountArr) #Number of figure = sum of number of axes for diffeent time windows
		ax_num = [1]*fig_num #Number of axes is 1 axes/per figure
		fig_title = titles_dict['title_win']
		titles_dict['title_win'] = [item for item, count in zip(fig_title, transCountArr) for i in range(count)] #Repeat 
	else:
		raise Exception('No such figure type! Choose single or multiple')

	# Create the axes and figures using the given parameters
	figs, axs = fig_generator(fig_num, ax_num, titles_dict = titles_dict, **fig_kw)
	
	mc_flatten = [mc_data for mc_window in mc_sheet['mc_data_mat'] for mc_data in mc_window]
	cluster_size = [i for cluster_size in mc_sheet['cluster_size_ls'] for i in cluster_size]
	for idx, ax in enumerate(axs):
		# print('idx is',idx, 'with data is',mc_flatten[idx])
		utils.plot_mc(mc_data = mc_flatten[idx], cluster_size = cluster_size[idx] ,plot_type = plot_type, ax=ax, **plot_kw) #Plot MC based on plot_type on current axe
	
	if save_pdf == True:
		# Save all the figures in this sheet into a single PDF
		resultFolderPath = kwargs['resultFolderPath'] if 'resultFolderPath' in kwargs.keys() else '' #Extract result folder path if there is such (a/b/c/ format)
		prefix = kwargs['prefix']+'-' if 'prefix' in kwargs.keys() else '' #Extract result file affix if there is such
		suffix = kwargs['suffix'] if 'suffix' in kwargs.keys() else '' #Extract result file suffix if there is such - this is applied to file name
		suffix = suffix + '-' if suffix else '' #Check if '-' is needed (not if empty)
		resultFilePath = resultFolderPath + prefix +titles_dict['title_sheet']+suffix+'.pdf' #Generate PDF file path

		fig2pdf(resultFilePath, fig_num ='all')
	else:
		plt.show()
		
	return None

def fig_generator(fig_num, ax_num, titles_dict, **kwargs):
	# Given set of figure parameters, return a list of (figure, ax) objects for plotting
	# Input:
		# fig_num: Number of figures
		# ax_num: A List, in which each entry is either a list or int which gives row&column number of grids for corresponding fig:
			# The composition of ax_num is dependent on fig_num and len(ax_num), which is # of time windows:
				# fig_num = 1 & len(ax_num) > 1: 
					# This is the case where there are multiple time windows (each can have multiple plots) need to be plotted (columns) on the same figure
					# Use plt.figure and fig.add_subplot
					# ax_num[0] is an array of number of rows for each column - transCountArr
					# ax_num[1] is the total column number - len(WindowArr)
				# fig_num != 1 or len(ax_num) = 1:
					# This corresponds to the case where each a single figure only contains graphs for a single time winodw (can contain multiple plots):
						# A: Only 1 time winodw -> fig_num =1 and len(ax_num) = 1 -> 
						# B: Multiple time window -> fig_num > 1 and len(ax_num) > 1
					# In this case, ax_num = transCountArr - List of # of plots in each time window
					# ax_num[i] is number of plots in each figure i
		# titles_dict: A dictionary contains titles for figures and axes:
			# 'title_win': List of titles for each time window
			# 'title_sheet': A single title for current sheet (the sheet contains all MCs of the same transitions
		# kwargs:
			# border_dist:
			# fig_size: 
			# ax_kw: A dictionary that contains keyword arguments that will be added to fig.add_subplot, plt.subplot2grid and plt.subplots
			# fig_kw: All other keyword arguments are used for figure parameters (plt.figure and plt.subplots)
	# Output:
		# figs, axs: List of figures and axes

	fig_kw = kwargs.copy() #Create a copy of figure keyword dict
	ax_kw = fig_kw.pop('ax_kw') if 'ax_kw' in fig_kw.keys() else {} #Remove and assign axes keyword dict (from fig_kw) and leaves only kwargs for the fig_kw
	fig_size = fig_kw.pop('fig_size') if 'fig_size' in fig_kw.keys() else (15,10) #Figure size (default (15,10))
	border_dist = fig_kw.pop('border_dist') if 'border_dist' in fig_kw.keys() else 0.2 #Distance between borders of subplots (default 0.2)
	suptitle_kw = fig_kw.pop('suptitle_kw') if 'suptitle_kw' in fig_kw.keys() else {} #Keyword arguments for suptitle of figure (default empty)

	# fig_size = kwargs['fig_size'] if 'fig_size' in kwargs.keys() else (15,10) #Figure size (default (15,10))
	# c_layout = kwargs['c_layout'] if 'c_layout' in kwargs.keys() else False #Indicator for constrained layout (default True)
	# t_layout = kwargs['t_layout'] if 't_layout' in kwargs.keys() else False #Indicator for tight layout (default True)
	
	
	figs = [] #A list of figures
	axs = [] #A list of axes
	plt.rcParams.update({'figure.max_open_warning': 0})
	if fig_num == 1 and len(ax_num)>1: #There is only one figure with with multiple time windows, i.e. fig_type = 'single'
		fig = plt.figure(figsize = fig_size, **fig_kw) #Build figure
		fig.suptitle(titles_dict['title_sheet'], **suptitle_kw) #Add the title to the figure
		ax_title = titles_dict['title_win'] #Get the list of titles for the axes, one for each time window/column

		# Compute number of rows and columns for grids on figure 
		grid_row_num = np.lcm.reduce(ax_num[0]) #Row # = Least common multiplier for all val in transCountArr
		grid_col_num = ax_num[1] #Column # = # of time windows

		#Create grid and set border distances
		gs = fig.add_gridspec(grid_row_num, grid_col_num, 
			left=border_dist, right=1-border_dist, top=1-border_dist, bottom=border_dist,
			wspace = border_dist, hspace = border_dist)  
		
		# Iterate over each col & row to add axes
		for i in range(grid_col_num): #Iterate over each column (time window)
			ax_row_num = ax_num[0][i] #Get the number of rows/MCs for current column
			row_num = int(grid_row_num/ax_row_num) #Compute number of rows occupied for each MC for current col
			# print(gs)
			for j in range(ax_row_num): #Iterate over each row (MC/plot)
				
				ax = fig.add_subplot(gs[j*row_num:(j+1)*row_num-1,i], label = str(i), **ax_kw) #Add the axe
				axs.append(ax)
			axs[-ax_row_num].set_title(ax_title[i]) #Only the 1st axe in column has title
		figs.append(fig)

	else: #There are multiple figures, i.e. fig_type = 'multiple', each figure corresponding to a time window
		# This also applies for a single time winodw for a single figure
		fig_title = titles_dict['title_win'] #fig_title is a list of title for each figure/time window - corresponding to a subtitle
		for i in range(fig_num): #Iterate over each figure
			fig = plt.figure(num = i, figsize = fig_size, **fig_kw) 
			fig_ax_num = ax_num[i] #Number of plots/MCs = # of trans mat on this figure
			ax_col_num = 2 #Max number of columns for axes (user-defined)
			if fig_ax_num > ax_col_num: #Multiple column setting
				# Number of figures is larger than column number -> Multi columns for most rows
				fig_row_num = ceil(fig_ax_num/ax_col_num) #Number of rows for axes
				extra_ax_num = fig_ax_num%ax_col_num #Number of axes leftover
				extra_row_num = bool(extra_ax_num) #Number of (extra) rows that are incompleted filled (with leftover axes), 0 or 1
				
				grid_col_num = np.lcm(ax_col_num,extra_ax_num) or ax_col_num #Number of columns for the grids to be placed is the least common multiplier for max # of axe columns and # of extra axes (if extra=0, this equal to ax_col_num)
				# print('grid_col_num=',grid_col_num,'extra_ax_num=',extra_ax_num,'ax_col_num=',ax_col_num)
				if extra_ax_num != 0:
					grid_col_span0 = int(grid_col_num/extra_ax_num) #Grid span for axes in rows that are incompletely filled
				grid_col_span1 = int(grid_col_num/ax_col_num) #Grid span for axes in rows that are completely filled

				for j in range(extra_ax_num): #Fill 1st row with the extra axes
					axs.append(plt.subplot2grid([fig_row_num,grid_col_num], [0,j*grid_col_span0], fig = fig, colspan = grid_col_span0, **ax_kw))

				for j in range(fig_ax_num-extra_ax_num): #Iterate over each axes that belongs to rows that are completely filled
					axs.append(plt.subplot2grid([fig_row_num,grid_col_num], 
						[floor(j/ax_col_num)+extra_row_num, (j%ax_col_num)*grid_col_span1], fig = fig,
						colspan = grid_col_span1, **ax_kw))
			else:
				# 1-Column figure case
				fig, ax = plt.subplots(nrows = fig_ax_num, ncols =1, num = i, figsize = fig_size, squeeze = False, subplot_kw = ax_kw, **fig_kw) #Build figure with only 1 column
				axs.extend(ax.flatten()) #ax is a 2d array of axes (squeeze=False) thus needs to be merged with axs (extend than append)
			################
			fig.suptitle(titles_dict['title_sheet']+fig_title[i], **suptitle_kw) #Get and assign the title to the figure (axes don't have a title)
			figs.append(fig)
	return figs, axs

def fig2pdf(file_path, fig_num = 'all', **kwargs):
	# Given the file path and figure numbers to be saved, save figures to a pdf
	# Input:
		# file_path: Path of file to be saved, must NOT end with '\' or '/'
		# fig_num: 'all' or list of int, list of figure numbers
	################################
	folder_path = os.path.split(file_path)[0] #Extract the folder path
	pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True) #Create the result folder (and parent folder) if not exists yet 
	
	pp = PdfPages(file_path) #Create the pdf file
	# Get the list of figures given by fig_num
	if fig_num == 'all':
		figs = list(map(plt.figure, plt.get_fignums())) #Get list of all figures opened right now
	else:
		figs = list(map(plt.figure, fig_num)) #Get list of figures given by fig_num
	for fig in figs: #Iterate over each figure
		pp.savefig(fig)

	pp.close() #Close and save the PDF
	plt.close('all') #Close all figures

	return None