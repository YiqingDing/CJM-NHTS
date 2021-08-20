import utils, collections, csv, random, os, pathlib, ast, uuid, time, copy
import pandas as pd
from math import *
import numpy as np
from bidict import bidict
import matplotlib.pyplot as plt, matplotlib.lines as ml
np.seterr('raise')

def trip_data_processing(raw_trip_path, processed_file_name = 'final_trip.csv'):
	# Read the raw trip data file and return a list of day trips and save the output trip file
	# Output:
		# trip: List of day trips, each item is a 2-tuple, with 1st item being timestamps, 2nd item being events
		# final_trip.csv: A csv file with all the day tips, every 2 lines form a day trip
	
	raw_data = pd.read_csv(raw_trip_path) #Read the data and return a pandas df
	previous_id = 0 #Initialize the ID of the day trip
	trip = []  #Create an empty trip segment list
	processed_trip_path = 'output/'+ processed_file_name #Default output folder
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
	csv_writer.writerow(current_trip[0])
	csv_writer.writerow(current_trip[1])
	
	current_trip = tuple([tuple(i) for i in current_trip])
	trip.append(current_trip)
	#return a list of day trips, each day trip is in the format of tuple(time tuple, location tuple)
	return trip

def trip_ls_input(file_name, mode = 'w'):
	# Given raw file name and mode, return either newly processed data or data stored in existing file
	# The file_name input is always the raw file name, depends on mode:
	# 	If 'w': raw file is read and processed. 
	# 	If 'r': processed file name is inferred and read.
	current_path = pathlib.Path(os.getcwd()) #Get the current working directory
	if mode == 'w': #Data writing mode: Process raw inputs and return and save generated data 
		processed_file_name = file_name.split('.csv')[0]+'_processed'+'.csv'
		raw_trip_path = str(current_path.parent.parent)+'/Data/'+file_name #Raw data file path
		trip_ls = trip_data_processing(raw_trip_path, processed_file_name)
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
def plot_vec_centers(trip_data_df, result_loc, raw_trip_ls, top_n = float("inf")):
	# Given set of centers in df format, plot and save all the centers
	# Input:
	# 	trip_data_df: Dataframe of trip centers with four columns [Trial No, Key, Best CJM, Score]
	# 	result_loc: Location of current file
	img_folder_name = 'IMG' #Name of folder to save images to
	img_folder_path =  result_loc +'/'+img_folder_name
	if not os.path.exists(img_folder_path):
		os.mkdir(img_folder_path)
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

	if not os.path.exists(img_folder_path):
		os.mkdir(img_folder_path)
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
	if not os.path.exists(img_folder_path):
		os.mkdir(img_folder_path)

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

#################################################################################################################################################
def id_modifier(new_val_ls, id_dict = bidict(), f_hash = utils.container_conv):
	# Modifier a dictionary of ids with the new input variable
	# Input:
	# 	new_val_ls: List of data points that are not in id_dict, must be unique from id_dict.inverse()!
	# 	id_dict: ID dictionary, default empty bidict()
	# 	f_hash: Function to change data point hashable (default list of lists to tuple)
	# Output:
	# 	id_dict: Dictionary of ids where key is the id#, value is the data (in case of hashable)
	for data in new_val_ls:
		# dict keys starts from start_id
		id_dict[uuid.uuid4().hex] = f_hash(data, tuple) #Change the data to hashable for dic keys
	return id_dict

def bayesian_clustering(mc_ls, alpha, s):
	# Given a list of data, perform Bayesian clustering 
	# input:
		# mc_ls: A list of list, where each list is a MC generated from an individual data point
		# alpha: Global precision
		# s: Number of states for Markov chain 
	################### Important Variables #####################
	# 	cluster_ls: List of clusters, each entry is a list of count matrices (in list format). Can convert to count_ls.
	# 	count_ls: List of count matrices, each entry is a count matrix combined from the corresponding cluster (list of count matrices)
	# 	prior_ls: List of prior matrices, each entry is a count matrix (generated from uniform_prior function)
	# 	id_dict: ID bidirectional dictionary, key is unique id, value is hased count matrix (tuple format)
	# 	dist_dict_mat: Distance dictionary, key is a pair of IDs for count matrices (from id_dict), value is the distance. This dict contains all past values (a repository)
	# 	dist_rank: Distance rank for current count_ls, in a tuple format (key_pair, distance) and ascending 
	# 	mc_temp: A single cluster, composed of several existing count matrices
	###################### Initialization #######################
	# last_time = time.time() #Timepoint of last step
	count_ls = mc_ls2mat(mc_ls, alpha, s)[1] #Get the unique count_ls

	id_dict, dist_dict_mat = KL_distance_input(count_ls) #If distances have not been computed previously 
	print('Initial distane computed!')
	# id_dict, dist_dict_mat = KL_distance_input('dist_dict_Bayesian.json') #If raw distances have been computed previously 
	dist_rank = sorted(dist_dict_mat.items(), key = lambda x: x[1]) #Sort the dictionary based on distance - output a tuple of (idx pair, distance)
	prior_ls = utils.uniform_prior_ls(count_ls, alpha) #Using the alpha, generate uniform prior with alpha_kij = 1/s for all i,j,k. Each entry in prior_ls is a count matrix (ls of ls)
	cluster_ls = [[k] for k in count_ls] #The initial clusters only have 1 count matrix per cluster

	p_new = posterior_Bayesian(cluster_ls, prior_ls) #Compute the initial posterior
	p_old = float('-inf') #Initial old posterior

	# print('The initial number of clusters is',len(cluster_ls))
	###################### Loop #######################
	print('Clustering Starts!')
	run_no1 = 0 #Debug index for outer loop
	while p_new > p_old: #Continue loop if if the previous run generated a better posterior
		p_old = p_new #Replace p_old with p_new, the best posterior from previous run
		idx = 0 #Index for inner loop (dist_rank) - restart for every loop
		###########################Test Variable###########################	
		run_no2 = 0 #Debug index for inner loop
		run_no1 += 1 #Debug index for outer loop
		print('----------------------------')
		print('External loop no.',run_no1, end = '')
		# last_time = time.time() #Timepoint of last step
		########################### Notes for Loop###########################
		# The following loop goes through every possible merging in current dist_rank setting
		# The following loop will produce a best posterior p_new but since dist_rank will be updated with clusters, this p_new may not be the best p_new (thus the outer loop)
		# In the inner loop, we simply loop over all the items in the dist_rank
		# If a merging happens, we would change the clusters (so is the dist_rank)
		########################### Notes for Loop###########################
		# print('The dist_rank has length', len(dist_rank))
		print(' And the number of clusters is', len(cluster_ls))
		while idx <= len(dist_rank)-1: #dist_rank is dynamically updated
			run_no2 += 1
			key_pair = dist_rank[idx][0] #Get the key pair
			###################### Debug #######################
			# print('The current run_no2 is', run_no2) #Basic print

			# key_pair_check = list(zip(*dist_rank))[0]
			# key_ls_check = list(set(item for sublist in key_pair_check for item in sublist)) #All the keys in dist_rank
			# count_ls_check = [utils.id2count(key_temp, id_dict) for key_temp in key_ls_check] #Corresponding count mat from dist_rank
			# key_ls = [utils.count2id(item, id_dict) for item in count_ls] #All the keys in count_ls
			
			# if not utils.check_item_in_ls(key_ls, key_ls_check): #Chech if all the keys in dist_rank are also in count_ls
			# # if not utils.check_item_in_ls(count_ls, count_ls_check): #Chech if all matrices in dist_rank are also in count_ls
			# 	print('This is breaking No. 0 happens at run_no2',run_no2)
			# 	break
			###################### Debug #######################

			id1 = count_ls.index(utils.container_conv(id_dict[key_pair[0]], list)) #Find 1st index in key pair in count_ls
			id2 = count_ls.index(utils.container_conv(id_dict[key_pair[1]], list)) #Find the 2nd index in key pair

			cluster_ls_temp, mc_temp = merge_cluster(cluster_ls, id1, id2) #Merge the cluster in cluster_ls and produce a temporary cluster list and generated mc
			prior_ls_temp = merge_count(prior_ls, id1, id2) #Merge two clusters' priors and produce a temporary prior list

			p_temp = posterior_Bayesian(cluster_ls_temp, prior_ls_temp) #Compute the temporary posterior using the temp cluster_ls and prior_ls     
			
			if p_temp > p_new: #If the merged clusters (temp) have a higher posterior, we would accept
				###################### Debug #######################
				print('The current run_no2 is', run_no2, end = '') 

				print(' The original p_new is',p_new, 'and the new p_new is',p_temp, 'and the number of new cluster centers is', len(cluster_ls_temp))

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
				dist_rank = dist_rank_temp #The temp var is for printing purpose

				# Update the id_dict with the newly generated MC (mc_temp)
				new_count_ls = [utils.cluster2count(mc_temp)] #Convert the new MC chain (i.e. a cluster, a list of count mat) to a single count mat and then pack it in a list

				id_dict = id_modifier(new_count_ls, id_dict) #Update the original id dictionary with the new count_ls
				id_temp = id_dict.inverse[utils.container_conv(new_count_ls[0], tuple)] #Get id for the new cluster
				
				# Compute the distance between newly generated MC (mc_temp) and the cluster_ls (cluster_ls), save it to a temporary dictionary 
				dist_dict_mat_temp = calc_MC_distance(mat_ls1 = new_count_ls, mat_ls2 = count_ls, dist_dict_mat = collections.defaultdict(float), id_dict = id_dict) #Generate the dist dict between mc_temp and cluster_ls
				
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
	return cluster_ls

def KL_distance_input(input_obj):
	# Initialization of distances for input
	if isinstance(input_obj, str): #Read file
		# The input object is a string, thus treat it as the file name for reading
		id_dict, dist_dict_mat = utils.json2dict('output/'+input_obj)
		id_dict = bidict(utils.dict_val2tuple(id_dict)) #When onverting to dict from bidict (to save to json), the tuples are converted to ls, thus needs to be converted back to tuple for bidict
		dist_dict_mat = utils.dict_key2tuple(dist_dict_mat) #The keys of dist_dict_mat are in str format, thus needs to be converted
		return id_dict, dist_dict_mat
	elif isinstance(input_obj, list): #Compute and save data to file
		# The input object is a list, thus treat it as the list of count matrices
		
		# start_comp = time.time()
		id_dict = id_modifier(input_obj) #Create id_dict from input matrices
		dist_dict_mat = calc_MC_distance(input_obj, input_obj, id_dict,dist_dict_mat = collections.defaultdict(float), mat_type = 'count') #Compute the distance dictionary
		utils.dict2json('output/dist_dict_Bayesian.json', dict(id_dict), dist_dict_mat) #Save the dist dictionary and id dictionary
		# utils.dict2json('output/dist_dict_Bayesian.json', dist_dict_mat)
		return id_dict, dist_dict_mat

def calc_MC_distance(mat_ls1, mat_ls2, id_dict, f_hash=utils.container_conv ,dist_dict_mat = collections.defaultdict(float), mat_type = 'count', p_out = False):
	# Given two lists of count matrices, compute the KL distances between their count matrices and save them to a dictionary
	# Input:
	# 	mat_ls1, mat_ls2: Lists of matrices, each element is an np array, they can be count or trans matrices (given by mat_type)
	# 	id_dict: Dictionary for count matrices
	# 	f_hash: Function to change data point hashable (default list of lists to tuple) for id_dict
	# 	dist_dict_mat: Dictionary of distances, default 0.
	# 	mat_type: Type of matrices mat_ls are ('count' or 'trans'). Must be same across id_dict and dist_dict_mat keys
	# Output:
	# 	dist_dict_mat:
	# 	-Dictionary of distances, key is a tuple of ids
	
	if mat_type == 'count': #Generate the transitional matrices if necessary
		trans_ls1 = [utils.count2trans(nmat) for nmat in mat_ls1]
		trans_ls2 = [utils.count2trans(nmat) for nmat in mat_ls2]
	else: #Input is transitional matrix list
		trans_ls1 = mat_ls1
		trans_ls2 = mat_ls2
	
	for idx1, mat1 in enumerate(mat_ls1): #Iterate over 1st matrix list (use original mat_ls than trans_ls for id purposes)
		for idx2, mat2 in enumerate(mat_ls2): #Iterate over 2nd matrix list
			if not np.array_equal(mat1, mat2):
				# Computes dist only if two matrices are different 
				id1 = id_dict.inverse[f_hash(mat1, tuple)] #Generate id for current data 
				id2 = id_dict.inverse[f_hash(mat2, tuple)]
				if dist_dict_mat[(id1, id2)] == 0:
					# And there are no existing distances stored in dictionary
					trans1 = trans_ls1[idx1] #Find the transitional matrix from list
					trans2 = trans_ls2[idx2]
					KL_dist = utils.mat_KL_dist(trans1, trans2) #input has to be trans matrices
					dist_dict_mat[(id1, id2)] = KL_dist
					dist_dict_mat[(id2, id1)] = KL_dist
					if p_out:
						print(id1, id2)
	return dist_dict_mat

def matlab_data_cluster(data_ls):
	# This function processes raw matlab data and returns a dictionary for clusters
	# Input:
		# data_ls: A list/tuple of data, each entry is a time series generated by a certain MC chain
		# m: Total number of actual MC chains
		# d: A list of integers, each entry represents the number of time series generated by a certain MC chain
			# If d is a single integer, it is assumed all MC chains generated the same number of time series
	# Notes: m and d are given in the function
	# Output:
		# cluster_dict: A dictionary where the key is an integer 
	m = 8
	d = 10
	trans_ls = utils.csv_read('SimulatedData/matlab_trans.csv', output_type = tuple)
	s = len(trans_ls[0])
	if len(trans_ls)%s != 0:
		raise Exception('Size of transitional matrix unmatch!')

	# Build d array if it is not given in a list format
	if isinstance(d,int) or len(d) == 1:
		d = [d]*m
	# Check if total length of d matches length of data_ls
	if sum(d) != len(data_ls):
		raise Exception('Mismatch length for data list!!!')
	
	length = 0
	cluster_dict = collections.defaultdict(list) #Create an empty dictionary
	for idx, l_i in enumerate(d):
		trans_mat = trans_ls[s*idx:s*idx+s]
		cluster_dict[trans_mat] = data_ls[length:length+l_i]
		length += l_i
	return cluster_dict

def tripls2df(trip_ls, t_interval):
	# Convert trip_ls to a df with columns for each time interval given by the time interval
	# Input:
		# trip_ls: List of lists, in which each entry is a tuple of trip: ((time), (activities))
		# t_interval: Time window width (individual trip will be assigned to based on window)
	####################################################
	trip_df = pd.DataFrame()
	for ind_trip in trip_ls:
		# We re-use the trip_translator fn to produce the df with column names as time intervals
		activity_df = utils.trip_translator(input_trip = ind_trip, t_interval = t_interval, default_val= 0)

		trip_df = pd.concat([trip_df,activity_df]) #Append to the existing df
	return trip_df

def tripdf2datals(trip_df, mc_len):
	# Given a trip df and MC length, generate data lists for the specific window length over the entire df
	# Input:
		# trip_df: A dataframe of trips, where each row is a trip and activities are assigned to columns of corresponding times
		# mc_len: Integer, defines how long a MC is, or how long the window to crop from
	# Output:
		# data_ls: A list of list, in which each entry is a data list
			# A data list is a list of list, in which each entry is a chain of activities
				# A chain is a list of activities 
	#####################################################
	i_max = trip_df.shape[1]-1 #Maximum index of trip_df column
	for i in np.arange(0,trip_df.shape[1]-1,mc_len):
		i_end = min(i+mc_len, i_max) #Find out the end index for cropping (avoid out of index with min fn)
		ind_df = trip_df.iloc[:,i:i_end] #Crop the specific columns out of trip_df
		ind_ls = trip

def mc_ls2mat(mc_ls, alpha, s):
	# Given a list of mc chains, convert all of them to transitional matrices
	# input:
	# 	data_ls: List of markov chains
	# output: 
	# 	trans_ls: A list of transitional matrices, index corresponds to mc_ls
	# 	count_ls: A list of count matrices, index corresponds to mc_ls
	trans_ls = []
	count_ls = []
	m0 = len(mc_ls) #This is the raw m > real m since there are repetitive count mat
	
	for mc in mc_ls:
		pmat, nmat = utils.mc2mat(mc, alpha, m0, s) #Input alpha and m0 to generate nmat such that sum of columns for each row is not equal=0 
		trans_ls.append(pmat)
		count_ls.append(nmat)
	count_ls = utils.unique_ls(count_ls) #Get list of unique count matrices 
	trans_ls = utils.unique_ls(trans_ls) #Get list of unique trans matrices 
	return trans_ls, count_ls

def posterior_Bayesian(cluster_ls, prior_ls, mode = 'log'):
	# Compute the posteriori
	count_ls, m_ls = utils.cluster_ls2count_ls(cluster_ls) #Compute count list (a list of count matrices, each for one cluster) from the cluster list (each entry is a list of count matrices)
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
	new_cluster_ls[id1] = new_cluster_ls[id1] + new_cluster_ls[id2] #Combine the merged data
	del new_cluster_ls[id2] #Remove the one being merged
	# print('Merged cluster at',idx,'and resulted cluster is',new_cluster_ls[id1])
	
	# Return the new cluster list and the newly generated MC (merge resulted MC)
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

def result_clustering(cluster_ls):
	# This function processes the computed cluster list and produce the transitional matrices for the cluster
	# Input:
		# cluster_ls: List of clusters, each cluster is a list of count matrices
	# Output:
		# cluster_dict: A dictionary where the key is an integer 
	cluster_dict = collections.defaultdict(list) #Create an empty dictionary
	for cluster in cluster_ls:
		count_mat = utils.cluster2count(cluster)
		trans_mat = utils.count2trans(count_mat)
		trans_mat_hash = utils.container_conv(trans_mat, tuple)
		cluster_dict[trans_mat_hash] = cluster
	return cluster_dict
