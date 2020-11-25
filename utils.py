# Utilis file that contains all the essential function for data processing, etc
import numpy as np
from math import *
import matplotlib.pyplot as pl
import collections, random, bisect, json
import pandas as pd
# R change wd: setwd("~/Google Drive/School/Stanford/Research/Journey Map/Markov Chain Paper/Code & Data/NHTS")

def cal_dist(ind_trip, ind_top):
	#Given two trip segments, calculate their relative levienshtein distance
	#Input: Two trip segments in the form of ((time), (activities))
	#Output: Scalar Levienshtein distance
	
	Ea = ind_trip[1] #Extract events list (discard the time dimension)
	Er = ind_top[1]

	#Convert events list to a string
	Ea_str = [str(i) for i in Ea] #Convert list of numbers to list of strings
	Er_str = [str(i) for i in Er]
	Ja = ''.join(Ea_str) #Create a string of events (no demiliter)
	Jr = ''.join(Er_str)
	dist = cal_lev_dist(Ja, Jr) #Compute the Levienshtein distance
	return dist

def cal_lev_dist(Ja, Jr):
	#calculate the levienshtein distance between two journeys
	#Input: Two strings of data (no delimiter)
	#Ouput: A scalar
	size_x = len(Ja)+1
	size_y = len(Jr)+1
	matrix = np.zeros((size_x, size_y))
	for x in range(size_x): #Generate first row
		matrix [x, 0] = x
	for y in range(size_y):
		matrix [0, y] = y

	for x in range(1, size_x):
		for y in range(1, size_y):
			if Ja[x-1] == Jr[y-1]:
				matrix[x,y]= min(matrix[x-1, y]+1, matrix[x-1, y-1], matrix[x, y-1]+1)
			else:
				matrix [x,y] = min(matrix[x-1,y], matrix[x-1,y-1], matrix[x,y-1])+1
	return matrix[size_x - 1, size_y - 1]
	# return matrix

def cal_fitness(trip_ls, center_dict, dist_dict = {}):
	# Calculate the fitness function of the current cluster 
	# Input:
	# trip_ls: List of data entries (serving as constant)
	# center_dict: A dictionary of centers for current CJM, where if key is a center, returns a list of cluster samples, if key is a sample, returns its center (a tuple)
	# dist_dict: 2-d distance dictionary, default empty
	#Output: 
	# fitness: A scalar value
	lev_sum = 0
	len_sum = 0
	for sample in trip_ls: #Iterate over all data entries
		center = center_dict[sample] if isinstance(center_dict[sample], tuple) else center_dict[sample][0]
		lev_sum += dist_dict[center][sample] if bool(dist_dict) else cal_dist(sample, center) #Numerator sum
		len_sum += len(sample[0]) #Denominator sum
	# print('len_sum is '+str(len_sum))
	
	fitness = 1 - lev_sum/len_sum
	return fitness

def cal_NoR(center_dict):
	#Calculate the number of representatives(centers) in J_R
	NoR_min = 5
	NoR = 0
	for key in center_dict:
		if isinstance(center_dict[key], list):
			NoR += 1
	# NoR = min
	return NoR

def cal_Silhouette(center_ls, center_dict, dist_dict = {}):
	# Calculate Silhouette Index
	# Input: 
	# center_ls: List of centers
	# center_dict: Dictionary of centers, if key is center, returns a list of assignments, if not a center, return key's center
	# dist_dict: Dictionary of distances between different pairs, Default empty
	# Output: A dictionary of Silhouette Index, with mean Silhouette Index

	a_i = {} #Average distance of sample i to other samples within the same cluster (given a trip, find its a_i)
	b_i = {} #Smallest average distance of i to all points in another cluster (given a trip, find its b_i)
	s_i = {} #Silhouette index dictionary: For each sample i, there is an index

	center_ls1 = [center for center in center_ls if len(center_dict[center])!= 0]
	#We use center_ls since all samples are recorded in center_dict
	for center0 in center_ls1:
		ls0 = center_dict[center0] #Extract the list of entries belong to center_0 (current cluster)
		if len(ls0) == 1: #If there is only one item in the cluster (can be the center itself) 
			s_i[ls0[0]] = 0 #The Silhouette index for a single item is 0
		else: #If there are more than 1 item in the cluster
			for sample0 in ls0: #Extract the individual sample from the list for current cluster
				ls_a = ls0[:] #Copy the original cluster as cluster a
				ls_a.remove(sample0) #Remove the original item from cluster a
				d_a = cal_dist_sum(sample0, ls_a, dist_dict) #Calculate the sum of d_ij (sum of distances between sample 0 to other samples within the cluster)
				a_i[sample0] = d_a / (len(ls0) - 1) #Calculate and save a_i

				b_min = float('inf') #Initialize the b value
				center_ls_b = center_ls1[:] #Copy the original list of centers as list b
				center_ls_b.remove(center0) #Remove the current center from list b
				for center1 in center_ls_b: #Iterate over all centers except current one
					ls_1 = center_dict[center1] #Extract list of entries for center_1 cluster
					b_new = cal_dist_sum(sample0, ls_1, dist_dict)/len(ls_1) #Calculate average of distances from current trip to all trips in the new cluster
					b_min = min(b_new, b_min) #Update the b_min
				b_i[sample0] = b_min #Save the b_i

				s_i_temp = 0 if (b_i[sample0] == a_i[sample0]) else (b_i[sample0] - a_i[sample0])/max(b_i[sample0], a_i[sample0])
				s_i[sample0] = 0 if isnan(s_i_temp) else s_i_temp
	s_i['mean'] = sum(s_i.values())/float(len(s_i)) #Caclculate the average Silhouette index
	
	return s_i

def cal_dist_sum(sample0, sample_ls, dist_dict = {}):
	#Calculate sum of distances between one sample and all elements of sample_ls using cal_dist function
	d_sum = 0
	if not bool(dist_dict): #If dist_dict is empty (not distance dictionary given)
		for sample1 in sample_ls:
			d_sum += cal_dist(sample0, sample1) #Compute and sum distance
	else: #If dist_dict is not empty (given distance dictionary)
		for sample1 in sample_ls: 
			d_sum += dist_dict[sample0][sample1] #Extract and sum distance from given dictionary
	return d_sum

def cal_mutual_dist(data_ls, dist_dict =  collections.defaultdict(lambda: collections.defaultdict(int))):
	# Calculate the distances between different pairs of data entries and load them into dist_dict
	# Input: 
	# data_ls: List of data entries
	# dist_dict: Existing distance dictionary, default = empty 2d dictionary
	# Output: Dictionary of dstances between different pairs of data entries, dist_dict[item1][item2]
	for idx_1, item_1 in enumerate(data_ls[:-1]): #Iterate over the entire list until the item second to the last
		for item_2 in data_ls[idx_1+1:]: #Iterate over all the items after item_1 until the item second to the last
			dist = cal_dist(item_1, item_2)
			dist_dict[item_1][item_2] = dist
			dist_dict[item_2][item_1] = dist #Symmetry

	return dist_dict

def dd():
    return collections.defaultdict(int)

def cal_cross_dist(ls1, ls2, dist_dict = collections.defaultdict(lambda: collections.defaultdict(int))):
	# Calculate all the distance pairs between two lists of data entries and load them into dist_dict
	# Input: 
	# ls1, ls2: Lists of data entries
	# dist_dict: Existing distance dictionary, default = empty 2d dictionary
	# Ouput: Dictionary of dstances between different pairs of data entries, dist_dict[item1][item2]
	for item_1 in ls1:
		for item_2 in ls2:
			dist = cal_dist(item_1, item_2)
			dist_dict[item_1][item_2] = dist
			dist_dict[item_2][item_1] = dist #Symmetry
	return dist_dict

def ga_mutation_gene(ind_current, mode = 'addition', top_ls = [], n = 1):
	# Add a gene from top population (ppl-individual-gene-DNA) to current individual
	# Or delete a gene from current individual
	# Input:
	# ind_current: A list (an individual) of current genes - list of tuples
	# mode: 'addition' or 'deletion' of a gene
	# top_ls: A list of top genes to chose from for addition
	# n: Number of genes to be added or deleted
	# Output: A list of new genes (a new individual)

	ind_new = ind_current[:] #Make a copy rather than editing the current individual
	if mode == 'addition':
		if bool(top_ls): 
			ind_new.append(random.choice(top_ls))
		else:
			print('Please input the top population!')
	elif mode == 'deletion' and len(ind_new)> 1:
		ind_new.remove(random.choice(ind_current))
	return ind_new

def ga_mutation_dna(**kwargs):
	# Add or delete a DNA to/from a gene within the current individual (ppl-individual-gene-DNA)
	# kwargs['ind_current']: The list of current genes (an individual)
	# kwargs['mode']: 'addition' or 'deletion' of a dna
	# kwargs['n']: Number of DNAs to be added/deleted to/from the family
	# Output an updated individual (a CJM - list of CJs, each as a tuple)
	ind_new = kwargs['ind_current'][:] #Make a copy rather than editing the current individual
	gene_idx = random.randint(0,len(ind_new)-1) #Randomly choose a gene to edit (need index for later insertion)
	gene = tuple2ls(kwargs['ind_current'][gene_idx]) #Get the chosen gene (CJ) (make a list copy, originally tuple)

	if kwargs['mode'] == 'addition':
		event_new = random.randint(1,19) #Generate a new event
		time_new = int(random.randint(gene[0][0]*2, gene[0][-1]*2)/2) #Generate a random time within the current gene
		dna_idx = bisect.bisect(gene[0], time_new) #Locate the index of insertion for dna on the gene
		gene[0].insert(dna_idx, time_new) #Insert time dna into gene
		gene[1].insert(dna_idx, event_new) #Insert event dna into gene
	elif kwargs['mode'] == 'deletion' and len(gene[0])> 1:
		dna_idx = random.randint(0,len(gene[0])-1) #Randomly choose a dna for deletion
		del gene[0][dna_idx] #Remove time entry
		del gene[1][dna_idx] #Remove event entry

	ind_new[gene_idx] = ls2tuple(gene) #Update the individual with gene tuple
	return ind_new

def ga_operations(cjm_current, cjm_elite_ls, top_n, ga_action_no, prob_action):
	#Execute GA operations on current CJM
	tot_action = 0 #Number of action has been taken
	for idx, rep_no in enumerate(ga_action_no): #idx: index of ga action, rep_no: repetition of current ga action
		#Determine parameters for GA operations
		top_ppl_ls = {} #Empty top population list
		if idx == 0 or idx == 1 or idx == 2: #Determine the ga action
			ga_action = ga_mutation_gene #gene action
		else:
			ga_action = ga_mutation_dna #dna action
		
		if idx == 0 or idx == 1 or idx == 3: #Determine the ga mode
			ga_mode = 'addition'
		else:
			ga_mode = 'deletion'
		
		if idx == 0: #Determine the top population list
			top_ppl_ls = top_n #Top_n population
		elif idx==1:
			top_ppl_ls = cjm_elite_ls #Elite population
		
		# Perform the GA operations with repetition
		for j in range(rep_no): #Repetition of current ga action
			if random.uniform(0, 1)<= prob_action: #Probability of action being taken
				tot_action += 1 #One action is taken
				cjm_current = ga_action(ind_current = cjm_current, mode = ga_mode, top_ls = top_ppl_ls) #Build a new cjm_current with ga action
	
	#Ensure that at least one GA action is performed on current CJM
	if tot_action == 0: #If no action being done
		return ga_operations(cjm_current, cjm_elite_ls, top_n, ga_action_no, prob_action)
	else: #If at least an action was performed
		return cjm_current

def ls2tuple(ls):
	#Convert list of lists to tuple of tuples
	return tuple(tuple(i) for i in ls)

def tuple2ls(tp):
	#Convert tuple of tuples to list of lists
	return [list(i) for i in tp]

def ls2len_dict(ls):
	#Give a list, convert to a dictionary where keys are length of list entry
	#ls: A list of data entries
	len_dict = collections.defaultdict(list)
	for item in ls:
		len_dict[len(item[0])].append(item) #item[0] since item has two parts ((time), (events))
	return len_dict

def ls_split(ls, n):
	#Split one list into n lists where each list contain several entries of the original list
	#The first n-1 entries are equal length
	k, m = divmod(len(ls), n)
	return [ls[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def merge_dict(a, b, path=None):
    # Merges dictionary b into a
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def trip_translator(input_trip, book, single_col = False):
	# Translate input trip in number format to a list of individual activities in words
	# input_trip: A single tuple of ((time), (activity))
	# book: A dictionary translating individual code for activities to strings of activities
	# single_col: 
	# 			- If the output would be a single cell dataframe or multiple cells
	# 			- If single cell, only location is recorded
	# 			- If multiple cells, each cell represent the loation at each 30 minutes (total 24 hours in step of .5 hours)
	# output:
	# 			-If single_col, output a single list of activities with no time information
	# 			-If not signle_col, output a list of 49 values where activities are inserted at the index of their times
	time_ls = input_trip[0]
	loc_ls = input_trip[1]
	
	time_idx = [x*0.5 for x in range(0,49)]
	if single_col == False: #Multiple cells
		activity_ls = ['Nothing']*49 #Create an empty list of 49 99s (total 24 hours in step of .5 hours)
		for idx, time in enumerate(time_ls):
			activity_ls[int(time*2)] = book[loc_ls[idx]] #time*2 is the index in 49 cells (0-48 index)
		# trip_df = pd.DataFrame([trip_df]) #Convert the list of activities to a dataframe row (column idx is time in 30 min)
		# trip_df.columns = time_idx #Change the col label to hours
	else:
		activity_ls = [] #Create an empty list of activities
		for loc in loc_ls:
			activity_ls.append(book[loc])
		# trip_df = pd.DataFrame(columns = ['CJM']) #Create an empty dataframe with column named "CJM"
		# trip_df.at[0,'CJM'] = trip_cell #Fill the dataframe cell with list of activities
	return activity_ls


def plot():
	yticklabel = ['Regular home activities', 'Work from home (paid)', 'Work', 'Work-related meeting / trip', 'Volunteer activities (not paid)', 'Drop off /pick up someone', 'Change type of transportation', 'Attend school as a student','Attend child care','Attend adult care', 'Buy goods (groceries, clothes, appliances, gas)', 'Buy services (dry cleaners, banking, service a car, pet care)', 'Buy meals (go out for a meal, snack, carry-out)', 'Other general errands (post office, library)', 'Recreational activities (visit parks, movies, bars, museums)', 'Exercise (go for a jog, walk, walk the dog, go to the gym)', 'Visit friends or relatives', 'Health care visit (medical, dental, therapy)','Religious or other community activities']
	ytick_dict = {i+1: yticklabel[i] for i in range(len(yticklabel))}
	ytick_dict[97] = 'Something else'

	# plt.style.use(utils.matplotlib_params())
	for ind_trip in trip:
		plt.plot(ind_trip[0],ind_trip[1])
		print(ind_trip)
	
	# plt.savefig()
	ytick = [i for i in range(1,20)]
	ytick.append(97)
	plt.yticks(ytick, [ytick_dict[i] for i in ytick])
	plt.xlim([0,24])
	plt.show()

def matplotlib_params():
	rc_params = {
	'xtick.labelsize':	40, #Tick on x-axe
	'ytick.labelsize':	40, #Tick on y-axe
	'legend.fontsize': 45,
	'axes.labelsize': 50,
	'axes.titlesize': 60,
	'font.size': 50,

	'figure.figsize': (30,20),
	'lines.linewidth': 3,
	
	# 'lines.markersize': 15,
	'axes.grid': 1,
	'axes.grid.axis': 'x',
	# 'text.usetex': True,
	'legend.loc': 'lower left'
	# 'markers.fillstyle': 'none'
	}
	return rc_params