# Utilis file that contains all the essential function for data processing, etc
import numpy as np
from math import *
import matplotlib.pyplot as plt, matplotlib.lines as ml
import collections, random, bisect, ujson, csv,ast, os, sys
from sklearn.neighbors import KernelDensity
import pandas as pd
from scipy.special import gamma, factorial, binom, gammaln
import scipy.stats
# np.seterr(divide='ignore')
# R change wd: setwd("~/Google Drive/School/Stanford/Research/Journey Map/Markov Chain Paper/Code & Data/NHTS")

# 4 different sections:
# 1. Baseline func, 2. Bayeisan clustering func, 3. General mini func, 4. Plot func

##############################Baseline Functions################################
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
	# center_dict: A dictionary of centers for current CJM;
		# If key is a cluster center, returns a list of samples blong to this center (a list); 
		# If key is an individual sample, returns its center (a tuple)
	# dist_dict: 2-d distance dictionary, default empty
	#Output: 
	# fitness: A scalar value
	lev_sum = 0
	len_sum = 0
	for sample in trip_ls: #Iterate over all data entries, compute and sum its distance with centers
		center = center_dict[sample] if isinstance(center_dict[sample], tuple) else center_dict[sample][0] #Identify the cluster center for current sample (if itself is a center, then its center is itself, else its center is given by center_dict)
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
	gene = container_conv(kwargs['ind_current'][gene_idx], list) #Get the chosen gene (CJ) (make a list copy, originally tuple)

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

	ind_new[gene_idx] = container_conv(gene, tuple) #Update the individual with gene tuple
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

def trip_translator(input_trip, book = {}, single_col = False, t_interval = 0.5, default_val = 'Nothing'):
	# Translate input trip in number format to a list of individual activities in words
	# Input:
		# input_trip: A single tuple of ((time), (activity))
		# book: A dictionary translating individual code for activities to strings of activities
		# single_col: 
			# - Whether the output would be a single cell dataframe or multiple cells
			# - If single cell (true), only activities are recorded
			# - If multiple cells (fasle), each cell represent the loation at each 30 minutes (total 24 hours in step of .5 hours)
	# Output:
		# activity_df:
			# -If single_col, output a single df of activities with no time information
			# -If not signle_col, output a df of values where activities are inserted at the index of their times, and column name is the time widow (in str)

	time_ls = input_trip[0]
	loc_ls = input_trip[1]
	
	col_tot_no = ceil(24/t_interval) #Total number of columns
	time_array = np.arange(0,24, step = t_interval).tolist() #Time array with [0,24), i.e. 24 not included, and diff of 2 entries=t_interval
	
	if single_col == False: #Multiple cells
		activity_ls = [default_val]*col_tot_no #Create an empty list of 99s (total 24 hours in step t_interval)
		#Each list entry represents a 30 min interval

		for idx, time in enumerate(time_ls): #Iterate over all the activities based on time
			# time in the range of [0, 24]
			
			# Identify the location of the activity being inserted in the df
			time_idx = min( bisect.bisect_left(time_array, time), col_tot_no-1) #min fn to ensure not out of bound
			#Translate the activity and insert in the list
			activity_ls[time_idx] = book[loc_ls[idx]] if book else loc_ls[idx] #If book is empty/default, use the original value
		
		activity_df = multi_ls2df(activity_ls, t_interval) #Convert the list into df with col names computed by t_interval
		# activity_ls = [[i] for i in activity_ls] #Convert to list of lists
	else: #Single cell
		activity_ls = [] #Create an empty list of activities
		for loc in loc_ls:
			activity_ls.append(book[loc])

		activity_df = pd.DataFrame([[activity_ls]], columns = ['Activity List'])
	return activity_df
##########################################################Bayesian Functions############################################################
#################################################################################################################################################
def KL_dist_nonsym(p1i, p2i):
	# Compute nonsymmetric KL distance between two discrete prob distributions
	# input:
		# p1i, p2i: Two discrete prob dists in vec format (they are in same length)
	d = 0 #Initializes the nonsymmetric KL distance
	for idx, p1ij in enumerate(p1i): #Iterate over each element
		p2ij = p2i[idx] #Finding the corresponding element in p2i
		if p2ij > 0 and p1ij >0: 
			# if p2ij=0, then p1ij is implied=0
			# if p1ij=0, then the term is assumed to=0 (lim=0)
			d += p1ij*log(p1ij/p2ij)
		elif p1ij == 0:
			d += 0
		elif p2ij == 0:
			d += float('inf') #Proper KL divergence definition
			# d+= 0 
	return d

def KL_dist_sym(p1i, p2i):
	# Compute symmetric KL distance between two discrete prob distributions
	# input:
		# p1i, p2i: Two discrete prob dists in vec format (they are in same length)
	d1 = KL_dist_nonsym(p1i, p2i)
	d2 = KL_dist_nonsym(p2i, p1i)
	# print('d1 = '+str(d1)+' d2 = '+str(d2))
	d_sym = (d1+d2)/2
	return d_sym

def mat_KL_dist(p1, p2):
	# Given 2 transitional matrices, compute their symmetric KL distance
	# Input:
		# p1, p2: Two array/list with the same size, both are transitional matrices
	# Output:
		# D: Symmetric KL distance
	D = 0 #Initialization
	s = len(p1) if isinstance(p1, (tuple, list)) else p1.shape[0] #s is number of states

	for idx, p1i in enumerate(p1): #Iterate over each row
		p2i = p2[idx] #Find the corresponding row in p2
		D += KL_dist_sym(p1i, p2i)/s #Compute the KL distance summand for current row
	return D

# def tripls2datals(trip_ls):
# 	# Convert a trip_ls to data_ls, (loses time information, converts certain indices)
# 	# Input:
# 		# trip_ls: A list of list object, in which each entry(list) is a two tuple (time, activities)
# 	# Output:
# 		# data_ls: A list of list, where each list is a chain generated from an individual trip
# 			# A chain is a list of activities as [activity1, activity2, activity3, ...]
# 	###########################################################################################
# 	# A chain generated by a trip is comprised of states from 1 to 21
# 	data_ls_raw = [list(i) for i in zip(*trip_ls)][1] #Flatten the list and extract all the activities
# 	data_ls = []
# 	for data_raw in data_ls_raw: #Replace all the 97 and 99 in data with 20 and 21
# 		data = [20 if x==97 else x for x in data_raw]
# 		data = [21 if x==99 else x for x in data]
# 		data_ls.append(data)
# 	return data_ls

def uniform_prior_ls(cluster_ls, alpha):
	# Given a list of clusters (each is a list of count mat), generate a list of prior count matrices
	# Input:
		# cluster_ls: List of clusters, where each cluster is a list of count mat
		# alpha: Global precision
	# Output:
		# prior_ls: A list of priors, length = length of cluster_ls
	m = len([item for cluster in cluster_ls for item in cluster]) #Length of dataset (flatten the cluster_ls)
	# Number of states (dep if count mat is in numpy array format or list format)
	s = len(cluster_ls[0][0]) if isinstance(cluster_ls[0][0], (tuple, list)) else cluster_ls[0][0].shape[0]
	# return [np.ones((s,s))*alpha/(m*s**2)]*m #Return a np array
	prior_ls = []
	prior_mat = np.asarray(uniform_prior_mat(m, s, alpha)) #Generate an np array as an individual prior matrix
	for cluster in cluster_ls: #Iterate over all clusters
		n_count = len(cluster) #Number of count mat in current cluster
		prior_ls.append((prior_mat*n_count).tolist()) #Append a prior mat equal to n_count*single_mat
	return prior_ls

def dev_prior_mat(count_ls, s, prior_ratio):
	# Sum the nmat and normalize them based on prior_ratio
	prior_mat = np.zeros([s,s], dtype = int) #Initialize the combined count mat
	f_conv = np.asarray if isinstance(count_ls[0], (list, tuple)) else lambda x: x #Determine the conversion function for data format for count mat to be np array
	for nmat in count_ls:
		# prior_mat += np.asarray(nmat) if isinstance()
		prior_mat += f_conv(nmat) #Add the converted np.ndarray mat to prior_mat
	prior_mat = discrete_normal_dist(prior_mat)*prior_ratio #Normalize the prior_mat
	prior_mat[prior_mat == 0] = sys.float_info.min
	return prior_mat.tolist()

def initial_cluster_ls(count_ls):
	# Given the initial count_ls, generate the initial cluster_ls
	# Input:
		# count_ls: A list of count mats, can have duplicates
			# Individual count mat can be np.array or list (conversion will be done)
	# Output:
		# cluster_ls: A list of clusters, each cluster is a list of count matrices
	# Notes: This function finds duplications in count_ls and merge them into one cluster
	count_ls = count_ls if isinstance(count_ls[0],(tuple, list)) else [container_conv(i, list) for i in count_ls] #Convert count mat in count_ls to list format if they are not
	cluster_ls = []
	count_ls_unique = unique_ls(count_ls) #Find the list of unique count mat
	if len(count_ls_unique) == len(count_ls): #If there are no duplicates in count_ls
		cluster_ls = [[k] for k in count_ls]
	else: #If there are duplicates in count_ls
		for nmat in count_ls_unique: #Iterate over unique count matrices
			n = count_ls.count(nmat) #Find the number of occurances for nmat in count_ls
			cluster_ls.append([nmat]*n) #Append the count mats as a cluster with length n
	return cluster_ls

def data2mc(data):
	# Given a data/chain, convert to a Markov chain
	# Input:
		# data: A list of activities where idx corresponds to time, values corresponds to activities at the time
			# If no activities ocurred at certain times, 0 is placed
			# For example: [0,3,0,4] indicates event 3 at time 1 and event 4 at time 3
	# Output:
		# mc: A Markov chain generated from the data
			# A Markov chain is a list of activities as [activity1, activity2, activity3, ...]
	# This function performs the following operation:
		# - Zero padding: Replace intermediate zeros with the activity preceeding the zero, e.g. [0,3,0,0,4,0,5] -> [0,3,3,3,4,4,5]
		# - Replaces certain discontinued values with continus values: 97 -> 20; 99 -> 21
	###########################################################################################
	mc = []
	zero_padding = 1
	#Zero padding
	last_activity = 0 #Initialize last activity
	if zero_padding == 1: #Zero padding 
		for activity in data: #Iterate over different activities within the data
			if activity != 0:
				last_activity = activity #Records current activity as the new last activity
			mc.append(last_activity) 
	else: #No zero padding
		mc = data

	#Replace all the 97 and 99 in mc with 20 and 21
	mc = [20 if x==97 else x for x in mc] 
	mc = [21 if x==99 else x for x in mc]

	return mc

def mc2mat(mc, s):
	# Convert a MC to transitional and count matrices based on the frequency of different states
	# Input:
		# mc: A list of states, states are translated to integers starting from 1
		# s: Number of states
		# Suspended Inputs:
			# alpha: Global precision (sum of prior), for null transitions
			# m: Number of chains, for null transitions
	# Output:
		# pmat: Transitional matrix
		# nmat: Count matrix
	###########################################################################################
	# The following lines create a zero nmat np array and a prior array
	# If in the end any of the rows of nmat are completely zeros, the prior array will be added
	n = len(mc) #Length of mc (time)
	nmat = np.zeros((s,s),dtype=int) #Create a zero matrix
	
	# Loop over each state in mc and record the transitions in count matrix
	for t in range(n-1): #There are n-1 transitions
		i = mc[t] #Starting state
		j = mc[t+1] #Ending state
		nmat[i-1][j-1] += 1 #-1 for python 0 index

	# Suspended function: Add an uniform prior to the all zero rows (need alpha input)
	# nmat_prior = np.asarray(uniform_prior_mat(m, s, alpha)) #Create a prior matrix (no rows are completely zeros)
	# if np.where(~nmat.any(axis=1))[0].size!=0: #Check if any of the row is completely zeros (null transition row)
	# 	# if it's completely zeros, we add the prior
	# 	nmat = (nmat +nmat_prior)
	pmat = count2trans(nmat)
	return pmat, nmat.tolist()

def count2trans(nmat):
	# Convert a matrix of counting transitions to a transitional matrix
	nmat = np.asarray(nmat) if isinstance(nmat, (list, tuple)) else nmat #Convert to array if it's a list
	
	row_sum = np.sum(nmat,1) #Generate row sum for the count matrix

	s = nmat.shape[0] #Size of state space
	pmat = [] #Initialize pmat as a list
	for idx, row in enumerate(nmat): #Iterate over each row of nmat
		if row_sum[idx] != 0: #If there are transitions out of this state in count mat
			pmat.append((row/row_sum[idx]).tolist()) #Append the normalized transitional probability
		else:
			pmat.append([0]*s) #Append the zero vector (size = s)

	# pmat = pmat if isinstance(pmat, (list, tuple)) else pmat.tolist() #Convert to list
	return pmat #Return a list

def uniform_prior_mat(m, s, alpha):
	# Uniform prior = a matrix with equal values equal to alpha/(m*s^2)
	return (np.ones((s,s))*alpha/(m*s**2)).tolist()

def f1_comp(cluster_ls, prior_ls, mode='log'):
	# Compute f(S, C)
	count_ls, m_ls = cluster_ls2count_ls(cluster_ls) #Convert cluster_ls to count_ls and m_ls (list of # of time series in a cluster)

	# Convert both prior and count mat to array if they are lists
	count_ls = np.asarray(count_ls)
	prior_ls = np.asarray(prior_ls)
	
	alpha = np.sum(prior_ls) #sum of all values in prior
	alpha_k = np.sum(prior_ls,(1,2)) #Sum of all values in cluster k's prior

	# f1 = gamma(alpha)/gamma(alpha+sum(m_ls))
	# m = sum(m_ls)
	# for k in range(len(count_ls)):
	# 	alpha_k = np.sum(prior_ls[k]) #Current sum of prior values (# of transitions in prior for current cluster)
	# 	m_k = m_ls[k]
	# 	f1 = f1* gamma(alpha_k + m_k)/gamma(alpha_k)
	# return f1
	# print('F1: term 1 = '+ str(gamma(alpha)/gamma(alpha+sum(m_ls))) + ' and term 2 = '+str(np.prod(gamma(alpha_k+m_ls)/gamma(alpha_k))))
	if mode == 'log':
		# print(np.log(gamma(alpha)/gamma(alpha+sum(m_ls))))
		# return gammaln(alpha) - gammaln(alpha+sum(m_ls)) + np.sum(np.log(gamma(alpha_k+m_ls)/gamma(alpha_k)))
		return gammaln(alpha) - gammaln(alpha+sum(m_ls)) + np.sum(gammaln(alpha_k+m_ls) - gammaln(alpha_k))
	else:
		return gamma(alpha)/gamma(alpha+sum(m_ls))  * np.prod(gamma(alpha_k+m_ls)/gamma(alpha_k))

def f2_comp(count_ls, prior_ls, mode='log'):
	# Compute f(S,X_t-1, X_t, C)
	# Input:
	# 	count_ls: List of count matrices
	# 	prior_ls; 
	###########################################################################################
	# Convert both prior and count mat to array if they are lists
	count_ls = np.asarray(count_ls)
	prior_ls = np.asarray(prior_ls)

	if mode == 'log':
		# If the value is too close to 0, we use natural log to compute the value (sum instead of product)
		# term_wo_ijk = np.log(gamma(count_ls+prior_ls)/gamma(prior_ls))
		# print('max is', prior_ls.max(),'and min is',prior_ls.min())
		term_wo_ijk = gammaln(count_ls+prior_ls) - gammaln(prior_ls)

		# term_wo_ik = np.log(gamma(np.sum(prior_ls,2)) / gamma( np.sum(prior_ls,2)+np.sum(count_ls, 2))) + np.sum(term_wo_ijk,2)
		term_wo_ik = gammaln(np.sum(prior_ls,2)) - gammaln( np.sum(prior_ls,2)+np.sum(count_ls, 2)) + np.sum(term_wo_ijk,2)
		
		return np.sum(term_wo_ik)

		# term_wo_ijk = np.prod( gamma(count_ls+prior_ls)/gamma(prior_ls), 2)
		# term_wo_ik = gamma(np.sum(prior_ls,2)) / gamma( np.sum(prior_ls,2)+np.sum(count_ls, 2) ) * term_wo_ijk
		# return np.sum(np.log(term_wo_ik))
	else:
		return np.prod( gamma(np.sum(prior_ls,2)) / gamma( np.sum(prior_ls,2)+np.sum(count_ls, 2) ) * np.prod( gamma(count_ls+prior_ls)/gamma(prior_ls), 2) )

def cluster_ls2count_ls(cluster_ls):
	# Convert cluster list to count list
	# Input:
		# Cluster list is a list where each entry is a cluster, i.e. a list of count matrices
	# Output:
		# Count list is a list where each entry is a count matrices combined from original count matrices
		# m_ls: List of number of MCs in each cluster
	###########################################################################################
	count_ls = []
	m_ls = []
	for cluster in cluster_ls:
		# cluster is a list of count matrices!
		count_ls.append(cluster2count(cluster))
		# count_ls.append(np.sum(np.asarray(cluster), 0).tolist())
		# cluster2count returns a single matrix
		m_ls.append(len(cluster))
	return count_ls, m_ls

def cluster2count(cluster):
	# Conver a list of count matrix to a count matrix
	# Input:
		# cluster: A single cluster, a list of count matrix, each entry is a count matrix
	# Ouput:
		# A single count matrix
	###########################################################################################
	return np.sum(np.asarray(cluster), 0).tolist()

def merge_p(prior_ls, count_ls):
	m = len(count_ls)
	p_est_ls = []
	# Convert both prior and count mat to array if they are lists
	count_ls = np.asarray(count_ls)
	prior_ls = np.asarray(prior_ls)
	for k in range(m):
		# Iterate over each cluster	
		prior = prior_ls[k]
		nmat = count_ls[k]

		s = nmat.shape[0] #Compute s (# of states)
		prior_row_sum = np.sum(prior,1) #alpha_ki
		nmat_row_sum = np.sum(nmat,1) #n_ki
		p_est_ls.append((nmat+prior)/(prior_row_sum+nmat_row_sum).reshape(s,1))
	return p_est_ls

def check_item_in_ls(original_ls, item_ls, p_out=True):
	# Given a list of items, check if all of them are in list
	for idx, item in enumerate(item_ls):
		if item not in original_ls:
			if p_out: #Print error message
				print('The item '+str(item)+' is not in the list and its index is',idx)
			return False
			# raise Exception('The key '+str(key)+' is not in the list')
	return True

def check_dist_rank_keys(dist_rank, *keys):
	# ChEck if a set of keys are in the dist_rank
	key_pair_check = list(zip(*dist_rank))[0] #Get tuples of keys as a list
	key_ls_check = list(set(item for sublist in key_pair_check for item in sublist)) #Flatten the list to a set of keys
	return check_item_in_ls(key_ls_check, keys, False)

def count2id(count_mat, id_dict):
	# Find the id of a count_mat from the id_dict
	count_mat_hash = container_conv(count_mat, tuple)
	return id_dict.inverse[count_mat_hash]

def id2count(id_mat, id_dict):
	# Find the count matrix given the id
	count_mat_hash = id_dict[id_mat]
	return container_conv(count_mat_hash, list)


##########################################################General Mini Functions############################################################
def ls2trip_ls(ls, output_type = tuple):
	# This function extracts odd and even rows in ls (e.g. read for csv) and form them into a list of trips or tuple of tuples 
	# Given a list of list type object, convert it to a trip_ls with output_type:
		# ls: list of list where even entry list is time list, odd entry list is activity list
			# trip_ls is a list of list object, each entry(list) is a two tuple (time, activities)
		# The output_type determines the output object type down to the bot level

	ls = container_conv(ls, output_type) #Convert original list of lists to speicifc output_type
	time = ls[0::2] #Get the even rows
	activities = ls[1::2] #Get the odd rows
	return output_type(output_type(i) for i in zip(time, activities))

def csv_read(file_path, output_type = list):
	# Read a csv file given by file_path and return a container of rows in output_type
	# In the returned container, each entry is a row in the csv file
	with open(file_path, newline = '') as f:
		reader = csv.reader(f)
		data_ls = []
		for row in reader:
			data_ls.append([ast.literal_eval(i) for i in row])
		return container_conv(data_ls, output_type)

def dd():
    return collections.defaultdict(int)

# def ls2tuple(ls):
# 	#Convert list of lists to tuple of tuples
# 	return tuple(tuple(i) for i in ls)

# def tuple2ls(tp):
# 	#Convert tuple of tuples to list of lists
# 	return [list(i) for i in tp]

def container_conv(data, output_type):
	# Given the data and output type, convert data to specific type
	# Current accepting the following conversion:
		# container of container of val ---> list of list of val
		# container of container of val ---> tuple of tuple of val
	# Examples: 
		# trans_mat ---> trans_tuple
		# ((time), (activities)) ---> [[time],[activities]]
	return output_type(output_type(i) for i in data)

def ls2len_dict(ls):
	#Give a list, convert to a dictionary where keys are length of list entry
	#ls: A list of data entries
	len_dict = collections.defaultdict(list)
	for item in ls:
		len_dict[len(item[0])].append(item) #item[0] since item has two parts ((time), (events))
	return len_dict

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

def dict_val2tuple(dict0):
	# Convert dict with list of list values to tuple of tuple values (works for 1 lv dict) for bidict
	return {k: container_conv(v, tuple) for k, v in dict0.items()}

def find_in_lsofls(ls, val):
	# Given a list of lists, find value's index so that ls[i][j] = val
	# val has to be a list or number or str
	if not isinstance(ls[0],list) and not isinstance(ls[0],tuple):
		raise Exception('The input is not a list of lists!')

	for i, ls1 in enumerate(ls):

		if val in ls1:
			return (i, ls1.index(val))

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

def multi_ls2df(multi_col_ls, t_interval):
	#Given a multiple column ls, convert to a df with the given column names based on time
	# Input:

	# Output:
	column_names = col_names_time(t_interval) #Create column names
	return pd.DataFrame(multi_col_ls, index = column_names).transpose()

def col_names_30min():
	# Create a list of strings with 30 mins interval from 0000 - 2400
	start_dec = 0
	column_names = []
	for i in range(48):
		start_time = int(int(start_dec)*100+ (start_dec*60) % 60)
		end_time = int(int(start_dec+0.5)*100+ (start_dec*60+30) % 60)
		column_names.append(f"{start_time:04} - {end_time:04}")
		start_dec += 0.5

	return column_names

def col_names_time(t_interval = 0.5):
	# Create a list of strings from 0000 - 2400 with input time interval 
	# Input:
		# t_interval: Time interval between 2 times, default in hour
	# Output:
		# column_names: A list of strings, each as a 'hhmm - hhmm' string time interval, e.g. '0000 - 0030'

	start_dec = 0 #Start time in decimal representation
	column_names = [] #Columns
	col_tot_no = ceil(24/t_interval) #Number of columns 
	for i in range(col_tot_no):
		start_time = int(int(start_dec)*100+ (start_dec*60) % 60) #Start time in hhmm numerical format
		end_dec = min((i+1)*t_interval, 24) #End time in decimal representation
		end_time = int(int(end_dec)*100+ (end_dec*60) % 60) #End time in hhmm numerical format
		column_names.append(f"{start_time:04} - {end_time:04}") #Convert start and end times to interval hour and append
		start_dec = end_dec #Start time in decimal representation
	
	return column_names

def NHTS():
	NHTS_book = {1: 'Regular home activities (chores, sleep)', 2: 'Work from home (paid)', 3: 'Work', 4: 'Work-related meeting / trip', 
	5: 'Volunteer activities (not paid)', 6: 'Drop off /pick up someone', 7: 'Change type of transportation', 
	8: 'Attend school as a student', 9: 'Attend child care', 10: 'Attend adult care', 11: 'Buy goods (groceries, clothes, appliances, gas)',
	12: 'Buy services (dry cleaners, banking, service a car, pet care)', 13: 'Buy meals (go out for a meal, snack, carry-out)',
	14: 'Other general errands (post office, library)', 15: 'Recreational activities (visit parks, movies, bars, museums)',
	16: 'Exercise (go for a jog, walk, walk the dog, go to the gym)', 17: 'Visit friends or relatives', 18: 'Health care visit (medical, dental, therapy)',
	19: 'Religious or other community activities', 97: 'Something else', 99: 'Nothing',}

	return NHTS_book

def dict2json(file, *data):
	# Save the input data as a list into json file
	json_data = ujson.dumps(data)

	if os.path.exists(file):
		os.remove(file)
		# print('Existing File Removed! ')
	
	f = open(file,"w")
	f.write(json_data)
	f.close()
	# print('File '+file+' Saved!!!')
	
def json2dict(file):
	# Load the input data 
	f = open(file,"r")
	# print(f)
	data = ujson.load(f)
	f.close()
	return data

def unique_ls(list1): 
    # Given a list, return a list of all the unique values in the list
    # Input:
    	# list1: Must be a list
    # intilize a null list 
    unique_list = [] 
    # traverse for all elements 
    for x in list1:
        # check if exists in unique_list or not 
        if x not in unique_list:
        	unique_list.append(x) #Append the list
    return unique_list

def ls_diffs(ls1, ls2, mode = 'both'):
	# Given two lists, find differences between two lists
	if len(ls1)> len(ls2):
		ls_long = ls1
		ls_short = ls2
	elif len(ls1)<=len(ls2):
		ls_long = ls2
		ls_short = ls1
	
	common = [item for item in ls_short if item in ls_long] #Find all the common items (iterate over all items in ls_short)
	unique_short = []
	unique_long = []
	# Find unique values in short list that are not in long list
	for item in ls_short:
		if item not in common:
			unique_short.append(item)

	# Find unique values in long list that are not in long list
	for item in ls_long:
		if item not in common:
			unique_long.append(item)
	
	# Build the unique list based on given mode
	if mode == 'both': #If we want unique values in both list
		return unique_short + unique_long
	elif mode == 'short': #If we only want unique values in the short list
		return unique_short
	elif mode == 'long': #If we only want unique values in the long list
		return unique_long
	else:
		raise Exception('Invalid Mode!!!')

def dict_lvs(dict0, n = 1):
	# Check how deep a dictionary is 
	curr_val = random.choice(list(dict0.values()))
	if isinstance(curr_val, dict):
		return dict_lvs(curr_val ,n+1)
	else:
		return n

def discrete_normal_dist(prior_mat, kernel = 'gaussian'):
	# Given a matrix, perform kernel density approximation on each row and converts back to PMF
	# Input:
		# prior_mat: Input matrix
		# kernel: kernel in kernel density approximation
	# Output:
		# norm_prior_mat: A matrix where each row is a distribution with nonzero entries
	# This function essentially converts discrete dist to continuous dist then back to discrete dist to remove any zero PMFs in original discrete dist.
	prior_mat = np.asarray(prior_mat) if isinstance(prior_mat, (tuple, list)) else prior_mat #Conver to a np ndarray for easy computation
	s = prior_mat.shape[0]
	
	box_interval = np.linspace(1.5,s-0.5,s-1) #Divide (-inf, +inf) into intervals for integrate pdf (prob approximation)
	box_interval = np.insert(box_interval,0,float('-inf')) #Add -inf at the end
	box_interval = np.append(box_interval,float('inf')) #Add inf at the end
	norm_prior_mat = np.empty((0,s)) #Prior matrix initialization
	# state_space = np.linspace(1,s,s, dtype = int).reshape(-1,1)
	# max_moment = 2*s-1 #Max moment for the linear equation
	# C =[]
	for i, row in enumerate(prior_mat):
		if sum(row) == 0: #If there are no transitions recorded, we will use a uniform dist instead
			pmf = np.ones([1,s])/s #pmf would be uniform
		else:
			if np.nonzero(row)[0].shape[0]: #If there are only 1 state available and has n transitions, we will use a unimodal normal dist with std=2*s for each point (n point in tot)
				n = sum(row)
				std = 2*s/(sqrt(n)) #sqrt(((s/2)**2)/n)
				kde = scipy.stats.norm(loc = np.nonzero(row)[0][0]+1,scale = std) #A normal dist fit
				f_cdf = lambda x, y: kde.cdf(y)-kde.cdf(x) #Probability func between x and y
			else:
				row_data = np.asarray([i for idx, count in enumerate(row) for i in [(idx+1)]*count]) #Generate the overall data
				kde = scipy.stats.gaussian_kde(row_data) 
				f_cdf = kde.integrate_box_1d #Generate the prob fun between two values
			pmf = np.asarray([f_cdf(box_interval[i],box_interval[i+1]) for i in range(len(box_interval)-1)]).reshape(1,s) #Integrate to get the cdf -> pmf for the interval
		norm_prior_mat = np.append(norm_prior_mat, pmf, axis = 0) #Append the pmf to the existing matrix 
		
	# The following code uses algorithm from https://www.jstor.org/stable/2631060
	# 	moment_row = scipy.stats.moment(row_data,list(range(max_moment+1)))
	# 	a = np.zeros([s,s]) #Coeffficients of linear system
	# 	a = np.stack([moment_row[i:i+s] for i in range(s)], axis = 0)
	# 	b = np.array(-moment_row[s:max_moment+1]) #Intercept of linear system
	# 	C.append(np.linalg.solve(a,b))

		# kde = KernelDensity(kernel, bandwidth=0.2).fit(row) #Perform kenel density estimation on discrete data
	return norm_prior_mat


##########################################################Plot Functions############################################################
def raw_plot(raw_trip_ls = []):
	# Plot raw trip list as points. This function returns a set of axes object that will be reused to plot 
	# Input:
	# 	raw_trip_ls
	fig, (ax1, ax2) = axs_raw()
	for ind_trip in raw_trip_ls:
		sc1 = ax1.scatter(ind_trip[0],ind_trip[1])
		sc2 = ax2.scatter(ind_trip[0],ind_trip[1])
	# fig.savefig('Test')
	return fig, (ax1, ax2)

def axs_raw():
	#Generate raw axes
	ytick_dict = {1: 'Regular home activities (chores, sleep)', 2: 'Work from home (paid)', 3: 'Work', 4: 'Work-related meeting / trip', 5: 'Volunteer activities (not paid)', 6: 'Drop off /pick up someone', 7: 'Change type of transportation', 8: 'Attend school as a student', 9: 'Attend child care', 10: 'Attend adult care', 11: 'Buy goods (groceries, clothes, appliances, gas)',	12: 'Buy services (dry cleaners, banking, service a car, pet care)', 13: 'Buy meals (go out for a meal, snack, carry-out)',	14: 'Other general errands (post office, library)', 15: 'Recreational activities (visit parks, movies, bars, museums)',	16: 'Exercise (go for a jog, walk, walk the dog, go to the gym)', 17: 'Visit friends or relatives', 18: 'Health care visit (medical, dental, therapy)',	19: 'Religious or other community activities', 97: 'Something else'}
	ytick = [i for i in range(1,20)] #y tick value
	ytick.append(97) #y tick

	fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20,10), gridspec_kw={'height_ratios': [1, 9]})

	fig.subplots_adjust(left=0.21, right=0.97, bottom=0.1, top=0.95)

	tick_label_rot = 15
	ax2.set_yticks(ytick)
	ax2.set_yticklabels([ytick_dict[i] for i in ytick], rotation = tick_label_rot) #Set the y tick using labels
	ax.set_yticks(ytick)
	ax.set_yticklabels([ytick_dict[i] for i in ytick], rotation = tick_label_rot) #Set the y tick using labels
	
	ax.set_ylim(96.5, 97.5) 
	ax2.set_ylim(0, 21)

	ax.spines['bottom'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax.xaxis.tick_top()
	ax.tick_params(labeltop=False)  # don't put tick labels at the top
	ax2.xaxis.tick_bottom()

	ax.set_xlim([0,25])
	ax2.set_xlim([0,25])
	ax2.set_xticks(range(0,25))

	ax2.set_xlabel('Hour')
	return fig, (ax, ax2)

def ax3d_plot_bar(x,y,z,dx,dy,dz,xlabel, ylabel, label_size = 5):
	fig = plt.figure(figsize=(20, 10))
	ax1 = fig.add_subplot(111, projection='3d')
	tick_label_rot = 30

	ax1.bar3d(x, y, z, dx, dy, dz)
	
	ax1.set_xticks(range(1, len(xlabel)+1))
	ax1.set_xticklabels(xlabel,rotation= tick_label_rot)

	ax1.set_yticks(range(1,len(ylabel)+1))
	ax1.set_yticklabels(ylabel)
	ax1.tick_params(axis='both', which='major', labelsize=label_size)
	# a1.set_xlabel('Time')
	# a1.set_ylabel()
	return fig, ax1

def ax3d_plot_heat(data, xlabel, ylabel,label_size = 8):
	fig, ax1 = plt.subplots(figsize=(20, 10))
	fig.subplots_adjust(left=0.18, right=0.97, bottom=0.01, top=0.99)
	tick_label_rot = 90

	ax1.imshow(data,cmap="YlGn" )
	
	ax1.set_xticks(range(0, len(xlabel)))
	ax1.set_xticklabels(xlabel,rotation= tick_label_rot)

	ax1.set_yticks(range(0,len(ylabel)))
	ax1.set_yticklabels(ylabel)
	ax1.tick_params(axis='both', which='major', labelsize=label_size)

	for i in range(len(ylabel)):
	    for j in range(len(xlabel)):
	        text = ax1.text(j, i, data[i, j],
	                       ha="center", va="center", color="r", fontsize = 7)
	# a1.set_xlabel('Time')
	# a1.set_ylabel()
	return fig, ax1

def vec_plot(trip_ls, axs = [], raw_trip_ls = []):
	#Given a list of trips (a list of lists), plot every list in the list of lists in a graph and return the axes
	# Input:
	# 	trip_ls: A list of list, each individual list is a 2-element list (time vec, event vec). This represents a set individual centers, which will be plotted as a line
	# 	raw_trip_ls: A set of individual journeys, the raw journey, which will be plotted as colored points.
	####################################################################	
	if axs:
		ax = axs[0]
		ax2 = axs[1]
	else:
		fig, (ax, ax2) = raw_plot(raw_trip_ls)

	for ind_trip in trip_ls: #Plot the centers as lines on figure
		ax.plot(ind_trip[0],ind_trip[1],'D-',markersize=10)
		ax2.plot(ind_trip[0],ind_trip[1],'D-',markersize=10)
	return fig
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