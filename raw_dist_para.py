import numpy as np
import pandas as pd
from math import *
from mpi4py import MPI
import func, collections, utils, importlib,ujson 
###################################################Input##################################################################
########################Dataset#########################
# Server Dataset
# file_trip  ='/home/users/yiqingd/run/Data/trippub.csv' #Use the server complete dataset
# file_trip  ='/home/users/yiqingd/run/Data/trippub_top2k.csv' #Use the server 2k dataset 
# Local Dataset
# file_trip  ='trippub.csv'
file_trip  = 'trippub_top2k.csv' 
#####################Folder Paths#######################
output_file_path = 'output/dist_dict0.json'
#####################Raw Trips#######################
trip_ls  = func.data_input(file_trip, 'r') #Generate a list of day trips
trip_ls_div = utils.ls_split(trip_ls, p-1) #Divide the list of trips into (n-1) separate lists, each for parallel computing
###################################################Input##################################################################
#Parallel computing parameters
comm = MPI.COMM_WORLD #Get the communicator
my_rank = comm.Get_rank() #Get current process's rank
p = comm.Get_size() #Get the total number of processes

#The parallel computing process
if my_rank != 0: #If the process is not root, compute the distances in this process and send it to others
	#Given the rank of the current process, extract the data for this process, compute distances between this data and others
	data_0 = trip_ls_div[my_rank-1] #The individual list of data for current process
	data_1 = [j for i in trip_ls_div[my_rank-1:] for j in i] #Extract a list of entries start from data_0 (inclusive) to the end
	#Compute the distances among different trips in current list/segment
	dist_dict_curr = func.cal_mutual_dist_para(data_0, data_1)
	comm.send(dist_dict_curr, dest = 0) #Send the distance dictionary to the root process
else: #If this is the root computation - compile all the distances computed from other processes
	dist_dict0 = collections.defaultdict(lambda: collections.defaultdict(int)) #Create empty dictionary 
	for procid in range(1,p): #Lopp over all the other processes
		dist_dict_curr = comm.recv(source = procid) #Receive the message
		dist_dict0 = utils.merge_dict(dist_dict0, dist_dict_curr) 

	f = open(output_file_path, 'w+')
	f.write(ujson.dumps(dist_dict0)) #Save the distance dictionary to a file for future uses
##################################################################
# # Computing the genetic algorithm
# center_dict_all = {} #Create a dictionary of centers
# cjm_score = collections.defaultdict(list) #Create a dictionary of scores, where key is the identifier, item is list of scores (each entry is a score for the generation)

# for t in range(gen_max):
# 	#Assign CJs to current ppl of CJMs and evaluate current population (CJMs) and compute&sore them based on scores
# 	#The following loop is for one CJM - assign  CJs to all the representatives within the current CJM
# 	for key, center_ls in ppl.items(): #center_ls is a CJM (a list of centers/representatives (- CJs) in the CJM), key is identifier in [2, top_num]
# 		#Note that for initial ppl, key is length of individual journey (i in CJM_i).
# 		dist_dict1 = utils.cal_cross_dist(trip_ls, center_ls, dist_dict0) #Compute the distances between all trips and current centers
# 		center_dict_all[key] = func.cjm_assign(trip_ls, center_ls, dist_dict1) #Assign each journey to the closest centers/cluster in CJM_i
# 		score = func.cjm_eval(trip_ls, center_ls, center_dict_all[key], dist_dict1) #Compute score for the current CJM
# 		cjm_score[key].append(score)
# 		# print('Score for CJM '+str(key) + ' at time ' + str(t) + ' is ' + str(score))
# 	cjm_sort_ls = sorted(cjm_score, key = lambda x: cjm_score[x][-1], reverse = 1) #Sort the CJM by their latest score in descending order
# 	print('The best CJM at time ' + str(t) + ' is CJM #' + str(cjm_sort_ls[0]) +' and its score is '+ str(cjm_score[cjm_sort_ls[0]][t]))
# 	ppl = func.ga_CJM(ppl, cjm_score, top_n) #Genetic operations

# # scp utils.py yiqingd@sherlock.stanford.edu:run
# # Final evaluation
# for key, center_ls in ppl.items(): #center_ls is a CJM (a list of centers/CJs in the CJM), key is identifier in [2, top_num]
# 	#Note that for initial ppl, key is length of individual journey (i in CJM_i).
# 	dist_dict1 = utils.cal_cross_dist(trip_ls, center_ls, dist_dict0) #Compute the distances between all trips and current centers
# 	center_dict_all[key] = func.cjm_assign(trip_ls, center_ls, dist_dict1) #Assign each journey to the closest centers/cluster in CJM_i
# 	score = func.cjm_eval(trip_ls, center_ls, center_dict_all[key], dist_dict1) #Compute score for the current CJM
# 	cjm_score[key].append(score)
# cjm_sort_ls = sorted(cjm_score, key = lambda x: cjm_score[x][-1], reverse = 1) #Sort the CJM by their latest score in descending order
# print('The best CJM at time ' + str(t) + ' is CJM #' + str(cjm_sort_ls[0]) +' and its score is '+ str(cjm_score[cjm_sort_ls[0]][t]))
# print('The final CJM is #'+ str(cjm_sort_ls[0]) +': '+str(ppl[cjm_sort_ls[0]]))