from math import *
from datetime import date
import func, collections, pathlib, os, utils, time
###################### Test Packages #######################
import numpy as np
import csv
###################### Input #######################
# current_path = pathlib.Path(os.getcwd()) #Get the current working directory
# raw_trip_file = 'trippub_top2k.csv' #File name of the 2k data
# trip_ls_raw  = func.trip_ls_input(raw_trip_file,'r')
# data_ls = utils.tripls2datals(trip_ls_raw)
# s = 21
# alpha = 100 #Global precision (this val equals to 1/s for alpha_kij)

raw_trip_file = 'SimulatedData/matlab_data.csv' #File name of the matlab file
data_ls = utils.csv_read(raw_trip_file, output_type = tuple)
alpha = 80 #Global precision (this val equals to 1/s for alpha_kij)
s = 5
################### Important Variables #####################
# 	cluster_ls: List of clusters, each entry is a list of count matrices (in list format). Can convert to count_ls.
# 	count_ls: List of count matrices, each entry is a count matrix combined from the corresponding cluster (list of count matrices)
# 	prior_ls: List of prior matrices, each entry is a count matrix (generated from uniform_prior function)
# 	id_dict: ID bidirectional dictionary, key is unique id, value is hased count matrix (tuple format)
# 	dist_dict_mat: Distance dictionary, key is a pair of IDs for count matrices (from id_dict), value is the distance. This dict contains all past values (a repository)
# 	dist_rank: Distance rank for current count_ls, in a tuple format (key_pair, distance) and ascending 
# 	mc_temp: A single cluster, composed of several existing count matrices
###################### Initialization #######################

# last_time = time.time()
count_ls = func.datals2MC(data_ls, alpha, s)[1] #Get the unique count_ls

id_dict, dist_dict_mat = func.KL_distance_input(count_ls) #If distances have not been computed previously 
# id_dict, dist_dict_mat = func.KL_distance_input('dist_dict_Bayesian.json') #If raw distances have been computed previously 
dist_rank = sorted(dist_dict_mat.items(), key = lambda x: x[1]) #Sort the dictionary based on distance - output a tuple of (idx pair, distance)

prior_ls = utils.uniform_prior_ls(count_ls, alpha) #Using the alpha, generate uniform prior with alpha_kij = 1/s for all i,j,k. Each entry in prior_ls is a count matrix (ls of ls)
cluster_ls = [[k] for k in count_ls] #The initial clusters only have 1 count matrix per cluster

p_new = func.posterior_Bayesian(cluster_ls, prior_ls) #Compute the initial posterior
p_old = float('-inf') #Initial old posterior

last_time = time.time() #Timepoint of last step
print('The initial number of clusters is',len(cluster_ls))
###################### Loop #######################
run_no1 = 0 #Debug index for outer loop
while p_new > p_old: #Continue loop if if the previous run generated a better posterior
	p_old = p_new #Replace p_old with p_new, the best posterior from previous run
	idx = 0 #Index for inner loop (dist_rank) - restart for every loop
	###########################Test Variable###########################	
	run_no2 = 0 #Debug index for inner loop
	run_no1 += 1 #Debug index for outer loop
	print('The external loop no.',run_no1, 'and the time for last loop is',time.time()-last_time)
	last_time = time.time() #Timepoint of last step
	########################### Notes for Loop###########################
	# The following loop goes through every possible merging in current dist_rank setting
	# The following loop will produce a best posterior p_new but since dist_rank will be updated with clusters, this p_new may not be the best p_new (thus the outer loop)
	# In the inner loop, we simply loop over all the items in the dist_rank
	# If a merging happens, we would change the clusters (so is the dist_rank)
	########################### Notes for Loop###########################
	print('The dist_rank has length', len(dist_rank))
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

		cluster_ls_temp, mc_temp = func.merge_cluster(cluster_ls, id1, id2) #Merge the cluster in cluster_ls and produce a temporary cluster list and generated mc
		prior_ls_temp = func.merge_count(prior_ls, id1, id2) #Merge two clusters' priors and produce a temporary prior list

		p_temp = func.posterior_Bayesian(cluster_ls_temp, prior_ls_temp) #Compute the temporary posterior using the temp cluster_ls and prior_ls     
		
		if p_temp > p_new: #If the merged clusters (temp) have a higher posterior
			###################### Debug #######################
			# print('The current run_no2 is', run_no2) 

			# print(' The original p_new is',p_new, 'and the new p_new is',p_temp)

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

			id_dict = func.id_modifier(new_count_ls, id_dict) #Update the original id dictionary with the new count_ls
			id_temp = id_dict.inverse[utils.container_conv(new_count_ls[0], tuple)] #Get id for the new cluster
			
			# Compute the distance between newly generated MC (mc_temp) and the cluster_ls (cluster_ls), save it to a temporary dictionary 
			dist_dict_mat_temp = func.calc_MC_distance(mat_ls1 = new_count_ls, mat_ls2 = count_ls, dist_dict_mat = collections.defaultdict(float), id_dict = id_dict) #Generate the dist dict between mc_temp and cluster_ls
			
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
print('--------------Loop Completed--------------')
print('The best p is', p_old, 'and the number of clusters is',len(cluster_ls))
# print('Time of loop',time.time()-last_time)

# # Check if the clustering is conducted correctly
# cluster_dict_raw = func.matlab_data_cluster(data_ls)
# cluster_dict_data = func.result_clustering(cluster_ls)
# utils.dict2json('temp/cluster_dict_raw.json', cluster_dict_raw)
# utils.dict2json('temp/cluster_dict_data.json', cluster_dict_data)
