import func, utils, xlsxwriter, pandas, pathlib, os
###################### Test Packages #######################
###################### Input #######################
#Input for the clustering algorithm:
	# mc_ls: List of lists, each entry is a discrete time Markov chain 
	# s: Cardinality of state space
	# alpha: Global precision
#################### Matlab Data Input ##################
# raw_trip_file = 'SimulatedData/matlab_data.csv' #File name of the matlab file
# mc_ls = utils.csv_read(raw_trip_file, output_type = tuple)
# s = 5
# alpha = 80 #Global precision (this val equals to 1/s for alpha_kij)
# mc_crop_ls = [mc_ls]
# print('MC crop list generated!')
#################### NHTS Data Input ####################
t_interval = 0.5
raw_trip_file = 'trippub_top2k.csv' #File name of the 2k data
complete_trip_file = 'trippub.csv' #File name of the complete data
trip_ls_raw  = func.trip_ls_input(raw_trip_file,'w') #Generate the day trips for dataset
trip_df = func.tripls2df(trip_ls_raw, t_interval) #Convert trips into df where col are time windows
s = 21
alpha = 10 #Global precision (this val equals to 1/s for alpha_kij)
# The following 2 lines apply for LM 
loop_iter = range(0,23) #Max transition number available
loop_iter = range(23,47) #Max transition number available
##### Complete Dataset for Prior Generation #####
trip_df_complete = pandas.read_csv('trip_df_complete.csv')
trip_df_select = trip_df_complete.sample(3000) #Choose 3000 samples (select the size of prior dataset - how much prior info given)
trip_df_prior = trip_df_complete #Use trip_df_complete or trip_df_select
#################################################
# Write to an excel in Parent/Results/Bayesian/Bayesian_Clustering_Results.xlsx
workbook = xlsxwriter.Workbook(str(pathlib.Path(os.getcwd()).parent)+'/Results/Bayesian/Bayesian_Clustering_Results.xlsx')
worksheet = workbook.add_worksheet('General Results')
for i in loop_iter: #Iterate over different number of transitions
	# mc_len = 4 #Test mc_len value
	mc_len = i+1 #Number of transitions in desired Markov chains
	mc_crop_ls = func.tripdf2mcls(trip_df, mc_len) #Convert trip df to mc lists of list using number of transitions
	mc_crop_ls_prior = func.tripdf2mcls(trip_df_prior, mc_len) #Convert the complete trip df to mc lists of list
	print('MC crop list generated for mc_len=',mc_len,'!')
	###################### Input #######################
	set_no = len(mc_crop_ls) #Total number of cluster sets for one day
	cluster_len_ls = [] #List of cluster length
	for idx, mc_ls in enumerate(mc_crop_ls): #Iterate over all the time windows
		mc_ls = mc_crop_ls[idx] #Choose the time window corresponding to idx
		# Each time window contains a list of MCs
		print('--------------Clustering Starts for No.'+str(idx+1)+' out of '+str(set_no) +' sets--for MC '+str(mc_len)+'---------')
		# Perform Bayesian clustering (prior using the dataset )
		prior_input_dev = ['dev',mc_crop_ls_prior[idx]] #Form the prior input for dev prior
		cluster_ls = func.bayesian_clustering(mc_ls,alpha, s, prior_input = prior_input_dev)
		# cluster_ls = func.bayesian_clustering(mc_ls,alpha, s)
		cluster_len_ls.append(len(cluster_ls))
		print('The number of clusters is',cluster_len_ls[idx])

		# if cluster_ls_len != 1:
		# 	print('The cluster with meaningful result is No.'+str(idx+1))
		# 	break
	idx_meaningful = [i+1 for i, e in enumerate(cluster_len_ls) if e != 1]
	print('The number of clusters for this division are',cluster_len_ls,'and the index of meaningful clusters are',idx_meaningful)
	# Saves the file
	worksheet.write(i,0,str(mc_len)+' transitions:')
	worksheet.write(i,1,str(cluster_len_ls))
	worksheet.write(i,2,str(idx_meaningful))
###################### Test #######################
workbook.close()