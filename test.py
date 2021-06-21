import numpy as np
import utils, func, pandas, time,xlsxwriter, pathlib, os

last_time = time.time()
s = 21
alpha = 10 #Global precision (this val equals to 1/s for alpha_kij)
mc_len = 2 #Number of transitions
idx = 24 #Time window
t_interval = 0.5 #Default time window
cluster_len_ls = []
# ##################################################
raw_trip_file = 'trippub_top2k.csv' #File name of the 2k data
trip_ls_raw  = func.trip_ls_input(raw_trip_file,'r') #Generate the day trips for dataset
trip_df = func.tripls2df(trip_ls_raw, t_interval)

##### Complete Dataset for Prior Generation #####
####Dataset generation - From start########
# raw_trip_file_complete = 'trippub.csv' #File name of the 2k data
# trip_ls_raw_complete  = func.trip_ls_input(raw_trip_file_complete,'w',save_file = False) #Generate the day trips for the complete dataset
# trip_df = func.tripls2df(trip_ls_raw_complete, t_interval)
# trip_df.to_csv('trip_df_complete.csv', index = False)
####Dataset Read&Selection########
trip_df_complete = pandas.read_csv('trip_df_complete.csv').iloc[trip_df.shape[0]:,] #Only use the rows not belong to test dataset
trip_df_select = trip_df_complete.sample(3000) #Choose 3000 samples (select the size of prior dataset - how much prior info given)
trip_df_prior = trip_df_complete #Use trip_df_complete or trip_df_select


mc_crop_ls, mc_title_ls = func.tripdf2mcls(trip_df, mc_len) #Convert trip df to mc lists of list using number of transitions
mc_crop_ls_prior = func.tripdf2mcls(trip_df_prior, mc_len)[0] #Convert the complete trip df to mc lists of list

set_no = len(mc_crop_ls) #Total number of cluster sets for one day
# ##################################################
# Test workbook writing
workbook = xlsxwriter.Workbook(str(pathlib.Path(os.getcwd()).parent)+'/Results/Bayesian/Bayesian_Clustering_Results_'+os.environ.get('USER')
+'.xlsx')
worksheet_1 = workbook.add_worksheet(str(mc_len)+' Transition') #Added new sheet to record result for current mc_len
last_row_no = 1 # Last row number used for writing in worksheet_1
##################################################

for idx, mc_ls in enumerate(mc_crop_ls): #Iterate over all the time windows
	mc_ls = mc_crop_ls[idx] 
	# Each time window contains a list of MCs
	print('--------------Clustering Starts for No.'+str(idx+1)+' out of '+str(set_no) +' sets--for MC '+str(mc_len)+'---------')
	prior_input_dev = ['dev',mc_crop_ls_prior[idx]] #Generate the prior input for dev prior
	# Perform Bayesian clustering (prior using the dataset )
	cluster_ls, trans_ls = func.bayesian_clustering(mc_ls,alpha, s, prior_input = prior_input_dev)
	# cluster_ls = func.bayesian_clustering(mc_ls,alpha, s)
	cluster_len_ls.append(len(cluster_ls))
	# 	#################################
	# saving to worksheet starts from row 1 (1st row reserved for general result)
	# row_0 = 1+ idx * (s+1) #Starting row number of current saving 
	worksheet_1.write(last_row_no,0,'No. '+str(idx)) #Write current title (time window), at row 1, 23, etc.
	worksheet_1.write_row(last_row_no,1,mc_title_ls[idx]) #Write current title (time window), at row 1, 23, etc.
	if len(cluster_ls)>1: #Only saves trans_ls if the clustering result is meaningful
		trans_ls_zip = list(zip(*trans_ls)) #Convert flat trans_ls to a list in which each entry is a list of row values for all trans matrices
		k0 = 0 #Index for row no of trans_ls_zip - relative row number (reset for each time window)
		for j in range(last_row_no+1,last_row_no+s+1): #Absolute row number in excel
			# j starts from last_row_no+1 (2, 24, etc.) and ends at last_row_no+s+1 (22, 44, etc.)
			# k0 points to each row within trans_ls_zip and k0 increments with j
			for k, row_ls in enumerate(trans_ls_zip[k0]): #Iterate over list of rows for all trans_mat
				worksheet_1.write_row(j,k*(s+1),row_ls) #Write the one row for one matrix and goes to the next matrix
			k0 +=1 #Update the relative index
		last_row_no += s+1 #Update the last_row_no with the new index
	else:
		worksheet_1.write(last_row_no+1,1,'No meaning clustering result generated!')
		last_row_no += 2
# 	#################################
	print('The number of clusters is',cluster_len_ls[idx])
workbook.close()